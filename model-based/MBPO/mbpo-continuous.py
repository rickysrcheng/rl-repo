import gymnasium as gym
import math, random, os
import numpy as np

from dataclasses import dataclass
import tyro

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, AffineTransform, TransformedDistribution

from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict, merge_tensordicts
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    tracking: bool = False
    wandb_entity: str = ""
    wandb_project: str = f"mbpo"
    wandb_mode: str = "disabled"

    env_id: str = "Walker2d-v5"
    total_timesteps: int = 1_000_000
    use_vf: bool = True
    num_envs: int = 1
    capture_video: bool = False
    seed: int = 42

    # MBPO hyperparameters
    num_ensembles: int = 7
    hidden_dim: int = 200
    n_epochs: int = 300
    e_envsteps: int = 1000
    m_rollouts: int = 400
    g_updates: int = 20

    k_start: int = 1
    k_end: int = 1
    k_epoch_start: int = 20
    k_epoch_end: int = 100

    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2

    env_buffer_size: int = 1_000_000
    model_buffer_size: int = 400_000
    batch_size: int = 256
    warmup_timesteps: int = 5_000

    eps_begin: float = 0.4
    eps_end: float = 0.1
    eps_decay_steps: int = 500_000

    update_policy_frequency: int = 2

    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

class DynamicsNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DynamicsNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Predict delta states, not absolute next states
        self.delta_state_mean_head = nn.Linear(hidden_dim, output_dim)
        self.delta_state_logstd_head = nn.Linear(hidden_dim, output_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.network(x)
        delta_mean = self.delta_state_mean_head(x)
        logstd = self.delta_state_logstd_head(x)
        reward = self.reward_head(x)
        done_logit = self.done_head(x)

        return delta_mean, logstd, reward, done_logit

class EnsembleDynamicsNetwork(nn.Module):
    def __init__(self, env, args):
        super(EnsembleDynamicsNetwork, self).__init__()
        n_observations = env.single_observation_space.shape[0]
        n_actions = env.single_action_space.shape[0]
        self.num_ensembles = args.num_ensembles
        self.n_observations = n_observations
        
        # Store normalization statistics
        self.register_buffer("state_mean", torch.zeros(n_observations))
        self.register_buffer("state_std", torch.ones(n_observations))
        self.register_buffer("action_mean", torch.zeros(n_actions))
        self.register_buffer("action_std", torch.ones(n_actions))
        self.register_buffer("delta_mean", torch.zeros(n_observations))
        self.register_buffer("delta_std", torch.ones(n_observations))
        
        self.network = nn.ModuleList([
            DynamicsNetwork(n_observations + n_actions, args.hidden_dim, n_observations) 
            for _ in range(args.num_ensembles)
        ])
    
    def update_normalizers(self, states, actions, next_states):
        with torch.no_grad():
            deltas = next_states - states
            
            self.state_mean = states.mean(0)
            self.state_std = states.std(0) + 1e-8
              
            self.action_mean = actions.mean(0)
            self.action_std = actions.std(0) + 1e-8
            
            self.delta_mean = deltas.mean(0)
            self.delta_std = deltas.std(0) + 1e-8
    
    def normalize_inputs(self, states, actions):
        norm_states = (states - self.state_mean) / self.state_std
        norm_actions = (actions - self.action_mean) / self.action_std
        return norm_states, norm_actions
    
    def denormalize_deltas(self, norm_deltas):
        return norm_deltas * self.delta_std + self.delta_mean
    
    def forward(self, states, actions):
        LOG_STD_MIN = -10 
        LOG_STD_MAX = 0.5
        
        # normalize inputs first
        norm_states, norm_actions = self.normalize_inputs(states, actions)
        x = torch.cat((norm_states, norm_actions), dim=1)
        
        means, logstds, rewards, done_logits = [], [], [], []

        for model in self.network:
            delta_mean, logstd, reward, done_logit = model(x)
            means.append(delta_mean)
            logstds.append(logstd)
            rewards.append(reward)
            done_logits.append(done_logit)
        
        means = torch.stack(means, dim=0)
        logstds = torch.stack(logstds, dim=0)
        rewards = torch.stack(rewards, dim=0)
        done_logits = torch.stack(done_logits, dim=0)

        logstds = torch.clamp(logstds, LOG_STD_MIN, LOG_STD_MAX)
        
        denorm_means = self.denormalize_deltas(means)
        denorm_stds = logstds.exp() * self.delta_std.unsqueeze(0)
        
        return denorm_means, denorm_stds, rewards, done_logits

    def sample_from_one_model(self, states, actions, model_idx):
        norm_states, norm_actions = self.normalize_inputs(states, actions)
        x = torch.cat((norm_states, norm_actions), dim=1)
        
        delta_mean, logstd, reward, done_logit = self.network[model_idx](x)
        
        denorm_delta_mean = self.denormalize_deltas(delta_mean)
        denorm_std = logstd.exp() * self.delta_std
        
        dist = Normal(denorm_delta_mean, denorm_std)
        delta = dist.rsample()
        
        next_state = states + delta

        done_prob = torch.sigmoid(done_logit)
        done = torch.bernoulli(done_prob)
        return next_state, reward, done

class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        n_observations = env.single_observation_space.shape[0]
        n_actions = env.single_action_space.shape[0]
        self.network = nn.Sequential(
            nn.Linear(n_observations + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network(x)

class Policy(nn.Module):
    def __init__(self, env):
        super(Policy, self).__init__()
        n_observations = env.single_observation_space.shape[0]
        n_actions = env.single_action_space.shape[0]
        self.network = nn.Sequential(
            nn.Linear(n_observations, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU()
        )

        self.mean_head = nn.Linear(256, n_actions)
        self.logstd_head = nn.Linear(256, n_actions)

        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        LOG_STD_MIN = -5
        LOG_STD_MAX = 2
        x = self.network(x)
        mean = self.mean_head(x)
        logstd = self.logstd_head(x)
        
        logstd = torch.tanh(logstd)
        logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logstd + 1)
        return mean, logstd
    
    def select_action(self, x):
        mean, logstd = self(x)
        std = logstd.exp()
        normal = Normal(mean, std)
        u = normal.rsample()
        u_tanh = torch.tanh(u)
        action = u_tanh * self.action_scale + self.action_bias
        log_prob = normal.log_prob(u)
        log_prob -= torch.log(self.action_scale * (1 - u_tanh.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


def soft_update(net, target_net, args):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

def tuple_to_tensordict(state, action, reward, next_state, done, n_envs=None):
    return TensorDict({
        "state": torch.tensor(state, dtype=torch.float32),
        "action": torch.tensor(action),
        "reward": torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
        "next_state": torch.tensor(next_state, dtype=torch.float32),
        "done": torch.tensor(done, dtype=torch.float32).unsqueeze(0)
    }, batch_size=[n_envs] if n_envs else [1])

def tensor_to_tensordict(state, action, reward, next_state, done, n_batch=None):
    return TensorDict({
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "done": done
    }, batch_size=[n_batch] if n_batch else [1])

if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
    )

    if args.tracking:
        import wandb
        run = wandb.init(
            entity=args.wandb_entity,
            project=f"{args.wandb_project}-{args.env_id}",
            config=vars(args),
            mode=args.wandb_mode,
            sync_tensorboard=True
        )
    
    writer = SummaryWriter(f"runs/{args.env_id}_{int(time.time())}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    
    dynamics_net = EnsembleDynamicsNetwork(envs, args).to(device)
    policy_net = Policy(envs).to(device)

    q_net1 = QNetwork(envs).to(device)    
    q_net2 = QNetwork(envs).to(device)

    target_q_net1 = QNetwork(envs).to(device)    
    target_q_net2 = QNetwork(envs).to(device)
    target_q_net1.load_state_dict(q_net1.state_dict())
    target_q_net2.load_state_dict(q_net2.state_dict())

    dynamics_optimizer = optim.Adam(dynamics_net.parameters(), lr=args.learning_rate)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    q1_optimizer = optim.Adam(q_net1.parameters(), lr=args.learning_rate)
    q2_optimizer = optim.Adam(q_net2.parameters(), lr=args.learning_rate)

    rb_env = ReplayBuffer(storage=LazyTensorStorage(args.env_buffer_size))
    rb_model = ReplayBuffer(storage=LazyTensorStorage(args.model_buffer_size), 
                      batch_size=math.floor(0.95*args.batch_size))
    
    state, _ = envs.reset(seed=args.seed)
    episodes = 0
    global_step = 0
    
    # FIXED: Collect more warmup data for better normalization
    warmup_states, warmup_actions, warmup_next_states = [], [], []
    
    for warmup in range(args.warmup_timesteps):
        action = envs.action_space.sample()
        next_state, reward, terminated, truncated, infos = envs.step(action)

        real_next_state = next_state.copy()
        if "_final_info" in infos:
            for i in range(envs.num_envs):
                if infos["_final_info"][i]:
                    episodes += 1
                    real_next_state[i] = infos["final_obs"][i]
                    writer.add_scalar("metric/episodic_return", infos['final_info']['episode']['r'][i], global_step)
                    writer.add_scalar("metric/episodic_length", infos['final_info']['episode']['l'][i], global_step)
                    writer.add_scalar("metric/episodes", episodes, global_step)
        
        # Collect data for normalization
        warmup_states.append(state.copy())
        warmup_actions.append(action.copy())
        warmup_next_states.append(real_next_state.copy())
            
        obj = tuple_to_tensordict(state, action, reward, real_next_state, terminated, args.num_envs)
        rb_env.extend(obj)
        
        state = next_state
        global_step += 1
    
    # Initialize normalizers with warmup data
    warmup_states = torch.tensor(np.vstack(warmup_states), dtype=torch.float32).to(device)
    warmup_actions = torch.tensor(np.vstack(warmup_actions), dtype=torch.float32).to(device)
    warmup_next_states = torch.tensor(np.vstack(warmup_next_states), dtype=torch.float32).to(device)
    dynamics_net.update_normalizers(warmup_states, warmup_actions, warmup_next_states)

    for epochs in range(args.n_epochs):
        # Train dynamics model with proper delta prediction
        for env_training in range(10):
            env_batch = rb_env.sample(batch_size=args.batch_size)

            state_env_batch = env_batch['state'].to(device)
            next_state_env_batch = env_batch['next_state'].to(device)
            action_env_batch = env_batch['action'].to(device)
            reward_env_batch = env_batch['reward'].to(device)
            done_env_batch = env_batch['done'].to(device)

            delta_targets = next_state_env_batch - state_env_batch
            
            b_delta_mean, b_std, b_reward, b_done_logits = dynamics_net(state_env_batch, action_env_batch)
            
            delta_targets_exp = delta_targets.unsqueeze(0).expand_as(b_delta_mean)
            model_loss = ((b_delta_mean - delta_targets_exp) ** 2 / (2 * b_std ** 2) + 
                         torch.log(b_std)).mean()
            
            done_env_batch_exp = done_env_batch.unsqueeze(0).expand_as(b_done_logits)
            reward_env_batch_exp = reward_env_batch.unsqueeze(0).expand_as(b_reward)
            
            reward_loss = F.mse_loss(b_reward, reward_env_batch_exp, reduction='mean')
            done_loss = F.binary_cross_entropy_with_logits(b_done_logits, done_env_batch_exp)

            total_model_loss = model_loss + reward_loss + done_loss

            dynamics_optimizer.zero_grad()
            total_model_loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(dynamics_net.parameters(), 1.0)
            dynamics_optimizer.step()

        writer.add_scalar("losses/model_loss", model_loss.mean().item(), epochs)
        writer.add_scalar("losses/reward_loss", reward_loss.mean().item(), epochs)
        writer.add_scalar("losses/done_loss", done_loss.mean().item(), epochs)
        writer.add_scalar("losses/total_model_loss", total_model_loss.mean().item(), epochs)

        # Better k scheduling
        if epochs < args.k_epoch_start:
            k_end = args.k_start
        elif epochs >= args.k_epoch_end:
            k_end = args.k_end
        else:
            progress = (epochs - args.k_epoch_start) / (args.k_epoch_end - args.k_epoch_start)
            k_end = int(args.k_start + progress * (args.k_end - args.k_start))

        for e_steps in range(args.e_envsteps):
            action, logp = policy_net.select_action(torch.tensor(state,dtype=torch.float32, device=device))
            action = action.cpu().detach().numpy()

            next_state, reward, terminated, truncated, infos = envs.step(action)

            real_next_state = next_state.copy()
            if "_final_info" in infos:
                for i in range(envs.num_envs):
                    if infos["_final_info"][i]:
                        episodes += 1
                        real_next_state[i] = infos["final_obs"][i]
                        writer.add_scalar("metric/episodic_return", infos['final_info']['episode']['r'][i], global_step)
                        writer.add_scalar("metric/episodic_length", infos['final_info']['episode']['l'][i], global_step)
                        writer.add_scalar("metric/episodes", episodes, global_step)
                
            obj = tuple_to_tensordict(state, action, reward, real_next_state, terminated, args.num_envs)
            rb_env.extend(obj)
            state = next_state

            with torch.no_grad():
                for m in range(args.m_rollouts):
                    m_sample = rb_env.sample(batch_size=1)
                    m_state = m_sample['state'].to(device)

                    model_idx = np.random.randint(args.num_ensembles)
                    for k in range(k_end):
                        m_action, m_logp = policy_net.select_action(m_state)

                        m_next_states, m_rewards, m_done = dynamics_net.sample_from_one_model(m_state, m_action, model_idx)
                        m_obj = tensor_to_tensordict(m_state, m_action, m_rewards, m_next_states, m_done)
                        rb_model.extend(m_obj)
                        
                        # Early termination if done
                        if m_done.item() > 0.5:
                            break
                            
                        m_state = m_next_states

            for g_updates in range(args.g_updates):
                env_batch = rb_env.sample(batch_size=math.ceil(0.05*args.batch_size))
                model_batch = rb_model.sample()
                g_batch = torch.cat([env_batch, model_batch], dim=0)

                state_model_batch = g_batch['state'].to(device)
                next_state_model_batch = g_batch['next_state'].to(device)
                action_model_batch = g_batch['action'].to(device)
                reward_model_batch = g_batch['reward'].to(device)
                done_model_batch = g_batch['done'].to(device)

                with torch.no_grad():
                    next_state_actions, next_logp = policy_net.select_action(next_state_model_batch)
                    q_values1 = target_q_net1(next_state_model_batch, next_state_actions)
                    q_values2 = target_q_net2(next_state_model_batch, next_state_actions)
                    q_values = torch.min(q_values1, q_values2)
                    next_state_values =  q_values - args.alpha*next_logp
                    td_target = reward_model_batch + args.gamma * next_state_values * (1 - done_model_batch)

                state_action_values1 = q_net1(state_model_batch, action_model_batch)
                state_action_values2 = q_net2(state_model_batch, action_model_batch)

                q1_loss = F.mse_loss(state_action_values1, td_target)
                q2_loss = F.mse_loss(state_action_values2, td_target)
                avg_loss = (q1_loss + q2_loss)/2

                q1_optimizer.zero_grad()
                q1_loss.backward()
                q1_optimizer.step()
                q2_optimizer.zero_grad()
                q2_loss.backward()
                q2_optimizer.step()

                action, logp = policy_net.select_action(state_model_batch)
                min_q = torch.min(q_net1(state_model_batch, action), q_net2(state_model_batch, action))
                actor_loss = (args.alpha*logp - min_q).mean()
                
                policy_optimizer.zero_grad()
                actor_loss.backward()
                policy_optimizer.step()
            
                soft_update(q_net1, target_q_net1, args)
                soft_update(q_net2, target_q_net2, args)

            writer.add_scalar("losses/q1_val", state_action_values1.mean().item(), global_step)
            writer.add_scalar("losses/q1_loss", q1_loss.item(), global_step)
            writer.add_scalar("losses/q2_val", state_action_values2.mean().item(), global_step)
            writer.add_scalar("losses/q2_loss", q2_loss.item(), global_step)
            writer.add_scalar("losses/avg_q_loss", avg_loss, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            global_step += 1
    envs.close()