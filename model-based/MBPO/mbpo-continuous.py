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
from tensordict import TensorDict
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

    env_buffer_size: int = 100_000
    model_buffer_size: int = 20_000
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

        self.next_state_mean_head = nn.Linear(hidden_dim, output_dim)
        self.next_state_logstd_head = nn.Linear(hidden_dim, output_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.network(x)
        mean = self.next_state_mean_head(x)
        logstd = self.next_state_logstd_head(x)
        reward = self.reward_head(x)
        return mean, logstd, reward

# simple looped version, could use batched mm
class EnsembleDynamicsNetwork(nn.Module):
    def __init__(self, env, args):
        super(EnsembleDynamicsNetwork, self).__init__()
        n_observations = env.single_observation_space.shape[0]
        n_actions = env.single_action_space.shape[0]
        self.num_ensembles = args.num_ensembles
        self.network = nn.ModuleList([
            DynamicsNetwork(n_observations + n_actions, args.hidden_dim, n_observations) 
            for _ in range(args.num_ensembles)
        ])
    
    def forward(self, states, actions):
        LOG_STD_MIN = -5
        LOG_STD_MAX = 2
        x = torch.cat((states, actions), dim=1)
        means, logstds, rewards = [], [], []

        for model in self.network:
            mean, logstd, reward = model(x)
            means.append(mean)
            logstds.append(logstd)
            rewards.append(reward)
        
        means = torch.stack(means, dim=0)
        logstds = torch.stack(logstds, dim=0)
        rewards = torch.stack(rewards, dim=0)

        logstds = torch.tanh(logstds)
        logstds = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logstds + 1)
        return means, logstds, rewards

    def sample_from_one_model(self, states, actions, model_idx):
        x = torch.cat((states, actions), dim=1)
        mean, logstd, reward = self.network[model_idx](x)
        std = logstd.exp()
        dist = Normal(mean, std)
        next_state = dist.rsample()
        return next_state, reward

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
        
        #logstd = logstd.clamp(-20, 2)

        # not sure why cleanrl does this, but it seems to be a way to bound the logstd?
        logstd = torch.tanh(logstd)
        logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logstd + 1)
        return mean, logstd
    
    def select_action(self, x):
        mean, logstd = self(x)
        std = logstd.exp()
        # create squashed Gaussian distribution
        normal = Normal(mean, std)
        u = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        u_tanh = torch.tanh(u)
        action = u_tanh * self.action_scale + self.action_bias
        log_prob = normal.log_prob(u)
        # Enforcing Action Bound
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
        "reward": torch.tensor(reward, dtype=torch.float32),
        "next_state": torch.tensor(next_state, dtype=torch.float32),
        "done": torch.tensor(done, dtype=torch.float32)
    }, batch_size=[n_envs] if n_envs else [])

def tensor_to_tensordict(state, action, reward, next_state, n_batch=None):
    return TensorDict({
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state
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
    
    # really learned a lot on logging practices from cleanrl
    writer = SummaryWriter(f"runs/{args.env_id}_{int(time.time())}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # if GPU is to be used
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

    # switch to torchrl buffer
    rb_env = ReplayBuffer(storage=LazyTensorStorage(args.env_buffer_size), 
                      batch_size=args.batch_size)
    rb_model = ReplayBuffer(storage=LazyTensorStorage(args.model_buffer_size), 
                      batch_size=args.batch_size)
    
    state, _ = envs.reset(seed=args.seed)
    episodes = 0
    global_step = 0
    for warmup in range(args.warmup_timesteps):
        action = envs.action_space.sample()
        next_state, reward, terminated, truncated, infos = envs.step(action)

        real_next_state = next_state.copy()
        if "_final_info" in infos:
            for i in range(envs.num_envs):
                if infos["_final_info"][i]: # this env has finished this step
                    episodes += 1
                    real_next_state[i] = infos["final_obs"][i]
                    writer.add_scalar("metric/episodic_return", infos['final_info']['episode']['r'][i], global_step)
                    writer.add_scalar("metric/episodic_length", infos['final_info']['episode']['l'][i], global_step)
                    writer.add_scalar("metric/episodes", episodes, global_step)
            
        # if truncated, use real_next_state
        obj = tuple_to_tensordict(state, action, reward, real_next_state, terminated, args.num_envs)
        rb_env.extend(obj)
        # bootstrap model buffers with real data. It'll be cycled out pretty fast
        rb_model.extend(obj.exclude("done"))
        
        state = next_state
        global_step += 1

    for epochs in range(args.n_epochs):
        # first train dynamics model
        env_batch = rb_env.sample()

        state_env_batch = env_batch['state'].to(device)
        next_state_env_batch = env_batch['next_state'].to(device)
        action_env_batch = env_batch['action'].to(device)
        reward_env_batch = env_batch['reward'].unsqueeze(1).to(device)
        done_env_batch = env_batch['done'].unsqueeze(1).to(device)

        b_mean, b_logstd, b_reward = dynamics_net(state_env_batch, action_env_batch)
        b_std = b_logstd.exp()
        
        next_state_env_batch_exp = next_state_env_batch.unsqueeze(0).expand_as(b_mean)
        model_loss = F.gaussian_nll_loss(b_mean, next_state_env_batch_exp, b_std**2)
        
        reward_env_batch_exp = reward_env_batch.unsqueeze(0).expand_as(b_reward)
        reward_loss = F.mse_loss(b_reward, reward_env_batch_exp, reduction='mean')

        total_model_loss = model_loss + reward_loss

        writer.add_scalar("losses/model_loss", model_loss.mean().item(), epochs)
        writer.add_scalar("losses/reward_loss", reward_loss.mean().item(), epochs)
        writer.add_scalar("losses/total_model_loss", total_model_loss.mean().item(), epochs)

        dynamics_optimizer.zero_grad()
        total_model_loss.backward()
        dynamics_optimizer.step()

        if args.k_end == 1:
            k_end = 1
        else:
            k_end = min(max(args.k_start + (epochs - args.k_epoch_start)/(args.k_epoch_end - args.k_epoch_start)*(args.k_end - args.k_start), args.k_start), args.k_end)

        for e_steps in range(args.e_envsteps):
            # Take a single environment step, add to env_buffer
            action, logp = policy_net.select_action(torch.tensor(state,dtype=torch.float32, device=device))
            action = action.cpu().detach().numpy()

            next_state, reward, terminated, truncated, infos = envs.step(action)

            real_next_state = next_state.copy()
            if "_final_info" in infos:
                for i in range(envs.num_envs):
                    if infos["_final_info"][i]: # this env has finished this step
                        episodes += 1
                        real_next_state[i] = infos["final_obs"][i]
                        writer.add_scalar("metric/episodic_return", infos['final_info']['episode']['r'][i], global_step)
                        writer.add_scalar("metric/episodic_length", infos['final_info']['episode']['l'][i], global_step)
                        writer.add_scalar("metric/episodes", episodes, global_step)
                
            # if truncated, use real_next_state
            obj = tuple_to_tensordict(state, action, reward, real_next_state, terminated, args.num_envs)
            rb_env.extend(obj)
            state = next_state

            # first do naive loop over m rollouts, can probably batch this later
            with torch.no_grad():
                for m in range(args.m_rollouts):
                    m_sample = rb_env.sample(batch_size=1)
                    m_state = m_sample['state'].to(device)

                    model_idx = np.random.randint(args.num_ensembles)
                    for k in range(k_end):
                        m_action, m_logp = policy_net.select_action(m_state)

                        m_next_states, m_rewards = dynamics_net.sample_from_one_model(m_state, m_action, model_idx)
                        m_obj = tensor_to_tensordict(m_state, m_action, m_rewards, m_next_states)
                        rb_model.extend(m_obj)
                        m_state = m_next_states

            # gradient update phase
            for g_updates in range(args.g_updates):
                model_batch = rb_model.sample()

                state_model_batch = model_batch['state'].to(device)
                next_state_model_batch = model_batch['next_state'].to(device)
                action_model_batch = model_batch['action'].to(device)
                reward_model_batch = model_batch['reward'].unsqueeze(1).to(device)

                # Compute TD Target
                with torch.no_grad():
                    # Let Policy net choose actions
                    next_state_actions, next_logp = policy_net.select_action(next_state_model_batch)

                    # Clipped Double Q-learning
                    q_values1 = target_q_net1(next_state_model_batch, next_state_actions)
                    q_values2 = target_q_net2(next_state_model_batch, next_state_actions)
                    q_values = torch.min(q_values1, q_values2)

                    next_state_values =  q_values - args.alpha*next_logp
                    td_target = reward_model_batch + args.gamma * next_state_values

                state_action_values1 = q_net1(state_model_batch, action_model_batch)
                state_action_values2 = q_net2(state_model_batch, action_model_batch)

                # Compute q_loss
                q1_loss = F.mse_loss(state_action_values1, td_target)
                q2_loss = F.mse_loss(state_action_values2, td_target)

                avg_loss = (q1_loss + q2_loss)/2

                # Optimize q networks
                q1_optimizer.zero_grad()
                q1_loss.backward()
                q1_optimizer.step()
                q2_optimizer.zero_grad()
                q2_loss.backward()
                q2_optimizer.step()

                # Optimize actor
                action, logp = policy_net.select_action(state_model_batch)
                min_q = torch.min(q_net1(state_model_batch, action), q_net2(state_model_batch, action))

                actor_loss = (args.alpha*logp - min_q).mean()
                
                policy_optimizer.zero_grad()
                actor_loss.backward()
                #torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                policy_optimizer.step()
            
                # soft update only when updating actor weights
                soft_update(q_net1, target_q_net1, args)
                soft_update(q_net2, target_q_net2, args)

            # record just the last episode
            writer.add_scalar("losses/q1_val", state_action_values1.mean().item(), global_step)
            writer.add_scalar("losses/q1_loss", q1_loss.item(), global_step)
            writer.add_scalar("losses/q2_val", state_action_values2.mean().item(), global_step)
            writer.add_scalar("losses/q2_loss", q2_loss.item(), global_step)
            writer.add_scalar("losses/avg_q_loss", avg_loss, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            global_step += 1
    envs.close()
    


