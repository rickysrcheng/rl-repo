import gymnasium as gym
import random, os
import numpy as np

from dataclasses import dataclass
import tyro

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.modules import BatchRenorm1d
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    tracking: bool = False
    wandb_entity: str = ""
    wandb_project: str = f"crossq"
    wandb_mode: str = "disabled"

    env_id: str = "Walker2d-v5"
    num_envs: int = 1
    capture_video: bool = False

    total_timesteps: int = 1_000_000
    critic_hidden_size: int = 1024
    actor_hidden_size: int = 256
    autotune: bool = True
    seed: int = 42

    learning_rate: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    adam_betas: tuple = (0.5, 0.999)
    bn_momentum: float = 0.01 # PyTorch uses (1 - Paper's Value) for momentum
    bn_warmup: int = 100_000

    buffer_size: int = 100_000
    batch_size: int = 256
    warmup_timesteps: int = 10_000

    update_policy_frequency: int = 3

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

class QNetwork(nn.Module):
    def __init__(self, env, args):
        super(QNetwork, self).__init__()
        n_observations = env.observation_space.shape[1]
        n_actions = env.action_space.shape[1]
        hidden_size = args.critic_hidden_size
        momentum = args.bn_momentum
        self.network = nn.Sequential(
            BatchRenorm1d(n_observations + n_actions, momentum=momentum, warmup_steps=args.bn_warmup),
            nn.Linear(n_observations + n_actions, hidden_size),
            BatchRenorm1d(hidden_size, momentum=momentum, warmup_steps=args.bn_warmup),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            BatchRenorm1d(hidden_size, momentum=momentum, warmup_steps=args.bn_warmup),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network(x)

class Policy(nn.Module):
    def __init__(self, env, args):
        super(Policy, self).__init__()
        n_observations = env.observation_space.shape[1]
        n_actions = env.action_space.shape[1]
        hidden_size = args.actor_hidden_size
        momentum = args.bn_momentum
        self.network = nn.Sequential(
            BatchRenorm1d(n_observations, momentum=momentum, warmup_steps=args.bn_warmup),
            nn.Linear(n_observations, hidden_size),
            BatchRenorm1d(hidden_size, momentum=momentum, warmup_steps=args.bn_warmup),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            BatchRenorm1d(hidden_size, momentum=momentum, warmup_steps=args.bn_warmup),
            nn.ReLU()
        )

        self.mean_head = nn.Linear(hidden_size, n_actions)
        self.logstd_head = nn.Linear(hidden_size, n_actions)

        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high[0] - env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high[0] + env.action_space.low[0]) / 2.0, dtype=torch.float32)
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

def tuple_to_tensordict(state, action, reward, next_state, done, n_envs=None):
    return TensorDict({
        "state": torch.tensor(state, dtype=torch.float32),
        "action": torch.tensor(action),
        "reward": torch.tensor(reward, dtype=torch.float32),
        "next_state": torch.tensor(next_state, dtype=torch.float32),
        "done": torch.tensor(done, dtype=torch.float32)
    }, batch_size=[n_envs] if n_envs else [])

if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
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

    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.action_space.shape[1]).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.learning_rate)
    else:
        alpha = args.alpha

    policy_net = Policy(envs, args).to(device)
    policy_net.eval()

    q_net1 = QNetwork(envs, args).to(device)  
    q_net2 = QNetwork(envs, args).to(device)
    q_net1.eval()
    q_net2.eval()   

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate, betas=args.adam_betas)
    q1_optimizer = optim.Adam(q_net1.parameters(), lr=args.learning_rate, betas=args.adam_betas)
    q2_optimizer = optim.Adam(q_net2.parameters(), lr=args.learning_rate, betas=args.adam_betas)

    # switch to torchrl buffer
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size), 
                      batch_size=args.batch_size)
    
    state, _ = envs.reset(seed=args.seed)
    episode_return = 0
    episode_length = 0
    episodes = 0
    for global_step in range(args.total_timesteps):

        if global_step < args.warmup_timesteps:
            action = envs.action_space.sample()
        else:
            learning_step = global_step - args.warmup_timesteps
            action, logp = policy_net.select_action(torch.tensor(state,dtype=torch.float32, device=device))
            action = action.cpu().detach().numpy()

        next_state, reward, terminated, truncated, infos = envs.step(action)
        done = np.logical_or(terminated, truncated)

        if "episode" in infos:
            for idx in range(args.num_envs):
                if infos["_episode"][idx]:
                    #print(f"global_step={global_step}, episodic_return={infos['episode']['r'][idx]}")
                    writer.add_scalar("metric/episodic_return", infos["episode"]["r"][idx], global_step)
                    writer.add_scalar("metric/episodic_length", infos["episode"]["l"][idx], global_step)

        obj = tuple_to_tensordict(state, action, reward, next_state, done, args.num_envs)
        rb.extend(obj)
        state = next_state
        if global_step >= args.warmup_timesteps:
            batch = rb.sample()

            state_batch = batch['state'].to(device)
            next_state_batch = batch['next_state'].to(device)
            action_batch = batch['action'].to(device)
            reward_batch = batch['reward'].unsqueeze(1).to(device)
            done_batch = batch['done'].unsqueeze(1).to(device)

            # Compute TD Target
            with torch.no_grad():
                # Let Policy net choose actions
                next_state_actions, next_logp = policy_net.select_action(next_state_batch)

            combined_state_batch = torch.concat((state_batch, next_state_batch), dim=0)
            combined_action_batch = torch.concat((action_batch, next_state_actions), dim=0)

            # Combined Clipped Double Q-learning
            q_net1.train()
            q_net2.train()
            combined_action_values1 = q_net1(combined_state_batch, combined_action_batch)
            combined_action_values2 = q_net2(combined_state_batch, combined_action_batch)
            q_net1.eval()
            q_net2.eval()   

            # Extract current q values and next Q-values
            q_values1, next_q_values1 = torch.chunk(combined_action_values1, 2, dim=0)
            q_values2, next_q_values2 = torch.chunk(combined_action_values2, 2, dim=0)

            # calculate TD Target
            next_q_values = torch.min(next_q_values1, next_q_values2)
            next_state_values =  next_q_values - alpha*next_logp


            td_target = reward_batch + args.gamma * next_state_values * (1 - done_batch) 
            td_target = td_target.detach()           
            # Compute q_loss
            q1_loss = F.mse_loss(q_values1, td_target)
            q2_loss = F.mse_loss(q_values2, td_target)

            avg_loss = (q1_loss + q2_loss)/2

            writer.add_scalar("losses/q1_val", q_values1.mean().item(), global_step)
            writer.add_scalar("losses/q1_loss", q1_loss.item(), global_step)
            writer.add_scalar("losses/q2_val", q_values2.mean().item(), global_step)
            writer.add_scalar("losses/q2_loss", q2_loss.item(), global_step)
            writer.add_scalar("losses/avg_q_loss", avg_loss, global_step)

            # Optimize q networks
            q1_optimizer.zero_grad()
            q1_loss.backward()
            q1_optimizer.step()
            q2_optimizer.zero_grad()
            q2_loss.backward()
            q2_optimizer.step()

            # Policy Delay
            if global_step % args.update_policy_frequency == 0:
                # Optimize actor
                policy_net.train()
                action, logp = policy_net.select_action(state_batch)
                policy_net.eval()

                min_q = torch.min(q_net1(state_batch, action), q_net2(state_batch, action))

                actor_loss = (alpha*logp - min_q).mean()
                
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                policy_optimizer.zero_grad()
                actor_loss.backward()
                #torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                policy_optimizer.step()
                if args.autotune:
                    with torch.no_grad():
                        action, logp = policy_net.select_action(state_batch)
                    alpha_loss = (-log_alpha.exp() * (logp + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()


        # if done:
        #     state, _ = env.reset()
        #     writer.add_scalar("metric/episodic_return", episode_return, global_step)
        #     writer.add_scalar("metric/episodic_length", episode_length, global_step)
        #     writer.add_scalar("metric/episodes", episodes, global_step)

        #     episode_return = 0
        #     episode_length = 0
        #     episodes += 1
    envs.close()
    


