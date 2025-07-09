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
    wandb_project: str = f"redq"
    wandb_mode: str = "disabled"

    env_id: str = "Walker2d-v5"
    total_timesteps: int = 1_000_000
    use_vf: bool = True
    num_envs: int = 1
    capture_video: bool = False
    seed: int = 42

    num_q_nets: int = 10
    m_q_samples: int = 2
    g_utd_ratio: int = 20

    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2

    buffer_size: int = 100_000
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

class RandomizedEnsembleQNetwork(nn.Module):
    def __init__(self, env, args):
        super(RandomizedEnsembleQNetwork, self).__init__()
        n_observations = env.single_observation_space.shape[0]
        n_actions = env.single_action_space.shape[0]

        self.num_q_nets = args.num_q_nets
        self.m_q_samples = args.m_q_samples

        self.network = nn.ModuleList([nn.Sequential(
            nn.Linear(n_observations + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ) for _ in range(self.num_q_nets)
        ])

    def forward(self, state, action, random_sample=False):
        x = torch.cat((state, action), dim=1)
        if random_sample:
            sampled_idx = random.sample(range(self.num_q_nets), self.m_q_samples)
            return torch.stack([self.network[i](x) for i in sampled_idx]) # returns shape (m, b, f)
        else:
            return torch.stack([self.network[i](x) for i in range(self.num_q_nets)]) # returns shape (n, b, f)

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

    policy_net = Policy(envs).to(device)

    redq_net = RandomizedEnsembleQNetwork(envs, args).to(device)    


    target_redq_net = RandomizedEnsembleQNetwork(envs, args).to(device)    
    target_redq_net.load_state_dict(redq_net.state_dict())

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    redq_optimizer = optim.Adam(redq_net.parameters(), lr=args.learning_rate)

    # switch to torchrl buffer
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size), 
                      batch_size=args.batch_size)
    
    state, _ = envs.reset(seed=args.seed)
    episodes = 0
    for global_step in range(args.total_timesteps):

        if global_step < args.warmup_timesteps:
            action = envs.action_space.sample()
        else:
            learning_step = global_step - args.warmup_timesteps
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
        rb.extend(obj)
        
        state = next_state

        if global_step >= args.warmup_timesteps:
            utd_q_val = 0.0
            utd_total_loss = 0.0

            for _ in range(args.g_utd_ratio):
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

                    # Clipped Double Q-learning
                    q_values = redq_net(next_state_batch, next_state_actions, random_sample=True)
                    q_values = torch.min(q_values, dim=0)[0]

                    next_state_values =  q_values - args.alpha*next_logp
                    td_target = reward_batch + args.gamma * next_state_values * (1 - done_batch)

                state_action_values = redq_net(state_batch, action_batch)

                # expand td_target to match the shape of state_action_values
                td_targets_expanded = td_target.unsqueeze(0).expand_as(state_action_values)

                # Compute q_loss
                losses = F.mse_loss(state_action_values, td_targets_expanded, reduction='none')
                total_loss = losses.mean(dim=(1,2)).sum()

                utd_q_val += state_action_values.mean().item()
                utd_total_loss += total_loss.item()
                # Optimize q networks
                redq_optimizer.zero_grad()
                total_loss.backward()
                redq_optimizer.step()
               
                soft_update(redq_net, target_redq_net, args)

            writer.add_scalar("losses/avg_q_val", utd_q_val/args.g_utd_ratio, global_step)
            writer.add_scalar("losses/total_q_loss", utd_total_loss/args.g_utd_ratio, global_step)

            # Optimize actor
            action, logp = policy_net.select_action(state_batch)

            redq_val = redq_net(state_batch, action)
            avg_q_val = torch.mean(redq_val, dim=0)

            actor_loss = (args.alpha*logp - avg_q_val).mean()
            
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            policy_optimizer.zero_grad()
            actor_loss.backward()
            #torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            policy_optimizer.step()

    envs.close()
    


