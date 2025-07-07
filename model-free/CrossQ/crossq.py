import gymnasium as gym
import math, random
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
    tracking: bool = False
    wandb_entity: str = ""
    wandb_project: str = f"crossq"
    wandb_mode: str = "disabled"

    env_id: str = "Walker2d-v5"
    total_timesteps: int = 1_000_000
    use_vf: bool = True
    critic_hidden_size: int = 256
    actor_hidden_size: int = 256
    autotune: bool = True
    seed: int = 42

    learning_rate: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    adam_betas: tuple = (0.5, 0.999)
    bn_momentum: float = 0.01
    bn_warmup: int = 100_000

    buffer_size: int = 100_000
    batch_size: int = 256
    warmup_timesteps: int = 10_000

    update_policy_frequency: int = 3

    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5


class QNetwork(nn.Module):
    def __init__(self, env, args):
        super(QNetwork, self).__init__()
        n_observations = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]
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
        n_observations = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]
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

def tuple_to_tensordict(state, action, reward, next_state, done):
    return TensorDict({
        "state": torch.tensor(state, dtype=torch.float32),
        "action": torch.tensor(action),
        "reward": torch.tensor(reward, dtype=torch.float32),
        "next_state": torch.tensor(next_state, dtype=torch.float32),
        "done": torch.tensor(done, dtype=torch.float32)
    }, batch_size=[])

if __name__ == "__main__":
    args = tyro.cli(Args)
    env = gym.make(args.env_id)
    env.action_space.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape[0]).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.learning_rate)
    else:
        alpha = args.alpha

    policy_net = Policy(env, args).to(device)
    policy_net.eval()

    q_net1 = QNetwork(env, args).to(device)  
    q_net2 = QNetwork(env, args).to(device)
    q_net1.eval()
    q_net2.eval()   

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate, betas=args.adam_betas)
    q1_optimizer = optim.Adam(q_net1.parameters(), lr=args.learning_rate, betas=args.adam_betas)
    q2_optimizer = optim.Adam(q_net2.parameters(), lr=args.learning_rate, betas=args.adam_betas)

    # switch to torchrl buffer
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size), 
                      batch_size=args.batch_size)
    
    state, _ = env.reset()
    episode_return = 0
    episode_length = 0
    episodes = 0
    for global_step in range(args.total_timesteps):

        if global_step < args.warmup_timesteps:
            action = env.action_space.sample()
        else:
            learning_step = global_step - args.warmup_timesteps
            action, logp = policy_net.select_action(torch.tensor(state,dtype=torch.float32, device=device).unsqueeze(0))
            action = action.squeeze(0).cpu().detach().numpy()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_return += reward
        episode_length += 1
        
        obj = tuple_to_tensordict(state, action, reward, next_state, float(done))
        rb.add(obj)
        state = next_state
        if global_step >= args.warmup_timesteps:
            batch = rb.sample()

            state_batch = batch['state'].to(device)
            next_state_batch = batch['next_state'].to(device)
            action_batch = batch['action'].to(device)
            reward_batch = batch['reward'].to(device)
            done_batch = batch['done'].to(device)

            # Compute TD Target
            with torch.no_grad():
                # Let Policy net choose actions
                next_state_actions, next_logp = policy_net.select_action(next_state_batch)

            combined_state_batch = torch.concat((state_batch, next_state_batch), dim=0)
            combined_action_batch = torch.concat((action_batch, next_state_actions), dim=0)

            # Combined Clipped Double Q-learning
            q_net1.train()
            q_net2.train()
            combined_action_values1 = q_net1(combined_state_batch, combined_action_batch).squeeze(1)
            combined_action_values2 = q_net2(combined_state_batch, combined_action_batch).squeeze(1)
            q_net1.eval()
            q_net2.eval()   

            # Extract current q values and next Q-values
            q_values1, next_q_values1 = torch.chunk(combined_action_values1, 2, dim=0)
            q_values2, next_q_values2 = torch.chunk(combined_action_values2, 2, dim=0)

            # calculate TD Target
            next_q_values = torch.min(next_q_values1, next_q_values2)
            next_state_values =  next_q_values - alpha*next_logp.squeeze(1)

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

                min_q = torch.min(q_net1(state_batch, action), q_net2(state_batch, action)).squeeze(1)

                actor_loss = (alpha*logp.squeeze(1) - min_q).mean()
                
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


        if done:
            state, _ = env.reset()
            writer.add_scalar("metric/episodic_return", episode_return, global_step)
            writer.add_scalar("metric/episodic_length", episode_length, global_step)
            writer.add_scalar("metric/episodes", episodes, global_step)

            episode_return = 0
            episode_length = 0
            episodes += 1
    env.close()
    


