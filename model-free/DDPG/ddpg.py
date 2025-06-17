import gymnasium as gym
import math

from dataclasses import dataclass
import tyro

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter


# originally wrote this script in a notebook but decided to migrate to pure python
# a lot of utility code was referenced from CleanRL since I relied heavily on global
#  variables in my notebook

@dataclass
class Args:
    tracking: bool = False
    wandb_entity: str = ""
    wandb_project: str = f"ddpg"
    wandb_mode: str = "disabled"

    env_id: str = "Walker2d-v5"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    buffer_size: int = 100_000
    batch_size: int = 128
    warmup_timesteps: int = 25_000

    eps_begin: float = 0.4
    eps_end: float = 0.1
    eps_decay_steps: int = 500_000

    update_policy_frequency: int = 2


class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        n_observations = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]
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
        n_observations = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]
        self.network = nn.Sequential(
            nn.Linear(n_observations, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Tanh()
        )
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        return self.network(x)*self.action_scale + self.action_bias

@torch.no_grad()
def select_action(state, policy_net, learning_step, env, args):
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    policy_net.eval()
    actions = policy_net(state)
    policy_net.train()

    # exploration noise decay
    eps = args.eps_end + (args.eps_begin - args.eps_end) * \
            math.exp(-1. * (learning_step) / args.eps_decay_steps)
    noise = torch.randn_like(actions) * eps

    ACTION_LOW = torch.tensor(env.action_space.low, dtype=torch.float).to(device)
    ACTION_HIGH = torch.tensor(env.action_space.high, dtype=torch.float).to(device)
    return torch.clamp(actions + noise, min=ACTION_LOW, max=ACTION_HIGH).squeeze(0).cpu().detach().numpy()

def soft_update(net, target_net, args):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

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

    policy_net = Policy(env).to(device)
    target_policy_net = Policy(env).to(device)
    target_policy_net.load_state_dict(policy_net.state_dict())

    q_net = QNetwork(env).to(device)
    target_q_net = QNetwork(env).to(device)
    target_q_net.load_state_dict(q_net.state_dict())

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    q_optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)

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
            action = select_action(state, policy_net, learning_step, env, args)

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
                next_state_actions = target_policy_net(next_state_batch)

                # Compute next state values
                next_q_values = target_q_net(next_state_batch, next_state_actions).squeeze(1)
                td_target =  reward_batch + (next_q_values * args.gamma) * (1 - done_batch)
            
            state_action_values = q_net(state_batch, action_batch).squeeze(1)

            # Compute q_loss
            q_loss = F.mse_loss(state_action_values, td_target)
            #run.log({"losses/q_loss": q_loss.item()}, step=global_step)
            #run.log({"losses/q_val": state_action_values.mean().item()}, step=global_step)
            writer.add_scalar("losses/q_val", state_action_values.mean().item(), global_step)
            writer.add_scalar("losses/q_loss", q_loss.item(), global_step)


            # Optimize critic
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            # Optimize actor
            if global_step % args.update_policy_frequency == 0:
                actor_loss = -q_net(state_batch, policy_net(state_batch)).mean()
                #run.log({"losses/actor_loss": actor_loss.item()}, step=global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                policy_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                policy_optimizer.step()
            
            soft_update(q_net, target_q_net, args)
            soft_update(policy_net, target_policy_net, args)

        if done:
            state, _ = env.reset()
            writer.add_scalar("charts/episodic_return", episode_return, global_step)
            writer.add_scalar("charts/episodic_length", episode_length, global_step)
            writer.add_scalar("charts/episodes", episodes, global_step)
            #run.log({"charts/episodic_return": episode_return}, step=episodes)
            #run.log({"charts/episodic_length": episode_length}, step=episodes)
            episode_return = 0
            episode_length = 0
            episodes += 1
    env.close()
    


