import gymnasium as gym
import math

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
    tracking: bool = False
    wandb_entity: str = ""
    wandb_project: str = f"sac-v1"
    wandb_mode: str = "disabled"

    env_id: str = "Walker2d-v5"
    total_timesteps: int = 1_000_000
    use_vf: bool = True


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
    
class VNetwork(nn.Module):
    def __init__(self, env):
        super(VNetwork, self).__init__()
        n_observations = env.observation_space.shape[0]
        self.network = nn.Sequential(
            nn.Linear(n_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
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
        # tanh_transform = TanhTransform(cache_size=1)
        # affine_transform = AffineTransform(loc=self.action_bias, scale=self.action_scale)
        # dist = TransformedDistribution(
        #     normal,
        #     [tanh_transform, affine_transform]
        # )

        # action = dist.rsample()
        # log_prob = dist.log_prob(action)
        # log_prob = torch.sum(log_prob, 1, keepdim=True)

        # used above for tanh transform, but i think it's numerically unstable
        # so used cleanrl's implementation instead
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

    v_net = VNetwork(env).to(device)
    target_v_net = VNetwork(env).to(device)
    target_v_net.load_state_dict(v_net.state_dict())

    q_net1 = QNetwork(env).to(device)    
    q_net2 = QNetwork(env).to(device)

    # policy_net = torch.compile(policy_net)
    # q_net1 = torch.compile(q_net1)
    # q_net2 = torch.compile(q_net2)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    v_optimizer = optim.Adam(v_net.parameters(), lr=args.learning_rate)
    q1_optimizer = optim.Adam(q_net1.parameters(), lr=args.learning_rate)
    q2_optimizer = optim.Adam(q_net2.parameters(), lr=args.learning_rate)

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

            # Compute Value Target
            with torch.no_grad():
                # Let Policy net choose actions
                state_actions, logp = policy_net.select_action(state_batch)

                # Clipped Double Q-learning
                q_values1 = q_net1(state_batch, state_actions).squeeze(1)
                q_values2 = q_net2(state_batch, state_actions).squeeze(1)
                q_values = torch.min(q_values1, q_values2)

                value_target =  q_values - args.alpha*logp.squeeze(1)

            # update value function
            state_values = v_net(state_batch).squeeze(1)
            value_loss = F.mse_loss(state_values, value_target)

            writer.add_scalar("losses/v_val", state_values.mean().item(), global_step)
            writer.add_scalar("losses/v_loss", value_loss.item(), global_step)

            v_optimizer.zero_grad()
            value_loss.backward()
            v_optimizer.step()

            # compute TD target
            with torch.no_grad():
                next_state_values = target_v_net(next_state_batch).squeeze(1)
                td_target = reward_batch + args.gamma * next_state_values * (1 - done_batch)

            state_action_values1 = q_net1(state_batch, action_batch).squeeze(1)
            state_action_values2 = q_net2(state_batch, action_batch).squeeze(1)

            # Compute q_loss
            q1_loss = F.mse_loss(state_action_values1, td_target)
            q2_loss = F.mse_loss(state_action_values2, td_target)

            avg_loss = (q1_loss + q2_loss)/2

            writer.add_scalar("losses/q1_val", state_action_values1.mean().item(), global_step)
            writer.add_scalar("losses/q1_loss", q1_loss.item(), global_step)
            writer.add_scalar("losses/q2_val", state_action_values2.mean().item(), global_step)
            writer.add_scalar("losses/q2_loss", q2_loss.item(), global_step)
            writer.add_scalar("losses/avg_q_loss", avg_loss, global_step)

            # Optimize q networks
            q1_optimizer.zero_grad()
            q1_loss.backward()
            q1_optimizer.step()
            q2_optimizer.zero_grad()
            q2_loss.backward()
            q2_optimizer.step()

            # Optimize actor
            action, logp = policy_net.select_action(state_batch)
            min_q = torch.min(q_net1(state_batch, action), q_net2(state_batch, action)).squeeze(1)

            actor_loss = (args.alpha*logp.squeeze(1) - min_q).mean()
            
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            policy_optimizer.zero_grad()
            actor_loss.backward()
            #torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            policy_optimizer.step()
        
            # soft update only when updating actor weights
            soft_update(v_net, target_v_net, args)


        if done:
            state, _ = env.reset()
            writer.add_scalar("metric/episodic_return", episode_return, global_step)
            writer.add_scalar("metric/episodic_length", episode_length, global_step)
            writer.add_scalar("metric/episodes", episodes, global_step)

            episode_return = 0
            episode_length = 0
            episodes += 1
    env.close()
    


