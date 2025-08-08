import gymnasium as gym
import math

from dataclasses import dataclass
import tyro
import numpy as np

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

@dataclass
class Args:
    tracking: bool = False
    wandb_entity: str = ""
    wandb_project: str = f"ppo"
    wandb_mode: str = "disabled"

    env_id: str = "Walker2d-v5"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    tau: float = 0.005


    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    num_envs: int = 1
    batch_size: int = 2048
    mini_batch_size: int = 64
    update_epochs: int = 10

class Critic(nn.Module):
    def __init__(self, env):
        super(Critic, self).__init__()
        n_observations = env.observation_space.shape[1]

        # didn't know ppo prefers tanh activations
        self.network = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        n_observations = env.observation_space.shape[1]
        n_actions = env.action_space.shape[1]
        self.mean = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
        )
        
        # log_std is a learnable parameter for global state-independent std deviation
        # we can also use a fixed or a learnable std
        self.log_std = nn.Parameter(torch.zeros(n_actions))

    def get_action_and_prob(self, x, action=None):
        action_mean = self.mean(x)
        action_logstd = self.log_std.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.normal.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

class PPOAgent(nn.Module):
    def __init__(self, env):
        super(PPOAgent, self).__init__()
        self.actor = Actor(env)
        self.critic = Critic(env)

    def forward(self, x):
        action, log_prob, entropy = self.actor.get_action_and_prob(x)
        value = self.critic(x)
        return action, log_prob, entropy, value

    def act(self, x):
        return self.actor.get_action_and_prob(x)[0]
    
    def evaluate(self, x, action):
        _, log_prob, entropy = self.actor.get_action_and_prob(x, action)
        return log_prob, entropy, self.critic(x)


class RolloutBuffer:
    """
    A simple rollout buffer to store observations, actions, rewards, dones, values, and log probabilities.
    It also computes advantages and returns using GAE.
    """
    def __init__(self, num_env, num_steps, env):
        self.obs_dim = env.observation_space.shape[1]
        self.action_dim = env.action_space.shape[1]

        self.obs = torch.zeros(num_steps, num_env, self.obs_dim)
        self.actions = torch.zeros(num_steps, num_env, self.action_dim)
        self.rewards = torch.zeros(num_steps, num_env)
        self.dones = torch.zeros(num_steps + 1, num_env)  # +1 for bootstrapped value
        self.values = torch.zeros(num_steps + 1, num_env) # +1 for bootstrapped value
        self.log_probs = torch.zeros(num_steps, num_env)

        # To be filled after rollout
        self.advantages = torch.zeros(num_steps, num_env)
        self.returns = torch.zeros(num_steps, num_env)

        self.ptr = 0  # pointer to current index

    def add(self, obs, action, reward, done, value, log_prob):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1
    
    def add_bootstrapped_value(self, value, done):
        self.values[self.ptr] = value  # store the last value for bootstrapping
        self.dones[self.ptr] = done 

    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95, normalize_advantages=True):

        # vector compute TD errors
        deltas = self.rewards + gamma * self.values[1:] * (1 - self.dones[1:]) - self.values[:-1]
        advantage = 0
        for t in reversed(range(self.ptr)):
            advantage = deltas[t] + gamma * gae_lambda * (1-self.dones[t + 1]) * advantage
            self.advantages[t] = advantage

        self.returns = self.advantages + self.values[:-1]

        # normalize advantages
        if normalize_advantages:
            advantages_mean = self.advantages.mean()
            advantages_std = self.advantages.std()
            self.advantages = (self.advantages - advantages_mean) / (advantages_std + 1e-8)
    
    def reset(self):
        # lazy reset
        self.ptr = 0

    def get_dataloader(self, batch_size):
        dataset = TensorDataset(
            self.obs.reshape(-1, self.obs_dim), 
            self.actions.reshape(-1, self.action_dim),
            self.log_probs.reshape(-1),
            self.advantages.reshape(-1),
            self.returns.reshape(-1),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True  # ensures sampling without replacement
        )
        return loader

if __name__ == "__main__":
    args = tyro.cli(Args)
    env = gym.make(args.env_id)
    # env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.ClipAction(env)
    # env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
    # env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
    # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

    # finally switched to vectorized environment used in cleanrl
    env = gym.vector.SyncVectorEnv([lambda: env for _ in range(args.num_envs)])

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

    agent = PPOAgent(env).to(device)
    agent_opt = optim.Adam(agent.parameters(), lr=args.learning_rate)

    num_fixed_steps = args.batch_size // args.num_envs
    if args.batch_size % args.num_envs != 0:
        raise ValueError("Batch size must be divisible by number of environments.")


    rollout_buffer = RolloutBuffer(args.num_envs, num_fixed_steps, env)

    iteration_count = args.total_timesteps // args.batch_size

    next_state, _ = env.reset()
    next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    global_step = 0
    start_time = time.time()
    for iteration in range(iteration_count):
        rollout_buffer.reset()
        episode_return = 0
        episode_length = 0
        # rollout policy and collect data
        for step in range(num_fixed_steps):
            global_step += args.num_envs
            state = next_state
            done = next_done
            with torch.no_grad():
                action, log_prob, entropy, value = agent(state)
            next_state, reward, terminated, truncated, infos = env.step(action.cpu().detach().numpy())
            next_done = torch.tensor(np.logical_or(terminated, truncated), dtype=torch.float32, device=device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)

            episode_return += reward
            episode_length += 1

            # add to buffer
            rollout_buffer.add(
                obs=state.squeeze(1),
                action=action.squeeze(1),
                reward=torch.tensor(reward, dtype=torch.float32, device=device),
                done=done,
                value=value.squeeze(1),
                log_prob=log_prob
            )
            if "episode" in infos:
                for idx in range(args.num_envs):
                    if infos["_episode"][idx]:
                        #print(f"global_step={global_step}, episodic_return={infos['episode']['r'][idx]}")
                        writer.add_scalar("metric/episodic_return", infos["episode"]["r"][idx], global_step)
                        writer.add_scalar("metric/episodic_length", infos["episode"]["l"][idx], global_step)

        # Bootstrap last value
        with torch.no_grad():
            last_value = agent.critic(next_state).squeeze(1)
            rollout_buffer.add_bootstrapped_value(last_value, next_done)

        rollout_buffer.compute_returns_and_advantages(
            gamma=args.gamma,
            gae_lambda=args.gae_lambda
        ) 
        
        # wrap buffer into a dataloader because I don't want to implement mini-batch sampling manually
        dataloader = rollout_buffer.get_dataloader(args.mini_batch_size)

        # update models
        clipfracs = []
        a = 0
        for epoch in range(args.update_epochs):
            # Sample a batch of data from the replay buffer
            for obs_mb, act_mb, old_logp_mb, adv_mb, ret_mb in dataloader:
                obs_mb = obs_mb.to(device)
                act_mb = act_mb.to(device)
                old_logp_mb = old_logp_mb.to(device)
                adv_mb = adv_mb.to(device)
                ret_mb = ret_mb.to(device)

                log_prob, entropy, value = agent.evaluate(obs_mb, act_mb)

                logratio = (log_prob - old_logp_mb)  # ratio of new and old policy probabilities
                ratio = logratio.exp()
                # if a == 0:
                #     print(ratio)
                #     print(log_prob, old_logp_mb)
                #     print()
                #     a = 1

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_epsilon).float().mean().item()]

                # Compute the surrogate loss
                surrogate_loss = ratio * adv_mb
                clipped_surrogate_loss = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * adv_mb
                actor_loss = -torch.min(surrogate_loss, clipped_surrogate_loss).mean()

                # Compute the value loss
                value_loss = F.mse_loss(value.squeeze(-1), ret_mb)

                # Compute the entropy loss
                entropy_loss = -entropy.mean()

                total_loss = actor_loss + args.value_loss_coef * value_loss - args.entropy_coef * entropy_loss

                agent_opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                agent_opt.step()

        writer.add_scalar("loss/actor_loss", actor_loss.item(), global_step)
        writer.add_scalar("loss/critic_loss", value_loss.item(), global_step)
        writer.add_scalar("loss/entropy_loss", entropy_loss.item(), global_step)
        writer.add_scalar("loss/total_loss", total_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        #print("SPS:", int(global_step / (time.time() - start_time)))

    env.close()
    


