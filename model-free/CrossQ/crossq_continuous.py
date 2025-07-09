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
# from torchrl.modules import BatchRenorm1d
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter
import pprint

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
    seed: int = 42

    total_timesteps: int = 1_000_000
    critic_hidden_size: int = 1024
    actor_hidden_size: int = 256
    autotune: bool = True

    learning_rate: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    adam_betas: tuple = (0.5, 0.999)
    bn_momentum: float = 0.01 # PyTorch uses (1 - Paper's Value) for momentum
    bn_warmup: int = 100_000
    bn_type: str = "brn"

    buffer_size: int = 100_000
    batch_size: int = 256
    warmup_timesteps: int = 5_000

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

# from https://github.com/danielpalenicek/stable-baselines3-contrib/blob/feat/crossq/sb3_contrib/common/torch_layers.py
# torchrl's BatchRenorm1d weirdly suffers from catastrophic forgetting after 2*bn_warmup steps
class BatchRenorm(torch.nn.Module):
    """
    BatchRenorm Module (https://arxiv.org/abs/1702.03275).
    Adapted to Pytorch from
    https://github.com/araffin/sbx/blob/master/sbx/common/jax_layers.py

    BatchRenorm is an improved version of vanilla BatchNorm. Contrary to BatchNorm,
    BatchRenorm uses the running statistics for normalizing the batches after a warmup phase.
    This makes it less prone to suffer from "outlier" batches that can happen
    during very long training runs and, therefore, is more robust during long training runs.

    During the warmup phase, it behaves exactly like a BatchNorm layer. After the warmup phase,
    the running statistics are used for normalization. The running statistics are updated during
    training mode. During evaluation mode, the running statistics are used for normalization but
    not updated.

    :param num_features: Number of features in the input tensor.
    :param eps: A value added to the variance for numerical stability.
    :param momentum: The value used for the ra_mean and ra_var (running average) computation.
        It controls the rate of convergence for the batch renormalization statistics.
    :param affine: A boolean value that when set to True, this module has learnable
            affine parameters. Default: True
    :param warmup_steps: Number of warum steps that are performed before the running statistics
            are used for normalization. During the warump phase, the batch statistics are used.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 0.001,
        momentum: float = 0.01,
        affine: bool = True,
        warmup_steps: int = 100_000,
    ):
        super().__init__()
        # Running average mean and variance
        self.register_buffer("ra_mean", torch.zeros(num_features, dtype=torch.float))
        self.register_buffer("ra_var", torch.ones(num_features, dtype=torch.float))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long))
        self.scale = torch.nn.Parameter(torch.ones(num_features, dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.zeros(num_features, dtype=torch.float))

        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum
        self.num_features = num_features
        # Clip scale and bias of the affine transform
        self.rmax = 3.0
        self.dmax = 5.0
        self.warmup_steps = warmup_steps

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor.

        :param x: Input tensor
        :return: Normalized tensor.
        """

        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)
            batch_std = (batch_var + self.eps).sqrt()

            # Use batch statistics during initial warm up phase.
            # Note: in the original paper, after some warmup phase (batch norm phase of 5k steps)
            # the constraints are linearly relaxed to r_max/d_max over 40k steps
            # Here we only have a warmup phase
            if self.steps > self.warmup_steps:

                running_std = (self.ra_var + self.eps).sqrt()
                # scale
                r = (batch_std / running_std).detach()
                r = r.clamp(1 / self.rmax, self.rmax)
                # bias
                d = ((batch_mean - self.ra_mean) / running_std).detach()
                d = d.clamp(-self.dmax, self.dmax)

                # BatchNorm normalization, using minibatch stats and running average stats
                custom_mean = batch_mean - d * batch_var.sqrt() / r
                custom_var = batch_var / (r**2)

            else:
                custom_mean, custom_var = batch_mean, batch_var

            # Update Running Statistics
            self.ra_mean += self.momentum * (batch_mean.detach() - self.ra_mean)
            self.ra_var += self.momentum * (batch_var.detach() - self.ra_var)
            self.steps += 1

        else:
            # Use running statistics during evaluation mode
            custom_mean, custom_var = self.ra_mean, self.ra_var

        # Normalize
        x = (x - custom_mean[None]) / (custom_var[None] + self.eps).sqrt()

        if self.affine:
            x = self.scale * x + self.bias

        return x

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, momentum={self.momentum}, "
            f"warmup_steps={self.warmup_steps}, affine={self.affine}"
        )

class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() == 1:
            raise ValueError(f"Expected 2D or 3D input (got {x.dim()}D input)")
        
class QNetwork(nn.Module):
    def __init__(self, env, args):
        super(QNetwork, self).__init__()
        n_observations = env.observation_space.shape[1]
        n_actions = env.action_space.shape[1]
        hidden_size = args.critic_hidden_size
        momentum = args.bn_momentum
        if args.bn_type == "brn":
            BN = BatchRenorm1d
        else:
            BN = nn.BatchNorm1d

        self.network = nn.Sequential(
            BN(n_observations + n_actions, momentum=momentum, warmup_steps=args.bn_warmup),
            nn.Linear(n_observations + n_actions, hidden_size),
            BN(hidden_size, momentum=momentum, warmup_steps=args.bn_warmup),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            BN(hidden_size, momentum=momentum, warmup_steps=args.bn_warmup),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action, train=False):
        if train:
            self.train()
        else:
            self.eval()
        x = torch.cat((state, action), dim=1)
        return self.network(x)

class Policy(nn.Module):
    def __init__(self, env, args):
        super(Policy, self).__init__()
        n_observations = env.observation_space.shape[1]
        n_actions = env.action_space.shape[1]
        hidden_size = args.actor_hidden_size
        momentum = args.bn_momentum

        if args.bn_type == "brn":
            BN = BatchRenorm1d
        else:
            BN = nn.BatchNorm1d

        self.network = nn.Sequential(
            BN(n_observations, momentum=momentum, warmup_steps=args.bn_warmup),
            nn.Linear(n_observations, hidden_size),
            BN(hidden_size, momentum=momentum, warmup_steps=args.bn_warmup),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            BN(hidden_size, momentum=momentum, warmup_steps=args.bn_warmup),
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

    def forward(self, x, train=False):
        if train:
            self.train()
        else:
            self.eval()
        LOG_STD_MIN = -5
        LOG_STD_MAX = 2
        x = self.network(x)
        mean = self.mean_head(x)
        logstd = self.logstd_head(x)

        logstd = torch.tanh(logstd)
        logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logstd + 1)
        return mean, logstd
    
    def select_action(self, x, train=False):
        mean, logstd = self(x, train)
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

    
    episodes = 0
    for global_step in range(args.total_timesteps):

        if global_step < args.warmup_timesteps:
            action = envs.action_space.sample()
        else:
            learning_step = global_step - args.warmup_timesteps
            action, logp = policy_net.select_action(torch.tensor(state,dtype=torch.float32, device=device))
            action = action.cpu().detach().numpy()

        next_state, reward, terminated, truncated, infos = envs.step(action)

        # https://farama.org/Vector-Autoreset-Mode, change to SAME_STEP autoreset mode
        #pprint.pprint(infos)  # Print the infos dictionary for debugging
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
            combined_action_values1 = q_net1(combined_state_batch, combined_action_batch, train=True)
            combined_action_values2 = q_net2(combined_state_batch, combined_action_batch, train=True)

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
                action, logp = policy_net.select_action(state_batch, train=True)

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

    envs.close()
    


