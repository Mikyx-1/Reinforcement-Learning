# -*- coding: utf-8 -*-
"""cartpole_ppo_separate.ipynb

Revised PPO code with separate Actor and Critic networks.
"""

from itertools import count
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
# Assume HelicopterControlEnv is defined elsewhere
from customised_env import HelicopterControlEnv
from torch import nn, optim
from torch.distributions.categorical import Categorical

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


####################################
# Positional Embedding Module
####################################
class PositionalEmbedding2D(nn.Module):
    def __init__(self, grid_size: int, embed_dim: int):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim  # Total embedding size per position
        self.half_dim = (
            embed_dim // 4
        )  # Number of frequency terms per coordinate (x or y)

        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D encoding"

        # Precompute sinusoidal frequencies
        div_term = torch.exp(
            torch.arange(0, self.half_dim, 1) * (-np.log(10000.0) / self.half_dim)
        ).to(DEVICE)
        self.register_buffer("div_term", div_term)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions: Tensor of shape (batch_size, 2) with [x, y] coordinates
        Returns: Tensor of shape (batch_size, embed_dim) with positional embeddings
        """
        batch_size = positions.size(0)
        x, y = positions[:, 0], positions[:, 1]  # Shape: (batch_size,)

        # Initialize embedding tensor
        pe = torch.zeros(batch_size, self.embed_dim, device=DEVICE)

        # X-coordinate encodings
        pos_x = x.unsqueeze(-1)  # (batch_size, 1)
        pe[:, 0 : self.half_dim] = torch.sin(pos_x * self.div_term)
        pe[:, self.half_dim : self.half_dim * 2] = torch.cos(pos_x * self.div_term)

        # Y-coordinate encodings
        pos_y = y.unsqueeze(-1)  # (batch_size, 1)
        pe[:, self.half_dim * 2 : self.half_dim * 3] = torch.sin(pos_y * self.div_term)
        pe[:, self.half_dim * 3 :] = torch.cos(pos_y * self.div_term)

        return pe


####################################
# Actor Network with Positional Embeddings
####################################
class ActorNetwork(nn.Module):
    def __init__(
        self,
        obs_space_size: int,
        action_space_size: int,
        grid_size: int = 50,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        # Positional embedding for agent and goal
        self.pos_embed = PositionalEmbedding2D(
            grid_size, embed_dim // 2
        )  # embed_dim // 2 per position

        # Input size: agent embedding + goal embedding + raw obs
        input_dim = (
            embed_dim // 2
        ) * 2 + obs_space_size  # Agent and goal embeddings + raw obs
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, action_space_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: Tensor of shape (batch_size, 4) with [x_agent, y_agent, x_goal, y_goal]
        Returns: Action logits
        """
        # Extract agent and goal positions
        agent_pos = obs[:, :2]  # (batch_size, 2)
        goal_pos = obs[:, 2:]  # (batch_size, 2)

        # Get positional embeddings
        agent_pe = self.pos_embed(agent_pos)  # (batch_size, embed_dim // 2)
        goal_pe = self.pos_embed(goal_pos)  # (batch_size, embed_dim // 2)

        # Concatenate embeddings and raw observations
        input_features = torch.cat(
            [agent_pe, goal_pe, obs], dim=-1
        )  # (batch_size, embed_dim + obs_space_size)

        return self.network(input_features)


####################################
# Critic Network with Positional Embeddings
####################################
class CriticNetwork(nn.Module):
    def __init__(self, obs_space_size: int, grid_size: int = 50, embed_dim: int = 128):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        # Positional embedding for agent and goal
        self.pos_embed = PositionalEmbedding2D(grid_size, embed_dim // 2)

        # Input size: agent embedding + goal embedding + raw obs
        input_dim = (embed_dim // 2) * 2 + obs_space_size
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: Tensor of shape (batch_size, 4) with [x_agent, y_agent, x_goal, y_goal]
        Returns: State value
        """
        # Extract agent and goal positions
        agent_pos = obs[:, :2]
        goal_pos = obs[:, 2:]

        # Get positional embeddings
        agent_pe = self.pos_embed(agent_pos)
        goal_pe = self.pos_embed(goal_pos)

        # Concatenate embeddings and raw observations
        input_features = torch.cat([agent_pe, goal_pe, obs], dim=-1)

        return self.network(input_features)


####################################
# PPO Trainer with Separate Actor and Critic
####################################
class PPOTrainer:
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        ppo_clip_val: float = 0.2,
        target_kl_div: float = 0.01,
        max_policy_train_iters: int = 80,
        value_train_iters: int = 80,
        policy_lr: float = 3e-4,
        value_lr: float = 1e-2,
    ):
        self.actor = actor
        self.critic = critic
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters

        self.policy_optim = optim.RMSprop(self.actor.parameters(), lr=policy_lr)
        self.value_optim = optim.RMSprop(self.critic.parameters(), lr=value_lr)

    def train_policy(
        self,
        obs: torch.Tensor,
        acts: torch.Tensor,
        old_log_probs: torch.Tensor,
        gaes: torch.Tensor,
    ) -> None:
        for _ in range(self.max_policy_train_iters):
            self.policy_optim.zero_grad()

            logits = self.actor(obs)
            dist = Categorical(logits=logits)
            dist_entropy = dist.entropy()
            new_log_probs = dist.log_prob(acts)

            policy_ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = policy_ratio.clamp(
                1 - self.ppo_clip_val, 1 + self.ppo_clip_val
            )

            loss_unclipped = policy_ratio * gaes
            loss_clipped = clipped_ratio * gaes
            policy_loss = (
                -torch.min(loss_unclipped, loss_clipped).mean()
                - 0.01 * dist_entropy.mean()
            )

            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
            self.policy_optim.step()

    def train_value(self, obs: torch.Tensor, returns: torch.Tensor) -> None:
        for _ in range(self.value_train_iters):
            self.value_optim.zero_grad()

            values = self.critic(obs)
            value_loss = F.huber_loss(values, returns.unsqueeze(1))
            value_loss.backward()
            self.value_optim.step()


####################################
# Helper functions: Discounted rewards and GAEs
####################################
def discount_rewards(rewards: List, gamma: float = 0.99) -> np.ndarray:
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards) - 1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])


def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
    """
    Compute Generalized Advantage Estimates.
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [
        rew + gamma * next_val - val
        for rew, val, next_val in zip(rewards, values, next_values)
    ]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas) - 1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])
    return np.array(gaes[::-1])


####################################
# Rollout function
####################################
def rollout(actor: nn.Module, critic: nn.Module, env, max_steps=1000):
    """
    Performs a single rollout.
    Returns training data and cumulative reward.
    Data: obs, act, reward, values, act_log_probs
    """
    train_data = [[], [], [], [], []]  # obs, act, reward, values, act_log_probs
    obs, _ = env.reset()
    ep_reward = 0

    for _ in range(max_steps):
        # Convert obs to tensor
        obs_tensor = (
            torch.from_numpy(obs).to(dtype=torch.float32, device=DEVICE).view(1, -1)
        )
        logits = actor(obs_tensor)
        dist = Categorical(logits=logits)
        act = dist.sample()
        act_log_prob = dist.log_prob(act).item()

        # Critic gives the state value
        val = critic(obs_tensor).item()

        act = act.item()
        next_obs, reward, done, truncated, _ = env.step(act)

        for i, item in enumerate((obs, act, reward, val, act_log_prob)):
            train_data[i].append(item)

        obs = next_obs
        ep_reward += reward
        if done or truncated:
            break

    # Convert lists to numpy arrays
    train_data = [np.asarray(x) for x in train_data]
    # Compute GAEs for advantages using rewards and estimated values.
    train_data[3] = calculate_gaes(train_data[2], train_data[3])
    return train_data, ep_reward


####################################
# Main setup and training loop
####################################
env = HelicopterControlEnv()
obs_space = env.observation_space.shape[0]
action_space = env.action_space.n

print(f"obs_space: {obs_space}, action_space: {action_space}")

actor = ActorNetwork(obs_space, action_space).to(DEVICE)
critic = CriticNetwork(obs_space).to(DEVICE)

# Create a test rollout to verify
train_data, reward = rollout(actor, critic, env)
print("Test rollout complete. Reward:", reward)

# Define training parameters
n_episodes = 2000
print_freq = 20

ppo = PPOTrainer(
    actor,
    critic,
    policy_lr=3e-4,
    value_lr=1e-3,
    target_kl_div=0.02,
    max_policy_train_iters=5,
    value_train_iters=5,
)

ep_rewards = []
for episode_idx in count():
    train_data, reward = rollout(actor, critic, env)
    ep_rewards.append(reward)

    # Shuffle data indices
    permute_idxs = np.random.permutation(len(train_data[0]))

    # For actor training, stack observations
    obs_np = np.stack(train_data[0])
    obs_tensor = torch.tensor(obs_np[permute_idxs], dtype=torch.float32, device=DEVICE)
    acts_tensor = torch.tensor(
        train_data[1][permute_idxs], dtype=torch.int64, device=DEVICE
    )
    gaes_tensor = torch.tensor(
        train_data[3][permute_idxs], dtype=torch.float32, device=DEVICE
    )
    act_log_probs_tensor = torch.tensor(
        train_data[4][permute_idxs], dtype=torch.float32, device=DEVICE
    )

    # For critic training, compute returns
    returns = discount_rewards(train_data[2])[permute_idxs]
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

    # Train actor and critic
    ppo.train_policy(obs_tensor, acts_tensor, act_log_probs_tensor, gaes_tensor)
    ppo.train_value(obs_tensor, returns_tensor)

    if (episode_idx + 1) % print_freq == 0:
        print(
            "Episode {} | Avg Reward {:.1f}".format(
                episode_idx + 1, np.mean(ep_rewards[-print_freq:])
            )
        )

    if (episode_idx + 1) % 5000 == 0:
        print("Start saving actor's weights ")
        torch.save(actor.state_dict(), "customised_env_actor.pt")
