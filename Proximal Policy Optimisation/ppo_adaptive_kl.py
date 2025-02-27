import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical
from typing import Tuple, List
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence


DEVICE = "cpu"

# Policy and value model
class ActorCriticNetwork(nn.Module):
  def __init__(self, obs_space_size: int, action_space_size: int) -> None:
    super().__init__()

    self.shared_layers = nn.Sequential(
        nn.Linear(obs_space_size, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU())

    self.policy_layers = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, action_space_size))

    self.value_layers = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1))

  def value(self, obs: torch.Tensor) -> torch.Tensor:
    z = self.shared_layers(obs)
    value = self.value_layers(z)
    return value

  def policy(self, obs: torch.Tensor) -> torch.Tensor:
    z = self.shared_layers(obs)
    policy_logits = self.policy_layers(z)
    return policy_logits

  def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    z = self.shared_layers(obs)
    policy_logits = self.policy_layers(z)
    value = self.value_layers(z)
    return policy_logits, value
  

class PPOTrainer():
  def __init__(self, 
               actor_critic: nn.Module,
               beta: float = 7.5,
               d_target: float = 0.01,
               max_policy_train_iters: int = 80,
               value_train_iters: int = 80,
               policy_lr: float=3e-4,
               value_lr: float=1e-2):
  
    self.ac = actor_critic
    self.beta = beta
    self.d_target = d_target
    self.max_policy_train_iters = max_policy_train_iters
    self.value_train_iters = value_train_iters

    policy_params = list(self.ac.shared_layers.parameters()) + list(self.ac.policy_layers.parameters())
    value_params = list(self.ac.shared_layers.parameters()) + list(self.ac.value_layers.parameters())

    self.policy_optim = optim.Adam(policy_params, lr=policy_lr)
    self.value_optim = optim.Adam(value_params, lr=value_lr)

  
  def train_value(self, obs: torch.Tensor, 
                        returns: torch.Tensor) -> None:

    for _ in range(self.value_train_iters):
      self.value_optim.zero_grad()
      value = self.ac.value(obs)
      value_loss = F.mse_loss(value, returns.unsqueeze(-1))
      value_loss.backward()
      self.value_optim.step()
  
  def train_policy(self, obs: torch.Tensor,
                         acts: torch.Tensor,
                         old_log_probs: torch.Tensor,
                         gaes: torch.Tensor,
                         old_logits: torch.Tensor) -> None:
    
    total_kl = 0.0
    for _ in range(self.max_policy_train_iters):
      self.policy_optim.zero_grad()
      new_logits = self.ac.policy(obs)
      new_distribution = Categorical(logits=new_logits)
      old_distribution = Categorical(logits=old_logits)
      new_log_probs = new_distribution.log_prob(acts)

      policy_ratio = torch.exp(new_log_probs - old_log_probs)
      kl_distance = kl_divergence(new_distribution, old_distribution).mean()
      total_kl += kl_distance.item()

      # PPO with adaptive KL penalty
      policy_loss = -(policy_ratio * gaes - self.beta * kl_distance).mean()
      policy_loss.backward()
      self.policy_optim.step()

    # Adjust beta based on average KL divergence across iterations
    avg_kl = total_kl / self.max_policy_train_iters
    if avg_kl < self.d_target / 1.5:
      self.beta /= 2
    elif avg_kl > self.d_target * 1.5:
      self.beta *= 2
    self.beta = np.clip(self.beta, 1e-4, 10)


def discount_rewards(rewards: List, gamma: float =0.99) -> np.ndarray:
    """
    Return discounted rewards based on the given rewards and gamma param.
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])


def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])

@torch.no_grad()
def rollout(model, env, max_steps=1000):
    """
    Performs a single rollout.
    Returns training data in the shape (n_steps, observation_shape)
    and the cumulative reward.
    """
    ### Create data storage
    train_data = [[], [], [], [], [], []] # obs, act, reward, values, act_log_probs, old_logits
    obs, info = env.reset()

    ep_reward = 0
    for _ in range(max_steps):
        logits, val = model(torch.from_numpy(obs).to(DEVICE).float())
        act_distribution = Categorical(logits=logits)
        act = act_distribution.sample()
        act_log_prob = act_distribution.log_prob(act).item()

        act, val = act.item(), val.item()

        next_obs, reward, done, terminated, _ = env.step(act)

        for i, item in enumerate((obs, act, reward, val, act_log_prob, logits)):
          train_data[i].append(item)

        obs = next_obs
        ep_reward += reward
        if done or terminated:
            break

    train_data = [np.asarray(x) for x in train_data]

    ### Do train data filtering
    train_data[3] = calculate_gaes(train_data[2], train_data[3])

    return train_data, ep_reward

env = gym.make('CartPole-v1')
model = ActorCriticNetwork(env.observation_space.shape[0], env.action_space.n)
model = model.to(DEVICE)
train_data, reward = rollout(model, env) # Test rollout function

# Define training params
n_episodes = 1000
print_freq = 10

ppo = PPOTrainer(
    model,
    policy_lr = 3e-4,
    value_lr = 1e-3,
    max_policy_train_iters = 20,
    value_train_iters = 20)

# Training loop
ep_rewards = []
for episode_idx in range(n_episodes):
  # Perform rollout
  train_data, reward = rollout(model, env)
  ep_rewards.append(reward)

  # Shuffle
  permute_idxs = np.random.permutation(len(train_data[0]))

  # Policy data
  obs = torch.tensor(train_data[0][permute_idxs],
                     dtype=torch.float32, device=DEVICE)
  acts = torch.tensor(train_data[1][permute_idxs],
                      dtype=torch.int32, device=DEVICE)
  gaes = torch.tensor(train_data[3][permute_idxs],
                      dtype=torch.float32, device=DEVICE)
  act_log_probs = torch.tensor(train_data[4][permute_idxs],
                               dtype=torch.float32, device=DEVICE)
  
  old_logits = torch.tensor(train_data[5][permute_idxs],
                               dtype=torch.float32, device=DEVICE)

  # Value data
  returns = discount_rewards(train_data[2])[permute_idxs]
  returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

  # Train model
  ppo.train_policy(obs, acts, act_log_probs, gaes, old_logits)
  ppo.train_value(obs, returns)

  if (episode_idx + 1) % print_freq == 0:
    print('Episode {} | Avg Reward {:.1f}'.format(
        episode_idx + 1, np.mean(ep_rewards[-print_freq:])))