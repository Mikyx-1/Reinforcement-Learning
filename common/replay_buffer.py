"""
Memory buffers for RL agents.

ReplayBuffer   – off-policy uniform experience replay (DQN, DDPG, SAC, TD3)
RolloutBuffer  – on-policy episodic rollout storage (REINFORCE, PPO)
"""

import random
from collections import deque
from typing import Iterator

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Off-policy: Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────


class ReplayBuffer:
    """
    Circular buffer storing (s, a, r, s', done) transitions.

    Usage:
        buf = ReplayBuffer(capacity=100_000)
        buf.push(obs, action, reward, next_obs, done)
        batch = buf.sample(batch_size=256)
    """

    def __init__(self, capacity: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if len(self._buf) < batch_size:
            raise ValueError(
                f"Buffer has {len(self._buf)} transitions, need {batch_size}."
            )
        batch = random.sample(self._buf, batch_size)
        obs, actions, rewards, next_obs, dones = map(np.array, zip(*batch))
        return {
            "obs": torch.as_tensor(obs, dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(
                actions, dtype=torch.float32, device=self.device
            ),
            "rewards": torch.as_tensor(
                rewards, dtype=torch.float32, device=self.device
            ).unsqueeze(1),
            "next_obs": torch.as_tensor(
                next_obs, dtype=torch.float32, device=self.device
            ),
            "dones": torch.as_tensor(
                dones, dtype=torch.float32, device=self.device
            ).unsqueeze(1),
        }

    def __len__(self) -> int:
        return len(self._buf)

    def is_ready(self, min_size: int) -> bool:
        return len(self) >= min_size


# ─────────────────────────────────────────────────────────────────────────────
# On-policy: Rollout Buffer
# ─────────────────────────────────────────────────────────────────────────────


class RolloutBuffer:
    """
    Stores a complete on-policy rollout (one or more episodes).

    Supports:
      * Monte-Carlo returns   (discount_rewards)
      * Generalised Advantage Estimation  (compute_gae – used by PPO)

    Usage (REINFORCE):
        buf = RolloutBuffer()
        buf.push(obs, action, log_prob, reward, done)
        ...
        batch = buf.get(gamma=0.99)   # returns dict with 'returns'
        buf.clear()
    """

    def __init__(self):
        self.clear()

    def clear(self) -> None:
        self.obs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[float] = []  # filled by actor-critic agents

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float = 0.0,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    # ------------------------------------------------------------------
    # Return computation
    # ------------------------------------------------------------------

    def discount_rewards(self, gamma: float) -> np.ndarray:
        """Monte-Carlo discounted returns (no bootstrapping)."""
        returns = np.zeros(len(self.rewards))
        running = 0.0
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                running = 0.0
            running = self.rewards[t] + gamma * running
            returns[t] = running
        return returns

    def compute_gae(
        self, gamma: float, lam: float, last_value: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generalised Advantage Estimation (Schulman et al., 2016).
        Returns (advantages, returns).  Used by PPO.
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones, dtype=float)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + np.array(self.values)
        return advantages, returns

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def get(self, gamma: float = 0.99, device: str = "cpu") -> dict[str, torch.Tensor]:
        """Convert buffer to a batch dict ready for agent.update()."""
        dev = torch.device(device)
        returns = self.discount_rewards(gamma)

        # Normalise returns (reduces variance, standard practice)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return {
            "obs": torch.as_tensor(np.array(self.obs), dtype=torch.float32, device=dev),
            "actions": torch.as_tensor(
                np.array(self.actions), dtype=torch.float32, device=dev
            ),
            "log_probs": torch.as_tensor(
                np.array(self.log_probs), dtype=torch.float32, device=dev
            ),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=dev),
        }

    def __len__(self) -> int:
        return len(self.rewards)

    def __iter__(self) -> Iterator:
        """Iterate step-by-step (obs, action, log_prob, reward, done)."""
        return zip(self.obs, self.actions, self.log_probs, self.rewards, self.dones)
