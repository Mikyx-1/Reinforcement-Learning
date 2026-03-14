"""
PPO rollout buffer.

Collects exactly `rollout_steps` environment steps, then:
  1. Bootstraps the last value for incomplete episodes
  2. Computes GAE advantages and lambda-returns in-place
  3. Yields shuffled mini-batches for the K-epoch update loop

This is fundamentally different from RolloutBuffer (used by REINFORCE and
Actor-Critic) in three ways:
  - Fixed capacity (T steps), not episode-aligned
  - Stores V(s) at every step (required for GAE)
  - Supports mini-batch iteration (PPO reuses data across K epochs)
"""

import numpy as np
import torch


class PPORolloutBuffer:
    """
    Fixed-length rollout buffer for PPO.

    Args:
        rollout_steps : Number of env steps collected per update cycle (T).
        obs_dim       : Observation dimensionality.
        act_dim       : Action dimensionality (1 for discrete).
        gamma         : Discount factor γ.
        lam           : GAE λ (lambda).
        device        : Torch device for returned tensors.
    """

    def __init__(
        self,
        rollout_steps: int,
        obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: str = "cpu",
    ):
        self.rollout_steps = rollout_steps
        self.gamma = gamma
        self.lam = lam
        self.device = torch.device(device)
        self._ptr = 0
        self._full = False

        # Pre-allocate contiguous arrays (avoids per-step allocation overhead)
        self.obs = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, act_dim), dtype=np.float32)
        self.log_probs = np.zeros(rollout_steps, dtype=np.float32)
        self.rewards = np.zeros(rollout_steps, dtype=np.float32)
        self.dones = np.zeros(rollout_steps, dtype=np.float32)
        self.values = np.zeros(rollout_steps, dtype=np.float32)

        # Filled by compute_gae()
        self.advantages = np.zeros(rollout_steps, dtype=np.float32)
        self.returns = np.zeros(rollout_steps, dtype=np.float32)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        assert (
            self._ptr < self.rollout_steps
        ), "Buffer full — call compute_gae() then clear() before pushing more steps."
        t = self._ptr
        self.obs[t] = obs
        self.actions[t] = np.atleast_1d(action)
        self.log_probs[t] = log_prob
        self.rewards[t] = reward
        self.dones[t] = float(done)
        self.values[t] = value
        self._ptr += 1
        if self._ptr == self.rollout_steps:
            self._full = True

    def is_full(self) -> bool:
        return self._full

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_gae(self, last_value: float) -> None:
        """
        Compute GAE advantages and lambda-returns in-place.

        Call once after the rollout is full, before get_batches().

        GAE:  δ_t = r_t + γ·V(s_{t+1})·(1−d_t) − V(s_t)
              Â_t = δ_t + (γλ)·(1−d_t)·Â_{t+1}

        Returns = Advantages + Values  (used as critic targets).

        Args:
            last_value: Bootstrap value V(s_T). Pass 0.0 if last step was
                        terminal, else pass V(next_obs) from the agent.
        """
        gae = 0.0
        for t in reversed(range(self.rollout_steps)):
            mask = 1.0 - self.dones[t]
            next_val = self.values[t + 1] if t + 1 < self.rollout_steps else last_value
            delta = self.rewards[t] + self.gamma * next_val * mask - self.values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

        # Normalise advantages over the whole rollout (reduces sensitivity
        # to reward scale; standard PPO practice)
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)

    # ------------------------------------------------------------------
    # Mini-batch iteration
    # ------------------------------------------------------------------

    def get_batches(self, batch_size: int):
        """
        Yield random mini-batches of size `batch_size`.

        Typically called inside a K-epoch loop:
            for epoch in range(K):
                for batch in buffer.get_batches(batch_size):
                    metrics = agent._update_minibatch(batch)

        Yields dicts with keys:
            obs, actions, old_log_probs, advantages, returns
        """
        assert self._full, "Buffer not full — collect more steps first."
        indices = np.random.permutation(self.rollout_steps)

        for start in range(0, self.rollout_steps, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs": self._t(self.obs[idx]),
                "actions": self._t(self.actions[idx]),
                "old_log_probs": self._t(self.log_probs[idx]),
                "advantages": self._t(self.advantages[idx]),
                "returns": self._t(self.returns[idx]),
            }

    def _t(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._ptr = 0
        self._full = False

    def __len__(self) -> int:
        return self._ptr
