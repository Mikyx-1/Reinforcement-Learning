"""
PPO rollout buffer for graph observations.

Identical contract and GAE/mini-batch logic to agents/ppo/buffer.py's
PPORolloutBuffer — the only difference is that a Dict observation
(node_features, edge_features, action_mask) is stored as three arrays
instead of one flat obs array. `edge_index` isn't stored per step since the
topology is static — it's already baked into the network as a buffer.
"""

import numpy as np
import torch


class GraphPPORolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_nodes: int,
        num_edges: int,
        max_degree: int,
        node_feat_dim: int = 3,
        edge_feat_dim: int = 3,
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

        self.node_features = np.zeros((rollout_steps, num_nodes, node_feat_dim), dtype=np.float32)
        self.edge_features = np.zeros((rollout_steps, num_edges, edge_feat_dim), dtype=np.float32)
        self.action_mask = np.zeros((rollout_steps, max_degree), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, 1), dtype=np.float32)
        self.log_probs = np.zeros(rollout_steps, dtype=np.float32)
        self.rewards = np.zeros(rollout_steps, dtype=np.float32)
        self.dones = np.zeros(rollout_steps, dtype=np.float32)
        self.values = np.zeros(rollout_steps, dtype=np.float32)

        self.advantages = np.zeros(rollout_steps, dtype=np.float32)
        self.returns = np.zeros(rollout_steps, dtype=np.float32)

    def push(self, obs: dict, action, log_prob: float, reward: float, done: bool, value: float) -> None:
        assert self._ptr < self.rollout_steps, "Buffer full — call compute_gae() then clear() before pushing more."
        t = self._ptr
        self.node_features[t] = obs["node_features"]
        self.edge_features[t] = obs["edge_features"]
        self.action_mask[t] = obs["action_mask"]
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

    def compute_gae(self, last_value: float) -> None:
        gae = 0.0
        for t in reversed(range(self.rollout_steps)):
            mask = 1.0 - self.dones[t]
            next_val = self.values[t + 1] if t + 1 < self.rollout_steps else last_value
            delta = self.rewards[t] + self.gamma * next_val * mask - self.values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)

    def get_batches(self, batch_size: int):
        assert self._full, "Buffer not full — collect more steps first."
        indices = np.random.permutation(self.rollout_steps)

        for start in range(0, self.rollout_steps, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs": {
                    "node_features": self._t(self.node_features[idx]),
                    "edge_features": self._t(self.edge_features[idx]),
                    "action_mask": self._t(self.action_mask[idx]),
                },
                "actions": self._t(self.actions[idx]),
                "old_log_probs": self._t(self.log_probs[idx]),
                "advantages": self._t(self.advantages[idx]),
                "returns": self._t(self.returns[idx]),
            }

    def _t(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def clear(self) -> None:
        self._ptr = 0
        self._full = False

    def __len__(self) -> int:
        return self._ptr
