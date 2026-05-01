"""
Deep Q-Network – DQN
Mnih et al., 2015  –  https://www.nature.com/articles/nature14236

With the following standard improvements included by default:
  • Double DQN  (van Hasselt et al., 2016)
  • Dueling architecture  (Wang et al., 2016)   [optional, default: off]
  • Target network with periodic hard update

Algorithm (one step):
    1. Observe s_t; select a_t via ε-greedy policy
    2. Execute a_t, observe r_t, s_{t+1}, done_t
    3. Store (s_t, a_t, r_t, s_{t+1}, done_t) in replay buffer
    4. Sample random mini-batch B from buffer
    5. Compute TD targets:
         Vanilla DQN:
           y_t = r_t + γ · max_a Q_target(s_{t+1}, a) · (1 − done_t)
         Double DQN:
           a* = argmax_a Q_online(s_{t+1}, a)           ← online net picks action
           y_t = r_t + γ · Q_target(s_{t+1}, a*) · (1 − done_t)  ← target net evaluates
    6. Loss: L = MSE(Q_online(s_t, a_t), y_t)
    7. Gradient step on Q_online; periodically copy Q_online → Q_target

Key design:
  - ε decays linearly from eps_start to eps_end over eps_decay_steps
  - Target network is updated by hard copy every target_update_freq steps
  - on_step_end() is the epsilon-decay hook, called by the Trainer each step

References:
    Mnih, V. et al. (2015). Human-level control through deep reinforcement
    learning. Nature, 518, 529–533.
    van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning
    with Double Q-learning. AAAI.
    Wang, Z. et al. (2016). Dueling Network Architectures for Deep Reinforcement
    Learning. ICML.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from agents.dqn.networks import DuelingQNetwork, QNetwork
from common.schedulers import LinearSchedule
from common.utils import hard_update, soft_update


class DQNAgent(BaseAgent):
    """
    DQN with Double DQN and optional Dueling architecture.

    Only works with discrete action spaces.

    Args:
        env:                  gymnasium.Env (used for space inference only).
        hidden_dims:          Hidden layer sizes for the Q-network.
        lr:                   Adam learning rate.
        gamma:                Discount factor γ.
        eps_start:            Initial ε for ε-greedy exploration.
        eps_end:              Final ε after annealing.
        eps_decay_steps:      Number of steps to linearly anneal ε.
        target_update_freq:   Steps between hard target network updates.
        use_double:           If True, use Double DQN target computation.
        use_dueling:          If True, use the Dueling Q-network architecture.
        device:               'cpu' or 'cuda'.
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_dims: list[int] = [128, 128],
        lr: float = 1e-3,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 10_000,
        target_update_freq: int = 500,
        use_double: bool = True,
        use_dueling: bool = False,
        device: str = "cpu",
    ):
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "DQNAgent only supports discrete action spaces."
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = env.action_space.n

        super().__init__(obs_dim=obs_dim, act_dim=act_dim, device=device)

        self.gamma = gamma
        self.use_double = use_double
        self.target_update_freq = target_update_freq

        # ε-greedy schedule
        self.eps_schedule = LinearSchedule(
            start=eps_start,
            end=eps_end,
            duration=eps_decay_steps,
        )
        self.epsilon = eps_start

        # Q-networks (online + frozen target)
        NetClass = DuelingQNetwork if use_dueling else QNetwork
        self.q_net = NetClass(obs_dim, act_dim, hidden_dims).to(self.device)
        self.q_target = NetClass(obs_dim, act_dim, hidden_dims).to(self.device)
        hard_update(self.q_target, self.q_net)  # start with identical weights
        self.q_target.eval()  # target never trains directly

        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr, amsgrad=True)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss — less sensitive to outliers

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        ε-greedy action selection.

        Args:
            obs:           (obs_dim,) observation.
            deterministic: If True, always take the greedy action (eval mode).

        Returns:
            action: scalar int array.
        """
        if not deterministic and np.random.random() < self.epsilon:
            return np.array(np.random.randint(self.act_dim))

        obs_t = self.to_tensor(obs).unsqueeze(0)  # (1, obs_dim)
        with torch.no_grad():
            q_values = self.q_net(obs_t)  # (1, act_dim)
        return np.array(q_values.argmax(dim=1).item())

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        One gradient step on a sampled mini-batch.

        Expected batch keys (from ReplayBuffer.sample()):
            obs      : (B, obs_dim)
            actions  : (B, 1)   float32 — will be cast to long
            rewards  : (B, 1)
            next_obs : (B, obs_dim)
            dones    : (B, 1)

        Returns:
            metrics: {'loss': float, 'epsilon': float, 'mean_q': float}
        """
        obs = batch["obs"]
        actions = (
            batch["actions"].long().reshape(-1)
        )  # (B,) int64 — safe for (B,) or (B,1)
        rewards = batch["rewards"].reshape(-1, 1)  # (B, 1)
        next_obs = batch["next_obs"]
        dones = batch["dones"].reshape(-1, 1)  # (B, 1)

        # ── Current Q-values for taken actions ──────────────────────────────
        q_values = self.q_net(obs)  # (B, act_dim)
        q_taken = q_values.gather(1, actions.unsqueeze(1))  # (B, 1)

        # ── TD targets ───────────────────────────────────────────────────────
        with torch.no_grad():
            if self.use_double:
                # Double DQN: online net selects action, target net evaluates it
                next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)  # (B,1)
                next_q = self.q_target(next_obs).gather(1, next_actions)  # (B,1)
            else:
                # Vanilla DQN: target net both selects and evaluates
                next_q = (
                    self.q_target(next_obs).max(dim=1, keepdim=True).values
                )  # (B,1)

            targets = rewards + self.gamma * next_q * (1.0 - dones)  # (B,1)

        # ── Loss and gradient step ───────────────────────────────────────────
        loss = self.loss_fn(q_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (important for stability with large replay buffers)
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_step += 1

        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "mean_q": q_taken.mean().item(),
        }

    def on_step_end(self, step: int) -> None:
        """Decay ε and periodically sync the target network."""
        self.epsilon = self.eps_schedule.value(step)
        if step % self.target_update_freq == 0:
            soft_update(self.q_target, self.q_net, tau=0.005)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net_state_dict": self.q_net.state_dict(),
                "q_target_state_dict": self.q_target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_step": self.training_step,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net_state_dict"])
        self.q_target.load_state_dict(ckpt["q_target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.training_step = ckpt.get("training_step", 0)
        self.epsilon = ckpt.get("epsilon", self.eps_schedule.end)
        print(f"[DQNAgent] Loaded checkpoint from {path}")
