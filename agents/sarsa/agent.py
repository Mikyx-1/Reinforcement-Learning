"""
SARSA – State-Action-Reward-State-Action
Rummery & Niranjan, 1994; Sutton & Barto, 2018 (Chapter 6.4)

SARSA is the on-policy counterpart of Q-learning. The crucial difference:

    Q-learning (off-policy):
        y = r + γ · max_{a'} Q(s', a')   ← hypothetical greedy action

    SARSA (on-policy):
        y = r + γ · Q(s', a')            ← a' is the ACTUAL next action taken

Because the update uses the action that the agent *will actually take* (sampled
from the same ε-greedy policy), SARSA learns the value of the *behaviour policy*,
not the optimal policy. This makes it:
  - Safer in stochastic/dangerous environments (avoids cliff edges, etc.)
  - More conservative under ε-greedy exploration
  - Convergent to the optimal policy as ε → 0

Algorithm (one step):
    1. Observe s_t; select a_t via ε-greedy  (this is the behaviour policy)
    2. Execute a_t; observe r_t, s_{t+1}
    3. Select a_{t+1} via ε-greedy from s_{t+1}   ← must do BEFORE update
    4. TD target:  y_t = r_t + γ · Q(s_{t+1}, a_{t+1}) · (1 − done_t)
    5. Loss:       L   = MSE(Q(s_t, a_t),  y_t)
    6. Gradient step
    7. s_t ← s_{t+1},  a_t ← a_{t+1}   (carry forward the chosen next action)

Training loop implication:
    The standard Trainer.train_off_policy() cannot be used because it does
    not pass `next_action` to update().  Trainer.train_sarsa() handles this
    by maintaining the (s, a, r, s', a') quintuple within the loop.

References:
    Rummery, G. A., & Niranjan, M. (1994). On-line Q-learning using
    connectionist systems. Technical Report CUED/F-INFENG/TR 166.

    Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning:
    An Introduction (2nd ed.), Chapter 6.4.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from agents.sarsa.networks import QNetwork
from common.schedulers import LinearSchedule


class SarsaAgent(BaseAgent):
    """
    SARSA with neural function approximation and ε-greedy exploration.

    Discrete action spaces only.

    Args:
        env:              gymnasium.Env (used for space inference only).
        hidden_dims:      Hidden layer sizes. Use [] for linear approximation.
        lr:               Adam learning rate.
        gamma:            Discount factor γ.
        eps_start:        Initial ε for ε-greedy exploration.
        eps_end:          Final ε after annealing.
        eps_decay_steps:  Steps over which ε decays linearly to eps_end.
        device:           'cpu' or 'cuda'.
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
        device: str = "cpu",
    ):
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "SarsaAgent only supports discrete action spaces."
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = env.action_space.n

        super().__init__(obs_dim=obs_dim, act_dim=act_dim, device=device)

        self.gamma = gamma

        # ε-greedy schedule (same interface as DQN for easy comparison)
        self.eps_schedule = LinearSchedule(
            start=eps_start,
            end=eps_end,
            duration=eps_decay_steps,
        )
        self.epsilon = eps_start

        # Single Q-network (no target network — on-policy TD is more stable)
        self.q_net = QNetwork(obs_dim, act_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        ε-greedy action selection.

        This method is used both for the *current* step action and for
        selecting the *next* action a_{t+1} in the SARSA loop — the same
        policy is used in both places, which is the defining property of
        on-policy learning.

        Args:
            obs:           (obs_dim,) observation.
            deterministic: If True, always greedy (used at eval time).

        Returns:
            action: scalar int array.
        """
        if not deterministic and np.random.random() < self.epsilon:
            return np.array(np.random.randint(self.act_dim))

        obs_t = self.to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_t)  # (1, act_dim)
        return np.array(q_values.argmax(dim=1).item())

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        One SARSA TD update.

        Expected batch keys — NOTE the extra 'next_actions' key, which
        distinguishes SARSA from DQN and requires Trainer.train_sarsa():

            obs          : (B, obs_dim)
            actions      : (B, 1)   float32, cast to long
            rewards      : (B, 1)
            next_obs     : (B, obs_dim)
            dones        : (B, 1)
            next_actions : (B, 1)   float32, cast to long  ← SARSA-specific

        Returns:
            metrics: {'loss': float, 'epsilon': float, 'mean_q': float}
        """
        obs = batch["obs"]
        actions = batch["actions"].long().reshape(-1)  # (B,) — safe for (B,) or (B,1)
        rewards = batch["rewards"].reshape(-1, 1)  # (B, 1)
        next_obs = batch["next_obs"]
        dones = batch["dones"].reshape(-1, 1)  # (B, 1)
        next_actions = batch["next_actions"].long().reshape(-1)  # (B,)

        # ── Current Q-values for the taken actions ───────────────────────────
        q_values = self.q_net(obs)  # (B, act_dim)
        q_taken = q_values.gather(1, actions.unsqueeze(1))  # (B, 1)

        # ── SARSA TD target: use Q(s', a') where a' is the ACTUAL next action ─
        with torch.no_grad():
            next_q_values = self.q_net(next_obs)  # (B, act_dim)
            next_q_taken = next_q_values.gather(1, next_actions.unsqueeze(1))  # (B, 1)
            targets = rewards + self.gamma * next_q_taken * (1.0 - dones)

        # ── Loss and gradient step ───────────────────────────────────────────
        loss = self.loss_fn(q_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_step += 1

        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "mean_q": q_taken.mean().item(),
        }

    def on_step_end(self, step: int) -> None:
        """Decay ε after every environment step."""
        self.epsilon = self.eps_schedule.value(step)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net_state_dict": self.q_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_step": self.training_step,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.training_step = ckpt.get("training_step", 0)
        self.epsilon = ckpt.get("epsilon", self.eps_schedule.end)
        print(f"[SarsaAgent] Loaded checkpoint from {path}")
