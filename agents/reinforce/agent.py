"""
REINFORCE – Monte-Carlo Policy Gradient
(Williams, 1992)

Algorithm:
    For each episode:
        1. Collect trajectory τ = {s₀, a₀, r₀, ..., s_T}  under π_θ
        2. Compute discounted returns  G_t = Σ_{k≥t} γ^{k-t} r_k
        3. Optionally normalise G_t  (reduces variance)
        4. Policy gradient:
               ∇_θ J(θ) ≈ Σ_t G_t · ∇_θ log π_θ(a_t | s_t)
        5. Gradient ascent:  θ ← θ + α · ∇_θ J(θ)

Variants included:
    entropy_coef > 0  → entropy regularisation (encourages exploration)

References:
    Williams, R. J. (1992). Simple statistical gradient-following algorithms
    for connectionist reinforcement learning. Machine Learning, 8, 229–256.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from agents.reinforce.networks import CategoricalPolicy, GaussianPolicy


class ReinforceAgent(BaseAgent):
    """
    REINFORCE with optional entropy regularisation.

    Works for both discrete (CartPole) and continuous (Pendulum) envs.
    Action space is inferred automatically from `env`.

    Args:
        env:            gymnasium.Env instance (used only for space info).
        hidden_dims:    Hidden layer sizes for the policy MLP.
        lr:             Policy network learning rate.
        gamma:          Discount factor.
        entropy_coef:   Entropy bonus coefficient (0 = vanilla REINFORCE).
        device:         'cpu' or 'cuda'.
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_dims: list[int] = [64, 64],
        lr: float = 3e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.0,
        device: str = "cpu",
    ):
        obs_dim = int(np.prod(env.observation_space.shape))
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        act_dim = (
            env.action_space.n
            if self.discrete
            else int(np.prod(env.action_space.shape))
        )

        super().__init__(obs_dim=obs_dim, act_dim=act_dim, device=device)

        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # Policy network
        if self.discrete:
            self.policy = CategoricalPolicy(obs_dim, act_dim, hidden_dims).to(
                self.device
            )
        else:
            self.policy = GaussianPolicy(obs_dim, act_dim, hidden_dims).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, float]:
        """
        Returns (action, log_prob).
        The Trainer's on-policy loop unpacks this tuple.
        """
        obs_t = self.to_tensor(obs).unsqueeze(0)  # (1, obs_dim)

        with torch.no_grad():
            action, log_prob, _ = self.policy.act(obs_t, deterministic=deterministic)

        action_np = action.cpu().numpy().squeeze()
        log_prob_np = log_prob.cpu().item()
        return action_np, log_prob_np

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        One gradient step using a full episode's worth of data.

        Expected batch keys (from RolloutBuffer.get()):
            obs       : (T, obs_dim)
            actions   : (T,) or (T, act_dim)
            log_probs : (T,)  — stale; we recompute them for correctness
            returns   : (T,)  — normalised Monte-Carlo returns

        Returns:
            metrics dict with 'policy_loss' and 'entropy'.
        """
        obs = batch["obs"]
        actions = batch["actions"]
        returns = batch["returns"]

        # Re-evaluate log probs under current policy (important if
        # the rollout was collected many steps ago, though for
        # vanilla REINFORCE with 1-episode rollouts this is fresh).
        if self.discrete:
            dist = self.policy(obs)
            log_probs = dist.log_prob(actions.long().squeeze(-1))
            entropy = dist.entropy().mean()
        else:
            dist = self.policy(obs)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

        # Policy gradient loss (we minimise negative because optimisers descend)
        policy_loss = -(log_probs * returns).mean()

        # Entropy bonus (maximise entropy = minimise negative entropy)
        loss = policy_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (prevents rare large updates on long episodes)
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.training_step += 1

        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_step": self.training_step,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        print(f"[ReinforceAgent] Loaded checkpoint from {path}")
