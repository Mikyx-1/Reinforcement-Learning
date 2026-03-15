"""
Deep Deterministic Policy Gradient – DDPG
Lillicrap et al., 2016  –  https://arxiv.org/abs/1509.02971

DDPG extends DQN to continuous action spaces by combining:
  1. A deterministic policy (actor)  μ_θ(s) → a
  2. An action-value function (critic)  Q_φ(s, a) → scalar
  3. Experience replay (off-policy, like DQN)
  4. Soft target networks via Polyak averaging (unlike DQN's hard copy)
  5. Exploration via additive noise on the actor output

Algorithm (one step):
    Collect:
        a_t = clip( μ_θ(s_t) + ε_t,  a_low, a_high )   ε ~ OUNoise
        store (s_t, a_t, r_t, s_{t+1}, done) in B

    Update critic (minimise Bellman error):
        a'   = μ_θ'(s_{t+1})                     target actor
        y    = r + γ · Q_φ'(s_{t+1}, a') · (1-d) TD target
        L_Q  = MSE(Q_φ(s_t, a_t), y)
        φ ← φ - α_Q ∇_φ L_Q

    Update actor (maximise Q):
        L_μ  = -mean( Q_φ(s_t, μ_θ(s_t)) )
        θ ← θ - α_μ ∇_θ L_μ

    Soft update target networks:
        φ' ← τ φ + (1-τ) φ'
        θ' ← τ θ + (1-τ) θ'

Key properties:
  - Continuous action spaces only (actor outputs act_dim real values)
  - Off-policy — uses train_off_policy() from Trainer (no new loop needed)
  - Two optimisers: separate lr for actor and critic is standard practice
  - Soft target update (τ ≈ 0.005) instead of periodic hard copy
  - Action bounds enforced by actor Tanh + act_limit scaling

References:
    Lillicrap, T. P. et al. (2016). Continuous control with deep reinforcement
    learning. ICLR. arXiv:1509.02971.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from agents.ddpg.networks import Actor, Critic
from agents.ddpg.noise import GaussianNoise, OUNoise
from common.utils import hard_update, soft_update


class DDPGAgent(BaseAgent):
    """
    DDPG with soft target networks and OUNoise exploration.

    Continuous action spaces only.

    Args:
        env:             gymnasium.Env (used for space inference only).
        hidden_dims:     Hidden layer sizes for both actor and critic.
        actor_lr:        Actor learning rate.
        critic_lr:       Critic learning rate.
        gamma:           Discount factor γ.
        tau:             Polyak averaging coefficient for target networks.
                         θ_target ← τ·θ + (1−τ)·θ_target
        noise_type:      'ou' for Ornstein-Uhlenbeck, 'gaussian' for i.i.d. noise.
        noise_sigma:     Initial noise standard deviation / OU sigma.
        noise_sigma_min: Minimum noise sigma (Gaussian decay only).
        noise_sigma_decay: Per-episode multiplicative decay (Gaussian only).
        device:          'cpu' or 'cuda'.
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_dims: list[int] = [400, 300],
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 5e-3,
        noise_type: str = "ou",
        noise_sigma: float = 0.2,
        noise_sigma_min: float = 0.02,
        noise_sigma_decay: float = 0.999,
        device: str = "cpu",
    ):
        assert isinstance(
            env.action_space, gym.spaces.Box
        ), "DDPGAgent only supports continuous (Box) action spaces."

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))

        super().__init__(obs_dim=obs_dim, act_dim=act_dim, device=device)

        self.gamma = gamma
        self.tau = tau

        # Action bounds — used to scale actor output and clip actions
        self.act_limit = float(
            env.action_space.high.flat[0]
        )  # assumes symmetric bounds
        self.act_low = torch.as_tensor(
            env.action_space.low, dtype=torch.float32, device=self.device
        )
        self.act_high = torch.as_tensor(
            env.action_space.high, dtype=torch.float32, device=self.device
        )

        # ── Networks ─────────────────────────────────────────────────────────
        self.actor = Actor(obs_dim, act_dim, hidden_dims, self.act_limit).to(
            self.device
        )
        self.actor_target = Actor(obs_dim, act_dim, hidden_dims, self.act_limit).to(
            self.device
        )

        self.critic = Critic(obs_dim, act_dim, hidden_dims).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, hidden_dims).to(self.device)

        # Targets start as exact copies of the online networks
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Targets are never trained directly
        self.actor_target.eval()
        self.critic_target.eval()

        # ── Optimisers ───────────────────────────────────────────────────────
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.critic_loss_fn = nn.MSELoss()

        # ── Exploration noise ────────────────────────────────────────────────
        if noise_type == "ou":
            self.noise = OUNoise(act_dim, sigma=noise_sigma)
        elif noise_type == "gaussian":
            self.noise = GaussianNoise(
                act_dim,
                sigma=noise_sigma,
                sigma_min=noise_sigma_min,
                sigma_decay=noise_sigma_decay,
            )
        else:
            raise ValueError(
                f"noise_type must be 'ou' or 'gaussian', got '{noise_type}'."
            )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Deterministic action from actor + optional exploration noise.

        Args:
            obs:           (obs_dim,) observation.
            deterministic: If True, no noise is added (eval / greedy mode).

        Returns:
            action: (act_dim,) numpy array clipped to action bounds.
        """
        obs_t = self.to_tensor(obs).unsqueeze(0)  # (1, obs_dim)

        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().squeeze(0)  # (act_dim,)

        if not deterministic:
            action = action + self.noise.sample()

        # Hard clip to valid action range
        return np.clip(action, self.act_low.cpu().numpy(), self.act_high.cpu().numpy())

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        One gradient step for both actor and critic.

        Expected batch keys (from ReplayBuffer.sample()):
            obs      : (B, obs_dim)
            actions  : (B, act_dim)
            rewards  : (B, 1)
            next_obs : (B, obs_dim)
            dones    : (B, 1)

        Returns:
            metrics: {'critic_loss', 'actor_loss', 'mean_q'}
        """
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"].reshape(-1, 1)
        next_obs = batch["next_obs"]
        dones = batch["dones"].reshape(-1, 1)

        # ── Critic update ─────────────────────────────────────────────────────
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)  # (B, act_dim)
            q_next = self.critic_target(next_obs, next_actions)  # (B, 1)
            targets = rewards + self.gamma * q_next * (1.0 - dones)  # (B, 1)

        q_pred = self.critic(obs, actions)  # (B, 1)
        critic_loss = self.critic_loss_fn(q_pred, targets)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # ── Actor update ──────────────────────────────────────────────────────
        # Freeze critic parameters during actor update to save computation
        for p in self.critic.parameters():
            p.requires_grad = False

        actor_loss = -self.critic(obs, self.actor(obs)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # Unfreeze critic
        for p in self.critic.parameters():
            p.requires_grad = True

        # ── Soft target update ───────────────────────────────────────────────
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        self.training_step += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_q": q_pred.mean().item(),
        }

    def on_episode_end(self, episode: int, info: dict[str, Any]) -> None:
        """Reset OU noise state at episode boundaries; decay Gaussian sigma."""
        self.noise.reset()
        if hasattr(self.noise, "decay"):
            self.noise.decay()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_opt_state_dict": self.actor_opt.state_dict(),
                "critic_opt_state_dict": self.critic_opt.state_dict(),
                "training_step": self.training_step,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.actor_target.load_state_dict(ckpt["actor_target_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.critic_target.load_state_dict(ckpt["critic_target_state_dict"])
        self.actor_opt.load_state_dict(ckpt["actor_opt_state_dict"])
        self.critic_opt.load_state_dict(ckpt["critic_opt_state_dict"])
        self.training_step = ckpt.get("training_step", 0)
        print(f"[DDPGAgent] Loaded checkpoint from {path}")
