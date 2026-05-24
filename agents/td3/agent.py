"""
Twin Delayed Deep Deterministic Policy Gradient – TD3
Fujimoto, van Hoof & Meger, 2018  –  https://arxiv.org/abs/1802.09477

TD3 fixes three failure modes of DDPG:

  1. Overestimation bias of the critic.
     DDPG's TD target  y = r + γ Q'(s', μ'(s'))  is biased upward because
     the max operator (implicit in the actor maximising Q) amplifies noise.
     TD3 keeps two critics Q₁, Q₂ and uses the minimum:
         y = r + γ · min(Q₁'(s', ã'), Q₂'(s', ã')) · (1 − d)

  2. High-variance actor updates from noisy critic Q-values.
     TD3 updates the actor (and all target networks) only every
     `policy_delay` critic updates — the critic converges enough between
     actor updates for the policy gradient to be meaningful.

  3. Overfitting to narrow ridges in Q.
     The critic regresses (s, a) → y. If Q has a sharp peak in `a`, the
     deterministic actor exploits it even when it is an artifact.
     Target-policy smoothing adds *clipped* Gaussian noise to ã' so the
     critic fits a small neighbourhood, not a point:
         ã' = clip( μ'(s') + clip(ε, −c, c),  a_low, a_high ),  ε ~ N(0, σ̃²)

Algorithm (one update step):
    sample mini-batch {(s, a, r, s', d)} from B

    # Target action with smoothing
    ε      ~ N(0, σ̃²)                       (per-element)
    ε      = clip(ε, −noise_clip, +noise_clip)
    ã'     = clip( μ_θ'(s') + ε,  a_low,  a_high )

    # Twin critic target (min reduces overestimation)
    y      = r + γ · min(Q_{φ₁}'(s', ã'), Q_{φ₂}'(s', ã')) · (1 − d)

    # Critic update — both critics regress to the same target
    L_Q    = MSE(Q_{φ₁}(s,a), y) + MSE(Q_{φ₂}(s,a), y)
    φ      ← φ − α_Q ∇_φ L_Q

    # Delayed actor + target update
    if t mod policy_delay == 0:
        L_μ    = −mean( Q_{φ₁}(s, μ_θ(s)) )      # only Q₁ — paper convention
        θ      ← θ − α_μ ∇_θ L_μ
        φ'  ← τ φ  + (1−τ) φ'
        θ'  ← τ θ  + (1−τ) θ'

Why these three changes work together:
  - Twin critic gives the actor a *conservative* gradient (won't chase
    overestimated regions).
  - Delayed updates ensure the actor only sees critic values from a
    near-converged Q.
  - Target smoothing prevents the converged Q from being a spiky function
    that the actor can over-exploit.

References:
    Fujimoto, S., van Hoof, H. & Meger, D. (2018). Addressing Function
    Approximation Error in Actor-Critic Methods. ICML. arXiv:1802.09477.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from agents.ddpg.noise import GaussianNoise, OUNoise
from agents.td3.networks import Actor, TwinCritic
from common.utils import hard_update, soft_update


class TD3Agent(BaseAgent):
    """
    TD3: DDPG + twin critics + delayed updates + target-policy smoothing.

    Continuous action spaces only.

    Args:
        env:               gymnasium.Env (used for space inference).
        hidden_dims:       Hidden layer sizes for actor and both critics.
        actor_lr:          Actor learning rate.
        critic_lr:         Critic learning rate (shared by Q₁, Q₂).
        gamma:             Discount factor γ.
        tau:               Polyak averaging coefficient for target networks.
        policy_delay:      Apply actor + target updates once per this many
                           critic updates. TD3 paper uses 2.
        target_noise_sigma: Std of the Gaussian noise added to target actions
                           for smoothing. Paper uses 0.2 (× act_limit).
        target_noise_clip: Clip range for the smoothing noise. Paper uses 0.5.
        noise_type:        Exploration noise for behaviour policy:
                           'gaussian' (paper) or 'ou' (DDPG-style).
        noise_sigma:       Std (Gaussian) or volatility (OU).
        noise_sigma_min:   Floor for Gaussian decay (unused for OU).
        noise_sigma_decay: Per-episode multiplicative decay (Gaussian only).
        device:            'cpu' or 'cuda'.
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_dims: list[int] = [400, 300],
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        policy_delay: int = 2,
        target_noise_sigma: float = 0.2,
        target_noise_clip: float = 0.5,
        noise_type: str = "gaussian",
        noise_sigma: float = 0.1,
        noise_sigma_min: float = 0.01,
        noise_sigma_decay: float = 1.0,
        device: str = "cpu",
    ):
        assert isinstance(
            env.action_space, gym.spaces.Box
        ), "TD3Agent only supports continuous (Box) action spaces."

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))

        super().__init__(obs_dim=obs_dim, act_dim=act_dim, device=device)

        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        # Both σ̃ and the clip are expressed in *action* units (i.e. already
        # scaled by act_limit). The paper specifies σ̃=0.2, c=0.5 for
        # act_limit=1 tasks; scaling here keeps the same relative magnitude
        # for envs like Pendulum where act_limit=2.
        self.act_limit = float(env.action_space.high.flat[0])
        self.target_noise_sigma = target_noise_sigma * self.act_limit
        self.target_noise_clip = target_noise_clip * self.act_limit

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

        self.critic = TwinCritic(obs_dim, act_dim, hidden_dims).to(self.device)
        self.critic_target = TwinCritic(obs_dim, act_dim, hidden_dims).to(self.device)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self.actor_target.eval()
        self.critic_target.eval()

        # ── Optimisers ───────────────────────────────────────────────────────
        # One optimiser covers both Q₁ and Q₂ because they live inside the
        # same nn.Module — gradients from L_Q₁ + L_Q₂ update both at once.
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.critic_loss_fn = nn.MSELoss()

        # ── Exploration noise ────────────────────────────────────────────────
        if noise_type == "gaussian":
            self.noise = GaussianNoise(
                act_dim,
                sigma=noise_sigma * self.act_limit,
                sigma_min=noise_sigma_min * self.act_limit,
                sigma_decay=noise_sigma_decay,
            )
        elif noise_type == "ou":
            self.noise = OUNoise(act_dim, sigma=noise_sigma)
        else:
            raise ValueError(
                f"noise_type must be 'gaussian' or 'ou', got '{noise_type}'."
            )

        # Counter for delayed actor + target updates. Tracks number of
        # critic updates so we can fire actor+target every `policy_delay`-th.
        self._critic_updates = 0
        # Cache the most recent actor loss so logs are stable on non-actor
        # update steps (otherwise the metric would disappear in W&B charts).
        self._last_actor_loss: float = 0.0

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = self.to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().squeeze(0)

        if not deterministic:
            action = action + self.noise.sample()

        return np.clip(action, self.act_low.cpu().numpy(), self.act_high.cpu().numpy())

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        One TD3 step:
          - critic update every call
          - actor + target update every `policy_delay`-th call
        """
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"].reshape(-1, 1)
        next_obs = batch["next_obs"]
        dones = batch["dones"].reshape(-1, 1)

        # ── Build target (smoothed action + min over twin critics) ──────────
        with torch.no_grad():
            # Target-policy smoothing: clipped Gaussian noise on target action
            noise = torch.randn_like(actions) * self.target_noise_sigma
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)

            next_actions = self.actor_target(next_obs) + noise
            next_actions = torch.max(
                torch.min(next_actions, self.act_high), self.act_low
            )

            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next)
            targets = rewards + self.gamma * q_next * (1.0 - dones)

        # ── Critic update (both Q-networks regress to the same target) ─────
        q1_pred, q2_pred = self.critic(obs, actions)
        critic_loss = self.critic_loss_fn(q1_pred, targets) + self.critic_loss_fn(
            q2_pred, targets
        )

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        self._critic_updates += 1
        self.training_step += 1

        metrics: dict[str, float] = {
            "critic_loss": critic_loss.item(),
            "mean_q1": q1_pred.mean().item(),
            "mean_q2": q2_pred.mean().item(),
            "actor_loss": self._last_actor_loss,
        }

        # ── Delayed actor + target update ───────────────────────────────────
        if self._critic_updates % self.policy_delay == 0:
            # Freeze critic params during the actor update to save compute.
            for p in self.critic.parameters():
                p.requires_grad = False

            # Paper uses Q₁ only for the actor loss; Q₂ exists purely to make
            # the *target* conservative. Using min(Q₁, Q₂) here too would
            # bias the actor away from genuinely good actions.
            actor_loss = -self.critic.q1_only(obs, self.actor(obs)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_opt.step()

            for p in self.critic.parameters():
                p.requires_grad = True

            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

            self._last_actor_loss = actor_loss.item()
            metrics["actor_loss"] = self._last_actor_loss

        return metrics

    def on_episode_end(self, episode: int, info: dict[str, Any]) -> None:
        """Reset OU state (no-op for Gaussian); decay Gaussian sigma if enabled."""
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
                "critic_updates": self._critic_updates,
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
        self._critic_updates = ckpt.get("critic_updates", 0)
        print(f"[TD3Agent] Loaded checkpoint from {path}")
