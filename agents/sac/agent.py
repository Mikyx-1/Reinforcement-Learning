"""
Soft Actor-Critic – SAC (v2, with automatic entropy tuning)
Haarnoja et al., 2018  –  https://arxiv.org/abs/1801.01290
                          https://arxiv.org/abs/1812.05905  (auto-α addendum)

SAC is an off-policy actor-critic algorithm for continuous control that
optimises a maximum-entropy objective:

        J(π) = Σ_t E_{(s,a)∼π} [ r(s, a) + α · H(π(·|s)) ]

The entropy bonus α · H encourages exploration and yields multi-modal,
robust policies. SAC builds on the same off-policy machinery as DDPG/TD3
(replay buffer, twin critics, soft target update) but replaces three
DDPG pain-points with entropy regularisation:

  1. Exploration noise (DDPG: OU/Gaussian on the deterministic action)
     is replaced by sampling from a learned stochastic policy.
  2. Target-policy smoothing (TD3: clipped Gaussian on the target action)
     is unnecessary — the policy is already stochastic, so the target
     Q-network is queried at a *distribution* of actions rather than a
     spike, and the entropy term keeps π from collapsing.
  3. Policy delay (TD3) is unnecessary — the entropy bonus makes the Q
     surface smoother, so the actor gradient is stable enough to apply
     every step.

Algorithm (one update step):
    sample mini-batch {(s, a, r, s', d)} from B

    # Soft TD target — entropy of *next* action is included
    â', log π_θ(â'|s') ~ π_θ(·|s')
    y = r + γ · (1 − d) · [ min(Q_{φ₁}'(s', â'), Q_{φ₂}'(s', â'))
                            − α · log π_θ(â'|s') ]

    # Critic update — both Q-networks regress to the same target
    L_Q = MSE(Q_{φ₁}(s,a), y) + MSE(Q_{φ₂}(s,a), y)
    φ  ← φ − α_Q ∇_φ L_Q

    # Actor update (reparameterised; minimise expected α·logπ − Q)
    ã, log π_θ(ã|s) ~ π_θ(·|s)           (with rsample, grads to θ)
    L_π = mean( α · log π_θ(ã|s) − min(Q_{φ₁}(s,ã), Q_{φ₂}(s,ã)) )
    θ  ← θ − α_π ∇_θ L_π

    # Automatic temperature update (Haarnoja 2018 v2):
    L_α = mean( − log α · (log π_θ(ã|s) + H̄).detach() )
    log α ← log α − α_α ∇ L_α
    Heuristic target entropy:  H̄ = −act_dim

    # Soft Polyak update of critic targets
    φ' ← τ φ + (1 − τ) φ'                  (no actor target network)

Why no actor target?
  TD3 needs μ_θ' to build a deterministic target action. SAC's target is
  built by sampling from π_θ itself — the *current* actor — and the
  twin-critic min + entropy term provide enough regularisation that the
  bootstrap remains stable.

References:
    Haarnoja, T., Zhou, A., Abbeel, P. & Levine, S. (2018). Soft
    Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
    with a Stochastic Actor. ICML. arXiv:1801.01290.

    Haarnoja, T. et al. (2018). Soft Actor-Critic Algorithms and
    Applications. arXiv:1812.05905. (Adds the automatic α tuning used
    here; the original paper treated α as a fixed hyperparameter.)
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from agents.sac.networks import SquashedGaussianActor, TwinCritic
from common.utils import hard_update, soft_update


class SACAgent(BaseAgent):
    """
    Soft Actor-Critic with twin critics and (optionally) automatic α tuning.

    Continuous action spaces only.

    Args:
        env:               gymnasium.Env (used for space inference).
        hidden_dims:       Hidden layer sizes for actor and both critics.
        actor_lr:          Actor learning rate.
        critic_lr:         Critic learning rate (shared by Q₁, Q₂).
        alpha_lr:          Learning rate for log α (only used when
                           autotune_alpha=True).
        gamma:             Discount factor γ.
        tau:               Polyak averaging coefficient for critic targets.
        init_alpha:        Initial entropy coefficient α. When
                           autotune_alpha=True this is just the starting
                           point; when False it stays fixed.
        autotune_alpha:    If True (SAC v2, recommended), learn log α to
                           match `target_entropy`. If False, α is fixed
                           at `init_alpha`.
        target_entropy:    Target entropy H̄. If None, defaults to the
                           paper heuristic H̄ = −act_dim.
        device:            'cpu' or 'cuda'.
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_dims: list[int] = [256, 256],
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        init_alpha: float = 0.2,
        autotune_alpha: bool = True,
        target_entropy: float | None = None,
        device: str = "cpu",
    ):
        assert isinstance(
            env.action_space, gym.spaces.Box
        ), "SACAgent only supports continuous (Box) action spaces."

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))

        super().__init__(obs_dim=obs_dim, act_dim=act_dim, device=device)

        self.gamma = gamma
        self.tau = tau
        self.act_limit = float(env.action_space.high.flat[0])

        self.act_low = torch.as_tensor(
            env.action_space.low, dtype=torch.float32, device=self.device
        )
        self.act_high = torch.as_tensor(
            env.action_space.high, dtype=torch.float32, device=self.device
        )

        # ── Networks ─────────────────────────────────────────────────────────
        self.actor = SquashedGaussianActor(
            obs_dim, act_dim, hidden_dims, self.act_limit
        ).to(self.device)

        self.critic = TwinCritic(obs_dim, act_dim, hidden_dims).to(self.device)
        self.critic_target = TwinCritic(obs_dim, act_dim, hidden_dims).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.critic_target.eval()

        # ── Optimisers ───────────────────────────────────────────────────────
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_loss_fn = nn.MSELoss()

        # ── Temperature α ────────────────────────────────────────────────────
        # We optimise log α (rather than α directly) so the parameter is
        # unconstrained and α = exp(log α) stays strictly positive.
        self.autotune_alpha = autotune_alpha
        if autotune_alpha:
            self.target_entropy = (
                float(target_entropy) if target_entropy is not None else -float(act_dim)
            )
            self.log_alpha = torch.tensor(
                float(np.log(init_alpha)),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            # Fixed-α path: keep log α as a non-trainable buffer for a
            # uniform .alpha accessor below.
            self.target_entropy = 0.0
            self.log_alpha = torch.tensor(
                float(np.log(init_alpha)),
                dtype=torch.float32,
                device=self.device,
            )
            self.alpha_opt = None

        self._act_low_np = env.action_space.low
        self._act_high_np = env.action_space.high

    @property
    def alpha(self) -> torch.Tensor:
        """Current temperature, as a tensor (detached if α is fixed)."""
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = self.to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                # Greedy / evaluation action: mean of the squashed Gaussian
                # (no sampling, no entropy).
                _, _, mean_action = self.actor.sample(obs_t)
                action = mean_action
            else:
                action, _, _ = self.actor.sample(obs_t)
        action = action.cpu().numpy().squeeze(0)
        # Defensive clip — tanh output is already inside ±act_limit, but
        # downstream envs sometimes check strictly.
        return np.clip(action, self._act_low_np, self._act_high_np)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        One SAC update step: critic → actor → α → soft-update critic target.
        """
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"].reshape(-1, 1)
        next_obs = batch["next_obs"]
        dones = batch["dones"].reshape(-1, 1)

        # ── Critic update ───────────────────────────────────────────────────
        # Soft TD target: bootstrapped Q with entropy bonus on next action.
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            targets = rewards + self.gamma * q_next * (1.0 - dones)

        q1_pred, q2_pred = self.critic(obs, actions)
        critic_loss = self.critic_loss_fn(q1_pred, targets) + self.critic_loss_fn(
            q2_pred, targets
        )

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # ── Actor update ────────────────────────────────────────────────────
        # Freeze critic params for the actor pass — gradients flow only to
        # the actor through the reparameterised action.
        for p in self.critic.parameters():
            p.requires_grad = False

        sampled_action, log_prob, _ = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, sampled_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        # ── Temperature update ──────────────────────────────────────────────
        if self.autotune_alpha:
            # log α gradient:  ∂ L_α / ∂ log α = − (log π + H̄)
            # Detaching log π means α only adapts to the *current* entropy,
            # not back-propagating through the actor.
            alpha_loss = -(
                self.log_alpha * (log_prob.detach() + self.target_entropy)
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_val = alpha_loss.item()
        else:
            alpha_loss_val = 0.0

        # ── Soft target update ──────────────────────────────────────────────
        # No actor target network exists — SAC's target uses the *current*
        # actor; only the critic target is Polyak-averaged.
        soft_update(self.critic_target, self.critic, self.tau)

        self.training_step += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss_val,
            "alpha": self.alpha.item(),
            "mean_q1": q1_pred.mean().item(),
            "mean_q2": q2_pred.mean().item(),
            "mean_log_prob": log_prob.mean().item(),
            "entropy": -log_prob.mean().item(),
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_opt_state_dict": self.actor_opt.state_dict(),
            "critic_opt_state_dict": self.critic_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "training_step": self.training_step,
            "autotune_alpha": self.autotune_alpha,
            "target_entropy": self.target_entropy,
        }
        if self.autotune_alpha:
            ckpt["alpha_opt_state_dict"] = self.alpha_opt.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.critic_target.load_state_dict(ckpt["critic_target_state_dict"])
        self.actor_opt.load_state_dict(ckpt["actor_opt_state_dict"])
        self.critic_opt.load_state_dict(ckpt["critic_opt_state_dict"])

        # log α may be saved as a 0-dim tensor; copy into the live param
        # so optimiser state (if any) stays attached to the original.
        saved_log_alpha = ckpt["log_alpha"].to(self.device)
        with torch.no_grad():
            self.log_alpha.copy_(saved_log_alpha)

        if self.autotune_alpha and "alpha_opt_state_dict" in ckpt:
            self.alpha_opt.load_state_dict(ckpt["alpha_opt_state_dict"])

        self.training_step = ckpt.get("training_step", 0)
        print(f"[SACAgent] Loaded checkpoint from {path}")
