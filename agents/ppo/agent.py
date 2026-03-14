"""
Proximal Policy Optimisation – PPO-Clip
Schulman et al., 2017  –  https://arxiv.org/abs/1707.06347

Algorithm (one update cycle):
    1. Collect exactly T env steps under π_θ_old
       storing (obs, action, log_prob, reward, done, V(s))
    2. Bootstrap last value; compute GAE advantages Â_t and returns R_t
    3. For K epochs, iterate over shuffled mini-batches of size M:
         a. Re-evaluate actions under current π_θ  →  log π_θ, V_θ, H
         b. Ratio:  r_t = exp(log π_θ(a|s) − log π_old(a|s))
         c. Clipped surrogate:
              L^CLIP = E[min(r_t·Â_t,  clip(r_t, 1−ε, 1+ε)·Â_t)]
         d. Value loss (optionally clipped):
              L^VF = 0.5·E[(V_θ(s) − R_t)²]
         e. Entropy bonus:  L^H = E[H[π_θ(·|s)]]
         f. Total:  L = −L^CLIP + c_vf·L^VF − c_ent·L^H
         g. Gradient step with global norm clipping
         h. Optional early stop if approx KL > 1.5 × target_kl

What makes PPO different from Actor-Critic:
  - Clipped ratio prevents large policy updates (stability)
  - Fixed-T rollouts (not episode-aligned)
  - K update epochs per rollout (sample reuse)
  - Separate actor/critic networks (no shared trunk)
  - GAE for advantage estimation (not raw A_t = R_t − V(s))

References:
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
    Proximal Policy Optimization Algorithms. arXiv:1707.06347.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from agents.ppo.buffer import PPORolloutBuffer
from agents.ppo.networks import CategoricalActorCritic, GaussianActorCritic
from common.utils import explained_variance


class PPOAgent(BaseAgent):
    """
    PPO-Clip with separate Actor and Critic networks.

    Supports both discrete and continuous action spaces.

    Args:
        env:             gymnasium.Env (used for space inference only).
        hidden_dims:     MLP hidden layer sizes for both actor and critic.
        lr:              Learning rate (Adam, shared for actor and critic).
        gamma:           Discount factor γ.
        lam:             GAE λ.  Higher = less bias, more variance.
        clip_eps:        Clipping parameter ε.  Typically 0.1–0.2.
        n_epochs:        Update epochs per rollout (K).
        batch_size:      Mini-batch size within each epoch.
        rollout_steps:   Env steps collected before each update (T).
        vf_coef:         Value loss coefficient c_vf.
        ent_coef:        Entropy bonus coefficient c_ent.
        max_grad_norm:   Gradient clipping max norm.
        clip_value_loss: Apply PPO-style value clipping.
        target_kl:       Early-stop KL threshold (None = disabled).
        device:          'cpu' or 'cuda'.
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_dims: list[int] = [64, 64],
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        rollout_steps: int = 2048,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        clip_value_loss: bool = True,
        target_kl: float | None = 0.015,
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

        # Hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.clip_value_loss = clip_value_loss
        self.target_kl = target_kl

        # Separate actor and critic networks
        NetClass = CategoricalActorCritic if self.discrete else GaussianActorCritic
        self.ac = NetClass(obs_dim, act_dim, hidden_dims).to(self.device)

        # Single Adam optimiser over both networks
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)

        # Fixed-length rollout buffer
        self.buffer = PPORolloutBuffer(
            rollout_steps=rollout_steps,
            obs_dim=obs_dim,
            act_dim=1 if self.discrete else act_dim,
            gamma=gamma,
            lam=lam,
            device=device,
        )

        # Cached from last select_action() call
        self._last_value: float = 0.0

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, float]:
        """
        Sample action and cache value estimate.

        Returns (action, log_prob) — same contract as other on-policy agents,
        so the Trainer can call this uniformly.
        The cached value is used inside collect_step().
        """
        obs_t = self.to_tensor(obs).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _, value = self.ac.act(obs_t, deterministic=deterministic)

        self._last_value = float(value.cpu().item())
        action_np = action.cpu().numpy().squeeze()
        log_prob_np = float(log_prob.cpu().item())
        return action_np, log_prob_np

    def collect_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
    ) -> None:
        """
        Store one transition in the rollout buffer.
        Must be called immediately after select_action() (which caches the value).
        """
        self.buffer.push(
            obs=obs,
            action=action,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=self._last_value,
        )

    def finish_rollout(self, last_obs: np.ndarray, last_done: bool) -> None:
        """
        Bootstrap the value of the state after the last collected step,
        then compute GAE advantages and lambda-returns.

        Call once the buffer is full (buffer.is_full()), before update().

        Args:
            last_obs:  The observation after the last env step.
            last_done: Whether the last step terminated the episode.
        """
        with torch.no_grad():
            last_obs_t = self.to_tensor(last_obs).unsqueeze(0)
            _, _, _, last_value = self.ac.act(last_obs_t)
        bootstrap = 0.0 if last_done else float(last_value.cpu().item())
        self.buffer.compute_gae(last_value=bootstrap)

    def update(self, batch: dict | None = None) -> dict[str, float]:
        """
        Run K PPO update epochs over the collected rollout.

        `batch` parameter is ignored — PPO manages its own buffer.
        Call finish_rollout() before update().

        Returns:
            Averaged metrics across all epochs × mini-batches.
        """
        metrics_acc: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
            "explained_variance": [],
        }

        for _ in range(self.n_epochs):
            for mini_batch in self.buffer.get_batches(self.batch_size):
                step_metrics = self._update_minibatch(mini_batch)
                for k, v in step_metrics.items():
                    metrics_acc[k].append(v)

                # Early stop if KL blows up
                if self.target_kl is not None:
                    if step_metrics["approx_kl"] > 1.5 * self.target_kl:
                        break

        self.training_step += 1
        self.buffer.clear()

        return {k: float(np.mean(v)) for k, v in metrics_acc.items() if v}

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ac_state_dict": self.ac.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_step": self.training_step,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["ac_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.training_step = ckpt.get("training_step", 0)
        print(f"[PPOAgent] Loaded checkpoint from {path}")

    # ------------------------------------------------------------------
    # Private: single mini-batch update
    # ------------------------------------------------------------------

    def _update_minibatch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        # Re-evaluate current policy on stored observations
        log_probs, entropy, values = self.ac.evaluate(obs, actions)

        # ── Clipped surrogate objective ──────────────────────────────────────
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # ── Value loss (optionally clipped) ──────────────────────────────────
        if self.clip_value_loss:
            # Reconstruct old values from returns and (un-normalised) advantages
            # Returns = old_values + old_advantages; we use returns as the target
            v_clipped = (
                returns
                - advantages
                + torch.clamp(
                    values - (returns - advantages), -self.clip_eps, self.clip_eps
                )
            )
            value_loss = (
                0.5
                * torch.max(
                    (values - returns).pow(2),
                    (v_clipped - returns).pow(2),
                ).mean()
            )
        else:
            value_loss = 0.5 * (values - returns).pow(2).mean()

        # ── Entropy bonus ────────────────────────────────────────────────────
        entropy_loss = entropy.mean()

        # ── Combined loss ─────────────────────────────────────────────────────
        total_loss = (
            policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # ── Diagnostics ───────────────────────────────────────────────────────
        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean().item()
            clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean().item()
            ev = explained_variance(
                values.detach().cpu().numpy(),
                returns.detach().cpu().numpy(),
            )

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_frac,
            "explained_variance": ev,
        }
