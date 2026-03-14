"""
Actor-Critic (A2C – Advantage Actor-Critic)
Mnih et al., 2016  –  https://arxiv.org/abs/1602.01783

Algorithm (one episode = one update):
    1. Collect full episode τ under π_θ
    2. Compute discounted returns  R_t = Σ_{k≥t} γ^{k-t} r_k
    3. Compute TD advantages       A_t = R_t − V_φ(s_t)
    4. Normalise A_t  (reduces variance)
    5. Actor  loss: L_π  = −E[A_t · log π_θ(a_t|s_t)]
    6. Critic loss: L_V  =  E[(R_t − V_φ(s_t))²]
    7. Entropy bonus:  L_H = −E[H[π_θ(·|s_t)]]
    8. Combined:  L = L_π + c_vf·L_V + c_ent·L_H
    9. Single gradient step on all parameters simultaneously

Key differences from REINFORCE:
  - Critic V(s) subtracts a state-dependent baseline → much lower variance
  - Actor and Critic share a feature trunk → joint representation learning
  - Single network, single optimiser

Key differences from PPO:
  - No clipping, no importance-sampling ratio
  - One gradient step per episode (no K-epoch reuse)
  - Per-episode rollout (not fixed T steps)
  - Simpler, faster, but less sample-efficient

References:
    Mnih, V. et al. (2016). Asynchronous Methods for Deep Reinforcement
    Learning. ICML. arXiv:1602.01783.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.actor_critic.networks import (CategoricalActorCritic,
                                          GaussianActorCritic)
from agents.base_agent import BaseAgent


class ActorCriticAgent(BaseAgent):
    """
    A2C-style Actor-Critic with shared trunk.

    Supports both discrete (CartPole, LunarLander) and continuous
    (Pendulum, MuJoCo) environments. Action space is inferred from `env`.

    Args:
        env:          gymnasium.Env (used for space inference only).
        hidden_dims:  MLP hidden layer sizes (last dim = shared trunk output).
        lr:           Learning rate for the combined actor-critic network.
        gamma:        Discount factor γ.
        vf_coef:      Weight on the critic (value) loss term.
        ent_coef:     Weight on the entropy bonus (encourages exploration).
        max_grad_norm: Gradient clipping norm.
        device:       'cpu' or 'cuda'.
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_dims: list[int] = [128, 128],
        lr: float = 7e-4,
        gamma: float = 0.99,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
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
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        # Shared Actor-Critic network
        NetClass = CategoricalActorCritic if self.discrete else GaussianActorCritic
        self.ac = NetClass(obs_dim, act_dim, hidden_dims).to(self.device)

        # Single optimiser for both actor and critic
        self.optimizer = optim.RMSprop(self.ac.parameters(), lr=lr, eps=1e-5)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, float]:
        """
        Sample action from the current policy.

        Returns (action, log_prob) so the Trainer's on-policy loop can
        store the log_prob in the rollout buffer.
        Also caches the value estimate for use in collect_step().
        """
        obs_t = self.to_tensor(obs).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _, value = self.ac.act(obs_t, deterministic=deterministic)

        self._last_value = float(value.cpu().item())
        action_np = action.cpu().numpy().squeeze()
        log_prob_np = float(log_prob.cpu().item())
        return action_np, log_prob_np

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        One gradient step using a full episode's transitions.

        Expected batch keys (produced by RolloutBuffer.get_with_values()):
            obs       : (T, obs_dim)
            actions   : (T,) or (T, act_dim)
            returns   : (T,)   Monte-Carlo discounted returns R_t

        Returns:
            metrics dict with actor_loss, critic_loss, entropy, total_loss.
        """
        obs = batch["obs"]  # (T, obs_dim)
        actions = batch["actions"]  # (T,) or (T, act_dim)
        returns = batch["returns"]  # (T,)

        # ── Re-evaluate under current policy ────────────────────────────────
        if self.discrete:
            dist, values = self.ac(obs)
            log_probs = dist.log_prob(actions.long().squeeze(-1))
            entropy = dist.entropy()
        else:
            dist, values = self.ac(obs)
            log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)

        # ── Advantage = Returns − V(s)  (detach critic from actor loss) ─────
        advantages = returns - values.detach()

        # Normalise advantages over the episode (variance reduction)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Losses ───────────────────────────────────────────────────────────
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = 0.5 * (returns - values).pow(2).mean()  # MSE
        entropy_loss = entropy.mean()

        total_loss = (
            actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss
        )

        # ── Gradient step ────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.training_step += 1

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy_loss.item(),
            "total_loss": total_loss.item(),
        }

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
        print(f"[ActorCriticAgent] Loaded checkpoint from {path}")
