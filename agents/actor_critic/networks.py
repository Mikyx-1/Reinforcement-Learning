"""
Actor-Critic networks.

Design:
  - Actor and Critic share a feature-extraction trunk (shared_net).
    This is the classic A2C/A3C design — fewer parameters, faster training,
    and the shared representation benefits both heads.
  - Actor head  : trunk → logits (discrete) or mean+log_std (continuous)
  - Critic head : trunk → scalar V(s)

Why shared trunk here, versus separate nets in PPO?
  Actor-Critic (A2C style) is typically used with small networks and fast
  environments where shared features are a net win.  PPO uses longer rollouts
  and multiple update epochs, where shared parameters can cause conflicting
  gradients — separate nets are safer there.

Discrete  → CategoricalActorCritic
Continuous → GaussianActorCritic
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from common.utils import build_mlp, init_weights


class CategoricalActorCritic(nn.Module):
    """
    Shared-trunk Actor-Critic for discrete action spaces.

    Architecture:
        obs → shared MLP → ┬→ actor head → logits → Categorical
                           └→ critic head → V(s)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] = [128, 128],
    ):
        super().__init__()

        # Shared feature extractor
        self.shared = build_mlp(
            obs_dim, hidden_dims[-1], hidden_dims[:-1], activation=nn.Tanh
        )

        # Actor head: hidden → logits
        self.actor_head = nn.Linear(hidden_dims[-1], act_dim)

        # Critic head: hidden → scalar
        self.critic_head = nn.Linear(hidden_dims[-1], 1)

        # Init: hidden layers gain=√2, actor out gain=0.01, critic out gain=1
        for m in self.shared.modules():
            if isinstance(m, nn.Linear):
                init_weights(m, gain=2**0.5)
        init_weights(self.actor_head, gain=0.01)
        init_weights(self.critic_head, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """Returns (Categorical distribution, value scalar per sample)."""
        feat = self.shared(obs)
        dist = Categorical(logits=self.actor_head(feat))
        value = self.critic_head(feat).squeeze(-1)
        return dist, value

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step action selection.

        Returns:
            action   : (B,) int64
            log_prob : (B,)
            entropy  : (B,)
            value    : (B,)
        """
        dist, value = self.forward(obs)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class GaussianActorCritic(nn.Module):
    """
    Shared-trunk Actor-Critic for continuous action spaces.

    Architecture:
        obs → shared MLP → ┬→ actor head → mean
                           │   log_std (free parameter, state-independent)
                           └→ critic head → V(s)
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] = [128, 128],
    ):
        super().__init__()

        self.shared = build_mlp(
            obs_dim, hidden_dims[-1], hidden_dims[:-1], activation=nn.Tanh
        )
        self.actor_head = nn.Linear(hidden_dims[-1], act_dim)
        self.critic_head = nn.Linear(hidden_dims[-1], 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        for m in self.shared.modules():
            if isinstance(m, nn.Linear):
                init_weights(m, gain=2**0.5)
        init_weights(self.actor_head, gain=0.01)
        init_weights(self.critic_head, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        feat = self.shared(obs)
        mean = self.actor_head(feat)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        dist = Normal(mean, log_std.exp())
        value = self.critic_head(feat).squeeze(-1)
        return dist, value

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value
