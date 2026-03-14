"""
PPO Actor-Critic networks.

Design philosophy (distinct from agents/actor_critic/networks.py):
  - SEPARATE actor and critic networks (no shared trunk).
  - PPO runs K update epochs per rollout — shared trunks cause conflicting
    gradients between the policy and value objectives across epochs,
    destabilising training. Separate parameters eliminate this coupling.
  - Orthogonal initialisation: gain=√2 for hidden layers,
    gain=0.01 for policy output (near-uniform distribution at init),
    gain=1.0 for value output.

Discrete  → CategoricalActorCritic
Continuous → GaussianActorCritic

Both expose the same interface:
    act(obs, deterministic) → (action, log_prob, entropy, value)
    evaluate(obs, actions)  → (log_prob, entropy, value)    ← used in update()
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from common.utils import build_mlp, init_weights


def _init_mlp(net: nn.Sequential, hidden_gain: float, output_gain: float) -> None:
    """Apply orthogonal init to every Linear layer in a Sequential MLP."""
    linear_layers = [m for m in net if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linear_layers):
        gain = output_gain if i == len(linear_layers) - 1 else hidden_gain
        init_weights(layer, gain=gain)


class CategoricalActorCritic(nn.Module):
    """
    Separate-parameter Actor-Critic for discrete action spaces (PPO).

    actor  : obs → MLP → logits → Categorical distribution
    critic : obs → MLP → scalar  V(s)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] = [64, 64],
    ):
        super().__init__()
        self.actor = build_mlp(obs_dim, act_dim, hidden_dims, activation=nn.Tanh)
        self.critic = build_mlp(obs_dim, 1, hidden_dims, activation=nn.Tanh)

        _init_mlp(self.actor, hidden_gain=2**0.5, output_gain=0.01)
        _init_mlp(self.critic, hidden_gain=2**0.5, output_gain=1.0)

    def _dist_and_value(self, obs: torch.Tensor):
        return Categorical(logits=self.actor(obs)), self.critic(obs).squeeze(-1)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action   : (B,) int64
            log_prob : (B,)
            entropy  : (B,)
            value    : (B,)
        """
        dist, value = self._dist_and_value(obs)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored actions under the *current* policy.
        Called inside PPO's K-epoch update loop.

        Returns: (log_prob, entropy, value)  all shape (B,)
        """
        dist, value = self._dist_and_value(obs)
        log_prob = dist.log_prob(actions.long().squeeze(-1))
        return log_prob, dist.entropy(), value


class GaussianActorCritic(nn.Module):
    """
    Separate-parameter Actor-Critic for continuous action spaces (PPO).

    actor  : obs → MLP → mean;  log_std is a free parameter vector
    critic : obs → MLP → scalar  V(s)
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] = [64, 64],
    ):
        super().__init__()
        self.actor_mean = build_mlp(obs_dim, act_dim, hidden_dims, activation=nn.Tanh)
        self.critic = build_mlp(obs_dim, 1, hidden_dims, activation=nn.Tanh)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        _init_mlp(self.actor_mean, hidden_gain=2**0.5, output_gain=0.01)
        _init_mlp(self.critic, hidden_gain=2**0.5, output_gain=1.0)

    def _dist_and_value(self, obs: torch.Tensor):
        mean = self.actor_mean(obs)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Normal(mean, log_std.exp()), self.critic(obs).squeeze(-1)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self._dist_and_value(obs)
        action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self._dist_and_value(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value
