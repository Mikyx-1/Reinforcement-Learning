"""
Actor and Critic networks for DDPG.

Actor  : obs → μ(s) ∈ ℝ^{act_dim}     deterministic policy, output tanh-scaled
Critic : (obs, action) → Q(s,a) ∈ ℝ   action-value function

Key design decisions:

Actor output activation — Tanh:
  DDPG targets continuous action spaces that are bounded (e.g. Pendulum
  actions ∈ [-2, 2]).  We output tanh(x) ∈ (-1, 1) and scale by act_limit
  so the output lives in (-act_limit, act_limit).  This avoids the policy
  ever saturating the action bounds (which would zero the gradient).

Critic architecture — action injected at the first hidden layer:
  The original DDPG paper injects the action after the first hidden layer
  rather than at the input.  This lets the first layer learn a nonlinear
  encoding of the observation before mixing in the action signal.
  Concretely: obs → Linear → ReLU → concat(hidden, action) → ... → Q

Weight initialisation:
  - Hidden layers: fan-in uniform initialisation  1/√fan_in  (original paper)
  - Final layers:  small uniform init  ±3e-3  to ensure near-zero initial
    Q-values and near-zero initial actions, preventing early divergence.
"""

import numpy as np
import torch
import torch.nn as nn


def _fanin_init(layer: nn.Linear) -> None:
    """Uniform initialisation in [-1/√fan_in, +1/√fan_in]."""
    fan_in = layer.weight.data.size(1)
    bound = 1.0 / np.sqrt(fan_in)
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


def _final_init(layer: nn.Linear, bound: float = 3e-3) -> None:
    """Small uniform init for output layers — keeps initial outputs near zero."""
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


class Actor(nn.Module):
    """
    Deterministic policy  μ_θ(s) → a.

    Architecture:
        obs → Linear(obs_dim, h1) → ReLU
            → Linear(h1, h2)      → ReLU
            → Linear(h2, act_dim) → Tanh → scale by act_limit

    Args:
        obs_dim:    Observation dimensionality.
        act_dim:    Action dimensionality.
        hidden_dims: [h1, h2, ...] hidden layer sizes.
        act_limit:  Action bound — output will be in [-act_limit, act_limit].
                    For multi-dimensional spaces, pass a scalar (same bound
                    per dimension) or a np.ndarray of per-dim limits.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] = [400, 300],
        act_limit: float = 1.0,
    ):
        super().__init__()
        self.act_limit = act_limit

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, act_dim)

        # Initialise
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                _fanin_init(layer)
        _final_init(self.out)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_dim)
        Returns:
            action: (B, act_dim)  clipped to (-act_limit, act_limit)
        """
        return torch.tanh(self.out(self.trunk(obs))) * self.act_limit


class Critic(nn.Module):
    """
    Action-value function  Q_φ(s, a) → scalar.

    Architecture (action injected after first hidden layer):
        obs → Linear(obs_dim, h1) → ReLU
            → concat with action
            → Linear(h1 + act_dim, h2) → ReLU
            → Linear(h2, 1)

    Args:
        obs_dim:     Observation dimensionality.
        act_dim:     Action dimensionality.
        hidden_dims: [h1, h2, ...] hidden layer sizes (minimum 2 elements).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] = [400, 300],
    ):
        super().__init__()
        assert len(hidden_dims) >= 2, "Critic needs at least 2 hidden layers."

        h1 = hidden_dims[0]
        self.obs_layer = nn.Linear(obs_dim, h1)

        # Subsequent layers receive concatenated (hidden, action)
        in_dim = h1 + act_dim
        mid_layers: list[nn.Module] = []
        for h in hidden_dims[1:]:
            mid_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.mid = nn.Sequential(*mid_layers)
        self.out = nn.Linear(in_dim, 1)

        # Initialise
        _fanin_init(self.obs_layer)
        for layer in self.mid:
            if isinstance(layer, nn.Linear):
                _fanin_init(layer)
        _final_init(self.out)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:    (B, obs_dim)
            action: (B, act_dim)
        Returns:
            Q value: (B, 1)
        """
        h = torch.relu(self.obs_layer(obs))  # (B, h1)
        x = torch.cat([h, action], dim=-1)  # (B, h1 + act_dim)
        return self.out(self.mid(x))  # (B, 1)
