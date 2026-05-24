"""
Actor and Twin-Critic networks for TD3.

The Actor is identical to DDPG (deterministic μ_θ(s) → a, Tanh-scaled).
The TwinCritic holds two independent Q-networks Q₁, Q₂ that share no parameters.

Why two critics?
  TD3 mitigates DDPG's overestimation bias by taking the minimum of the two
  target Q-values when forming the TD target:

      y = r + γ · min(Q₁'(s', ã'), Q₂'(s', ã')) · (1 − d)

  Both critics are trained to regress to this same target, but their
  independent initialisations + parameter trajectories make their errors
  partially uncorrelated, so the min acts as a conservative estimate.
"""

import numpy as np
import torch
import torch.nn as nn


def _fanin_init(layer: nn.Linear) -> None:
    fan_in = layer.weight.data.size(1)
    bound = 1.0 / np.sqrt(fan_in)
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


def _final_init(layer: nn.Linear, bound: float = 3e-3) -> None:
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


class Actor(nn.Module):
    """
    Deterministic policy  μ_θ(s) → a, output Tanh-scaled to [-act_limit, act_limit].

    Same architecture as DDPG; we keep a local copy so TD3 stays a
    self-contained package.
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

        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                _fanin_init(layer)
        _final_init(self.out)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.out(self.trunk(obs))) * self.act_limit


class _SingleCritic(nn.Module):
    """One Q-network: (obs, action) → scalar. Action injected after first layer."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int]):
        super().__init__()
        assert len(hidden_dims) >= 2, "Critic needs at least 2 hidden layers."

        h1 = hidden_dims[0]
        self.obs_layer = nn.Linear(obs_dim, h1)

        in_dim = h1 + act_dim
        mid: list[nn.Module] = []
        for h in hidden_dims[1:]:
            mid += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.mid = nn.Sequential(*mid)
        self.out = nn.Linear(in_dim, 1)

        _fanin_init(self.obs_layer)
        for layer in self.mid:
            if isinstance(layer, nn.Linear):
                _fanin_init(layer)
        _final_init(self.out)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.obs_layer(obs))
        x = torch.cat([h, action], dim=-1)
        return self.out(self.mid(x))


class TwinCritic(nn.Module):
    """
    Two independent Q-networks. Returns (Q₁(s,a), Q₂(s,a)).

    Bundling both critics in one nn.Module keeps the optimiser, target
    network, save/load, and Polyak update logic clean — we only ever pass
    `twin_critic.parameters()` around, never two separate handles.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] = [400, 300],
    ):
        super().__init__()
        self.q1 = _SingleCritic(obs_dim, act_dim, hidden_dims)
        self.q2 = _SingleCritic(obs_dim, act_dim, hidden_dims)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)

    def q1_only(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Used in the actor loss — gradient flows through Q₁ only (paper convention)."""
        return self.q1(obs, action)
