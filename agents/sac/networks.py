"""
Squashed-Gaussian Actor and Twin-Critic networks for SAC.

The Actor outputs the parameters (μ, log σ) of a diagonal Gaussian over
pre-tanh actions u. Sampled actions are squashed to (-1, 1) with tanh
and then scaled by `act_limit`:

        u ~ N(μ_θ(s), σ_θ(s)²)
        a = tanh(u) · act_limit

The change-of-variables correction for log π(a|s) is required so the
entropy term in SAC has its usual meaning:

        log π(a|s) = log N(u|μ,σ²)
                     − Σ_i log( 1 − tanh(u_i)² )
                     − act_dim · log(act_limit)        (constant; included
                                                       for numerical correctness)

The TwinCritic is structurally identical to TD3's — two independent
Q-networks Q₁, Q₂ : (s, a) → ℝ bundled in one nn.Module so the
optimiser, target network, and Polyak update logic stay simple.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Bounds on log σ. The lower bound stops the Gaussian collapsing to a
# delta (which would make log π → ∞ on the squashed action). The upper
# bound stops it exploding early in training when Q is uninformative.
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _fanin_init(layer: nn.Linear) -> None:
    fan_in = layer.weight.data.size(1)
    bound = 1.0 / np.sqrt(fan_in)
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


def _final_init(layer: nn.Linear, bound: float = 3e-3) -> None:
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


class SquashedGaussianActor(nn.Module):
    """
    π_θ(·|s) = tanh-squashed diagonal Gaussian, scaled to ±act_limit.

    forward(obs) returns (mean_action, log_std, pre_tanh_mean). The
    `sample()` method draws an action with the reparameterisation trick
    and returns (action, log_prob, mean_action), where:
      - action       has gradients flowing through the rsample (used in
                     actor loss),
      - log_prob     includes the tanh Jacobian correction,
      - mean_action  is the deterministic (greedy) action — tanh of μ,
                     used at evaluation time.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] = [256, 256],
        act_limit: float = 1.0,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.act_limit = act_limit

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        self.mean_head = nn.Linear(in_dim, act_dim)
        self.log_std_head = nn.Linear(in_dim, act_dim)

        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                _fanin_init(layer)
        _final_init(self.mean_head)
        _final_init(self.log_std_head)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reparameterised sample with the squashing-correction for log_prob.

        Returns:
            action       : tanh(u) · act_limit, where u = μ + σ·ε, ε~N(0,I)
            log_prob     : log π(action | obs), summed over action dims,
                           shape (B, 1).
            mean_action  : deterministic policy output  tanh(μ) · act_limit
                           — used at evaluation time.
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # Reparameterised sample u = μ + σ · ε  (grads flow through μ, σ).
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()

        # log N(u | μ, σ²)  — diagonal, summed over action dims.
        log_prob_u = normal.log_prob(u).sum(dim=-1, keepdim=True)

        # tanh correction: a = tanh(u),  d a / d u = 1 − tanh(u)².
        # Numerically stable form of  log(1 − tanh(u)²):
        #     2 · ( log 2 − u − softplus(−2u) )
        tanh_correction = (
            2.0 * (np.log(2.0) - u - F.softplus(-2.0 * u))
        ).sum(dim=-1, keepdim=True)

        # Constant shift for the act_limit scaling. Gradient is zero
        # (act_limit is a constant) but we include it so the reported
        # entropy and α-loss have correct magnitudes.
        scale_correction = self.act_dim * float(np.log(self.act_limit))

        log_prob = log_prob_u - tanh_correction - scale_correction

        action = torch.tanh(u) * self.act_limit
        mean_action = torch.tanh(mean) * self.act_limit
        return action, log_prob, mean_action


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

    SAC takes min(Q₁, Q₂) in the TD target to reduce overestimation
    bias — the same trick TD3 uses. Both critics are trained with the
    same target.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] = [256, 256],
    ):
        super().__init__()
        self.q1 = _SingleCritic(obs_dim, act_dim, hidden_dims)
        self.q2 = _SingleCritic(obs_dim, act_dim, hidden_dims)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)
