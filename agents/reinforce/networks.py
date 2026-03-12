"""
Policy networks for REINFORCE.

Discrete action spaces  → CategoricalPolicy (softmax output, Categorical distribution)
Continuous action spaces → GaussianPolicy   (mean + log_std, Normal distribution)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from common.utils import build_mlp, init_weights


class CategoricalPolicy(nn.Module):
    """
    Stochastic policy for discrete action spaces.

    Forward pass returns a Categorical distribution over actions.
    The agent samples from it during training and takes the argmax at eval.

    Architecture: obs → MLP → logits → Categorical
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] = [64, 64],
    ):
        super().__init__()
        self.net = build_mlp(obs_dim, act_dim, hidden_dims, activation=nn.Tanh)
        self.apply(
            lambda m: init_weights(m, gain=0.01) if isinstance(m, nn.Linear) else None
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample an action (training) or return the greedy action (eval).

        Returns:
            action:   int tensor, shape ()
            log_prob: scalar tensor
            entropy:  scalar tensor (useful for entropy regularisation)
        """
        dist = self.forward(obs)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class GaussianPolicy(nn.Module):
    """
    Stochastic policy for continuous action spaces.

    Outputs a diagonal Gaussian: mean from the network, log_std as a
    learnable parameter (shared across all states – simplest variant).

    Architecture: obs → MLP → mean;  log_std (free parameter)
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
        self.mean_net = build_mlp(obs_dim, act_dim, hidden_dims, activation=nn.Tanh)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.apply(
            lambda m: init_weights(m, gain=0.01) if isinstance(m, nn.Linear) else None
        )

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Normal(mean, log_std.exp())

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self.forward(obs)
        action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # sum over action dims
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy
