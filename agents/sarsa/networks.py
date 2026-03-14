"""
Q-Network for SARSA.

Identical to DQN's QNetwork in structure — a plain MLP mapping
observations to Q-values over all actions simultaneously.

It is a separate file from agents/dqn/networks.py to keep each
algorithm folder fully self-contained and independently readable.

SARSA does not use a Dueling architecture or a target network:
  - No target network: SARSA is on-policy TD(0); the update is
    inherently bootstrapped from the *same* policy, so the moving
    target problem is less severe than in off-policy Q-learning.
  - No dueling: the advantage/value decomposition is most useful
    when the optimal action changes little across states; SARSA is
    typically used in simpler environments where vanilla Q suffices.
"""

import torch
import torch.nn as nn

from common.utils import build_mlp, init_weights


class QNetwork(nn.Module):
    """
    MLP Q-network for SARSA.

    obs → [hidden_dims, ReLU] → Q(s, a₀), Q(s, a₁), …, Q(s, aₙ₋₁)

    Args:
        obs_dim:     Observation dimensionality.
        act_dim:     Number of discrete actions.
        hidden_dims: List of hidden layer sizes.
                     Use [] for a linear (tabular-like) function approximator.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] = [128, 128],
    ):
        super().__init__()

        if hidden_dims:
            self.net = build_mlp(obs_dim, act_dim, hidden_dims, activation=nn.ReLU)
            # Orthogonal init: √2 for hidden layers, 1.0 for output
            linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
            for i, layer in enumerate(linear_layers):
                gain = 1.0 if i == len(linear_layers) - 1 else (2**0.5)
                init_weights(layer, gain=gain)
        else:
            # Linear function approximator — mirrors tabular SARSA with features
            self.net = nn.Linear(obs_dim, act_dim)
            init_weights(self.net, gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_dim)
        Returns:
            Q values: (B, act_dim)
        """
        return self.net(obs)
