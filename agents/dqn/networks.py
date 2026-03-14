"""
Q-Networks for DQN.

QNetwork        – vanilla MLP Q-network  Q(s, a) for all a simultaneously
DuelingQNetwork – Dueling architecture   Q(s,a) = V(s) + A(s,a) − mean(A)

Both take an observation and output one Q-value per discrete action,
so argmax gives the greedy action in O(1) (no need to evaluate each action
separately as in continuous Q-learning).

Architecture choices:
  - ReLU activations (standard for value functions; Tanh can saturate)
  - Orthogonal initialisation, gain=√2 for hidden, gain=1 for output
  - Dueling: separate value and advantage streams share the feature trunk,
    then recombine as  Q = V + (A − mean(A))  (Wang et al., 2016)
"""

import torch
import torch.nn as nn

from common.utils import build_mlp, init_weights


def _apply_init(net: nn.Sequential) -> None:
    """Orthogonal init: √2 for hidden Linear layers, 1.0 for the output layer."""
    linear_layers = [m for m in net if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linear_layers):
        gain = 1.0 if i == len(linear_layers) - 1 else (2**0.5)
        init_weights(layer, gain=gain)


class QNetwork(nn.Module):
    """
    Vanilla MLP Q-network.

    obs → [hidden_dims] → Q(s, a₀), Q(s, a₁), …, Q(s, a_{n-1})

    Args:
        obs_dim:     Dimensionality of the observation space.
        act_dim:     Number of discrete actions.
        hidden_dims: List of hidden layer sizes.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] = [128, 128],
    ):
        super().__init__()
        self.net = build_mlp(obs_dim, act_dim, hidden_dims, activation=nn.ReLU)
        _apply_init(self.net)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_dim)
        Returns:
            Q values: (B, act_dim)
        """
        return self.net(obs)


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-network (Wang et al., 2016).

    Splits the final layers into:
      - Value stream:     V(s)          → scalar
      - Advantage stream: A(s, a)       → one per action

    Recombined as:  Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]

    Subtracting the mean advantage makes the decomposition unique and
    stabilises training. The value stream alone can improve even when
    not all actions are visited.

    Architecture:
        obs → shared trunk → ┬→ value head    → V(s)
                             └→ advantage head → A(s, a₀…aₙ)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] = [128, 128],
    ):
        super().__init__()
        self.act_dim = act_dim

        # Shared feature trunk (all hidden layers except the last)
        trunk_dims = hidden_dims[:-1]
        stream_dim = hidden_dims[-1]

        self.trunk = (
            build_mlp(obs_dim, stream_dim, trunk_dims, activation=nn.ReLU)
            if trunk_dims
            else nn.Identity()
        )

        # Value and advantage heads
        self.value_head = nn.Linear(stream_dim, 1)
        self.advantage_head = nn.Linear(stream_dim, act_dim)

        # Init trunk
        if trunk_dims:
            for m in self.trunk.modules():
                if isinstance(m, nn.Linear):
                    init_weights(m, gain=2**0.5)
        # Init heads
        init_weights(self.value_head, gain=1.0)
        init_weights(self.advantage_head, gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Q values: (B, act_dim)
        """
        feat = self.trunk(obs) if not isinstance(self.trunk, nn.Identity) else obs
        value = self.value_head(feat)  # (B, 1)
        adv = self.advantage_head(feat)  # (B, act_dim)
        # Subtract mean advantage for identifiability
        q = value + adv - adv.mean(dim=-1, keepdim=True)  # (B, act_dim)
        return q
