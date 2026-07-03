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


def _apply_conv_init(net: nn.Sequential) -> None:
    """Orthogonal init, gain=√2, for every Conv2d layer (all are followed by ReLU)."""
    for m in net:
        if isinstance(m, nn.Conv2d):
            init_weights(m, gain=2**0.5)


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


def _conv_trunk(in_channels: int) -> nn.Sequential:
    """Nature DQN conv trunk (Mnih et al., 2015): 3 conv layers over an 84x84 input."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
    )


def _conv_out_dim(conv: nn.Sequential, in_channels: int, screen_size: int = 84) -> int:
    with torch.no_grad():
        return conv(torch.zeros(1, in_channels, screen_size, screen_size)).shape[1]


class CNNQNetwork(nn.Module):
    """
    Nature DQN Q-network (Mnih et al., 2015) for stacked-frame Atari observations.

    obs (in_channels, 84, 84) → conv trunk → FC 512 → Q(s, a) for all a.

    Pixel values are divided by 255 inside forward() so the replay buffer can
    keep storing raw uint8 frames (a 4x84x84 frame stack is ~28KB as uint8,
    4x that as float32) — normalisation happens only on the sampled
    mini-batch, not on every stored transition.

    Args:
        in_channels: Number of stacked frames (channel dim of the input).
        act_dim:     Number of discrete actions.
        fc_dim:      Hidden size of the FC layer between the conv trunk and output.
    """

    def __init__(self, in_channels: int, act_dim: int, fc_dim: int = 512):
        super().__init__()
        self.conv = _conv_trunk(in_channels)
        conv_out_dim = _conv_out_dim(self.conv, in_channels)
        self.head = nn.Sequential(
            nn.Linear(conv_out_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, act_dim),
        )
        _apply_conv_init(self.conv)
        _apply_init(self.head)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, in_channels, 84, 84), raw pixel values in [0, 255].
        Returns:
            Q values: (B, act_dim)
        """
        x = obs.float() / 255.0
        return self.head(self.conv(x))


class CNNDuelingQNetwork(nn.Module):
    """Dueling variant of CNNQNetwork — same conv trunk, split value/advantage FC heads."""

    def __init__(self, in_channels: int, act_dim: int, fc_dim: int = 512):
        super().__init__()
        self.conv = _conv_trunk(in_channels)
        conv_out_dim = _conv_out_dim(self.conv, in_channels)
        self.value_head = nn.Sequential(
            nn.Linear(conv_out_dim, fc_dim), nn.ReLU(), nn.Linear(fc_dim, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(conv_out_dim, fc_dim), nn.ReLU(), nn.Linear(fc_dim, act_dim)
        )
        _apply_conv_init(self.conv)
        _apply_init(self.value_head)
        _apply_init(self.advantage_head)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Q values: (B, act_dim)
        """
        x = obs.float() / 255.0
        feat = self.conv(x)
        value = self.value_head(feat)  # (B, 1)
        adv = self.advantage_head(feat)  # (B, act_dim)
        return value + adv - adv.mean(dim=-1, keepdim=True)  # (B, act_dim)
