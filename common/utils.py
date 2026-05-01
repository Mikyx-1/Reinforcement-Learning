"""
Shared utilities: seeding, network construction, weight init, etc.
"""

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (+ CUDA if available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Network builders
# ─────────────────────────────────────────────────────────────────────────────


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    activation: type[nn.Module] = nn.Tanh,
    output_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    """
    Build a fully-connected MLP.

    Args:
        input_dim:          Number of input features.
        output_dim:         Number of output features.
        hidden_dims:        List of hidden layer sizes, e.g. [64, 64].
        activation:         Activation after each hidden layer.
        output_activation:  Optional activation on the final layer.

    Returns:
        nn.Sequential MLP.

    Example:
        net = build_mlp(4, 2, [64, 64], activation=nn.ReLU)
    """
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == len(dims) - 2
        if is_last:
            if output_activation is not None:
                layers.append(output_activation())
        else:
            layers.append(activation())
    return nn.Sequential(*layers)


def init_weights(module: nn.Module, gain: float = 1.0) -> None:
    """
    Orthogonal initialisation (recommended for policy networks).
    Biases are set to zero.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


# ─────────────────────────────────────────────────────────────────────────────
# Config I/O
# ─────────────────────────────────────────────────────────────────────────────


def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


# ─────────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────────


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """
    Polyak averaging: θ_target ← τ·θ_source + (1−τ)·θ_target
    Used by DDPG, TD3, SAC for slowly updating target networks.
    """ 
    target_net_state_dict = target.state_dict()
    policy_net_state_dict = source.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[
            key
        ] * tau + target_net_state_dict[key] * (1 - tau)
    target.load_state_dict(target_net_state_dict)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Full weight copy: θ_target ← θ_source. Used by DQN."""
    target.load_state_dict(source.state_dict())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Fraction of variance in y_true explained by y_pred.
    1.0 = perfect prediction, 0.0 = no better than mean, <0 = worse.
    Useful diagnostic for value function quality (PPO).
    """
    var_y = np.var(y_true)
    return 0.0 if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)
