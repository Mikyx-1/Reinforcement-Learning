"""
Abstract base class for all RL agents.
Every algorithm must implement this interface to ensure compatibility
with the shared Trainer, Evaluator, and Benchmarker.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch


class BaseAgent(ABC):
    """
    Contract that every RL agent must fulfill.

    Subclasses implement:
        select_action  – given an observation, return an action
        update         – consume a batch of experience, return a loss dict
        save / load    – checkpoint the full agent state
    """

    def __init__(self, obs_dim: int, act_dim: int, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)
        self.training_step = 0  # incremented inside update()

    # ------------------------------------------------------------------
    # Core interface (must override)
    # ------------------------------------------------------------------

    @abstractmethod
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Map an observation to an action.

        Args:
            obs:           Shape (obs_dim,) numpy array.
            deterministic: If True, return the greedy / mean action (eval mode).

        Returns:
            action: Shape (act_dim,) or scalar numpy array.
        """

    @abstractmethod
    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        Perform one gradient update step.

        Args:
            batch: Dict with keys like 'obs', 'actions', 'rewards',
                   'next_obs', 'dones', 'log_probs', etc.
                   (Keys vary by algorithm; each agent documents its own.)

        Returns:
            metrics: Dict of scalar losses/stats for logging,
                     e.g. {"policy_loss": 0.42, "entropy": 1.1}
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist model weights + optimizer states to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore model weights + optimizer states from disk."""

    # ------------------------------------------------------------------
    # Optional hooks (override if the algorithm needs them)
    # ------------------------------------------------------------------

    def on_episode_end(self, episode: int, info: dict[str, Any]) -> None:
        """Called at the end of every episode. Useful for per-episode schedules."""

    def on_step_end(self, step: int) -> None:
        """Called after every environment step. Useful for epsilon decay, etc."""

    # ------------------------------------------------------------------
    # Convenience helpers (shared by all agents)
    # ------------------------------------------------------------------

    def to_tensor(self, x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
            f"device={self.device})"
        )
