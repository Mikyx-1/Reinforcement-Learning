"""
Exploration noise for DDPG and continuous-action off-policy algorithms.

OUNoise      – Ornstein-Uhlenbeck process (original DDPG paper)
GaussianNoise – Simpler i.i.d. Gaussian noise (often works just as well)

Why noise matters in DDPG:
  DDPG learns a deterministic policy μ(s). During training we need to
  explore the action space, so we add correlated noise to the action:

      a = clip(μ(s) + ε,  a_low,  a_high)

  OU noise produces temporally correlated perturbations (like physical
  momentum), which can help in environments where sustained directional
  exploration is useful (e.g. driving, robotic arm).
  Gaussian noise is simpler and often performs comparably in practice.

Both classes expose the same interface:
    noise.sample()   → np.ndarray of shape (act_dim,)
    noise.reset()    → restart the process (called at episode start)
"""

import numpy as np


class OUNoise:
    """
    Ornstein-Uhlenbeck process.

    dxₜ = θ(μ − xₜ)dt + σ dWₜ

    Discretised as:
        xₜ₊₁ = xₜ + θ(μ − xₜ) + σ · N(0, 1)

    Args:
        act_dim:  Dimensionality of the action space.
        mu:       Long-run mean (typically 0).
        theta:    Mean-reversion speed.  Higher → noise decays faster.
        sigma:    Volatility of the noise process.
        dt:       Time step size (scales theta and sigma together).
    """

    def __init__(
        self,
        act_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
    ):
        self.act_dim = act_dim
        self.mu = mu * np.ones(act_dim)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        """Reset state to the mean at the start of a new episode."""
        self.x = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(
            self.dt
        ) * np.random.randn(self.act_dim)
        self.x = self.x + dx
        return self.x.copy()


class GaussianNoise:
    """
    Independent Gaussian noise  ε ~ N(0, σ²I).

    Simpler than OU noise and equally effective in many environments.
    Decaying σ over training is supported via sigma_decay.

    Args:
        act_dim:      Dimensionality of the action space.
        sigma:        Initial standard deviation of the noise.
        sigma_min:    Minimum σ after decay.
        sigma_decay:  Multiplicative decay applied each call to decay().
    """

    def __init__(
        self,
        act_dim: int,
        sigma: float = 0.1,
        sigma_min: float = 0.01,
        sigma_decay: float = 1.0,  # 1.0 = no decay
    ):
        self.act_dim = act_dim
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay

    def reset(self) -> None:
        """No internal state — reset is a no-op."""

    def sample(self) -> np.ndarray:
        return np.random.randn(self.act_dim) * self.sigma

    def decay(self) -> None:
        """Call once per episode to decay σ towards sigma_min."""
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)
