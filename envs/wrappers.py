"""
Gym environment wrappers.

These follow the standard gymnasium.Wrapper API and can be stacked.
"""

from collections import deque

import gymnasium as gym
import numpy as np


class NormalizeObservation(gym.ObservationWrapper):
    """
    Running mean/std normalisation of observations.
    Critical for continuous control environments (MuJoCo, etc.).
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self._obs_rms_mean = np.zeros(env.observation_space.shape)
        self._obs_rms_var = np.ones(env.observation_space.shape)
        self._count = 0

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._update_stats(obs)
        return (obs - self._obs_rms_mean) / np.sqrt(self._obs_rms_var + self.epsilon)

    def _update_stats(self, obs: np.ndarray) -> None:
        """Welford online algorithm."""
        self._count += 1
        delta = obs - self._obs_rms_mean
        self._obs_rms_mean += delta / self._count
        self._obs_rms_var += delta * (obs - self._obs_rms_mean)


class ClipReward(gym.RewardWrapper):
    """Clip rewards to [-clip, +clip]. Standard for Atari."""

    def __init__(self, env: gym.Env, clip: float = 1.0):
        super().__init__(env)
        self.clip = clip

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, -self.clip, self.clip))


class ScaleReward(gym.RewardWrapper):
    """Multiply every reward by a constant scale factor."""

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        return reward * self.scale


class FrameStack(gym.ObservationWrapper):
    """
    Stack `n` consecutive frames along the channel axis.
    Useful for partially observable environments and Atari.
    """

    def __init__(self, env: gym.Env, n: int = 4):
        super().__init__(env)
        self.n = n
        self._frames: deque = deque(maxlen=n)
        low = np.repeat(env.observation_space.low, n, axis=-1)
        high = np.repeat(env.observation_space.high, n, axis=-1)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n):
            self._frames.append(obs)
        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._frames.append(obs)
        return np.concatenate(list(self._frames), axis=-1)


class RecordEpisodeStats(gym.Wrapper):
    """
    Tracks episode return and length; adds them to `info` at episode end.
    Works with both old and new Gym step APIs.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._episode_return = 0.0
        self._episode_length = 0

    def reset(self, **kwargs):
        self._episode_return = 0.0
        self._episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_return += float(reward)
        self._episode_length += 1
        done = terminated or truncated
        if done:
            info["episode"] = {
                "return": self._episode_return,
                "length": self._episode_length,
            }
        return obs, reward, terminated, truncated, info


def make_env(
    env_id: str,
    seed: int = 0,
    normalize_obs: bool = False,
    clip_reward: float | None = None,
    scale_reward: float | None = None,
    record_stats: bool = True,
) -> gym.Env:
    """
    Factory that builds and wraps an environment.

    Args:
        env_id:        Gymnasium environment ID, e.g. 'CartPole-v1'.
        seed:          RNG seed for reproducibility.
        normalize_obs: Apply running-mean normalisation to observations.
        clip_reward:   If set, clip rewards to ±clip_reward.
        scale_reward:  If set, multiply rewards by this factor.
        record_stats:  Attach RecordEpisodeStats wrapper.

    Returns:
        Wrapped gymnasium.Env.
    """
    env = gym.make(env_id)
    env.reset(seed=seed)

    if record_stats:
        env = RecordEpisodeStats(env)
    if normalize_obs:
        env = NormalizeObservation(env)
    if clip_reward is not None:
        env = ClipReward(env, clip=clip_reward)
    if scale_reward is not None:
        env = ScaleReward(env, scale=scale_reward)

    return env
