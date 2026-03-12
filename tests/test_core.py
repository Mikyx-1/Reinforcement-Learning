"""
Unit tests for core components.

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
import pytest
import torch

from common.replay_buffer import ReplayBuffer, RolloutBuffer
from common.schedulers import ExponentialSchedule, LinearSchedule
from common.utils import build_mlp, set_seed, soft_update
from envs.wrappers import RecordEpisodeStats, make_env

# ─────────────────────────────────────────────────────────────────────────────
# ReplayBuffer
# ─────────────────────────────────────────────────────────────────────────────


class TestReplayBuffer:
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(10):
            buf.push(np.zeros(4), np.zeros(2), 1.0, np.zeros(4), False)
        assert len(buf) == 10

    def test_capacity_overflow(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(np.zeros(4), np.zeros(2), float(i), np.zeros(4), False)
        assert len(buf) == 5

    def test_sample_shapes(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(50):
            buf.push(np.zeros(4), np.zeros(2), 1.0, np.zeros(4), False)
        batch = buf.sample(32)
        assert batch["obs"].shape == (32, 4)
        assert batch["actions"].shape == (32, 2)
        assert batch["rewards"].shape == (32, 1)
        assert batch["next_obs"].shape == (32, 4)
        assert batch["dones"].shape == (32, 1)

    def test_not_ready_before_min_size(self):
        buf = ReplayBuffer(capacity=100)
        buf.push(np.zeros(4), np.zeros(2), 1.0, np.zeros(4), False)
        assert not buf.is_ready(10)

    def test_raises_if_sample_too_large(self):
        buf = ReplayBuffer(capacity=100)
        buf.push(np.zeros(4), np.zeros(2), 1.0, np.zeros(4), False)
        with pytest.raises(ValueError):
            buf.sample(10)


# ─────────────────────────────────────────────────────────────────────────────
# RolloutBuffer
# ─────────────────────────────────────────────────────────────────────────────


class TestRolloutBuffer:
    def _fill(self, buf, n=5):
        for i in range(n):
            buf.push(np.zeros(4), np.array([0]), -0.1, float(i), i == n - 1)

    def test_len(self):
        buf = RolloutBuffer()
        self._fill(buf, 5)
        assert len(buf) == 5

    def test_clear(self):
        buf = RolloutBuffer()
        self._fill(buf, 5)
        buf.clear()
        assert len(buf) == 0

    def test_discount_rewards_no_done(self):
        buf = RolloutBuffer()
        rewards = [1.0, 1.0, 1.0]
        for r in rewards:
            buf.push(np.zeros(2), np.array([0]), 0.0, r, False)
        returns = buf.discount_rewards(gamma=1.0)
        # G_0 = 3, G_1 = 2, G_2 = 1
        np.testing.assert_allclose(returns, [3.0, 2.0, 1.0])

    def test_get_returns_tensors(self):
        buf = RolloutBuffer()
        self._fill(buf, 5)
        batch = buf.get(gamma=0.99)
        assert isinstance(batch["obs"], torch.Tensor)
        assert isinstance(batch["returns"], torch.Tensor)
        assert batch["returns"].shape == (5,)


# ─────────────────────────────────────────────────────────────────────────────
# Schedulers
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulers:
    def test_linear_start(self):
        s = LinearSchedule(1.0, 0.1, 100)
        assert s.value(0) == pytest.approx(1.0)

    def test_linear_end(self):
        s = LinearSchedule(1.0, 0.1, 100)
        assert s.value(100) == pytest.approx(0.1)

    def test_linear_clamps_beyond_duration(self):
        s = LinearSchedule(1.0, 0.1, 100)
        assert s.value(200) == pytest.approx(0.1)

    def test_exponential_decays(self):
        s = ExponentialSchedule(1.0, 0.01, decay=0.9)
        assert s.value(0) == pytest.approx(1.0)
        assert s.value(1) == pytest.approx(0.9)
        assert s.value(100) >= 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────


class TestUtils:
    def test_build_mlp_output_shape(self):
        net = build_mlp(4, 2, [64, 64])
        x = torch.zeros(8, 4)
        assert net(x).shape == (8, 2)

    def test_soft_update(self):
        import torch.nn as nn

        source = nn.Linear(4, 4)
        target = nn.Linear(4, 4)
        nn.init.constant_(source.weight, 1.0)
        nn.init.constant_(target.weight, 0.0)
        soft_update(target, source, tau=0.5)
        np.testing.assert_allclose(
            target.weight.data.numpy(), np.full((4, 4), 0.5), atol=1e-6
        )

    def test_set_seed_reproducibility(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# Reinforce Agent (smoke tests)
# ─────────────────────────────────────────────────────────────────────────────


class TestReinforceAgent:
    @pytest.fixture
    def cartpole_agent(self):
        env = gym.make("CartPole-v1")
        from agents.reinforce.agent import ReinforceAgent

        return ReinforceAgent(env, hidden_dims=[32, 32], lr=1e-3)

    def test_select_action_shape(self, cartpole_agent):
        env = gym.make("CartPole-v1")
        obs, _ = env.reset()
        action, log_prob = cartpole_agent.select_action(obs)
        assert isinstance(action, (int, np.integer, np.ndarray))
        assert isinstance(log_prob, float)

    def test_update_returns_metrics(self, cartpole_agent):
        env = gym.make("CartPole-v1")
        buf = RolloutBuffer()
        obs, _ = env.reset()
        for _ in range(10):
            action, log_prob = cartpole_agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            buf.push(obs, np.array([action]), log_prob, reward, done)
            obs = next_obs if not done else env.reset()[0]
        batch = buf.get(gamma=0.99)
        metrics = cartpole_agent.update(batch)
        assert "policy_loss" in metrics
        assert "entropy" in metrics

    def test_save_load(self, cartpole_agent, tmp_path):
        path = tmp_path / "ckpt.pt"
        cartpole_agent.save(path)
        env = gym.make("CartPole-v1")
        from agents.reinforce.agent import ReinforceAgent

        loaded = ReinforceAgent(env, hidden_dims=[32, 32])
        loaded.load(path)
        obs, _ = env.reset()
        a1, _ = cartpole_agent.select_action(obs, deterministic=True)
        a2, _ = loaded.select_action(obs, deterministic=True)
        assert int(a1) == int(a2)


# ─────────────────────────────────────────────────────────────────────────────
# Environment wrappers
# ─────────────────────────────────────────────────────────────────────────────


class TestEnvWrappers:
    def test_make_env_runs(self):
        env = make_env("CartPole-v1", seed=0, record_stats=True)
        obs, _ = env.reset()
        assert obs.shape == (4,)
        env.close()

    def test_record_episode_stats(self):
        env = make_env("CartPole-v1", seed=0, record_stats=True)
        env.reset()
        info = {}
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            done = terminated or truncated
        assert "episode" in info
        assert "return" in info["episode"]
        assert "length" in info["episode"]
        env.close()
