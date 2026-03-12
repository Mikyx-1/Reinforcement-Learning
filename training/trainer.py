"""
Generic training loop.

The Trainer decouples *how to train* from *what algorithm is being trained*.
It drives the environment interaction loop and delegates all learning logic
to the agent.

Supports:
  - On-policy agents (REINFORCE, PPO): collect full rollout → update
  - Off-policy agents (DQN, DDPG, SAC): step → push to buffer → sample → update
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from agents.base_agent import BaseAgent
from common.logger import Logger
from common.replay_buffer import ReplayBuffer, RolloutBuffer


class Trainer:
    """
    Drives the training loop for any BaseAgent.

    Args:
        agent:          An instantiated BaseAgent subclass.
        env:            Training environment (wrapped or raw gymnasium.Env).
        config:         Dict of training hyperparameters (see below).
        logger:         Logger instance. If None, a default one is created.

    Config keys (all optional with sensible defaults):
        max_steps     (int)   : Total env steps to train for.   [200_000]
        eval_interval (int)   : Eval every N episodes.          [50]
        save_interval (int)   : Save checkpoint every N episodes.[100]
        checkpoint_dir(str)   : Where to save checkpoints.      ["results/checkpoints"]
        gamma         (float) : Discount factor.                [0.99]
        rollout_len   (int)   : Steps per on-policy rollout.    [None → full episode]
        warmup_steps  (int)   : Random steps before learning.   [0]
        batch_size    (int)   : Off-policy mini-batch size.      [256]
        update_every  (int)   : Off-policy update frequency.     [1]
    """

    def __init__(
        self,
        agent: BaseAgent,
        env: gym.Env,
        config: dict[str, Any],
        logger: Logger | None = None,
    ):
        self.agent = agent
        self.env = env
        self.config = config
        self.logger = logger or Logger(
            log_dir=f"results/runs/{type(agent).__name__}",
            use_tb=True,
            print_freq=config.get("print_freq", 10),
        )

        # Hyperparameters
        self.max_steps = config.get("max_steps", 200_000)
        self.eval_interval = config.get("eval_interval", 50)
        self.save_interval = config.get("save_interval", 100)
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "results/checkpoints"))
        self.gamma = config.get("gamma", 0.99)
        self.warmup_steps = config.get("warmup_steps", 0)
        self.batch_size = config.get("batch_size", 256)
        self.update_every = config.get("update_every", 1)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.global_step = 0
        self.episode = 0
        self.episode_returns: list[float] = []

    # ------------------------------------------------------------------
    # On-policy training  (REINFORCE, PPO)
    # ------------------------------------------------------------------

    def train_on_policy(self) -> None:
        """
        Collect one full episode per iteration, then update.
        This is the standard loop for REINFORCE.
        """
        print(f"[Trainer] Starting on-policy training for {self.max_steps} steps.")
        buffer = RolloutBuffer()

        while self.global_step < self.max_steps:
            obs, _ = self.env.reset()
            buffer.clear()
            episode_return = 0.0
            done = False

            # ── Collect episode ──────────────────────────────────────
            while not done:
                action, log_prob = self._select_action_with_logprob(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                buffer.push(obs, action, log_prob, float(reward), done)
                obs = next_obs
                episode_return += float(reward)
                self.global_step += 1
                self.agent.on_step_end(self.global_step)

            self.episode += 1

            # ── Update ───────────────────────────────────────────────
            batch = buffer.get(gamma=self.gamma, device=str(self.agent.device))
            metrics = self.agent.update(batch)

            # ── Log ──────────────────────────────────────────────────
            self.episode_returns.append(episode_return)
            self.agent.on_episode_end(self.episode, info)

            log_data = {
                "train/episode_return": episode_return,
                "train/episode_length": len(buffer),
                **{f"train/{k}": v for k, v in metrics.items()},
            }
            self.logger.log(log_data, step=self.global_step)

            # ── Eval & checkpoint ────────────────────────────────────
            if self.episode % self.eval_interval == 0:
                self._eval_and_log()
            if self.episode % self.save_interval == 0:
                self._save_checkpoint()

        self.logger.close()
        print("[Trainer] Training complete.")

    # ------------------------------------------------------------------
    # Off-policy training  (DQN, DDPG, SAC, TD3)
    # ------------------------------------------------------------------

    def train_off_policy(self, replay_buffer: ReplayBuffer) -> None:
        """
        Step → push → (optionally) sample → update.
        Used by DQN, DDPG, TD3, SAC.
        """
        print(f"[Trainer] Starting off-policy training for {self.max_steps} steps.")
        obs, _ = self.env.reset()
        episode_return = 0.0

        for step in range(self.max_steps):
            self.global_step += 1

            # Random exploration during warmup
            if step < self.warmup_steps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.select_action(obs, deterministic=False)
                # Flatten in case agent returns a 1-D array
                if hasattr(action, "__len__") and len(action) == 1:
                    action = action[0]

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            replay_buffer.push(obs, action, float(reward), next_obs, done)
            obs = next_obs
            episode_return += float(reward)
            self.agent.on_step_end(self.global_step)

            # Update
            if (
                step >= self.warmup_steps
                and step % self.update_every == 0
                and replay_buffer.is_ready(self.batch_size)
            ):
                batch = replay_buffer.sample(self.batch_size)
                metrics = self.agent.update(batch)
                self.logger.log(
                    {f"train/{k}": v for k, v in metrics.items()},
                    step=self.global_step,
                )

            if done:
                self.episode += 1
                self.episode_returns.append(episode_return)
                self.logger.log(
                    {"train/episode_return": episode_return}, step=self.global_step
                )
                self.agent.on_episode_end(self.episode, info)

                if self.episode % self.eval_interval == 0:
                    self._eval_and_log()
                if self.episode % self.save_interval == 0:
                    self._save_checkpoint()

                obs, _ = self.env.reset()
                episode_return = 0.0

        self.logger.close()
        print("[Trainer] Training complete.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_action_with_logprob(self, obs: np.ndarray):
        """
        On-policy agents expose log_prob via a second return value.
        Falls back gracefully if the agent only returns actions.
        """
        result = self.agent.select_action(obs, deterministic=False)
        if isinstance(result, tuple):
            return result  # (action, log_prob)
        return result, 0.0  # off-policy agent used in on-policy loop

    def _eval_and_log(self, n_episodes: int = 5) -> float:
        """Quick greedy evaluation; logs mean return."""
        from evaluation.evaluator import evaluate_agent

        mean_return, std_return = evaluate_agent(
            self.agent, self.env, n_episodes=n_episodes
        )
        self.logger.log(
            {"eval/mean_return": mean_return, "eval/std_return": std_return},
            step=self.global_step,
        )
        return mean_return

    def _save_checkpoint(self) -> None:
        path = self.checkpoint_dir / f"{type(self.agent).__name__}_ep{self.episode}.pt"
        self.agent.save(path)
        print(f"[Trainer] Checkpoint saved → {path}")
