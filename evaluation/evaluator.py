"""
Evaluation utilities.

evaluate_agent  – run N episodes with a deterministic policy, return stats
Evaluator       – class wrapper for more structured evaluation + CSV export
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from agents.base_agent import BaseAgent


def evaluate_agent(
    agent: BaseAgent,
    env: gym.Env,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
) -> tuple[float, float]:
    """
    Run `n_episodes` greedy episodes and return (mean_return, std_return).

    Args:
        agent:        Trained agent.
        env:          Evaluation environment (may differ from training env).
        n_episodes:   Number of evaluation episodes.
        deterministic: If True, agent uses greedy/mean action.
        render:       If True, call env.render() each step.

    Returns:
        (mean_return, std_return) over all episodes.
    """
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        done = False

        while not done:
            if render:
                env.render()

            result = agent.select_action(obs, deterministic=deterministic)
            action = result[0] if isinstance(result, tuple) else result

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            done = terminated or truncated

        returns.append(episode_return)

    return float(np.mean(returns)), float(np.std(returns))


class Evaluator:
    """
    Structured evaluation: runs eval, computes statistics, saves to CSV.

    Usage:
        evaluator = Evaluator(agent, env, n_episodes=20)
        results = evaluator.run(step=50_000)
        evaluator.save_results("results/eval_reinforce.csv")
    """

    def __init__(
        self,
        agent: BaseAgent,
        env: gym.Env,
        n_episodes: int = 20,
        deterministic: bool = True,
    ):
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.deterministic = deterministic
        self._records: list[dict[str, Any]] = []

    def run(self, step: int = 0) -> dict[str, float]:
        """
        Evaluate the agent and return a metrics dict.

        Returns dict with keys:
            mean_return, std_return, min_return, max_return,
            median_return, step
        """
        returns = []
        lengths = []

        for _ in range(self.n_episodes):
            obs, _ = self.env.reset()
            ep_return, ep_length = 0.0, 0
            done = False

            while not done:
                result = self.agent.select_action(obs, deterministic=self.deterministic)
                action = result[0] if isinstance(result, tuple) else result
                obs, reward, terminated, truncated, _ = self.env.step(action)
                ep_return += float(reward)
                ep_length += 1
                done = terminated or truncated

            returns.append(ep_return)
            lengths.append(ep_length)

        metrics = {
            "step": step,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
            "median_return": float(np.median(returns)),
            "mean_length": float(np.mean(lengths)),
        }
        self._records.append(metrics)
        return metrics

    def save_results(self, path: str | Path) -> None:
        """Write all evaluation records to a CSV file."""
        if not self._records:
            print("[Evaluator] No records to save.")
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._records[0].keys())
            writer.writeheader()
            writer.writerows(self._records)
        print(f"[Evaluator] Results saved → {path}")

    def summary(self) -> None:
        """Print a formatted summary of the latest evaluation."""
        if not self._records:
            print("[Evaluator] No evaluations run yet.")
            return
        r = self._records[-1]
        print(
            f"\n{'─'*50}\n"
            f"  Evaluation  (step={r['step']:,})\n"
            f"{'─'*50}\n"
            f"  Mean return   : {r['mean_return']:8.2f} ± {r['std_return']:.2f}\n"
            f"  Min / Max     : {r['min_return']:.2f} / {r['max_return']:.2f}\n"
            f"  Median return : {r['median_return']:.2f}\n"
            f"  Mean length   : {r['mean_length']:.1f}\n"
            f"{'─'*50}\n"
        )
