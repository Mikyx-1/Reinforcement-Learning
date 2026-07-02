#!/usr/bin/env python
"""
Evaluate a saved checkpoint.

Usage:
    python scripts/evaluate.py \
        --config configs/ppo_cartpole.yaml \
        --checkpoint results/checkpoints/ppo_cartpole/PPOAgent_ep1400.pt \
        --n_episodes 20

    # With explicit seed and output directory (used by run_sweep.py):
    python scripts/evaluate.py \
        --config configs/ppo_cartpole.yaml \
        --checkpoint results/checkpoints/ppo_cartpole/PPOAgent_ep1400.pt \
        --seed 3 --out_dir results/evals/ppo_cartpole/seed_3 --n_episodes 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.registry import build_agent, infer_agent_name
from common.utils import load_config, set_seed
from envs.wrappers import make_env
from evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--render", action="store_true", help="Render episodes in a window as they run."
    )
    parser.add_argument("--seed", type=int, help="Override config seed.")
    parser.add_argument("--out_dir", help="Directory for per-episode CSV output.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    agent_name = infer_agent_name(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)

    set_seed(seed)
    render_mode = "human" if args.render else None
    env = make_env(cfg["env_id"], seed=seed, render_mode=render_mode)
    agent = build_agent(agent_name, env, cfg, device=args.device)
    agent.load(args.checkpoint)

    evaluator = Evaluator(agent, env, n_episodes=args.n_episodes)
    evaluator.run(step=0)
    evaluator.summary()

    # Aggregate results (saved next to the checkpoint by default)
    agg_path = Path(args.out_dir) / "eval_results.csv" if args.out_dir else \
        Path(args.checkpoint).parent / "eval_results.csv"
    evaluator.save_results(agg_path)

    # Per-episode returns (used by run_sweep.py + plotting)
    ep_path = Path(args.out_dir) / "episode_returns.csv" if args.out_dir else \
        Path(args.checkpoint).parent / "episode_returns.csv"
    evaluator.save_episode_returns(ep_path)

    env.close()


if __name__ == "__main__":
    main()
