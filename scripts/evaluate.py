#!/usr/bin/env python
"""
Evaluate a saved checkpoint.

Usage:
    python scripts/evaluate.py \\
        --config configs/reinforce_cartpole.yaml \\
        --checkpoint results/checkpoints/reinforce_cartpole/ReinforceAgent_ep200.pt \\
        --n_episodes 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.utils import load_config, set_seed
from envs.wrappers import make_env
from evaluation.evaluator import Evaluator


def build_agent(name: str, env, cfg: dict, device: str):
    if name == "reinforce":
        from agents.reinforce.agent import ReinforceAgent

        return ReinforceAgent(
            env=env,
            hidden_dims=cfg["agent"]["hidden_dims"],
            lr=cfg["agent"]["lr"],
            gamma=cfg["agent"]["gamma"],
            entropy_coef=cfg["agent"].get("entropy_coef", 0.0),
            device=device,
        )
    raise ValueError(f"Unknown agent '{name}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    agent_name = Path(args.config).stem.split("_")[0].lower()

    set_seed(cfg.get("seed", 0))
    env = make_env(cfg["env_id"], seed=cfg.get("seed", 0))
    agent = build_agent(agent_name, env, cfg, device=args.device)
    agent.load(args.checkpoint)

    evaluator = Evaluator(agent, env, n_episodes=args.n_episodes)
    results = evaluator.run(step=0)
    evaluator.summary()

    save_path = Path(args.checkpoint).parent / "eval_results.csv"
    evaluator.save_results(save_path)
    env.close()


if __name__ == "__main__":
    main()
