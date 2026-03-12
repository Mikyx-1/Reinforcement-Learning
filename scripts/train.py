#!/usr/bin/env python
"""
CLI training entry point.

Usage:
    python scripts/train.py --config configs/reinforce_cartpole.yaml
    python scripts/train.py --config configs/reinforce_cartpole.yaml --seed 123
    python scripts/train.py --config configs/dqn_cartpole.yaml --device cuda
"""

import argparse
import sys
from pathlib import Path

# Make repo root importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logger import Logger
from common.utils import load_config, set_seed
from envs.wrappers import make_env
from training.trainer import Trainer


# ─────────────────────────────────────────────────────────────────────────────
# Agent registry  – add new algorithms here
# ─────────────────────────────────────────────────────────────────────────────
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
    raise ValueError(f"Unknown agent '{name}'. Register it in scripts/train.py.")


def infer_agent_name(config_path: str) -> str:
    """Guess agent name from config filename, e.g. 'reinforce_cartpole.yaml' → 'reinforce'."""
    stem = Path(config_path).stem
    return stem.split("_")[0].lower()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train an RL agent.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--seed", type=int, help="Override config seed.")
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda'.")
    parser.add_argument("--agent", help="Override agent name.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    set_seed(seed)

    agent_name = args.agent or infer_agent_name(args.config)
    env_id = cfg["env_id"]

    print(
        f"[train.py] Agent={agent_name.upper()}  Env={env_id}  Seed={seed}  Device={args.device}"
    )

    # Build env
    env = make_env(env_id, seed=seed, record_stats=True)

    # Build agent
    agent = build_agent(agent_name, env, cfg, device=args.device)
    print(f"[train.py] {agent}")

    # Build logger
    log_cfg = cfg.get("logging", {})
    logger = Logger(
        log_dir=log_cfg.get("log_dir", f"results/runs/{agent_name}_{env_id}"),
        use_tb=log_cfg.get("use_tb", True),
        use_wandb=log_cfg.get("use_wandb", False),
        print_freq=cfg["training"].get("print_freq", 10),
    )
    logger.log_hparams({**cfg["agent"], "env": env_id, "seed": seed})

    # Train
    trainer = Trainer(
        agent=agent,
        env=env,
        config=cfg["training"],
        logger=logger,
    )

    # Route to correct training loop
    if agent_name in ("reinforce", "ppo"):
        trainer.train_on_policy()
    else:
        from common.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(
            capacity=cfg["training"].get("buffer_size", 100_000),
            device=args.device,
        )
        trainer.train_off_policy(buf)

    env.close()


if __name__ == "__main__":
    main()
