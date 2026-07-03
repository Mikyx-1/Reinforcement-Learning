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
import time
from pathlib import Path

# Make repo root importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logger import Logger
from common.registry import build_agent, infer_agent_name
from common.utils import load_config, set_seed
from envs.wrappers import make_env
from training.trainer import Trainer


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train an RL agent.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--seed", type=int, help="Override config seed.")
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda'.")
    parser.add_argument("--agent", help="Override agent name.")
    parser.add_argument(
        "--resume",
        help=(
            "Path to a checkpoint (.pt) to load weights/optimizer state from "
            "before training. Only restores the agent's networks/optimizer — "
            "the replay buffer starts empty and the Trainer's step/episode "
            "counters restart at 0, so point --config at a config with its "
            "own checkpoint_dir/log_dir to avoid overwriting the original run."
        ),
    )
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
    env = make_env(
        env_id,
        seed=seed,
        record_stats=True,
        env_kwargs=cfg.get("env_kwargs"),
        atari_preprocessing=cfg.get("atari_preprocessing", False),
        frame_stack=cfg.get("frame_stack", 4),
    )

    # Build agent
    agent = build_agent(agent_name, env, cfg, device=args.device)
    if args.resume:
        agent.load(args.resume)
        print(f"[train.py] Resumed weights from {args.resume}")
    print(f"[train.py] {agent}")

    # Build logger
    log_cfg = cfg.get("logging", {})

    # Extract wandb_config and set defaults
    wandb_kwargs = log_cfg.get("wandb_config", {}).copy()
    if "project" not in wandb_kwargs:
        wandb_kwargs["project"] = "Reinforcement-Learning"
    if "name" not in wandb_kwargs:
        # Format: <agent>/<environment>/seed_<seed>/<timestamp>
        timestamp = time.strftime("%b%d_%H:%M")
        wandb_kwargs["name"] = f"{agent_name}/{env_id}/seed_{seed}/{timestamp}"

    if "config" not in wandb_kwargs:
        wandb_kwargs["config"] = cfg

    logger = Logger(
        log_dir=log_cfg.get("log_dir", f"results/runs/{agent_name}_{env_id}"),
        use_wandb=log_cfg.get("use_wandb", False),
        wandb_kwargs=wandb_kwargs,
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
    if agent_name in ("reinforce", "actor"):
        trainer.train_on_policy()

    elif agent_name in ("ppo", "gnnppo"):
        trainer.train_ppo()

    elif agent_name == "sarsa":
        trainer.train_sarsa()
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
