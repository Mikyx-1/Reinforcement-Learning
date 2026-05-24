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
    elif name == "actor":
        from agents.actor_critic.agent import ActorCriticAgent
        return ActorCriticAgent(
            env=env,
            hidden_dims=cfg["agent"]["hidden_dims"],
            lr=cfg["agent"]["lr"],
            gamma=cfg["agent"]["gamma"],
            ent_coef=cfg["agent"].get("entropy_coef", 0.0),
            device=device,
        )
    elif name == "ppo":
        from agents.ppo.agent import PPOAgent
        return PPOAgent(
            env=env,
            hidden_dims=cfg["agent"]["hidden_dims"],
            lr=cfg["agent"]["lr"],
            gamma=cfg["agent"]["gamma"],
            ent_coef=cfg["agent"].get("entropy_coef", 0.0),
            clip_eps=cfg["agent"].get("clip_eps", 0.2),
            device=device,
        )
    elif name == "dqn":
        from agents.dqn.agent import DQNAgent
        return DQNAgent(
            env=env,
            hidden_dims=cfg["agent"]["hidden_dims"],
            lr=cfg["agent"]["lr"],
            gamma=cfg["agent"]["gamma"],
            eps_start=cfg["agent"].get("eps_start", 1.0),
            eps_end=cfg["agent"].get("eps_end", 0.01),
            eps_decay_steps=cfg["agent"].get("eps_decay_steps", 500),
            target_update_freq=cfg["agent"].get("target_update_freq", 1),
            device=device,
        )
    elif name == "sarsa":
        from agents.sarsa.agent import SarsaAgent
        return SarsaAgent(
            env=env,
            hidden_dims=cfg["agent"]["hidden_dims"],
            lr=cfg["agent"]["lr"],
            gamma=cfg["agent"]["gamma"],
            eps_start=cfg["agent"].get("eps_start", 1.0),
            eps_end=cfg["agent"].get("eps_end", 0.01),
            eps_decay_steps=cfg["agent"].get("eps_decay_steps", 500),
            device=device,
        )
    elif name == "ddpg":
        from agents.ddpg.agent import DDPGAgent
        return DDPGAgent(
            env=env,
            hidden_dims=cfg["agent"]["hidden_dims"],
            actor_lr=cfg["agent"]["actor_lr"],
            critic_lr=cfg["agent"]["critic_lr"],
            gamma=cfg["agent"]["gamma"],
            tau=cfg["agent"]["tau"],
            noise_type=cfg["agent"]["noise_type"],
            noise_sigma=cfg["agent"]["noise_sigma"],
            noise_sigma_min=cfg["agent"].get("noise_sigma_min", 0.02),
            noise_sigma_decay=cfg["agent"].get("noise_sigma_decay", 0.999),
            device=device,
        )
    elif name == "td3":
        from agents.td3.agent import TD3Agent
        return TD3Agent(
            env=env,
            hidden_dims=cfg["agent"]["hidden_dims"],
            actor_lr=cfg["agent"]["actor_lr"],
            critic_lr=cfg["agent"]["critic_lr"],
            gamma=cfg["agent"]["gamma"],
            tau=cfg["agent"]["tau"],
            policy_delay=cfg["agent"].get("policy_delay", 2),
            target_noise_sigma=cfg["agent"].get("target_noise_sigma", 0.2),
            target_noise_clip=cfg["agent"].get("target_noise_clip", 0.5),
            noise_type=cfg["agent"].get("noise_type", "gaussian"),
            noise_sigma=cfg["agent"].get("noise_sigma", 0.1),
            noise_sigma_min=cfg["agent"].get("noise_sigma_min", 0.01),
            noise_sigma_decay=cfg["agent"].get("noise_sigma_decay", 1.0),
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
    parser.add_argument("--seed", type=int, help="Override config seed.")
    parser.add_argument("--out_dir", help="Directory for per-episode CSV output.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    agent_name = Path(args.config).stem.split("_")[0].lower()
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)

    set_seed(seed)
    env = make_env(cfg["env_id"], seed=seed)
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
