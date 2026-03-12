#!/usr/bin/env python
"""
Record an MP4 video of a trained agent playing in its environment.

Uses gymnasium's built-in RecordVideo wrapper, which requires either:
  - `ffmpeg` installed on the system (recommended), OR
  - `imageio[ffmpeg]` pip package as a fallback

Usage examples:
    # Basic – record 3 episodes, save to results/videos/
    python scripts/record_video.py \\
        --config     configs/reinforce_cartpole.yaml \\
        --checkpoint results/checkpoints/reinforce_cartpole/ReinforceAgent_ep200.pt

    # Record 5 episodes, custom output dir, render in real-time as well
    python scripts/record_video.py \\
        --config     configs/reinforce_cartpole.yaml \\
        --checkpoint results/checkpoints/reinforce_cartpole/ReinforceAgent_ep200.pt \\
        --n_episodes 5 \\
        --output_dir results/videos/reinforce_cartpole \\
        --show

    # Stochastic policy (not greedy) – useful for visualising exploration
    python scripts/record_video.py \\
        --config     configs/reinforce_cartpole.yaml \\
        --checkpoint results/checkpoints/reinforce_cartpole/ReinforceAgent_ep200.pt \\
        --stochastic

Output:
    results/videos/<run_name>/rl-video-episode-0.mp4
    results/videos/<run_name>/rl-video-episode-1.mp4
    ...
    results/videos/<run_name>/episode_stats.csv   ← return & length per episode
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np

from common.utils import load_config, set_seed
from envs.wrappers import make_env

# ─────────────────────────────────────────────────────────────────────────────
# Agent registry (mirrors train.py – add algorithms here as they are built)
# ─────────────────────────────────────────────────────────────────────────────


def build_agent(name: str, env: gym.Env, cfg: dict, device: str):
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
    # ── add future algorithms below ──────────────────────────────────────────
    # if name == "dqn":
    #     from agents.dqn.agent import DQNAgent
    #     return DQNAgent(env=env, **cfg["agent"], device=device)
    raise ValueError(f"Unknown agent '{name}'. Register it in scripts/record_video.py.")


def infer_agent_name(config_path: str) -> str:
    return Path(config_path).stem.split("_")[0].lower()


# ─────────────────────────────────────────────────────────────────────────────
# Core recording logic
# ─────────────────────────────────────────────────────────────────────────────


def record(
    agent,
    env_id: str,
    output_dir: Path,
    n_episodes: int,
    deterministic: bool,
    seed: int,
    show: bool,
) -> list[dict]:
    """
    Wrap the environment with gymnasium's RecordVideo, run `n_episodes`
    episodes, and return per-episode stats.

    Args:
        agent:        Loaded, ready-to-use agent.
        env_id:       Gymnasium environment ID.
        output_dir:   Directory where MP4 files will be written.
        n_episodes:   Number of episodes to record.
        deterministic: Use greedy (True) or stochastic (False) policy.
        seed:         RNG seed for reproducibility.
        show:         If True, call env.render() for live preview (requires
                      a display; disable in headless environments).

    Returns:
        List of dicts with keys: episode, return, length.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Record every episode (episode_trigger=lambda ep: True)
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(output_dir),
        episode_trigger=lambda ep_id: True,  # record all episodes
        name_prefix="rl-video",
        disable_logger=True,
    )
    env.reset(seed=seed)

    stats = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        ep_length = 0
        done = False

        while not done:
            if show:
                env.render()

            # select_action returns (action, log_prob) for on-policy agents
            # and just action for off-policy agents – handle both
            result = agent.select_action(obs, deterministic=deterministic)
            action = result[0] if isinstance(result, tuple) else result

            # Discrete envs expect a plain Python int, not a numpy scalar
            if hasattr(action, "item"):
                action = action.item()

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_length += 1
            done = terminated or truncated

        stats.append({"episode": ep, "return": ep_return, "length": ep_length})
        print(
            f"  Episode {ep + 1:>2d}/{n_episodes}"
            f"  return={ep_return:8.2f}"
            f"  length={ep_length:4d}"
        )

    env.close()
    return stats


def save_stats(stats: list[dict], output_dir: Path) -> None:
    path = output_dir / "episode_stats.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "return", "length"])
        writer.writeheader()
        writer.writerows(stats)

    returns = [s["return"] for s in stats]
    print(
        f"\n  ── Summary ──────────────────────────────\n"
        f"  Episodes   : {len(stats)}\n"
        f"  Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}\n"
        f"  Min / Max  : {np.min(returns):.2f} / {np.max(returns):.2f}\n"
        f"  Stats saved: {path}\n"
        f"  ─────────────────────────────────────────"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Record MP4 videos of a trained RL agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, help="Path to the YAML config used during training."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a .pt checkpoint file produced by train.py.",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=3, help="Number of episodes to record."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to save videos. Defaults to results/videos/<config_stem>/.",
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device: 'cpu' or 'cuda'."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override the seed in the config file."
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy instead of greedy (deterministic=False).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Render frames to screen in real-time (requires a display).",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    agent_name = infer_agent_name(args.config)
    env_id = cfg["env_id"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    set_seed(seed)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results/videos") / Path(args.config).stem
    )

    print(
        f"\n[record_video.py]\n"
        f"  Agent      : {agent_name.upper()}\n"
        f"  Env        : {env_id}\n"
        f"  Checkpoint : {args.checkpoint}\n"
        f"  Episodes   : {args.n_episodes}\n"
        f"  Policy     : {'stochastic' if args.stochastic else 'deterministic'}\n"
        f"  Output dir : {output_dir}\n"
    )

    # ── Build agent (using a plain env just for space inference) ─────────────
    _tmp_env = gym.make(env_id)
    agent = build_agent(agent_name, _tmp_env, cfg, device=args.device)
    _tmp_env.close()

    agent.load(args.checkpoint)
    agent.policy.eval()  # disable dropout / batch-norm in training mode

    # ── Record ───────────────────────────────────────────────────────────────
    print(f"Recording {args.n_episodes} episode(s)...\n")
    stats = record(
        agent=agent,
        env_id=env_id,
        output_dir=output_dir,
        n_episodes=args.n_episodes,
        deterministic=not args.stochastic,
        seed=seed,
        show=args.show,
    )

    save_stats(stats, output_dir)
    print(f"Videos saved to: {output_dir}/\n")


if __name__ == "__main__":
    main()
