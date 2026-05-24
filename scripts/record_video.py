#!/usr/bin/env python
"""
Record an MP4 or GIF of a trained agent playing in its environment.

MP4 uses gymnasium's built-in RecordVideo wrapper (requires ffmpeg or
imageio[ffmpeg]).  GIF uses imageio directly — no ffmpeg needed.

Usage examples:
    # Basic – record 3 episodes as MP4
    python scripts/record_video.py \\
        --config     configs/reinforce_cartpole.yaml \\
        --checkpoint results/checkpoints/reinforce_cartpole/ReinforceAgent_ep200.pt

    # Export a GIF (capped at 300 frames, 24 fps)
    python scripts/record_video.py \\
        --config     configs/reinforce_cartpole.yaml \\
        --checkpoint results/checkpoints/reinforce_cartpole/ReinforceAgent_ep200.pt \\
        --format gif --max_frames 300 --fps 24

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

Output (MP4):
    results/videos/<run_name>/rl-video-episode-0.mp4  …
    results/videos/<run_name>/episode_stats.csv

Output (GIF):
    results/gifs/<run_name>/<AgentClass>_<env_id>.gif
    results/gifs/<run_name>/episode_stats.csv
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
    if name == "ppo":
        from agents.ppo.agent import PPOAgent

        a = cfg["agent"]
        return PPOAgent(
            env=env,
            hidden_dims=a["hidden_dims"],
            lr=a["lr"],
            gamma=a["gamma"],
            lam=a.get("lam", 0.95),
            clip_eps=a.get("clip_eps", 0.2),
            n_epochs=a.get("n_epochs", 10),
            batch_size=a.get("batch_size", 64),
            rollout_steps=a.get("rollout_steps", 2048),
            vf_coef=a.get("vf_coef", 0.5),
            ent_coef=a.get("ent_coef", 0.01),
            max_grad_norm=a.get("max_grad_norm", 0.5),
            clip_value_loss=a.get("clip_value_loss", True),
            target_kl=a.get("target_kl", 0.015),
            device=device,
        )
    if name == "ddpg":
        from agents.ddpg.agent import DDPGAgent

        a = cfg["agent"]
        return DDPGAgent(
            env=env,
            hidden_dims=a["hidden_dims"],
            actor_lr=a["actor_lr"],
            critic_lr=a["critic_lr"],
            gamma=a["gamma"],
            tau=a["tau"],
            noise_type=a["noise_type"],
            noise_sigma=a["noise_sigma"],
            noise_sigma_min=a.get("noise_sigma_min", 0.02),
            noise_sigma_decay=a.get("noise_sigma_decay", 0.999),
            device=device,
        )
    if name == "td3":
        from agents.td3.agent import TD3Agent

        a = cfg["agent"]
        return TD3Agent(
            env=env,
            hidden_dims=a["hidden_dims"],
            actor_lr=a["actor_lr"],
            critic_lr=a["critic_lr"],
            gamma=a["gamma"],
            tau=a["tau"],
            policy_delay=a.get("policy_delay", 2),
            target_noise_sigma=a.get("target_noise_sigma", 0.2),
            target_noise_clip=a.get("target_noise_clip", 0.5),
            noise_type=a.get("noise_type", "gaussian"),
            noise_sigma=a.get("noise_sigma", 0.1),
            noise_sigma_min=a.get("noise_sigma_min", 0.01),
            noise_sigma_decay=a.get("noise_sigma_decay", 1.0),
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


def _step(agent, obs, deterministic: bool):
    """Return a plain Python action from either (action, log_prob) or action."""
    result = agent.select_action(obs, deterministic=deterministic)
    action = result[0] if isinstance(result, tuple) else result
    # Unwrap 0-D numpy scalars from discrete agents (DQN etc.) to a Python int.
    # Leave 1-D+ arrays alone — continuous envs (Pendulum, LunarLanderContinuous)
    # expect an array, not a bare float.
    if hasattr(action, "ndim") and action.ndim == 0:
        action = action.item()
    return action


def record(
    agent,
    env_id: str,
    output_dir: Path,
    n_episodes: int,
    deterministic: bool,
    seed: int,
    show: bool,
) -> list[dict]:
    """Run n_episodes with gymnasium's RecordVideo wrapper (MP4 output)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(output_dir),
        episode_trigger=lambda ep_id: True,
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

            action = _step(agent, obs, deterministic)
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


def record_gif(
    agent,
    env_id: str,
    output_path: Path,
    n_episodes: int,
    deterministic: bool,
    seed: int,
    fps: int,
    max_frames: int | None,
) -> list[dict]:
    """Collect frames manually and write a single GIF via imageio."""
    try:
        import imageio
    except ImportError as e:
        raise SystemExit(
            "imageio is required for GIF export: pip install imageio"
        ) from e

    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)

    frames: list = []
    stats = []
    total_frames = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        ep_length = 0
        done = False

        while not done:
            if max_frames is not None and total_frames >= max_frames:
                done = True
                break

            frames.append(env.render())
            total_frames += 1

            action = _step(agent, obs, deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_length += 1
            done = terminated or truncated

        stats.append({"episode": ep, "return": ep_return, "length": ep_length})
        print(
            f"  Episode {ep + 1:>2d}/{n_episodes}"
            f"  return={ep_return:8.2f}"
            f"  length={ep_length:4d}"
            f"  frames={total_frames}"
        )

        if max_frames is not None and total_frames >= max_frames:
            print(f"  max_frames={max_frames} reached — stopping early.")
            break

    env.close()

    print(f"\nWriting GIF ({len(frames)} frames @ {fps} fps) → {output_path}")
    imageio.mimsave(str(output_path), frames, fps=fps)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"GIF size: {size_mb:.2f} MB")

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
        description="Record MP4 or GIF of a trained RL agent.",
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
        help=(
            "Where to save output. Defaults to results/videos/<config_stem>/ (MP4) "
            "or results/gifs/<config_stem>/ (GIF)."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["mp4", "gif"],
        default="mp4",
        help="Output format.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (GIF and MP4).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Hard cap on total frames collected (GIF only). Keeps file size small.",
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
        help="Render frames to screen in real-time (MP4 only; requires a display).",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    agent_name = infer_agent_name(args.config)
    env_id = cfg["env_id"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    set_seed(seed)

    config_stem = Path(args.config).stem
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.format == "gif":
        output_dir = Path("results/gifs") / config_stem
    else:
        output_dir = Path("results/videos") / config_stem

    print(
        f"\n[record_video.py]\n"
        f"  Agent      : {agent_name.upper()}\n"
        f"  Env        : {env_id}\n"
        f"  Checkpoint : {args.checkpoint}\n"
        f"  Format     : {args.format.upper()}\n"
        f"  Episodes   : {args.n_episodes}\n"
        f"  FPS        : {args.fps}\n"
        + (f"  Max frames : {args.max_frames}\n" if args.max_frames else "")
        + f"  Policy     : {'stochastic' if args.stochastic else 'deterministic'}\n"
        f"  Output dir : {output_dir}\n"
    )

    # ── Build agent (using a plain env just for space inference) ─────────────
    _tmp_env = gym.make(env_id)
    agent = build_agent(agent_name, _tmp_env, cfg, device=args.device)
    _tmp_env.close()

    agent.load(args.checkpoint)
    # Put whichever network attribute exists into eval mode
    net = getattr(agent, "policy", None) or getattr(agent, "ac", None)
    if net is not None:
        net.eval()

    # ── Record ───────────────────────────────────────────────────────────────
    print(f"Recording {args.n_episodes} episode(s)...\n")

    if args.format == "gif":
        gif_name = f"{type(agent).__name__}_{env_id}.gif"
        gif_path = output_dir / gif_name
        stats = record_gif(
            agent=agent,
            env_id=env_id,
            output_path=gif_path,
            n_episodes=args.n_episodes,
            deterministic=not args.stochastic,
            seed=seed,
            fps=args.fps,
            max_frames=args.max_frames,
        )
        save_stats(stats, output_dir)
        print(f"GIF saved to: {gif_path}\n")
    else:
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
