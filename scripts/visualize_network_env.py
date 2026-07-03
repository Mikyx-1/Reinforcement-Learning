#!/usr/bin/env python
"""
Render a GIF of NetworkRoutingEnv running a random (or shortest-path) policy —
no trained agent needed. Useful for sanity-checking the simulation and
topology/congestion visuals before wiring up a GNN agent.

Usage:
    python scripts/visualize_network_env.py
    python scripts/visualize_network_env.py --policy shortest_path --ticks 300
    python scripts/visualize_network_env.py --num_nodes 20 --arrival_rate 0.6

Output:
    results/gifs/network_routing/<policy>.gif
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import numpy as np

from envs.network_routing import NetworkRoutingEnv


def shortest_path_action(env: NetworkRoutingEnv, obs: dict, info: dict) -> int:
    """Baseline: greedily hop towards the destination via the delay-weighted shortest path."""
    current = info["current_node"]
    dest = info["destination_node"]
    path = nx.shortest_path(env.graph, current, dest, weight=lambda u, v, _: env.directed[u][v]["delay"])
    nxt = path[1]
    return env.neighbors[current].index(nxt)


def random_action(env: NetworkRoutingEnv, obs: dict, info: dict) -> int:
    valid = np.flatnonzero(obs["action_mask"])
    return int(env.np_random.choice(valid))


POLICIES = {"random": random_action, "shortest_path": shortest_path_action}


def main():
    parser = argparse.ArgumentParser(description="Visualize NetworkRoutingEnv as a GIF.")
    parser.add_argument("--policy", choices=list(POLICIES), default="random")
    parser.add_argument("--num_nodes", type=int, default=15)
    parser.add_argument("--arrival_rate", type=float, default=0.4)
    parser.add_argument("--ticks", type=int, default=200, help="max_ticks for the episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=1, help="low by default — each decision is otherwise too fast to follow")
    parser.add_argument("--render_every", type=int, default=8, help="render every Nth decision (keeps GIFs short)")
    parser.add_argument("--width", type=int, default=None, help="downscale frames to this width (keeps aspect ratio)")
    parser.add_argument("--gif_colors", type=int, default=128, help="palette size for GIF export")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    env = NetworkRoutingEnv(
        num_nodes=args.num_nodes,
        arrival_rate=args.arrival_rate,
        max_ticks=args.ticks,
        render_mode="rgb_array",
    )
    policy_fn = POLICIES[args.policy]

    obs, info = env.reset(seed=args.seed)
    frames = [env.render()]
    step_count = 0

    while info["tick"] < args.ticks:
        action = policy_fn(env, obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        if step_count % args.render_every == 0:
            frames.append(env.render())
        if terminated or truncated:
            break

    stats = env.stats
    print(
        f"policy={args.policy}  decisions={step_count}  "
        f"delivered={stats['delivered']}  dropped={stats['dropped']}  "
        f"avg_latency={stats['total_latency'] / max(stats['delivered'], 1):.2f}"
    )

    output_path = Path(args.output) if args.output else Path("results/gifs/network_routing") / f"{args.policy}.gif"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    pil_frames = [Image.fromarray(f) for f in frames]
    if args.width is not None:
        ratio = args.width / pil_frames[0].width
        size = (args.width, round(pil_frames[0].height * ratio))
        pil_frames = [f.resize(size, Image.LANCZOS) for f in pil_frames]
    pil_frames = [f.convert("P", palette=Image.ADAPTIVE, colors=args.gif_colors) for f in pil_frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=round(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    print(f"Saved {len(frames)} frames -> {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
