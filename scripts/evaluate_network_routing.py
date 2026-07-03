#!/usr/bin/env python
"""
Head-to-head comparison of routing policies on NetworkRoutingEnv, using the
SAME topology + traffic config for every policy (fair comparison).

Return isn't the right metric here (reward shaping — congestion penalty,
drop penalty — differs in scale from anything human-readable); the metrics
that actually matter are delivery rate, drop rate, and mean latency of
delivered packets.

Usage:
    python scripts/evaluate_network_routing.py \\
        --config configs/gnnppo_network_routing.yaml \\
        --checkpoint results/checkpoints/gnnppo_network_routing/GNNPPOAgent_ep950.pt \\
        --n_episodes 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import numpy as np

from common.registry import build_agent
from common.utils import load_config
from envs.network_routing import NetworkRoutingEnv


def random_policy(env, obs, info, agent=None):
    valid = np.flatnonzero(obs["action_mask"])
    return int(env.np_random.choice(valid))


def shortest_path_policy(env, obs, info, agent=None):
    current, dest = info["current_node"], info["destination_node"]
    path = nx.shortest_path(env.graph, current, dest, weight=lambda u, v, _: env.directed[u][v]["delay"])
    return env.neighbors[current].index(path[1])


def gnn_ppo_policy(env, obs, info, agent):
    action, _ = agent.select_action(obs, deterministic=True)
    return int(action)


POLICIES = {"random": random_policy, "shortest_path": shortest_path_policy, "gnn_ppo": gnn_ppo_policy}


def run_episode(env, policy_fn, agent, seed) -> dict:
    obs, info = env.reset(seed=seed)
    episode_return = 0.0
    decisions = 0
    while info["tick"] < env.max_ticks:
        action = policy_fn(env, obs, info, agent)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        decisions += 1
        if terminated or truncated:
            break

    delivered = env.stats["delivered"]
    dropped = env.stats["dropped"]
    total = delivered + dropped
    return {
        "return": episode_return,
        "decisions": decisions,
        "delivered": delivered,
        "dropped": dropped,
        "drop_rate": dropped / total if total else 0.0,
        "avg_latency": env.stats["total_latency"] / delivered if delivered else float("nan"),
    }


def summarize(name: str, episodes: list[dict]) -> None:
    def stat(key):
        vals = [e[key] for e in episodes]
        return np.mean(vals), np.std(vals)

    ret_m, ret_s = stat("return")
    del_m, del_s = stat("delivered")
    drop_m, drop_s = stat("dropped")
    dr_m, dr_s = stat("drop_rate")
    lat_m, lat_s = stat("avg_latency")

    print(
        f"{name:<15}"
        f"return={ret_m:9.1f}±{ret_s:<7.1f}"
        f"delivered={del_m:6.1f}±{del_s:<5.1f}"
        f"dropped={drop_m:6.1f}±{drop_s:<5.1f}"
        f"drop_rate={dr_m*100:5.1f}%±{dr_s*100:<4.1f}"
        f"avg_latency={lat_m:6.2f}±{lat_s:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Compare routing policies on NetworkRoutingEnv.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="GNN-PPO checkpoint to evaluate.")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--seed_start", type=int, default=10_000, help="Held-out seeds, disjoint from training.")
    parser.add_argument(
        "--arrival_rate", type=float, default=None,
        help="Override env_kwargs.arrival_rate — stress-test the same checkpoint under heavier load "
        "without retraining (topology/network weights don't depend on traffic dynamics).",
    )
    parser.add_argument(
        "--edge_capacity_max", type=int, default=None,
        help="Override the upper bound of env_kwargs.edge_capacity_range (tightens congestion).",
    )
    parser.add_argument("--max_ticks", type=int, default=None, help="Override env_kwargs.max_ticks (episode length).")
    parser.add_argument(
        "--policies", nargs="+", default=None, choices=list(POLICIES),
        help="Subset of policies to run (default: all). Useful to skip 'random' once it's established as the floor.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_kwargs = dict(cfg.get("env_kwargs", {}))
    if args.arrival_rate is not None:
        env_kwargs["arrival_rate"] = args.arrival_rate
    if args.edge_capacity_max is not None:
        lo = env_kwargs.get("edge_capacity_range", [3, 8])[0]
        env_kwargs["edge_capacity_range"] = [lo, args.edge_capacity_max]
    if args.max_ticks is not None:
        env_kwargs["max_ticks"] = args.max_ticks

    policies = {k: POLICIES[k] for k in (args.policies or POLICIES)}

    env = NetworkRoutingEnv(**env_kwargs)
    agent = build_agent("gnnppo", env, cfg, device="cpu")
    agent.load(args.checkpoint)

    print(
        f"\nEnv: num_nodes={env_kwargs.get('num_nodes')}  arrival_rate={env_kwargs.get('arrival_rate')}  "
        f"max_ticks={env_kwargs.get('max_ticks')}  n_episodes={args.n_episodes}\n"
    )
    print(f"{'policy':<15}{'return':<17}{'delivered':<13}{'dropped':<13}{'drop_rate':<14}{'avg_latency'}")
    print("-" * 95)

    for name, policy_fn in policies.items():
        episodes = [
            run_episode(env, policy_fn, agent, seed=args.seed_start + i) for i in range(args.n_episodes)
        ]
        summarize(name, episodes)


if __name__ == "__main__":
    main()
