#!/usr/bin/env python
"""
Run multi-seed evaluation of a checkpoint and produce a learning-curve PNG.

Usage:
    python scripts/run_sweep.py \
        --config configs/ppo_cartpole.yaml \
        --checkpoint results/checkpoints/ppo_cartpole/PPOAgent_ep1400.pt \
        --n_seeds 5 \
        --n_episodes 20
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit seed list.")
    parser.add_argument("--n_seeds", type=int, default=5, help="Run seeds 0..N-1.")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else list(range(args.n_seeds))
    run_name = Path(args.config).stem
    base_dir = Path(f"results/evals/{run_name}")

    print(f"[run_sweep.py] Config   : {args.config}")
    print(f"[run_sweep.py] Checkpoint: {args.checkpoint}")
    print(f"[run_sweep.py] Seeds    : {seeds}  ({args.n_episodes} episodes each)")
    print(f"[run_sweep.py] Output   : {base_dir}")

    statuses = {}
    for seed in seeds:
        out_dir = base_dir / f"seed_{seed}"
        print(f"\n{'='*60}")
        print(f"[run_sweep.py] Seed {seed} → {out_dir}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "scripts/evaluate.py",
            "--config", args.config,
            "--checkpoint", args.checkpoint,
            "--seed", str(seed),
            "--out_dir", str(out_dir),
            "--n_episodes", str(args.n_episodes),
            "--device", args.device,
        ]
        ret = subprocess.run(cmd)
        statuses[seed] = ret.returncode == 0

    print(f"\n{'='*60}")
    print("[run_sweep.py] Summary:")
    for seed, ok in statuses.items():
        print(f"  seed {seed}: {'OK' if ok else 'FAILED'}")

    successful = [s for s, ok in statuses.items() if ok]
    if len(successful) >= 2:
        from common.plotting import plot_learning_curves
        run_dirs = [base_dir / f"seed_{s}" for s in successful]
        out = base_dir / "learning_curves.png"
        plot_learning_curves(run_dirs, output_path=out)
        print(f"\n[run_sweep.py] Learning curve → {out}")
    else:
        print("[run_sweep.py] Not enough successful seeds to plot.")


if __name__ == "__main__":
    main()
