"""
Lightweight logger that writes to:
  - stdout (always)
  - W&B (if wandb is installed and init'd externally)
  - CSV metrics file

Usage:
    logger = Logger(log_dir="results/runs/reinforce_cartpole")
    logger.log({"policy_loss": 0.42, "episode_return": 195.0}, step=1000)
    logger.close()
"""

import sys
import time
from pathlib import Path
from typing import Any


class Logger:
    def __init__(
        self,
        log_dir: str | Path = "results/runs/default",
        use_wandb: bool = False,
        wandb_kwargs: dict[str, Any] | None = None,
        print_freq: int = 1,  # print every N calls to log()
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.print_freq = print_freq
        self._call_count = 0
        self._start_time = time.time()


        # W&B
        self._wandb = None
        if use_wandb:
            try:
                import wandb

                if wandb.run is None:
                    # Default initialization if not already done
                    init_args = wandb_kwargs or {}
                    wandb.init(**init_args)

                self._wandb = wandb
            except ImportError:
                print("[Logger] wandb not installed, skipping W&B logging.")

        # Plain-text log file
        self._txt = open(self.log_dir / "metrics.csv", "w")
        self._header_written = False

    # ------------------------------------------------------------------

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self._call_count += 1

        # CSV
        if not self._header_written:
            self._txt.write("step," + ",".join(metrics.keys()) + "\n")
            self._header_written = True
        row = str(step) + "," + ",".join(str(v) for v in metrics.values())
        self._txt.write(row + "\n")
        self._txt.flush()


        # W&B
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

        # Stdout
        if self._call_count % self.print_freq == 0:
            elapsed = time.time() - self._start_time
            parts = [f"step={step:>7d}", f"elapsed={elapsed:6.1f}s"]
            parts += [
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            ]
            print("  |  ".join(parts), flush=True)

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        """Log hyperparameters."""
        if self._wandb is not None:
            self._wandb.config.update(hparams)
        print("[Logger] Hyperparameters:", hparams)

    def close(self) -> None:
        self._txt.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
