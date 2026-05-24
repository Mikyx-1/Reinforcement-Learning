"""
Plotting utilities for multi-seed RL experiments.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _iqm(matrix: np.ndarray) -> np.ndarray:
    """Interquartile mean along axis=0 (across seeds) at each timestep."""
    q25 = np.percentile(matrix, 25, axis=0)
    q75 = np.percentile(matrix, 75, axis=0)
    result = []
    for t in range(matrix.shape[1]):
        col = matrix[:, t]
        mask = (col >= q25[t]) & (col <= q75[t])
        result.append(col[mask].mean() if mask.any() else col.mean())
    return np.array(result)


def plot_learning_curves(
    run_dirs,
    metric: str = "return",
    x_col: str = "episode",
    smooth: int = 1,
    output_path=None,
) -> plt.Figure:
    """
    Plot mean ± std and IQM curves from multi-seed runs.

    Args:
        run_dirs:    List of seed directories, each containing episode_returns.csv.
        metric:      Column name in the CSV for the y-axis (default: "return").
        x_col:       Column name for the x-axis (default: "episode").
        smooth:      Rolling-average window size (1 = no smoothing).
        output_path: Path to save the PNG. Defaults to run_dirs[0].parent/learning_curves.png.

    Returns:
        The matplotlib Figure.
    """
    run_dirs = [Path(d) for d in run_dirs]

    series = []
    for d in run_dirs:
        csv = d / "episode_returns.csv"
        if not csv.exists():
            print(f"[plotting] Warning: {csv} not found, skipping.")
            continue
        df = pd.read_csv(csv)
        if metric not in df.columns:
            print(f"[plotting] Warning: '{metric}' not in {csv}, skipping.")
            continue
        series.append(df[[x_col, metric]].dropna())

    if not series:
        raise ValueError(f"No valid data found for metric '{metric}'.")

    # Align all seeds to the shortest series
    min_len = min(len(s) for s in series)
    x = series[0][x_col].values[:min_len]
    matrix = np.stack([
        np.interp(x, s[x_col].values, s[metric].values)
        for s in series
    ])  # shape: [n_seeds, T]

    if smooth > 1:
        matrix = np.stack([_smooth(row, smooth) for row in matrix])

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    iqm_line = _iqm(matrix)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, label="Mean ± Std")
    ax.plot(x, mean, linewidth=2, label=f"Mean (n={len(series)})")
    ax.plot(x, iqm_line, linewidth=2, linestyle="--", label="IQM")
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel(metric.capitalize())
    title = f"{metric}  ({len(series)} seeds"
    if smooth > 1:
        title += f", smooth={smooth}"
    title += ")"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is None:
        output_path = run_dirs[0].parent / "learning_curves.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plotting] Saved → {output_path}")
    return fig
