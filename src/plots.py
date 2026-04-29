from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .config import OutputPaths


def _mean_sem(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    grouped = frame.groupby(["model", x])[y]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"].fillna(0.0) / out["count"].clip(lower=1) ** 0.5
    return out


def plot_error_vs_sequence_length(metrics: pd.DataFrame, out_path: Path) -> None:
    clean = metrics[(metrics["split"] == "test") & (metrics["noise_std"] == 0.0)]
    if clean.empty:
        return
    summary = _mean_sem(clean, "T", "position_rmse")
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)
    for model, group in summary.groupby("model"):
        group = group.sort_values("T")
        ax.errorbar(
            group["T"],
            group["mean"],
            yerr=group["sem"],
            marker="o",
            capsize=3,
            linewidth=1.8,
            label=model,
        )
    ax.set_xlabel("Sequence length T")
    ax.set_ylabel("Position RMSE")
    ax.set_title("Path-integration error vs sequence length")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_error_vs_noise(metrics: pd.DataFrame, out_path: Path) -> None:
    noisy = metrics[metrics["split"] == "test_noise"]
    if noisy.empty or noisy["noise_std"].nunique() < 2:
        return
    summary = _mean_sem(noisy, "noise_std", "position_rmse")
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)
    for model, group in summary.groupby("model"):
        group = group.sort_values("noise_std")
        ax.errorbar(
            group["noise_std"],
            group["mean"],
            yerr=group["sem"],
            marker="o",
            capsize=3,
            linewidth=1.8,
            label=model,
        )
    ax.set_xlabel("Input noise std")
    ax.set_ylabel("Position RMSE at T=200")
    ax.set_title("Noise robustness")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_plots(paths: OutputPaths) -> None:
    if not paths.metrics_by_seed_csv.exists():
        return
    metrics = pd.read_csv(paths.metrics_by_seed_csv)
    plot_error_vs_sequence_length(metrics, paths.error_vs_sequence_length_png)
    plot_error_vs_noise(metrics, paths.error_vs_noise_png)
