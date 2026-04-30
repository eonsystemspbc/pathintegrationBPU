from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .config import TASK_CX_POLAR_BUMP, OutputPaths


def _mean_sem(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    grouped = frame.groupby(["model", x])[y]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"].fillna(0.0) / out["count"].clip(lower=1) ** 0.5
    return out


def _is_polar_bump(metrics: pd.DataFrame) -> bool:
    return "task" in metrics.columns and set(metrics["task"].dropna().astype(str)) == {
        TASK_CX_POLAR_BUMP
    }


def _error_axis_label(metrics: pd.DataFrame) -> str:
    if _is_polar_bump(metrics):
        return "Home-distance RMSE"
    return "Position RMSE"


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
    ax.set_ylabel(_error_axis_label(metrics))
    title = (
        "Home-vector error vs sequence length"
        if _is_polar_bump(metrics)
        else "Path-integration error vs sequence length"
    )
    ax.set_title(title)
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
    ax.set_ylabel(f"{_error_axis_label(metrics)} at T=200")
    ax.set_title("Noise robustness")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_loss_curve(loss_history: pd.DataFrame, out_path: Path) -> None:
    if loss_history.empty:
        return
    required = {"model", "epoch", "train_mse", "val_mse"}
    if not required.issubset(loss_history.columns):
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)
    for model, group in loss_history.groupby("model"):
        group = group.sort_values("epoch")
        ax.plot(
            group["epoch"],
            group["train_mse"],
            marker="o",
            linewidth=1.6,
            linestyle="-",
            label=f"{model} train",
        )
        ax.plot(
            group["epoch"],
            group["val_mse"],
            marker="s",
            linewidth=1.6,
            linestyle="--",
            label=f"{model} val",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training and validation loss")
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
    if paths.loss_history_csv.exists():
        loss_history = pd.read_csv(paths.loss_history_csv)
        plot_loss_curve(loss_history, paths.loss_curve_png)
