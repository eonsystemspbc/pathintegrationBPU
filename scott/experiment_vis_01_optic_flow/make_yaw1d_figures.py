#!/usr/bin/env python
"""Figures for vis-01 subruns 03 (optic lobe) + 04 (mushroom body): the yaw-only learnability probe.

Two figures, both telling the same "floor vs ceiling" story:
  fig_yaw1d_training_curves.png -- per-condition held-out yaw-rate R2 vs epoch (every seed + median),
                                   the two GRU ceilings in their own panel, shared y-axis.
  fig_yaw1d_summary.png         -- per-seed BEST val R2 by substrate (strip), against the GRU ceiling
                                   band and the predict-the-mean floor.

Reads only the collected outputs (result.json + metrics_epochs.csv per run; gate_*_curve.json for the
GRU). No GPU, no network. Run after `collect.sh` (or run.py --collect) has pulled both subruns:

    uv run python scott/experiment_vis_01_optic_flow/make_yaw1d_figures.py

Design follows the repo's data-viz method: color by entity in fixed order (optic lobe = blue,
mb_full = aqua, mb_core_alpn = violet, GRU = red), one y-axis, thin seed traces + a bold median,
recessive grid, direct labels (no legend needed), and the naive floor / causal-GRU ceiling drawn
faintly across the connectome panels so the gap is legible at a glance.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OL_RUNS = HERE / "subruns" / "03_yaw1d" / "outputs" / "runs"
MB_RUNS = HERE / "subruns" / "04_mb_yaw1d" / "outputs" / "runs"
GATE_DIR = HERE / "subruns" / "03_yaw1d" / "outputs"          # curve-augmented GRU ceilings
OUT_DIR = HERE / "figures"

# entity -> (runs dir, color, human label, N neurons) in fixed categorical order
CONDITIONS = [
    ("ol_left",      OL_RUNS, "#2a78d6", "Optic lobe\n(48,894 neurons)"),
    ("mb_full",      MB_RUNS, "#1baf7a", "Mushroom body — full\n(14,025 neurons)"),
    ("mb_core_alpn", MB_RUNS, "#4a3aa7", "MB core + ALPN\n(6,014 neurons)"),
]
GRU_COLOR = "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a86", "#e6e5e1"
NAIVE_FLOOR = -0.3609        # predict-the-train-mean yaw_rate R2 (from gate naive baseline)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "figure.dpi": 130, "svg.fonttype": "none",
})


def load_curves(sub: str, runs_dir: Path):
    """Return list of per-epoch val_mean_r2 arrays and the per-seed best_val_r2 for one substrate."""
    curves, best = [], []
    for d in sorted(runs_dir.glob(f"{sub}_connectome_u*_hp*")):
        r = json.loads((d / "result.json").read_text())
        if r["substrate"] != sub:
            continue
        ys = []
        with open(d / "metrics_epochs.csv") as f:
            for row in csv.DictReader(f):
                ys.append(float(row["val_mean_r2"]))
        curves.append(np.array(ys))
        best.append(float(r["best_val_r2"]))
    return curves, best


def gru_curve(name: str):
    g = json.loads((GATE_DIR / name).read_text())["gate"]
    return np.array(g["curve_yaw_r2"]), g["per_dof_r2"]["yaw_rate"]


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)


# ------------------------------------------------------------------ figure 1: training curves
def fig_training_curves():
    data = {sub: load_curves(sub, rd) for sub, rd, _, _ in CONDITIONS}
    gru_bi, bi_final = gru_curve("gate_yaw1d_curve.json")
    gru_ca, ca_final = gru_curve("gate_yaw1d_causal_curve.json")

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharey=True)
    ylo, yhi = -0.45, 0.9

    for ax, (sub, _rd, color, label) in zip(axes[:3], CONDITIONS):
        style_ax(ax)
        curves, best = data[sub]
        # reference lines: naive floor + the FAIR (causal) GRU ceiling
        ax.axhline(ca_final, color=GRU_COLOR, linewidth=1.1, linestyle=(0, (4, 3)), alpha=0.7, zorder=2)
        ax.axhline(NAIVE_FLOOR, color=MUTED, linewidth=1.0, linestyle=(0, (1, 2)), zorder=2)
        # every seed, thin + faded
        for ys in curves:
            ax.plot(np.arange(1, len(ys) + 1), ys, color=color, linewidth=0.5, alpha=0.28, zorder=3)
        # median across seeds (clip to common length)
        L = min(len(ys) for ys in curves)
        med = np.median(np.stack([ys[:L] for ys in curves]), axis=0)
        ax.plot(np.arange(1, L + 1), med, color=color, linewidth=2.0, zorder=5)
        ax.set_title(label, fontsize=8.5, color=INK, loc="left", pad=6)
        ax.set_xlabel("epoch")
        ax.set_xlim(0, 300)
        ax.set_ylim(ylo, yhi)
        # direct labels
        ax.text(297, med[-1] + 0.02, f"median  best {max(best):.3f}", ha="right", va="bottom",
                color=color, fontsize=7.5, fontweight="bold")

    # GRU panel
    ax = axes[3]
    style_ax(ax)
    ax.plot(np.arange(1, len(gru_bi) + 1), gru_bi, color=GRU_COLOR, linewidth=2.0, zorder=5)
    ax.plot(np.arange(1, len(gru_ca) + 1), gru_ca, color=GRU_COLOR, linewidth=2.0,
            linestyle=(0, (4, 3)), zorder=5)
    ax.set_title("GRU ceiling\n(same yaw stimulus)", fontsize=8.5, color=INK, loc="left", pad=6)
    ax.set_xlabel("epoch")
    ax.set_xlim(0, 80)
    ax.set_ylim(ylo, yhi)
    ax.text(78, bi_final + 0.015, f"bidirectional  {bi_final:.2f}", ha="right", va="bottom",
            color=GRU_COLOR, fontsize=7.5, fontweight="bold")
    ax.text(78, ca_final - 0.02, f"causal  {ca_final:.2f}", ha="right", va="top",
            color=GRU_COLOR, fontsize=7.5, fontweight="bold")

    axes[0].set_ylabel("held-out yaw-rate  $R^2$")
    # a small shared annotation for the reference lines on the first panel
    axes[0].text(150, ca_final + 0.02, "fair GRU ceiling", color=GRU_COLOR, fontsize=6.8, alpha=0.8)
    axes[0].text(150, NAIVE_FLOOR - 0.06, "predict-mean floor", color=MUTED, fontsize=6.8)

    fig.suptitle("vis-01 yaw-only optomotor: connectome networks floor at $R^2\\!\\approx\\!0$ while a GRU "
                 "reaches 0.58–0.76 on the identical stimulus",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "fig_yaw1d_training_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------ figure 2: best-val summary
def fig_summary():
    gru_bi, bi_final = gru_curve("gate_yaw1d_curve.json")
    gru_ca, ca_final = gru_curve("gate_yaw1d_causal_curve.json")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    style_ax(ax)

    # ceiling band (causal..bidirectional) + naive floor
    ax.axhspan(ca_final, bi_final, color=GRU_COLOR, alpha=0.10, zorder=1)
    ax.axhline(bi_final, color=GRU_COLOR, linewidth=1.2, zorder=2)
    ax.axhline(ca_final, color=GRU_COLOR, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.axhline(NAIVE_FLOOR, color=MUTED, linewidth=1.0, linestyle=(0, (1, 2)), zorder=2)

    rng = np.random.default_rng(0)
    xs = []
    for i, (sub, rd, color, label) in enumerate(CONDITIONS):
        _c, best = load_curves(sub, rd)
        jit = (rng.random(len(best)) - 0.5) * 0.28
        ax.scatter(i + jit, best, s=26, color=color, alpha=0.75, edgecolor="white",
                   linewidth=0.5, zorder=4)
        med = float(np.median(best))
        ax.plot([i - 0.22, i + 0.22], [med, med], color=color, linewidth=2.6, zorder=5)
        ax.text(i, -0.02, f"median {med:.3f}", ha="center", va="top", color=color,
                fontsize=7.5, fontweight="bold")
        xs.append(label.replace("\n", " "))

    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([s.split(" (")[0] for s in xs], fontsize=8.5)
    ax.set_ylabel("best held-out yaw-rate  $R^2$  (per seed)")
    ax.set_ylim(-0.45, 0.9)
    ax.set_xlim(-0.5, len(CONDITIONS) - 0.5)
    ax.text(len(CONDITIONS) - 0.55, bi_final + 0.01, f"GRU bidirectional {bi_final:.2f}",
            ha="right", va="bottom", color=GRU_COLOR, fontsize=7.5, fontweight="bold")
    ax.text(len(CONDITIONS) - 0.55, ca_final - 0.015, f"GRU causal {ca_final:.2f}",
            ha="right", va="top", color=GRU_COLOR, fontsize=7.5, fontweight="bold")
    ax.text(len(CONDITIONS) - 0.55, NAIVE_FLOOR - 0.02, "predict-mean floor",
            ha="right", va="top", color=MUTED, fontsize=7.5)

    fig.suptitle("Every connectome seed lands in the $R^2\\!\\approx\\!0$ band — far below the GRU ceiling",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "fig_yaw1d_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_training_curves()
    fig_summary()
