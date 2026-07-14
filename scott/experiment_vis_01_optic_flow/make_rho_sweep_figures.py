#!/usr/bin/env python
"""Figures for vis-01 subrun 05: the spectral-radius (rho) sweep on mb_core_alpn (yaw-only).

The question the sweep answers is one-dimensional -- "does raising rho lift the connectome FlowRNN off the
R2 ~= 0 floor?" -- so the figures are read along the rho axis:

  fig_rho_sweep_summary.png        -- per-seed BEST held-out yaw R2 vs rho (strip + median line per rho),
                                      against the causal GRU ceiling and the predict-the-mean floor. THE
                                      headline: does any rho column climb off the floor toward the ceiling?
  fig_rho_sweep_curves.png         -- per-rho training curves (every seed thin + bold median), one panel
                                      per rho, shared y-axis -- shows WHETHER a rho learns and whether the
                                      high-rho columns DIVERGE (curves shoot negative) rather than learn.

Reads only the collected outputs (result.json + metrics_epochs.csv per run; gate_yaw1d_causal.json for the
fair GRU ceiling). No GPU, no network. Run after `run.py --collect` has pulled the fleet:

    uv run python scott/experiment_vis_01_optic_flow/make_rho_sweep_figures.py [OUTPUT_DIR]

Design follows the repo's data-viz method: color by the rho entity in fixed order (a single-hue sequential
ramp light->dark as rho rises, since rho is an ordered magnitude), one y-axis, thin seed traces + a bold
median, recessive grid, direct labels (no legend), the causal-GRU ceiling + predict-mean floor drawn
faintly so the gap is legible at a glance.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "subruns" / "05_rho_sweep" / "outputs"
FIG_DIR = HERE / "subruns" / "05_rho_sweep" / "figures"

# rho is an ORDERED magnitude -> a single-hue sequential ramp (light = low rho, dark = high rho).
RHO_COLORS = ["#9ec9e8", "#5fa0d8", "#2a78d6", "#1b4f8a"]   # 0.95 -> 1.2, light -> dark (validated blues)
GRU_COLOR = "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a86", "#e6e5e1"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "figure.dpi": 130, "svg.fonttype": "none",
})


def load_runs(out_dir: Path):
    """Group runs by rho_target. Returns {rho: {'best': [...], 'curves': [np.array,...]}} sorted by rho."""
    runs_dir = out_dir / "runs"
    by_rho: dict[float, dict] = {}
    for d in sorted(runs_dir.glob("*_connectome_u*")):
        rj = d / "result.json"
        if not rj.exists():
            continue
        r = json.loads(rj.read_text())
        rho = round(float(r.get("rho_target", 0.95)), 4)
        slot = by_rho.setdefault(rho, {"best": [], "curves": []})
        slot["best"].append(float(r["best_val_r2"]))
        ys = []
        mcsv = d / "metrics_epochs.csv"
        if mcsv.exists():
            with open(mcsv) as f:
                for row in csv.DictReader(f):
                    ys.append(float(row["val_mean_r2"]))
        if ys:
            slot["curves"].append(np.array(ys))
    return dict(sorted(by_rho.items()))


def causal_ceiling(out_dir: Path) -> float | None:
    p = out_dir / "gate_yaw1d_causal.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["gate"]["per_dof_r2"]["yaw_rate"])


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)


def color_for(i: int) -> str:
    return RHO_COLORS[i] if i < len(RHO_COLORS) else RHO_COLORS[-1]


def fig_summary(by_rho: dict, ceil: float | None, out_dir: Path):
    rhos = list(by_rho.keys())
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    style_ax(ax)
    if ceil is not None:
        ax.axhline(ceil, color=GRU_COLOR, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.text(len(rhos) - 0.55, ceil + 0.01, f"causal GRU ceiling {ceil:.2f}",
                ha="right", va="bottom", color=GRU_COLOR, fontsize=7.5, fontweight="bold")
    ax.text(0.0, 0.012, "predict-per-episode-mean floor", ha="left", va="bottom",
            color=MUTED, fontsize=7.0)

    rng = np.random.default_rng(0)
    for i, rho in enumerate(rhos):
        best = np.array(by_rho[rho]["best"], dtype=float)
        jit = (rng.random(len(best)) - 0.5) * 0.28
        ax.scatter(i + jit, best, s=26, color=color_for(i), alpha=0.8, edgecolor="white",
                   linewidth=0.5, zorder=4)
        med = float(np.median(best)) if len(best) else float("nan")
        ax.plot([i - 0.22, i + 0.22], [med, med], color=color_for(i), linewidth=2.6, zorder=5)
        ax.text(i, med, f"  med {med:.3f}", ha="left", va="center", color=color_for(i),
                fontsize=7.5, fontweight="bold")

    ax.set_xticks(range(len(rhos)))
    ax.set_xticklabels([f"ρ = {r:g}" for r in rhos], fontsize=9)
    ax.set_xlabel("recurrence spectral radius  ρ  (init, both arms)")
    ax.set_ylabel("best held-out yaw-rate  $R^2$  (per seed)")
    lo = min(-0.45, *(min(by_rho[r]["best"], default=0) for r in rhos)) - 0.05
    hi = max(0.9, (ceil or 0) + 0.1)
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.5, len(rhos) - 0.5)
    fig.suptitle("vis-01 subrun 05 · does raising ρ lift mb_core_alpn off the $R^2\\!\\approx\\!0$ floor?",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_rho_sweep_summary.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def fig_curves(by_rho: dict, ceil: float | None, out_dir: Path):
    rhos = list(by_rho.keys())
    n = len(rhos)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), sharey=True, squeeze=False)
    axes = axes[0]
    ylo, yhi = -0.6, max(0.9, (ceil or 0) + 0.1)
    for i, (ax, rho) in enumerate(zip(axes, rhos)):
        style_ax(ax)
        if ceil is not None:
            ax.axhline(ceil, color=GRU_COLOR, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.7, zorder=2)
        curves = by_rho[rho]["curves"]
        for ys in curves:
            ax.plot(np.arange(1, len(ys) + 1), ys, color=color_for(i), linewidth=0.5, alpha=0.3, zorder=3)
        if curves:
            L = min(len(ys) for ys in curves)
            med = np.median(np.stack([ys[:L] for ys in curves]), axis=0)
            ax.plot(np.arange(1, L + 1), med, color=color_for(i), linewidth=2.0, zorder=5)
            best = max(by_rho[rho]["best"])
            ax.text(0.97, 0.03, f"best {best:.3f}", transform=ax.transAxes, ha="right", va="bottom",
                    color=color_for(i), fontsize=8, fontweight="bold")
        ax.set_title(f"ρ = {rho:g}", fontsize=9.5, color=INK, loc="left", pad=6)
        ax.set_xlabel("epoch")
        ax.set_ylim(ylo, yhi)
    axes[0].set_ylabel("held-out yaw-rate  $R^2$")
    if ceil is not None:
        axes[0].text(0.03, 0.97, "causal GRU", transform=axes[0].transAxes, ha="left", va="top",
                     color=GRU_COLOR, fontsize=7, alpha=0.85)
    fig.suptitle("Per-ρ training curves — a column that climbs = ρ cleared the floor; a column diving "
                 "negative = divergence",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_rho_sweep_curves.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main(argv):
    out_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT
    if not (out_dir / "runs").exists():
        print(f"[make_rho_sweep_figures] no runs/ under {out_dir} yet -- run `run.py --collect` first.")
        return 0
    by_rho = load_runs(out_dir)
    if not by_rho:
        print(f"[make_rho_sweep_figures] no result.json found under {out_dir}/runs -- nothing to plot.")
        return 0
    ceil = causal_ceiling(out_dir)
    print("rho groups:", {r: len(v["best"]) for r, v in by_rho.items()}, "causal ceiling:", ceil)
    fig_summary(by_rho, ceil, out_dir)
    fig_curves(by_rho, ceil, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
