#!/usr/bin/env python
"""Figures for vis-01 subrun 07: the FAIR connectome-vs-degree-matched-control test, normalization OFF,
long run, activation-RMS-matched control -- read PER W_in gain.

The question is "at each W_in gain in {3,4,5}, does the real connectome beat a degree-matched random
rewiring on yaw regression, once the arms are matched on activity (not just rho)?" So each gain is a
paired connectome-vs-control comparison:

  fig_control_summary.png   -- per gain, two strips side by side: connectome seeds (10) vs control graphs
                               (10), per-seed BEST held-out yaw R2, with the connectome median and control
                               median marked, against the causal GRU ceiling and the predict-mean floor.
                               THE headline: at which gain does the connectome strip sit ABOVE the control?
  fig_control_curves.png    -- per gain, one panel: connectome median training curve (bold) over the
                               control band (min-max across control graphs). Shows whether the connectome
                               climbs above the null and whether either arm is still rising at the cap.

Reads only the collected outputs (result.json + metrics_epochs.csv per run; gate_yaw1d_causal.json for the
fair GRU ceiling). No GPU, no network. Run after `run.py --collect`:

    uv run python scott/experiment_vis_01_optic_flow/make_control_compare_figures.py [OUTPUT_DIR]

Design follows the repo's data-viz method: the connectome is ONE fixed entity (a single warm ink color at
every gain); the degree-matched control is the null (a single cool/grey color). One y-axis, thin per-seed
traces + a bold median, recessive grid, direct labels (no legend), causal-GRU ceiling + predict-mean floor
drawn faintly.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "subruns" / "07_normoff_control" / "outputs"
FIG_DIR = HERE / "subruns" / "07_normoff_control" / "figures"

CONN_COLOR = "#1b4f8a"     # connectome = one fixed warm-ink entity (deep blue)
CTRL_COLOR = "#9a9a95"     # degree-matched control = the null (neutral grey)
GRU_COLOR = "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a86", "#e6e5e1"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "figure.dpi": 130, "svg.fonttype": "none",
})

CONN, CTRL = "connectome", "degree_matched"


def load_runs(out_dir: Path):
    """Group by (w_in_gain, condition). Returns {gain: {cond: {'best':[...],'test':[...],'curves':[...]}}}."""
    runs_dir = out_dir / "runs"
    by: dict[float, dict] = {}
    for d in sorted(runs_dir.glob("*_u*")):
        rj = d / "result.json"
        if not rj.exists():
            continue
        r = json.loads(rj.read_text())
        cond = r.get("condition")
        if cond not in (CONN, CTRL):
            continue
        gain = round(float(r.get("w_in_gain", 1.0)), 4)
        slot = by.setdefault(gain, {CONN: {"best": [], "test": [], "curves": []},
                                    CTRL: {"best": [], "test": [], "curves": []}})[cond]
        slot["best"].append(float(r["best_val_r2"]))
        slot["test"].append(float(r["test_r2"]))
        ys = []
        mcsv = d / "metrics_epochs.csv"
        if mcsv.exists():
            with open(mcsv) as f:
                for row in csv.DictReader(f):
                    ys.append(float(row["val_mean_r2"]))
        if ys:
            slot["curves"].append(np.array(ys))
    return dict(sorted(by.items()))


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


def fig_summary(by: dict, ceil: float | None):
    gains = list(by.keys())
    fig, ax = plt.subplots(figsize=(1.9 * len(gains) + 2.2, 4.4))
    style_ax(ax)
    if ceil is not None:
        ax.axhline(ceil, color=GRU_COLOR, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.text(len(gains) - 0.5, ceil + 0.01, f"causal GRU ceiling {ceil:.2f}",
                ha="right", va="bottom", color=GRU_COLOR, fontsize=7.5, fontweight="bold")
    ax.text(0.0, 0.012, "predict-per-episode-mean floor", ha="left", va="bottom", color=MUTED, fontsize=7.0)

    rng = np.random.default_rng(0)
    for i, g in enumerate(gains):
        for dx, cond, col, lab in ((-0.17, CONN, CONN_COLOR, "connectome"),
                                   (0.17, CTRL, CTRL_COLOR, "control")):
            best = np.array(by[g][cond]["best"], dtype=float)
            if not len(best):
                continue
            jit = (rng.random(len(best)) - 0.5) * 0.16
            ax.scatter(i + dx + jit, best, s=24, color=col, alpha=0.85, edgecolor="white",
                       linewidth=0.5, zorder=4)
            med = float(np.median(best))
            ax.plot([i + dx - 0.13, i + dx + 0.13], [med, med], color=col, linewidth=2.6, zorder=5)
            ax.text(i + dx, med, f" {med:.3f}", ha="center", va="bottom" if cond == CONN else "top",
                    color=col, fontsize=7.0, fontweight="bold")
    # direct labels for the two entities, placed once at the top-left group
    ax.text(-0.17, ax.get_ylim()[1], "connectome", ha="center", va="bottom", color=CONN_COLOR,
            fontsize=8, fontweight="bold")
    ax.text(0.17, ax.get_ylim()[1], "control", ha="center", va="bottom", color=CTRL_COLOR,
            fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(gains)))
    ax.set_xticklabels([f"W_in × {g:g}" for g in gains], fontsize=9)
    ax.set_xlabel("input-drive gain  W_in  (normalization OFF; control activation-RMS-matched to connectome)")
    ax.set_ylabel("best held-out yaw-rate  $R^2$  (per seed / per control graph)")
    ax.set_xlim(-0.6, len(gains) - 0.4)
    fig.suptitle("vis-01 subrun 07 · does the connectome beat a degree-matched control? (norm OFF, 750 ep)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_control_summary.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def fig_curves(by: dict, ceil: float | None):
    gains = list(by.keys())
    n = len(gains)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.7), sharey=True, squeeze=False)
    axes = axes[0]
    yhi = max(0.9, (ceil or 0) + 0.1)
    for i, (ax, g) in enumerate(zip(axes, gains)):
        style_ax(ax)
        if ceil is not None:
            ax.axhline(ceil, color=GRU_COLOR, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.7, zorder=2)
        # control band (min-max across control graphs) + median
        for cond, col, z in ((CTRL, CTRL_COLOR, 3), (CONN, CONN_COLOR, 5)):
            curves = by[g][cond]["curves"]
            if not curves:
                continue
            L = min(len(c) for c in curves)
            stk = np.stack([c[:L] for c in curves])
            x = np.arange(1, L + 1)
            if cond == CTRL:
                ax.fill_between(x, stk.min(0), stk.max(0), color=col, alpha=0.30, linewidth=0, zorder=z)
            med = np.median(stk, axis=0)
            ax.plot(x, med, color=col, linewidth=2.2 if cond == CONN else 1.6, zorder=z + 1)
            best = max(by[g][cond]["best"])
            ax.text(0.97, 0.03 if cond == CONN else 0.10, f"{'conn' if cond==CONN else 'ctrl'} best {best:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom", color=col, fontsize=8,
                    fontweight="bold")
        ax.set_title(f"W_in × {g:g}", fontsize=9.5, color=INK, loc="left", pad=6)
        ax.set_xlabel("epoch")
        ax.set_ylim(-0.6, yhi)
    axes[0].set_ylabel("held-out yaw-rate  $R^2$")
    axes[0].text(0.03, 0.97, "connectome (bold) vs control band", transform=axes[0].transAxes,
                 ha="left", va="top", color=INK2, fontsize=7.5)
    fig.suptitle("Per-gain training curves — connectome median (bold) over the degree-matched control band",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_control_curves.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main(argv):
    out_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT
    if not (out_dir / "runs").exists():
        print(f"[make_control_compare_figures] no runs/ under {out_dir} yet -- run `run.py --collect` first.")
        return 0
    by = load_runs(out_dir)
    if not by:
        print(f"[make_control_compare_figures] no result.json found under {out_dir}/runs -- nothing to plot.")
        return 0
    ceil = causal_ceiling(out_dir)
    print("gains:", {g: {c: len(by[g][c]["best"]) for c in (CONN, CTRL)} for g in by},
          "causal ceiling:", ceil)
    fig_summary(by, ceil)
    fig_curves(by, ceil)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
