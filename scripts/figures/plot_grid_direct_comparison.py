#!/usr/bin/env python3
"""Direct head-to-head metric comparison (connectome vs each matched control) for the
NON-mqar grid tasks, where task ceilings are not near 1.0 so raw metric is fair to read.

flow / path are R² (regression); seq_mnist / mod_sum are accuracy (classification).
Prints mean±std across seeds for all 4 wirings per cell and renders a grouped-bar figure.
"""
from __future__ import annotations
import glob, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "outputs/runs/region_task_grid"
OUT = Path("docs/results/region_task_grid"); OUT.mkdir(parents=True, exist_ok=True)
REGIONS = ["OL", "MB", "CX"]
TASKS = ["flow", "path", "seq_mnist", "mod_sum"]            # non-mqar
METRIC = {"flow": "R²", "path": "R²", "seq_mnist": "acc", "mod_sum": "acc"}
CHANCE = {"flow": 0.0, "path": 0.0, "seq_mnist": 0.10, "mod_sum": 1/7}
MODELS = ["connectome", "degree_preserving", "weight_shuffle", "random_sparse"]
LABEL = {"connectome": "connectome", "degree_preserving": "degree-pres",
         "weight_shuffle": "wt-shuffle", "random_sparse": "rand-sparse"}
COL = {"connectome": "#c0392b", "degree_preserving": "#2980b9",
       "weight_shuffle": "#27ae60", "random_sparse": "#8e44ad"}
NATIVE = {("OL", "flow"), ("CX", "path")}                  # non-mqar natives


def load():
    by = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(f"{RUNS}/*.npz"):
        m = re.match(r"(OL|MB|CX)_(\w+?)_(connectome|degree_preserving|weight_shuffle|random_sparse)_s(\d+)$", Path(f).stem)
        if m:
            by[(m[1], m[2])][m[3]][int(m[4])] = float(np.load(f, allow_pickle=True)["best_metric"])
    return by


def main():
    by = load()
    print(f"{'cell':<14}{'metric':>7} | " + "".join(f"{LABEL[m]:>14}" for m in MODELS) + f"{'Δ(conn-ctrl)':>14}{'p':>9}")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8)); axes = axes.ravel()
    for ax, t in zip(axes, TASKS):
        x = np.arange(len(REGIONS)); w = 0.2
        for k, mdl in enumerate(MODELS):
            means, stds = [], []
            for r in REGIONS:
                v = np.array(sorted(by[(r, t)][mdl].values())) if mdl in by[(r, t)] else np.array([])
                means.append(v.mean() if v.size else np.nan); stds.append(v.std() if v.size else 0)
            ax.bar(x + (k - 1.5) * w, means, w, yerr=stds, capsize=2,
                   color=COL[mdl], label=LABEL[mdl], alpha=0.9, error_kw=dict(lw=0.8))
        ax.axhline(CHANCE[t], ls=":", c="gray", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(REGIONS); ax.set_title(f"{t}  ({METRIC[t]})")
        ax.set_ylabel(METRIC[t])
        for j, r in enumerate(REGIONS):
            if (r, t) in NATIVE: ax.text(j, ax.get_ylim()[1] * 0.02, "native", ha="center", fontsize=7, color="green")
        # table rows
        for r in REGIONS:
            seeds = sorted(by[(r, t)]["connectome"])                       # seed order — keep pairing aligned
            conn = np.array([by[(r, t)]["connectome"][s] for s in seeds])
            rowvals = {mdl: np.array([by[(r, t)][mdl][s] for s in sorted(by[(r, t)][mdl])]) for mdl in MODELS}
            # paired: connectome vs per-seed control mean (both in seed order)
            cmean = np.array([np.mean([by[(r, t)][c][s] for c in MODELS[1:] if c in by[(r, t)] and s in by[(r, t)][c]]) for s in seeds])
            ctrl_seedmean = cmean.mean()
            p = stats.ttest_rel(conn, cmean).pvalue
            cells = "".join(f"{rowvals[m].mean():>8.3f}±{rowvals[m].std():<4.2f}"[:14].rjust(14) for m in MODELS)
            star = "*" if p < 0.05 else " "
            print(f"{r+'x'+t:<14}{METRIC[t]:>7} | {cells}{conn.mean()-ctrl_seedmean:>+13.3f}{star}{p:>8.1e}")
    axes[0].legend(fontsize=8, ncol=2, loc="upper left")
    fig.suptitle("Direct metric comparison — connectome vs matched controls (non-MQAR tasks, mean±std over 10 seeds)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "grid_direct_nonmqar.png", dpi=150); plt.close(fig)
    print(f"\nwrote {OUT/'grid_direct_nonmqar.png'}")


if __name__ == "__main__":
    main()
