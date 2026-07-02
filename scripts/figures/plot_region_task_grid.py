#!/usr/bin/env python3
"""Region×task grid: does the connectome beat matched controls MORE on its native task?

Reads a dir of {region}_{task}_{model}_s{seed}.npz (from run_pool_gated_grid.py). For each
(region, task) cell computes the connectome's advantage over the mean of the matched controls
(degree_preserving / weight_shuffle / random_sparse), the paired significance, and renders two
heatmaps: (A) connectome absolute score, (B) connectome-minus-control advantage (the grid story).
Native diagonal = OL/flow, MB/mqar, CX/path.
"""
from __future__ import annotations
import glob, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = sys.argv[1] if len(sys.argv) > 1 else "outputs/runs/region_task_grid"
OUT = Path("docs/results/region_task_grid"); OUT.mkdir(parents=True, exist_ok=True)
REGIONS = ["OL", "MB", "CX"]; TASKS = ["flow", "mqar", "path", "seq_mnist", "mod_sum"]
NATIVE = {("OL", "flow"), ("MB", "mqar"), ("CX", "path")}
CONTROLS = ["degree_preserving", "weight_shuffle", "random_sparse"]


def load():
    by = defaultdict(lambda: defaultdict(dict))   # (region,task) -> model -> seed -> score
    for f in sorted(glob.glob(f"{RUNS}/**/*.npz", recursive=True)):
        m = re.match(r"(OL|MB|CX)_(\w+?)_(connectome|degree_preserving|weight_shuffle|random_sparse)_s(\d+)$", Path(f).stem)
        if not m: continue
        d = np.load(f, allow_pickle=True)
        by[(m.group(1), m.group(2))][m.group(3)][int(m.group(4))] = float(d["best_metric"])
    return by


def main():
    by = load()
    if not by:
        print(f"no runs in {RUNS}"); return 1
    absc = np.full((3, 5), np.nan); adv = np.full((3, 5), np.nan); zsc = np.full((3, 5), np.nan)
    pv = np.full((3, 5), np.nan); nseed = np.zeros((3, 5), int)
    # CHANCE per task (for the ceiling-normalized advantage): accuracy tasks have a floor; R2 tasks floor at 0
    CHANCE = {"flow": 0.0, "path": 0.0, "mqar": 1.0 / 8, "seq_mnist": 0.10, "mod_sum": 1.0 / 7}
    print(f"{'cell':<14}{'conn':>8}{'ctrl':>8}{'rawΔ':>8}{'z':>7}{'Δ/range':>9}{'p':>10}{'n':>4}")
    for i, r in enumerate(REGIONS):
        for j, t in enumerate(TASKS):
            cell = by.get((r, t));
            if not cell or "connectome" not in cell: continue
            cs = cell["connectome"]; seeds = sorted(cs)
            conn = np.array([cs[s] for s in seeds])
            # per-seed mean-over-controls (paired) AND the pooled control distribution (for the z-score / empirical effect size)
            ctrl_per_seed, ctrl_pool = [], []
            for s in seeds:
                vals = [cell[c][s] for c in CONTROLS if c in cell and s in cell[c]]
                if vals: ctrl_per_seed.append(np.mean(vals)); ctrl_pool.extend(vals)
            ctrl = np.array(ctrl_per_seed); pool = np.array(ctrl_pool)
            n = min(len(conn), len(ctrl))
            if n < 2 or pool.size < 2: continue
            absc[i, j] = conn.mean(); adv[i, j] = conn[:n].mean() - ctrl[:n].mean()
            pv[i, j] = stats.ttest_rel(conn[:n], ctrl[:n]).pvalue; nseed[i, j] = n
            # ceiling-robust: (a) z = connectome's position in the control distribution (scale-free, THE cross-cell metric)
            zsc[i, j] = (conn.mean() - pool.mean()) / (pool.std(ddof=1) + 1e-9)
            # (b) advantage as a fraction of the achievable range above chance (0/1-bounded metrics)
            rng = max(conn.mean(), ctrl.mean()) - CHANCE.get(t, 0.0)
            frac = adv[i, j] / rng if rng > 0.02 else np.nan
            star = "*" if pv[i, j] < 0.05 else ""
            tag = " [NATIVE]" if (r, t) in NATIVE else ""
            print(f"{r+'x'+t:<14}{conn.mean():>8.3f}{ctrl.mean():>8.3f}{adv[i,j]:>+8.3f}{zsc[i,j]:>+7.2f}"
                  f"{(frac if not np.isnan(frac) else float('nan')):>+9.2f}{pv[i,j]:>10.1e}{n:>4}{star}{tag}")

    # figure: two heatmaps -- absolute score + the ceiling-robust z-score advantage (primary)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2))
    for ax, M, title, cmap in [(a1, absc, "(A) connectome score (best metric)", "viridis"),
                               (a2, zsc, "(B) connectome advantage = z in control distribution (ceiling-robust)", "RdBu_r")]:
        vmax = np.nanmax(np.abs(M)) if ax is a2 else np.nanmax(M)
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=(-vmax if ax is a2 else np.nanmin(M)), vmax=vmax)
        ax.set_xticks(range(5)); ax.set_xticklabels(TASKS, rotation=30, ha="right")
        ax.set_yticks(range(3)); ax.set_yticklabels(REGIONS)
        for i in range(3):
            for j in range(5):
                if np.isnan(M[i, j]): continue
                s = "*" if (ax is a2 and pv[i, j] < 0.05) else ""
                box = dict(boxstyle="round,pad=0.1", fc="none", ec="lime", lw=2) if (REGIONS[i], TASKS[j]) in NATIVE else None
                ax.text(j, i, f"{M[i,j]:.2f}{s}", ha="center", va="center", fontsize=9,
                        color="white" if ax is a1 else "black", bbox=box)
        ax.set_title(title); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Region × task grid — biological input gating, trainable recurrent (green box = native)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "region_task_grid.png", dpi=150); plt.close(fig)
    print(f"\nwrote {OUT/'region_task_grid.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
