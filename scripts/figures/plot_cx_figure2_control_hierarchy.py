#!/usr/bin/env python3
"""Figure 2: CX path-integration control hierarchy.

Heading-bump angular error (rad, lower = better) vs sequence length T = {50, 100, 200},
for the connectome and its matched controls (weight-shuffled, degree-shuffled, random,
no-recurrence), in the FROZEN reservoir and OBSERVED-edge trainable regimes. Mean +/- SEM
over 3 seeds. Clean: no in-plot text except axis labels; single shared legend below.

Reads docs/results/cx_structure_polar/metrics_by_seed_{frozen,observed}.csv
Writes docs/results/cx_structure_polar/figure2_cx_control_hierarchy.png
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DDIR = ROOT / "docs/results/cx_structure_polar"
METRIC = "heading_bump_angular_error"
TS = [50, 100, 200]
# model -> (display label, colour, linewidth, linestyle, zorder)
MODELS = {
    "cx_bpu":         ("connectome",      "#d62728", 2.8, "-",  5),
    "weight_shuffle": ("weight-shuffled", "#1f77b4", 1.8, "-",  3),
    "degree_shuffle": ("degree-shuffled", "#2ca02c", 1.8, "-",  3),
    "random":         ("random",          "#ff7f0e", 1.8, "-",  3),
    "no_recurrence":  ("no recurrence",   "#8c8c8c", 1.8, "--", 2),
}


def load(regime):
    """(model, T) -> (mean, sem) of the metric over seeds, noiseless test split."""
    rows = list(csv.DictReader(open(DDIR / f"metrics_by_seed_{regime}.csv")))
    vals = defaultdict(list)
    for r in rows:
        if r["split"] != "test" or float(r["noise_std"]) != 0.0:
            continue
        vals[(r["model"], int(r["T"]))].append(float(r[METRIC]))
    out = {}
    for key, xs in vals.items():
        a = np.asarray(xs, float)
        out[key] = (a.mean(), a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0)
    return out


def main():
    data = {reg: load(reg) for reg in ("frozen", "observed")}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, (reg, xlab) in zip(axes, [("frozen", "sequence length  T   (frozen reservoir)"),
                                      ("observed", "sequence length  T   (observed · trainable)")]):
        d = data[reg]
        for model, (label, col, lw, ls, z) in MODELS.items():
            m = np.array([d[(model, t)][0] for t in TS])
            e = np.array([d[(model, t)][1] for t in TS])
            ax.errorbar(TS, m, yerr=e, color=col, lw=lw, ls=ls, marker="o",
                        ms=6.5 if model == "cx_bpu" else 5, capsize=3, capthick=lw,
                        elinewidth=lw, zorder=z, label=label,
                        mec="white", mew=0.8)
        ax.set_xticks(TS)
        ax.set_xlabel(xlab, fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("heading error  (rad)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = DDIR / "figure2_cx_control_hierarchy.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # console check: confirm connectome is best at every point
    print(f"{'regime':<9}{'T':>5}  " + "  ".join(f"{m:>15}" for m in MODELS))
    for reg in ("frozen", "observed"):
        for t in TS:
            row = "  ".join(f"{data[reg][(m, t)][0]:>15.3f}" for m in MODELS)
            best = min(MODELS, key=lambda m: data[reg][(m, t)][0])
            print(f"{reg:<9}{t:>5}  {row}   best={best}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
