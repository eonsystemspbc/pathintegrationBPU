#!/usr/bin/env python3
"""Input ports emerge during training.

Left  : task learning curve (reversal-probe accuracy) vs epoch.
Right : ROC-AUC that the input weight ||W_in[i]|| lands on the biological input cells
        (hemibrain projection neurons = the odor input pathway) vs epoch.
Both panels: connectome vs degree-matched random wiring, mean +/- 95% CI over seeds.
The input-port AUC rises from chance only for the connectome, and it rises as the task
is learned. Clean: no in-plot text except axis labels; one shared legend below.

Reads outputs/runs/mb_biology_assoc_20seed/hemibrain_{connectome,random}_s*.npz
Writes docs/results/mb_biology_convergence/input_ports_emerge.png
"""
from __future__ import annotations
import glob, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "outputs/runs/mb_biology_assoc_20seed"
OUT = ROOT / "docs/results/mb_biology_convergence/input_ports_emerge.png"
COND = {  # stem prefix -> (display label, colour)
    "hemibrain_connectome": ("connectome", "#d62728"),
    "hemibrain_random":     ("matched random", "#8c8c8c"),
}


def band(curves):
    a = np.stack(curves); m = a.mean(0)
    ci = 1.96 * a.std(0, ddof=1) / np.sqrt(a.shape[0]) if a.shape[0] > 1 else np.zeros_like(m)
    return m, ci


def collect():
    acc, auc, eps = defaultdict(list), defaultdict(list), None
    for cond in COND:
        for f in sorted(glob.glob(str(RUN_DIR / f"{cond}_s*.npz"))):
            d = np.load(f, allow_pickle=True)
            eps = d["snapshot_epochs"]
            acc[cond].append(np.asarray(d["reversal_acc"], float))
            ty = d["coarse_type"].astype(str)
            bio = (ty == "PN")                                   # biological input cells (odor pathway)
            win = np.asarray(d["win_snapshots"])                 # [n_snap, N, input_dim]
            nrm = np.linalg.norm(win, axis=2)                    # per-neuron ||W_in||
            auc[cond].append(np.array([roc_auc_score(bio, nrm[s]) for s in range(nrm.shape[0])]))
    return eps, acc, auc


def main():
    eps, acc, auc = collect()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
    for cond, (label, col) in COND.items():
        ma, ca = band(acc[cond]); mu, cu = band(auc[cond])
        axL.plot(eps, ma, color=col, lw=2.6, label=label)
        axL.fill_between(eps, ma - ca, ma + ca, color=col, alpha=0.18, lw=0)
        axR.plot(eps, mu, color=col, lw=2.6, label=label)
        axR.fill_between(eps, mu - cu, mu + cu, color=col, alpha=0.18, lw=0)
    axL.set_ylabel("task accuracy  (reversal probe)", fontsize=11)
    axR.set_ylabel("ROC-AUC:  input weight → input cells", fontsize=11)
    axR.axhline(0.5, color="k", ls=":", lw=1)                    # chance
    for ax in (axL, axR):
        ax.set_xlabel("training epoch", fontsize=11)
        ax.tick_params(labelsize=10); ax.grid(alpha=0.25, lw=0.6)
        ax.set_xlim(eps[0], eps[-1])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles, labels = axL.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT, dpi=200, bbox_inches="tight"); plt.close(fig)

    print(f"{'condition':<18}{'acc i->f':>16}{'input-AUC i->f':>18}")
    for cond, (label, _) in COND.items():
        ma, _ = band(acc[cond]); mu, _ = band(auc[cond])
        print(f"{label:<18}{ma[0]:>7.3f}->{ma[-1]:.3f}{mu[0]:>10.3f}->{mu[-1]:.3f}  (n={len(acc[cond])})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
