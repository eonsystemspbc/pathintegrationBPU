#!/usr/bin/env python3
"""Training-curve figure for the dense-trainable CX comparison.

Plots val-MSE vs epoch for each of the four models (connectome / eigvec-matched / spectrum-full /
random), each at ITS OWN best (lr, K) cell, mean over seeds with a min-max band. Shows how the four
initializations converge under full training. Reads the per-epoch curves CSV written by
run_hp_spectrum_sweep.py (curves_shard*.csv). Read-only.
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NICE = {"connectome_bpu": "connectome", "eigvec_matched": "eigvec-matched",
        "spectrum_full": "spectrum-full", "random": "random"}
COLOR = {"connectome_bpu": "#1f77b4", "eigvec_matched": "#2ca02c",
         "spectrum_full": "#9467bd", "random": "#7f7f7f"}
ORDER = ["connectome_bpu", "eigvec_matched", "spectrum_full", "random"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--curves", nargs="+",
                   default=["outputs/runs/hp_sweep/cx_dense_trainable_v2/curves_shard*.csv"])
    p.add_argument("--out", default="docs/results/cx_dense_trainable/training_curves.png")
    p.add_argument("--title-suffix", default="")
    a = p.parse_args()
    files = []
    for g in a.curves:
        files += glob.glob(g)
    if not files:
        print("no curves CSVs:", a.curves); return 1
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.val_mse.notna() & np.isfinite(df.val_mse)]

    # best (lr,K) per model = the cell with the lowest seed-mean final best_val_mse
    finals = (df.sort_values("epoch").groupby(["model", "lr", "K", "seed"])
              .best_val_mse.last().reset_index())
    best_hp = {}
    for m, sub in finals.groupby("model"):
        cell = sub.groupby(["lr", "K"]).best_val_mse.mean().idxmin()
        best_hp[m] = cell  # (lr, K)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    order = [m for m in ORDER if m in best_hp]
    for m in order:
        lr, K = best_hp[m]
        sub = df[(df.model == m) & (df.lr == lr) & (df.K == K)]
        piv = sub.pivot_table(index="epoch", columns="seed", values="val_mse")
        ep = piv.index.values
        mean = piv.mean(axis=1).values
        lo, hi = piv.min(axis=1).values, piv.max(axis=1).values
        # legend shows the reported metric = seed-mean of the per-epoch BEST (min) val-MSE
        best = sub.pivot_table(index="epoch", columns="seed", values="best_val_mse").iloc[-1].mean()
        ax.plot(ep, mean, color=COLOR[m], lw=2,
                label=f"{NICE[m]}  (lr={lr:.0e}, K={int(K)}) → best {best:.3f}")
        ax.fill_between(ep, lo, hi, color=COLOR[m], alpha=0.15)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation MSE (lower = better)")
    ax.set_title("CX dense-trainable: convergence by initialization" + a.title_suffix +
                 "\n(each model at its own best lr,K; band = seed min–max)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    yld = df.groupby("model").apply(lambda d: d[(d.lr==best_hp[d.name][0])&(d.K==best_hp[d.name][1])].val_mse.min())
    ax.set_ylim(max(0, float(yld.min()) - 0.03), None)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("best HP per model:", {NICE[m]: (f"{best_hp[m][0]:.0e}", int(best_hp[m][1])) for m in order})
    print("wrote", out)


if __name__ == "__main__":
    raise SystemExit(main())
