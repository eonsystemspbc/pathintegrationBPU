#!/usr/bin/env python3
"""Figure: the dense-TRAINABLE, parameter+density-matched CX comparison.

The frozen result (docs/results/cx_eigval_vs_eigvec) showed the connectome's path-integration edge
lives in its eigenVECTORS -- but that was a FROZEN reservoir (only input/readout train). This asks
the harder question: make every recurrent matrix dense AND fully trainable (~54M params each), so
connectome / eigvec-matched / spectrum-full / random differ ONLY in initialization. Does the
connectome initialization still buy anything once the whole matrix can move?

Produces docs/results/cx_dense_trainable/dense_trainable_bars.png. Read-only on the sweep CSVs.
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "results" / "cx_dense_trainable"
METRIC = "best_val_loss"
NICE = {"connectome_bpu": "connectome", "eigvec_matched": "eigvec-matched",
        "spectrum_full": "spectrum-full", "random": "random"}
COLOR = {"connectome_bpu": "#1f77b4", "eigvec_matched": "#2ca02c",
         "spectrum_full": "#9467bd", "random": "#7f7f7f"}
# frozen-reservoir reference values (docs/results/cx_eigval_vs_eigvec)
FROZEN_REF = {"connectome_bpu": 0.390, "eigvec_matched": 0.229, "spectrum_full": 0.456, "random": 0.410}


def best_per_model(globs):
    files = []
    for g in globs:
        files += glob.glob(g)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df[METRIC].notna() & np.isfinite(df[METRIC])]
    cell = df.groupby(["model", "lr", "K"])[METRIC].mean().reset_index()
    best, hp = {}, {}
    for m, sub in cell.groupby("model"):
        r = sub.loc[sub[METRIC].idxmin()]
        best[m] = float(r[METRIC]); hp[m] = (float(r["lr"]), int(r["K"]))
    return best, hp, df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+",
                   default=["outputs/runs/hp_sweep/cx_dense_trainable/results_shard*.csv"])
    a = p.parse_args()
    best, hp, df = best_per_model(a.results)
    order = [m for m in ["connectome_bpu", "eigvec_matched", "spectrum_full", "random"] if m in best]
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    x = np.arange(len(order))
    w = 0.38
    dense = [best[m] for m in order]
    frozen = [FROZEN_REF.get(m, np.nan) for m in order]
    b1 = ax.bar(x - w/2, dense, w, color=[COLOR[m] for m in order], label="dense-TRAINABLE (~54M params, matched)")
    b2 = ax.bar(x + w/2, frozen, w, color=[COLOR[m] for m in order], alpha=0.40,
                hatch="//", label="frozen reservoir (22,943 params)")
    rnd_dense = best.get("random", np.nan)
    ax.axhline(rnd_dense, color="#d62728", ls="--", lw=1, label=f"random (dense-trainable) = {rnd_dense:.3f}")
    for m, b in zip(order, b1):
        lr, K = hp[m]
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.004,
                f"{b.get_height():.3f}\nlr={lr:.0e},K={K}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([NICE[m] for m in order])
    ax.set_ylabel("best val-MSE (lower = better)")
    ax.set_title("CX → path integration: does the connectome INITIALIZATION survive full training?\n"
                 "all recurrent matrices dense + fully trainable, params & density matched — only the init differs",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, max([v for v in dense + frozen if np.isfinite(v)]) * 1.18)
    fig.tight_layout(); fig.savefig(OUT / "dense_trainable_bars.png", dpi=150); plt.close(fig)
    print("best (dense-trainable):", {NICE[m]: round(best[m], 4) for m in order})
    print("wrote", OUT / "dense_trainable_bars.png")


if __name__ == "__main__":
    raise SystemExit(main())
