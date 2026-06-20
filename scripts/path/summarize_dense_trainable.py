#!/usr/bin/env python3
"""Summarize the dense-TRAINABLE, parameter+density-matched CX comparison.

Every model here is a dense N x N recurrent matrix whose entries are ALL trainable (~54M params),
so connectome / eigvec_matched / spectrum_full / random differ ONLY in their initialization --
the sparse-vs-dense and trainable-parameter-count confounds are both removed. The question:

  with density and trainable-parameter count matched, does the connectome INITIALIZATION still
  buy anything over a random init (or over the eigenvalue/eigenvector-matched inits)?

Reads outputs/runs/hp_sweep/cx_dense_trainable/results_shard*.csv. Each model is scored at its OWN
best (lr, K) cell, val-MSE averaged over seeds (lower = better). Read-only.
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd

METRIC = "best_val_loss"
NICE = {"connectome_bpu": "connectome (init)", "eigvec_matched": "eigvec-matched (init)",
        "spectrum_full": "spectrum-full (init)", "random": "random (init)"}
INIT = {"connectome_bpu": "the real CX wiring", "eigvec_matched": "connectome eigenVECTORS, random eigenvalues",
        "spectrum_full": "connectome eigenVALUES, random eigenvectors", "random": "degree-matched random"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+",
                   default=["outputs/runs/hp_sweep/cx_dense_trainable/results_shard*.csv"])
    a = p.parse_args()
    files = []
    for g in a.results:
        files += glob.glob(g)
    if not files:
        print("no results yet:", a.results); return 1
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df[METRIC].notna() & np.isfinite(df[METRIC])]
    # mean over seeds for each (model, lr, K) cell, then best cell per model
    cell = df.groupby(["model", "lr", "K"])[METRIC].mean().reset_index()
    rows = []
    for m, sub in cell.groupby("model"):
        b = sub.loc[sub[METRIC].idxmin()]
        rows.append((m, float(b[METRIC]), float(b["lr"]), int(b["K"]),
                     int(df[df.model == m].seed.nunique())))
    rows.sort(key=lambda r: r[1])
    rnd = next((v for (m, v, *_ ) in rows if m == "random"), float("nan"))
    print(f"\n=== CX dense-TRAINABLE (param+density matched, ~54M trainable each) — {len(df)} cells ===")
    print(f"{'model':<24}{'best val-MSE':>13}{'vs random':>11}   best (lr,K)   init")
    for m, v, lr, K, ns in rows:
        delta = "" if not np.isfinite(rnd) or rnd == 0 else f"{100*(rnd-v)/rnd:+.1f}%"
        star = "  <- random" if m == "random" else ""
        print(f"{NICE.get(m,m):<24}{v:>13.4f}{delta:>11}   lr={lr:.0e} K={K}   {INIT.get(m,'')}{star}")
    print(f"\n(seeds per model: {rows[0][4]}; lower val-MSE = better; frozen connectome ref = 0.390)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
