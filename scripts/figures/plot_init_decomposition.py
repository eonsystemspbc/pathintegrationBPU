#!/usr/bin/env python3
"""Figure: decomposing the dense-trainable CX init advantage into density / eigenvectors / eigenvalues.

The `dense_random` control (random values, DENSE init, same 54M params, rho-matched) is the key that
separates "dense-init effect" from "structure". At the stable operating point (lr=3e-4, K=2),
mean over 3 seeds, val-MSE lower=better:

  sparse inits:  random ~0.073, connectome ~0.073           (tie -> raw wiring adds nothing)
  dense inits:   dense_random ~0.059  <- just density, no structure
                 eigvec_matched ~0.057 (dense + connectome eigenVECTORS)
                 spectrum_full  ~0.067 (dense + connectome eigenVALUES)

=> density is the DOMINANT effect (+~19% sparse->dense_random); eigenVECTORS add a small extra
(+~5% dense_random->eigvec); eigenVALUES add NOTHING (spectrum is WORSE than dense_random).
Reads cx_dense_trainable_v2 + cx_dense_random results. Read-only.
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
ORDER = ["random", "connectome_bpu", "dense_random", "spectrum_full", "eigvec_matched"]
NICE = {"random": "random\n(sparse)", "connectome_bpu": "connectome\n(sparse)",
        "dense_random": "dense-random\n(dense, no structure)", "spectrum_full": "spectrum-full\n(dense + eigenVALUES)",
        "eigvec_matched": "eigvec-matched\n(dense + eigenVECTORS)"}
COLOR = {"random": "#9e9e9e", "connectome_bpu": "#5a7fb0", "dense_random": "#444444",
         "spectrum_full": "#9467bd", "eigvec_matched": "#2ca02c"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", default=[
        "outputs/runs/hp_sweep/cx_dense_trainable_v2/results_shard*.csv",
        "outputs/runs/hp_sweep/cx_dense_random/results_shard*.csv"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--K", type=int, default=2)
    a = p.parse_args()
    files = []
    for g in a.results:
        files += glob.glob(g)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[np.isfinite(df.best_val_loss)]
    cell = df[(df.lr == a.lr) & (df.K == a.K)].groupby("model").best_val_loss.agg(["mean", "std"])
    order = [m for m in ORDER if m in cell.index]
    means = [cell.loc[m, "mean"] for m in order]
    stds = [cell.loc[m, "std"] for m in order]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(order))
    bars = ax.bar(x, means, 0.62, yerr=stds, capsize=4, color=[COLOR[m] for m in order],
                  edgecolor="black", linewidth=0.6)
    spar = cell.loc["random", "mean"] if "random" in cell.index else np.nan
    den = cell.loc["dense_random", "mean"] if "dense_random" in cell.index else np.nan
    ax.axhline(spar, color="#9e9e9e", ls=":", lw=1.2)
    ax.axhline(den, color="#444444", ls="--", lw=1.2)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(stds)*0.4 + 0.0009,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    # effect annotations
    def eff(a_, b_):  # % improvement going a_->b_ (both model keys)
        va, vb = cell.loc[a_, "mean"], cell.loc[b_, "mean"]
        return (va - vb) / va * 100
    if {"random", "dense_random"} <= set(cell.index):
        ax.annotate(f"DENSITY\n{eff('random','dense_random'):+.0f}%  (dominant)",
                    xy=(2, den), xytext=(1.0, (spar+den)/2 - 0.001),
                    fontsize=10, fontweight="bold", color="#b00000", ha="center")
        ax.annotate("", xy=(2, den), xytext=(1, spar),
                    arrowprops=dict(arrowstyle="->", color="#b00000", lw=1.6))
    if {"dense_random", "eigvec_matched"} <= set(cell.index):
        ax.text(4, cell.loc["eigvec_matched","mean"] - 0.004,
                f"eigenVECTORS\n{eff('dense_random','eigvec_matched'):+.0f}%", ha="center", fontsize=9,
                fontweight="bold", color="#2ca02c")
    if {"dense_random", "spectrum_full"} <= set(cell.index):
        ax.text(3, cell.loc["spectrum_full","mean"] + max(stds)+0.0016,
                f"eigenVALUES\n{eff('dense_random','spectrum_full'):+.0f}% (worse)", ha="center", fontsize=9,
                fontweight="bold", color="#9467bd")
    ax.text(0.5, spar + 0.0011, "sparse-init baseline", color="#7f7f7f", fontsize=8.5, ha="center")
    ax.text(3.0, den - 0.0016, "dense-init baseline (no structure)", color="#444444", fontsize=8.5, ha="center")
    ax.set_xticks(x); ax.set_xticklabels([NICE[m] for m in order], fontsize=8.5)
    ax.set_ylabel("best val-MSE (lower = better)")
    ax.set_ylim(min(means) - 0.006, max(means) + max(stds) + 0.004)
    ax.set_title(f"CX dense-trainable: what actually helps as an initialization?  (lr={a.lr:.0e}, K={a.K}, 3 seeds)\n"
                 "density dominates (+19%); eigenVECTORS add a little (+5%); eigenVALUES add nothing; raw wiring ties random",
                 fontsize=10.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT / "init_decomposition.png", dpi=150); plt.close(fig)
    print("decomposition @", f"lr={a.lr},K={a.K}:", {m: round(cell.loc[m,'mean'],4) for m in order})
    print("wrote", OUT / "init_decomposition.png")


if __name__ == "__main__":
    raise SystemExit(main())
