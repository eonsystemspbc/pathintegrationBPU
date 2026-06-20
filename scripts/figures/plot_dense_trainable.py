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

    x = np.arange(len(order))
    dense = [best[m] for m in order]
    frozen = [FROZEN_REF.get(m, np.nan) for m in order]
    rnd_dense = best.get("random", np.nan)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})

    # LEFT: frozen reservoir vs dense-trainable (training drops every model a lot)
    w = 0.38
    b1 = axL.bar(x - w/2, dense, w, color=[COLOR[m] for m in order], label="dense-TRAINABLE (~54M params)")
    axL.bar(x + w/2, frozen, w, color=[COLOR[m] for m in order], alpha=0.40, hatch="//",
            label="frozen reservoir (22,943 params)")
    axL.set_xticks(x); axL.set_xticklabels([NICE[m] for m in order], fontsize=9)
    axL.set_ylabel("best val-MSE (lower = better)")
    axL.set_title("frozen reservoir vs dense-trainable\n(training drops every init ~5–7×)", fontsize=10)
    axL.legend(fontsize=8, loc="upper left")
    axL.set_ylim(0, max([v for v in dense + frozen if np.isfinite(v)]) * 1.15)

    # RIGHT: zoomed dense-trainable only — where the v2 story lives (eigvec < spectrum < connectome ≈ random)
    bars = axR.bar(x, dense, 0.6, color=[COLOR[m] for m in order])
    axR.axhline(rnd_dense, color="#d62728", ls="--", lw=1.2, label=f"random = {rnd_dense:.3f}")
    for m, b in zip(order, bars):
        lr, K = hp[m]
        d = (rnd_dense - b.get_height()) / rnd_dense * 100 if np.isfinite(rnd_dense) else 0
        tag = "" if m == "random" else f"\n({d:+.0f}% vs rnd)"
        axR.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0006,
                 f"{b.get_height():.4f}\nlr={lr:.0e},K={K}{tag}", ha="center", va="bottom", fontsize=8)
    lo = min(dense) * 0.92
    axR.set_ylim(lo, max(dense) * 1.10)
    axR.set_xticks(x); axR.set_xticklabels([NICE[m] for m in order], fontsize=9)
    axR.set_ylabel("best val-MSE (zoomed)")
    axR.set_title("dense-trainable only (zoom): only eigvec-matched\nclears random; the raw connectome ties it",
                  fontsize=10)
    axR.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("CX → path integration: which INITIALIZATION survives full training? "
                 "(all dense + fully trainable, params & density matched — only the init differs)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "dense_trainable_bars.png", dpi=150); plt.close(fig)
    print("best (dense-trainable):", {NICE[m]: round(best[m], 4) for m in order})
    print("wrote", OUT / "dense_trainable_bars.png")


if __name__ == "__main__":
    raise SystemExit(main())
