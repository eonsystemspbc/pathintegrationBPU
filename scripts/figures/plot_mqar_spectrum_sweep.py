#!/usr/bin/env python3
"""Figure: MQAR + MB-connectome control hierarchy (mirror of the CX path-integration figure).

Same spectrum/HP control set as the CX sweep, applied to the MB's region-matched task (associative
recall). Bar chart of best test_acc per model (each at its own best LR), with chance and the random
control marked. Writes docs/results/mqar_mb_spectrum/mqar_control_hierarchy.png. Read-only on CSVs.
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
OUT = ROOT / "docs" / "results" / "mqar_mb_spectrum"
NICE = {"hemibrain_seeded": "MB connectome", "weight_shuffle": "weight-shuffle\n(topology kept)",
        "degree_preserving_random": "degree-matched\nrandom", "random_sparse": "random",
        "spectrum_full": "spectrum-full\n(eigenVALUES)", "spectrum_topk": "spectrum-topk"}
COLOR = {"hemibrain_seeded": "#1f77b4", "weight_shuffle": "#17becf",
         "degree_preserving_random": "#7f7f7f", "random_sparse": "#7f7f7f",
         "spectrum_full": "#9467bd", "spectrum_topk": "#9467bd"}
ORDER = ["hemibrain_seeded", "weight_shuffle", "degree_preserving_random", "spectrum_topk",
         "spectrum_full", "random_sparse"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+",
                   default=["outputs/runs/hp_sweep/mb_mqar/results_shard*.csv"])
    a = p.parse_args()
    files = []
    for g in a.results:
        files += glob.glob(g)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.test_acc.notna()]
    chance = float(df.chance.iloc[0])
    cell = df.groupby(["model", "lr"]).test_acc.mean().reset_index()
    best = {m: float(sub.test_acc.max()) for m, sub in cell.groupby("model")}
    order = [m for m in ORDER if m in best]
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    vals = [best[m] for m in order]
    bars = ax.bar([NICE.get(m, m) for m in order], vals, color=[COLOR[m] for m in order])
    rnd = best.get("random_sparse", np.nan)
    ax.axhline(rnd, color="#d62728", ls="--", lw=1, label=f"random = {rnd:.3f}")
    ax.axhline(chance, color="k", ls=":", lw=1, label=f"chance = {chance:.3f}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("best test accuracy (higher = better)")
    ax.set_title("MQAR (associative recall) → MB connectome: control hierarchy\n"
                 "each model at its own best LR; MB is the region matched to associative memory")
    ax.set_ylim(0, max(vals) * 1.15); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "mqar_control_hierarchy.png", dpi=150); plt.close(fig)
    print("best test_acc:", {NICE.get(m, m).replace(chr(10), ' '): round(best[m], 4) for m in order})
    print("wrote", OUT / "mqar_control_hierarchy.png")


if __name__ == "__main__":
    raise SystemExit(main())
