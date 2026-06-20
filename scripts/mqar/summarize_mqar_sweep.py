#!/usr/bin/env python3
"""Summarize the MQAR + MB-connectome spectrum/HP sweep (mirror of the CX summarizer).

MQAR (multi-query associative recall) is the associative-memory task; the mushroom body is the
fly's associative-learning centre, so this is the MB's region-matched task. Same control hierarchy
as the CX path-integration sweep, asking the same question for associative recall: is the MB
connectome's advantage its topology, its specific weights, or its spectrum -- and does any control
recover it? Higher test_acc = better (unlike CX val-MSE). Each model scored at its own best LR,
test_acc averaged over seeds. Reads outputs/runs/hp_sweep/mb_mqar/results_shard*.csv. Read-only.
"""
from __future__ import annotations
import argparse, glob
import numpy as np
import pandas as pd

NICE = {"hemibrain_seeded": "MB connectome", "weight_shuffle": "weight-shuffle (topology kept)",
        "degree_preserving_random": "degree-matched random", "random_sparse": "random",
        "spectrum_full": "spectrum-full (eigenVALUES)", "spectrum_topk": "spectrum-topk"}
WHAT = {"hemibrain_seeded": "the real MB wiring + weights", "weight_shuffle": "topology, weights shuffled",
        "degree_preserving_random": "in/out degree sequence only", "random_sparse": "nothing (ER random)",
        "spectrum_full": "connectome eigenVALUES, random eigenvectors",
        "spectrum_topk": "connectome top-k eigenVALUES"}
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
    if not files:
        print("no results yet"); return 1
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.test_acc.notna()]
    chance = float(df.chance.iloc[0]) if "chance" in df else float("nan")
    cell = df.groupby(["model", "lr"]).test_acc.mean().reset_index()   # mean over seeds per LR
    rows = []
    for m, sub in cell.groupby("model"):
        b = sub.loc[sub.test_acc.idxmax()]
        rows.append((m, float(b.test_acc), float(b.lr), int(df[df.model == m].seed.nunique())))
    rows.sort(key=lambda r: -r[1])
    rnd = next((v for (m, v, *_ ) in rows if m == "random_sparse"), float("nan"))
    print(f"\n=== MQAR + MB connectome — control hierarchy ({len(df)} cells; chance={chance:.3f}) ===")
    print(f"{'model':<30}{'best test_acc':>14}{'vs random':>11}   best LR   preserves")
    for m, v, lr, ns in rows:
        d = "" if not np.isfinite(rnd) or rnd == 0 else f"{(v-rnd)/rnd*100:+.0f}%"
        star = "  <- random" if m == "random_sparse" else ""
        print(f"{NICE.get(m,m):<30}{v:>14.4f}{d:>11}   {lr:.0e}   {WHAT.get(m,'')}{star}")
    cov = df.groupby("model").seed.nunique()
    incomplete = [NICE.get(m, m) for m in ORDER if m in cov.index and cov[m] < 2]
    if incomplete:
        print(f"\n[partial] <2 seeds for: {', '.join(incomplete)}")
    missing = [NICE.get(m, m) for m in ORDER if m not in cov.index]
    if missing:
        print(f"[partial] no cells yet for: {', '.join(missing)}")
    print(f"(higher test_acc = better; seeds/model up to {df.seed.nunique()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
