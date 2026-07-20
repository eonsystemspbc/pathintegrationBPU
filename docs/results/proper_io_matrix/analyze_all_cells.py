#!/usr/bin/env python3
"""Across every complete region x task cell: does a shortcut-matched control change the verdict?

THE QUESTION. A degree-preserving shuffle of a LAYERED circuit invents direct input->output edges
the real wiring forbids (MB 2680 vs 32 real, 84x; CX 6054 vs 279, 21.7x; AL 25428 vs 21382, 1.19x).
`degree_sm` is the same shuffle repaired by degree-preserving swaps until that count matches the
connectome's. So: how much does the connectome-vs-control VERDICT move when the control loses its
shortcuts, and does that track how many shortcuts it had?

UNIT OF ANALYSIS. One region x task CELL is the unit, not one seed. Within a cell, seeds are paired
across arms (same seed -> same data order/init), so the per-cell statistic is a seed-paired mean and
the per-cell p is an exact paired Wilcoxon. Across cells we then ask whether the layered regions
(MB, CX) moved more than the shallow one (AL).

WHY THE CELL IS THE UNIT. Seeds are training replicates on ONE connectome graph per region; pooling
them across cells would treat correlated runs as independent. Using the cell as the unit costs
power (n=9) but does not manufacture it.

SOURCE HYGIENE. Local runs (RTX Blackwell) and fleet runs (L4) are never mixed WITHIN a cell -- each
cell is taken whole from a single source -- because a hardware difference inside a paired comparison
would land in the difference. Which source each cell came from is reported.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ARMS = ["connectome", "degree", "degree_sm"]
# direct input->output edges: (degree shuffle, connectome) -- see shortcut_match_report.json
HANDICAP = {"MB": (2680, 32), "CX": (6054, 279), "AL": (25428, 21382)}
DEPTH = {"MB": 1.90, "CX": 1.81, "AL": 1.02}      # mean hops, pathway_depth.json


def load_sources(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.is_file():
            files = [p]
        else:
            files = [Path(f) for f in glob.glob(str(p / "**" / "*.csv"), recursive=True)]
        for f in files:
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            if {"task", "region", "arm", "seed", "score"}.issubset(d.columns):
                if "src" not in d:
                    d["src"] = p.name
                frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    return d.drop_duplicates(subset=["task", "region", "arm", "seed", "src"])


def cell_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (src, region, task), g in df.groupby(["src", "region", "task"]):
        if region not in HANDICAP:
            continue
        p = g.pivot_table(index="seed", columns="arm", values="score")
        if not set(ARMS).issubset(p.columns):
            continue
        p = p.dropna()
        if len(p) < 6:
            continue
        vsd = (p.connectome - p.degree).to_numpy()
        vss = (p.connectome - p.degree_sm).to_numpy()
        shift = (p.degree_sm - p.degree).to_numpy()
        before, after = HANDICAP[region]
        rows.append({
            "region": region, "task": task, "src": src, "n": len(p),
            "handicap_x": before / after, "depth_hops": DEPTH[region],
            "vs_degree": vsd.mean(), "p_degree": stats.wilcoxon(vsd).pvalue,
            "vs_degree_sm": vss.mean(), "p_degree_sm": stats.wilcoxon(vss).pvalue,
            "ctl_shift": shift.mean(), "p_shift": stats.wilcoxon(shift).pvalue,
            "shift_neg_seeds": int((shift < 0).sum()),
            "verdict_move": abs(vss.mean() - vsd.mean()),
        })
    # one cell per (region,task): prefer the source with the most seeds, tie-break local
    R = pd.DataFrame(rows)
    if R.empty:
        return R
    R["_pref"] = (R.src == "local").astype(int)
    R = (R.sort_values(["n", "_pref"], ascending=False)
           .drop_duplicates(subset=["region", "task"]).drop(columns="_pref"))
    return R.sort_values(["handicap_x", "region", "task"], ascending=[False, True, True])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "all_cells_summary.csv")
    a = ap.parse_args()

    df = load_sources(a.dirs)
    if df.empty:
        print("no runs found")
        return 1
    R = cell_stats(df)
    if R.empty:
        print("no complete cells (need all 3 arms x 6 seeds)")
        return 1

    pd.set_option("display.width", 220)
    print(f"=== {len(R)} complete cells (3 arms x 6 seeds each) ===\n")
    cols = ["region", "task", "src", "handicap_x", "vs_degree", "p_degree",
            "vs_degree_sm", "p_degree_sm", "ctl_shift", "p_shift", "verdict_move"]
    print(R[cols].round(4).to_string(index=False))

    print("\n=== Q1: did the control's shortcuts matter? (direction of ctl_shift) ===")
    print("negative shift = removing shortcuts made the control WORSE = the shortcuts were HELPING it")
    neg = int((R.ctl_shift < 0).sum())
    sig = R[R.p_shift < 0.05]
    print(f"  {neg}/{len(R)} cells shift negative; {len(sig)} reach p<0.05:")
    for _, r in sig.iterrows():
        print(f"    {r.region}x{r.task:5s} {r.handicap_x:6.1f}x  shift={r.ctl_shift:+.4f} "
              f"p={r.p_shift:.4f}  ({6-r.shift_neg_seeds}/6 seeds positive)")

    print("\n=== Q2: LAYERED vs SHALLOW -- did the verdict move more where there were shortcuts? ===")
    lay = R[R.region != "AL"].verdict_move.to_numpy()
    sha = R[R.region == "AL"].verdict_move.to_numpy()
    print(f"  layered (MB/CX, 22-84x)  n={len(lay)}  mean |change in margin| = {lay.mean():.4f}  {np.round(lay,4)}")
    print(f"  shallow (AL,     1.19x)  n={len(sha)}  mean |change in margin| = {sha.mean():.4f}  {np.round(sha,4)}")
    if len(lay) and len(sha):
        u = stats.mannwhitneyu(lay, sha, alternative="greater")
        print(f"  Mann-Whitney (layered > shallow, one-sided): U={u.statistic:.0f}, p={u.pvalue:.4f}   [n={len(lay)} vs {len(sha)} CELLS]")

    print("\n=== Q3: is it GRADED in the handicap, or just layered-vs-not? ===")
    x, y = np.log10(R.handicap_x.to_numpy()), R.verdict_move.to_numpy()
    sp, pr = stats.spearmanr(x, y), stats.pearsonr(x, y)
    print(f"  Spearman rho={sp.statistic:+.3f} p={sp.pvalue:.3f} | Pearson r={pr.statistic:+.3f} p={pr.pvalue:.3f}  (n={len(R)} cells)")
    mb = R[R.region == "MB"].verdict_move.mean()
    cx = R[R.region == "CX"].verdict_move.mean()
    print(f"  MB (83.8x) mean move = {mb:.4f} vs CX (21.7x) mean move = {cx:.4f}"
          f"  -> {'NOT graded: CX moves as much as MB' if cx >= mb*0.7 else 'graded'}")

    print("\n=== Q4: which verdicts actually CHANGED? ===")
    for _, r in R.iterrows():
        was = "sig" if r.p_degree < 0.05 else "n.s."
        now = "sig" if r.p_degree_sm < 0.05 else "n.s."
        if was != now:
            print(f"  {r.region}x{r.task:5s}: {r.vs_degree:+.4f} ({was}, p={r.p_degree:.3f})"
                  f"  ->  {r.vs_degree_sm:+.4f} ({now}, p={r.p_degree_sm:.3f})   VERDICT FLIPPED")
    R.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
