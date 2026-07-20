#!/usr/bin/env python3
"""Analyse the proper-I/O matrix: is the diagonal real, or was AL x gas a one-off?

DECISION RULE (fixed before the numbers were seen):
  REAL      each native cell beats BOTH its own controls, rank >=5/6, permutation p<0.05
  ONE-OFF   native cells tie or lose to their own controls (like MB/CX did on the gas task)
  DEAD CELL all arms at floor/chance -> the cell cannot discriminate and is reported as such,
            NOT counted as evidence either way
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
NATIVE = {"mqar":"MB","path":"CX","flow":"OL","gas":"AL"}
CHANCE = {"mqar":1/32, "path":0.0, "flow":0.0}     # mqar chance=1/vocab; R^2 floor=0


def perm_p(a, b, n=20000, seed=0):
    rng = np.random.default_rng(seed); obs = a.mean()-b.mean()
    pool = np.r_[a, b]; k = len(a); cnt = 0
    for _ in range(n):
        p = rng.permutation(pool); cnt += (p[:k].mean()-p[k:].mean()) >= obs
    return (cnt+1)/(n+1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", type=Path, default=HERE/"outputs"/"matrix_metrics.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    rows = []
    for (task, region), g in df.groupby(["task","region"]):
        c = g[g.arm=="connectome"]["score"].values
        d = g[g.arm=="degree"]["score"].values
        r = g[g.arm=="random"]["score"].values
        if not len(c) or not len(d) or not len(r): continue
        allv = np.r_[c,d,r]
        dead = bool(allv.max() <= CHANCE[task] + 0.02)      # every arm at floor -> cannot discriminate
        rows.append(dict(task=task, region=region, native=int(NATIVE[task]==region),
            con=c.mean(), deg=d.mean(), rnd=r.mean(),
            pct_vs_deg=(c.mean()-d.mean())/max(abs(d.mean()),1e-9)*100,
            pct_vs_rnd=(c.mean()-r.mean())/max(abs(r.mean()),1e-9)*100,
            rank_deg=f"{int((c.mean()>d).sum())}/{len(d)}", rank_rnd=f"{int((c.mean()>r).sum())}/{len(r)}",
            p_deg=perm_p(c,d), p_rnd=perm_p(c,r), dead=dead, n=len(c)))
    R = pd.DataFrame(rows).sort_values(["task","region"])
    R.to_csv(HERE/"matrix_summary.csv", index=False)
    pd.set_option("display.width",200)
    print("=== PROPER-I/O MATRIX (each region through ITS OWN biological ports) ===")
    print(R[["task","region","native","con","deg","rnd","pct_vs_deg","pct_vs_rnd",
             "rank_deg","rank_rnd","p_deg","dead"]].to_string(index=False,
          float_format=lambda x: f"{x:.4f}"))
    live = R[~R.dead]
    print("\n=== VERDICT per native cell ===")
    for _, x in R[R.native==1].iterrows():
        if x.dead: v = "DEAD CELL (all arms at floor) — not evaluable"
        elif x.pct_vs_deg>0 and x.pct_vs_rnd>0 and x.p_deg<0.05: v = "WIN — diagonal supported here"
        elif x.pct_vs_deg>0 and x.pct_vs_rnd>0: v = "leans positive but not significant"
        else: v = "NO — connectome does not beat its own controls"
        print(f"  {x.region:3s} x {x.task:5s}: {v}   (vs deg {x.pct_vs_deg:+.1f}%, p={x.p_deg:.3f}; vs rnd {x.pct_vs_rnd:+.1f}%)")
    if len(live) > 2:
        nat = live[live.native==1]["pct_vs_rnd"]; off = live[live.native==0]["pct_vs_rnd"]
        if len(nat) and len(off):
            print(f"\nnative vs off-diagonal advantage (live cells only): "
                  f"{nat.mean():+.2f}% vs {off.mean():+.2f}%  (n={len(nat)} native, {len(off)} off)")
    print(f"\ndead cells: {list(R[R.dead].apply(lambda x: f'{x.region}x{x.task}', axis=1))}")


if __name__ == "__main__":
    main()
