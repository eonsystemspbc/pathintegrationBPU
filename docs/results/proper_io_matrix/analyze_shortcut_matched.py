#!/usr/bin/env python3
"""Does the connectome's standing vs a degree control change once the control loses its shortcuts?

THE PROBLEM. A degree-preserving shuffle of a LAYERED circuit manufactures direct input->output
edges that the real wiring forbids. Measured here:

    region   degree-shuffle direct in->out   connectome   handicap
    MB                 2,680                     32          84x
    CX                 6,054                    279        21.7x
    AL                25,428                 21,382         1.19x

An 84x express lane lets the MB control skip the Kenyon-cell layer -- i.e. skip the computation the
circuit exists to perform. Any "the connectome loses to its degree control" verdict on MB/CX is
therefore confounded.

THE CONTROL. `degree_sm` starts from the same degree shuffle and repairs it with degree-preserving
double-edge swaps until the direct input->output count matches the connectome's own. Degrees are
preserved EXACTLY (every op is a double-edge swap), so it remains a strict degree control that
simply no longer gets free shortcuts.

THE BUILT-IN CONTROL FOR THE CONTROL. AL's handicap is only 1.19x, so fixing it should barely move
AL while substantially moving MB (84x) and CX (21.7x). If instead every region shifts, the shortcut
story is wrong and something else (e.g. the swaps themselves perturbing structure) is responsible.

STATS. Seeds are training replicates on ONE connectome, not independent draws of the graph, so a
t-test over seeds overstates evidence (pseudoreplication). The arms DO share seeds, so the honest
test is a paired one on the seed-matched differences, reported alongside the raw effect. We report
the paired Wilcoxon signed-rank exact p (n=6) and never claim more than "consistent across seeds".
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARMS = ["connectome", "degree", "degree_sm"]
# direct input->output edge counts, from build_shortcut_matched.py / shortcut_match_report.json
HANDICAP = {"MB": (2680, 32), "CX": (6054, 279), "AL": (25428, 21382)}


def load(dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.rglob("matrix_shard*.csv")) + sorted(d.rglob("matrix_all.csv")):
            try:
                rows.append(pd.read_csv(p))
            except Exception:
                pass
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    return df.drop_duplicates(subset=[c for c in ("task", "region", "arm", "seed") if c in df])


def wilcoxon_exact(d: np.ndarray) -> float:
    """Exact two-sided Wilcoxon signed-rank p. n<=6 so we enumerate all 2^n sign flips."""
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    r = pd.Series(np.abs(d)).rank().to_numpy()
    obs = float(r[d > 0].sum())
    tot = 0
    hits = 0
    for signs in itertools.product([0, 1], repeat=n):
        s = float(r[np.array(signs, bool)].sum())
        tot += 1
        # two-sided: as extreme as observed, relative to the null mean r.sum()/2
        if abs(s - r.sum() / 2) >= abs(obs - r.sum() / 2) - 1e-9:
            hits += 1
    return hits / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", type=Path,
                    default=[HERE / "outputs",
                             Path("/home/ec2-user/.claude/jobs/1d2b95e6/tmp/local_sm/mb"),
                             Path("/home/ec2-user/.claude/jobs/1d2b95e6/tmp/local_sm/al")])
    a = ap.parse_args()
    df = load(a.dirs)
    if df.empty:
        print("no results yet")
        return 1
    df = df[df.arm.isin(ARMS)]
    print(f"loaded {len(df)} runs\n")

    out = []
    for (task, region), g in df.groupby(["task", "region"]):
        piv = g.pivot_table(index="seed", columns="arm", values="score")
        if not {"connectome", "degree"}.issubset(piv.columns):
            continue
        piv = piv.dropna()
        if piv.empty:
            continue
        n = len(piv)
        row = {"task": task, "region": region, "n_seeds": n}
        for arm in ARMS:
            row[arm] = piv[arm].mean() if arm in piv else np.nan
        # paired, seed-matched: connectome vs each control
        for ctl in ("degree", "degree_sm"):
            if ctl not in piv:
                row[f"vs_{ctl}"] = np.nan
                row[f"p_{ctl}"] = np.nan
                continue
            d = (piv["connectome"] - piv[ctl]).to_numpy()
            row[f"vs_{ctl}"] = d.mean()
            row[f"p_{ctl}"] = wilcoxon_exact(d) if n >= 4 else np.nan
        if {"degree", "degree_sm"}.issubset(piv.columns):
            row["ctl_shift"] = (piv["degree_sm"] - piv["degree"]).mean()
        before, after = HANDICAP.get(region, (np.nan, np.nan))
        row["handicap_x"] = before / after if after else np.nan
        out.append(row)

    R = pd.DataFrame(out).sort_values("handicap_x", ascending=False)
    pd.set_option("display.width", 200)
    print("=== connectome vs its degree control, BEFORE and AFTER removing the shortcut handicap ===")
    print("vs_degree     = connectome - degree_shuffle      (seed-paired mean)")
    print("vs_degree_sm  = connectome - shortcut_matched     (seed-paired mean)")
    print("ctl_shift     = how much the CONTROL moved when its shortcuts were removed")
    print("higher score = better; p = exact paired Wilcoxon over seeds\n")
    cols = ["task", "region", "handicap_x", "n_seeds", "connectome", "degree", "degree_sm",
            "vs_degree", "p_degree", "vs_degree_sm", "p_degree_sm", "ctl_shift"]
    print(R[[c for c in cols if c in R]].round(4).to_string(index=False))

    print("\n=== reading ===")
    print("PREDICTION: fixing the control should move MB (84x) and CX (21.7x) but NOT AL (1.19x).")
    for _, r in R.iterrows():
        if np.isnan(r.get("ctl_shift", np.nan)):
            continue
        moved = "MOVED" if abs(r["ctl_shift"]) > 0.01 else "flat"
        print(f"  {r['region']:>3} x {r['task']:<5} handicap {r['handicap_x']:>6.1f}x -> "
              f"control {moved} by {r['ctl_shift']:+.4f}; "
              f"connectome vs fixed control {r['vs_degree_sm']:+.4f}")
    R.to_csv(HERE / "shortcut_matched_summary.csv", index=False)
    print(f"\nwrote {HERE/'shortcut_matched_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
