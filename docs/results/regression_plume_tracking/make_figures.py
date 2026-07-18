#!/usr/bin/env python3
"""Figures for the plume-tracking regression benchmark."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd

plt.rcParams.update({"figure.facecolor":"white","axes.titleweight":"bold","font.size":11,
                     "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
HERE = Path(__file__).resolve().parent
FIGS = HERE/"figures"; FIGS.mkdir(exist_ok=True)
COL = {"connectome":"#c0392b","degree":"#2980b9","random":"#27ae60","spectrum":"#8e44ad","dense":"#e67e22"}
LAB = {"connectome":"connectome","degree":"degree-matched","random":"edge-random",
       "spectrum":"spectrum-matched","dense":"dense-Gaussian"}
ARMS = list(COL)


def main(out_dir):
    out_dir = Path(out_dir)
    parts = sorted(out_dir.glob("metrics_shard*.csv"))
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True) if parts else pd.read_csv(out_dir/"metrics_by_run.csv")
    df.to_csv(HERE/"metrics_by_run.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    # A: R2 by arm (bio)
    for i, io in enumerate(["bio", "generic"]):
        m = [df[(df.io==io)&(df.arm==a)]["test_r2"].mean() for a in ARMS]
        e = [df[(df.io==io)&(df.arm==a)]["test_r2"].std() for a in ARMS]
        ax[0].bar(np.arange(len(ARMS))+i*0.38-0.19, m, 0.36, yerr=e, capsize=3,
                  color=[COL[a] for a in ARMS], alpha=1.0 if io=="bio" else 0.45,
                  label=f"{io} I/O")
    for r, c, ls in [("gru","#111111","--"), ("adapter_only","#888888",":")]:
        v = df[df.arm==r]["test_r2"]
        if len(v): ax[0].axhline(v.mean(), color=c, ls=ls, lw=1.4, label=f"{r} ref")
    ax[0].axhline(0, color="red", lw=1.0)
    ax[0].set_xticks(range(len(ARMS))); ax[0].set_xticklabels([LAB[a] for a in ARMS], rotation=30, ha="right", fontsize=8)
    ax[0].set_ylabel("test R² (tracking c(t))"); ax[0].set_title("Regression: variance explained", fontsize=10)
    ax[0].legend(fontsize=7); ax[0].grid(axis="y", alpha=0.25)
    # B: collapse diagnostic
    for i, io in enumerate(["bio","generic"]):
        m = [df[(df.io==io)&(df.arm==a)]["test_output_std_ratio"].mean() for a in ARMS]
        ax[1].bar(np.arange(len(ARMS))+i*0.38-0.19, m, 0.36, color=[COL[a] for a in ARMS],
                  alpha=1.0 if io=="bio" else 0.45, label=f"{io} I/O")
    ax[1].axhline(1.0, color="black", ls="--", lw=1.2, label="perfect amplitude")
    ax[1].set_xticks(range(len(ARMS))); ax[1].set_xticklabels([LAB[a] for a in ARMS], rotation=30, ha="right", fontsize=8)
    ax[1].set_ylabel("output std / target std"); ax[1].grid(axis="y", alpha=0.25)
    ax[1].set_title("Collapse diagnostic (<1 = under-responsive)", fontsize=10)
    ax[1].legend(fontsize=7)
    # C: R2 vs collapse scatter
    for a in ARMS:
        s = df[(df.io=="bio")&(df.arm==a)]
        ax[2].scatter(s["test_output_std_ratio"], s["test_r2"], color=COL[a], s=28, label=LAB[a])
    v = df[df.arm=="gru"]; ax[2].scatter(v["test_output_std_ratio"], v["test_r2"], color="#111", marker="*", s=90, label="GRU")
    ax[2].axhline(0, color="red", lw=1); ax[2].axvline(1, color="black", ls="--", lw=1)
    ax[2].set_xlabel("output std / target std"); ax[2].set_ylabel("test R²")
    ax[2].set_title("Tracking amplitude vs accuracy", fontsize=10); ax[2].legend(fontsize=7); ax[2].grid(alpha=0.25)
    fig.suptitle("Continuous-tracking REGRESSION: can a connectome follow a changing target?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(FIGS/"fig_regression_tracking.png", dpi=140, bbox_inches="tight")
    print("wrote", FIGS/"fig_regression_tracking.png")
    # console summary
    print("\n=== test R² (mean±sd) ===")
    for io in ["bio","generic"]:
        for a in ARMS:
            s = df[(df.io==io)&(df.arm==a)]["test_r2"]
            if len(s): print(f"  {io:7s} {LAB[a]:18s} R²={s.mean():.3f}±{s.std():.3f}  n={len(s)}")
    for r in ["gru","adapter_only"]:
        s = df[df.arm==r]["test_r2"]
        if len(s): print(f"  ref     {r:18s} R²={s.mean():.3f}±{s.std():.3f}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else HERE/"fleet_outputs")
