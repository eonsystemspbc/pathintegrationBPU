#!/usr/bin/env python3
"""Merge the drift-validation runs, make the figure, and append a drift section to the README."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "fleet_outputs"
FIGS = HERE / "figures"
ARM_COLORS = {"connectome": "#c0392b", "degree": "#2980b9", "random": "#27ae60",
              "spectrum": "#8e44ad", "dense": "#e67e22"}
ARM_LABEL = {"connectome": "connectome", "degree": "degree-matched", "random": "ER-random",
             "spectrum": "spectrum-matched", "dense": "dense-Gaussian"}
BATCHES = list(range(3, 11))


def load():
    parts = [p / "drift_metrics.csv" for p in (OUT / "drift_bio", OUT / "drift_gen")]
    df = pd.concat([pd.read_csv(p) for p in parts if p.exists()], ignore_index=True)
    df.to_csv(HERE / "drift_metrics.csv", index=False)
    return df


def per_batch_matrix(df, io, arm):
    rows = df[(df.io == io) & (df.arm == arm)]
    curves = []
    for _, r in rows.iterrows():
        d = json.loads(r["per_batch"])
        curves.append([d.get(str(b), np.nan) for b in BATCHES])
    return np.array(curves, float) if curves else np.zeros((0, len(BATCHES)))


def main():
    FIGS.mkdir(exist_ok=True)
    df = load()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    # A: per-batch accuracy (bio), connectome vs controls
    for arm in ARM_COLORS:
        m = per_batch_matrix(df, "bio", arm)
        if not len(m):
            continue
        axes[0].plot(BATCHES, np.nanmean(m, 0), marker="o", ms=4, color=ARM_COLORS[arm],
                     lw=2 if arm == "connectome" else 1.3, label=ARM_LABEL[arm])
    axes[0].set_xlabel("test batch (chronological →)"); axes[0].set_ylabel("accuracy")
    axes[0].set_title("Drift: per-batch accuracy (biological I/O)", fontsize=10)
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.25)
    # B: mean-per-batch accuracy bar by arm (bio)
    arms = list(ARM_COLORS)
    means = [df[(df.io == "bio") & (df.arm == a)]["test_acc_mean_per_batch"].mean() for a in arms]
    sds = [df[(df.io == "bio") & (df.arm == a)]["test_acc_mean_per_batch"].std() for a in arms]
    axes[1].bar(range(len(arms)), means, yerr=sds, capsize=3, color=[ARM_COLORS[a] for a in arms])
    axes[1].set_xticks(range(len(arms))); axes[1].set_xticklabels([ARM_LABEL[a] for a in arms],
                                                                  rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("mean-per-batch accuracy"); axes[1].set_title("Drift: chronological accuracy (bio)", fontsize=10)
    axes[1].grid(axis="y", alpha=0.25)
    # C: bio vs generic (connectome)
    for io, col, ls in [("bio", "#c0392b", "-"), ("generic", "#7f8c8d", "--")]:
        m = per_batch_matrix(df, io, "connectome")
        if len(m):
            axes[2].plot(BATCHES, np.nanmean(m, 0), marker="s", ms=4, color=col, ls=ls, label=f"{io} I/O")
    axes[2].set_xlabel("test batch (chronological →)"); axes[2].set_ylabel("accuracy")
    axes[2].set_title("Drift: biological vs free I/O (connectome)", fontsize=10)
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.25)
    fig.suptitle("External validation — UCI-270 long-term sensor drift (train early, test future)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGS / "fig_drift_validation.png", dpi=140, bbox_inches="tight")
    print(f"wrote {FIGS/'fig_drift_validation.png'}")

    # analysis + README section
    summ = {}
    for io in ("bio", "generic"):
        con = df[(df.io == io) & (df.arm == "connectome")]["test_acc_mean_per_batch"]
        row = {"connectome_mean": round(float(con.mean()), 4)}
        for c in ("degree", "random", "spectrum", "dense"):
            cv = df[(df.io == io) & (df.arm == c)]["test_acc_mean_per_batch"]
            if len(cv) and len(con):
                pooled = np.sqrt((con.var(ddof=1) + cv.var(ddof=1)) / 2 + 1e-12)
                row[f"{c}_mean"] = round(float(cv.mean()), 4)
                row[f"d_vs_{c}"] = round(float((con.mean() - cv.mean()) / (pooled + 1e-9)), 3)
        summ[io] = row
    (HERE / "drift_analysis.json").write_text(json.dumps(summ, indent=2))

    def tbl(io):
        lines = [f"| arm | mean-per-batch acc ({io} I/O) | overall acc | macro-F1 |", "|---|---|---|---|"]
        for a in ("connectome", "degree", "random", "spectrum", "dense"):
            s = df[(df.io == io) & (df.arm == a)]
            def f(col):
                return f"{s[col].mean():.3f}±{s[col].std():.3f}" if len(s) else "—"
            lines.append(f"| {ARM_LABEL[a]} | {f('test_acc_mean_per_batch')} | {f('test_acc_overall')} | {f('test_macro_f1')} |")
        return "\n".join(lines)
    b = summ["bio"]
    ds = ", ".join(f"{c} d={b.get(f'd_vs_{c}')}" for c in ("degree", "random", "spectrum", "dense")
                   if b.get(f"d_vs_{c}") is not None)
    section = ("\n\n### External validation — long-term drift (UCI 270)\n\n"
               "Train on the two earliest batches, test on batches 3–10 in chronological order "
               "(never random CV). 6-gas classification through the same AL substrate (128→glomerulus "
               "adapter, 6-way projection-neuron readout).\n\n"
               + tbl("bio") + "\n\n"
               f"Connectome vs controls (bio I/O, mean-per-batch acc, Cohen's *d*): {ds}.\n\n"
               "**This is a null for the connectome** — on drift it does *not* beat the matched "
               "controls (it trails ER-random). The AL connectome's advantage is **task-specific**: "
               "it helps on the turbulent low-concentration detection matched to its native "
               "divisive-normalization / onset-emphasis computations, but confers no benefit on the "
               "drift-shift 6-gas classification. An honest scope limit on the headline claim — the "
               "connectome is not a generically better graph, it is a better graph *for the "
               "computation it evolved to do*.\n\n"
               "![drift](figures/fig_drift_validation.png)\n")
    readme = HERE / "README.md"
    txt = readme.read_text()
    txt = re.split(r"\n### External validation.*", txt, flags=re.S)[0].rstrip()
    readme.write_text(txt + section)
    print(json.dumps(summ, indent=2))
    print("updated README with drift section")


if __name__ == "__main__":
    main()
