#!/usr/bin/env python3
"""Figure: what happens to the connectome's margin when the control stops getting free shortcuts.

Left  — the handicap itself: direct input->output edges, connectome vs a degree shuffle.
Right — the connectome's seed-paired margin against the OLD control vs the FIXED control.

The right panel is the result: the margin should collapse where the handicap was large (MB, 84x)
and hold where there was none (AL, 1.19x).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HANDICAP = {"MB": (2680, 32), "CX": (6054, 279), "AL": (25428, 21382)}


def main():
    f = HERE / "shortcut_matched_summary.csv"
    if not f.exists():
        sys.exit(f"missing {f} — run analyze_shortcut_matched.py first")
    R = pd.read_csv(f).dropna(subset=["vs_degree_sm"])
    if R.empty:
        sys.exit("no cells with a degree_sm arm yet")
    R = R.sort_values("handicap_x", ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # --- left: the handicap -------------------------------------------------------------
    regs = [r for r in ("MB", "CX", "AL") if r in set(R.region)]
    x = np.arange(len(regs)); w = 0.36
    conn = [HANDICAP[r][1] for r in regs]
    degr = [HANDICAP[r][0] for r in regs]
    ax1.bar(x - w/2, degr, w, label="degree shuffle", color="#c44e52")
    ax1.bar(x + w/2, conn, w, label="real connectome", color="#4c72b0")
    ax1.set_yscale("log")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{r}\n{HANDICAP[r][0]/HANDICAP[r][1]:.0f}×" for r in regs])
    ax1.set_ylabel("direct input→output edges (log)")
    ax1.set_title("The structural fact: shortcuts a degree shuffle\ninvents that the real wiring forbids",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    for xi, (c, d) in enumerate(zip(conn, degr)):
        ax1.text(xi - w/2, d, f"{d:,}", ha="center", va="bottom", fontsize=8)
        ax1.text(xi + w/2, c, f"{c:,}", ha="center", va="bottom", fontsize=8)

    # --- right: margin before vs after -------------------------------------------------
    lab = [f"{r.region}×{r.task}\n{r.handicap_x:.0f}×" for _, r in R.iterrows()]
    x = np.arange(len(R))
    ax2.bar(x - w/2, R.vs_degree, w, label="vs degree shuffle (old control)", color="#c44e52")
    ax2.bar(x + w/2, R.vs_degree_sm, w, label="vs shortcut-matched (fair control)", color="#55a868")
    ax2.axhline(0, color="black", lw=1)
    ax2.set_xticks(x); ax2.set_xticklabels(lab, fontsize=9)
    ax2.set_ylabel("connectome − control  (seed-paired mean)")
    ax2.set_title("The result: MB's margin does not survive a fair\ncontrol (p=0.094); AL's is unchanged",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    for xi, (_, r) in enumerate(R.iterrows()):
        pd_ = r.get("p_degree", float("nan")); ps_ = r.get("p_degree_sm", float("nan"))
        ax2.text(xi - w/2, r.vs_degree, f"{r.vs_degree:+.4f}\np={pd_:.3f}", ha="center",
                 va="bottom" if r.vs_degree >= 0 else "top", fontsize=8)
        tag = "" if ps_ < 0.05 else "  (n.s.)"
        ax2.text(xi + w/2, r.vs_degree_sm, f"{r.vs_degree_sm:+.4f}\np={ps_:.3f}{tag}", ha="center",
                 va="bottom" if r.vs_degree_sm >= 0 else "top", fontsize=8)

    fig.suptitle("Does the connectome's margin survive a shortcut-matched control?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    out = HERE / "figures" / "fig_shortcut_matched.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
