#!/usr/bin/env python3
"""Across every complete cell: how far does the connectome-vs-control verdict move when the control
loses its manufactured shortcuts?

Left  — per-cell margin before (vs degree shuffle) and after (vs shortcut-matched), with the two
        cells whose SIGNIFICANCE flipped called out. The point is that the fair control pulls the
        verdict toward zero from BOTH sides: it removes a spurious connectome win (MB x mqar) and a
        spurious connectome loss (MB x path).
Right — how far each cell's verdict moved, grouped layered (MB/CX) vs shallow (AL). Cells, not
        seeds, are the plotted units; n is small and the group test is only suggestive.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=HERE / "all_cells_summary.csv")
    ap.add_argument("--out", type=Path, default=HERE / "figures" / "fig_all_cells.png")
    a = ap.parse_args()
    R = pd.read_csv(a.summary).sort_values(["handicap_x", "region", "task"],
                                           ascending=[False, True, True]).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.4),
                                   gridspec_kw={"width_ratios": [1.65, 1]})

    # ---- left: margin before -> after, per cell -------------------------------------
    y = np.arange(len(R))[::-1]
    for i, (_, r) in enumerate(R.iterrows()):
        yy = y[i]
        flipped = (r.p_degree < 0.05) != (r.p_degree_sm < 0.05)
        ax1.plot([r.vs_degree, r.vs_degree_sm], [yy, yy],
                 color="#888" if not flipped else "#d1495b", lw=3 if flipped else 1.6,
                 zorder=2, alpha=.9, solid_capstyle="round")
        ax1.scatter(r.vs_degree, yy, s=95, color="#c44e52", zorder=3,
                    edgecolor="white", linewidth=1.1)
        ax1.scatter(r.vs_degree_sm, yy, s=95, color="#55a868", zorder=3, marker="D",
                    edgecolor="white", linewidth=1.1)
        star = lambda p: "*" if p < 0.05 else ""
        ax1.text(r.vs_degree, yy + .28, f"{r.vs_degree:+.4f}{star(r.p_degree)}",
                 ha="center", fontsize=7.4, color="#8c2f33")
        ax1.text(r.vs_degree_sm, yy - .42, f"{r.vs_degree_sm:+.4f}{star(r.p_degree_sm)}",
                 ha="center", fontsize=7.4, color="#2f6b45")
        if flipped:
            ax1.text(0.995, yy, "significance flipped", transform=ax1.get_yaxis_transform(),
                     ha="right", va="center", fontsize=7.8, style="italic", color="#d1495b")
    ax1.axvline(0, color="black", lw=1.1, zorder=1)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{r.region}×{r.task}   {r.handicap_x:.0f}×" for _, r in R.iterrows()],
                        fontsize=9.5)
    ax1.set_xlabel("connectome − control   (seed-paired mean; >0 = connectome better)")
    ax1.set_title("The fair control pulls the verdict toward zero from BOTH sides\n"
                  "removing a spurious win (MB×mqar) and a spurious loss (MB×path)",
                  fontsize=11, fontweight="bold")
    ax1.scatter([], [], s=95, color="#c44e52", label="vs degree shuffle (has shortcuts)")
    ax1.scatter([], [], s=95, color="#55a868", marker="D", label="vs shortcut-matched (fair)")
    ax1.legend(fontsize=8.6, loc="lower right", framealpha=.95)
    ax1.margins(x=.20); ax1.set_ylim(-0.9, len(R)-0.1)
    ax1.text(.5, -.13, "* = exact paired Wilcoxon p < 0.05 over 6 seeds (floor p = 0.031). "
                       "Seeds are TRAINING replicates on one connectome graph per region.",
             transform=ax1.transAxes, ha="center", fontsize=7.6, color="#444")

    # ---- right: verdict movement, layered vs shallow ---------------------------------
    lay = R[R.region != "AL"].verdict_move.to_numpy()
    sha = R[R.region == "AL"].verdict_move.to_numpy()
    for xi, (vals, lab, col) in enumerate([(lay, f"layered\nMB/CX (22–84×)\nn={len(lay)} cells", "#4c72b0"),
                                           (sha, f"shallow\nAL (1.19×)\nn={len(sha)} cells", "#dd8452")]):
        jit = np.linspace(-.11, .11, len(vals)) if len(vals) > 1 else [0]
        ax2.scatter(np.full(len(vals), xi) + jit, vals, s=115, color=col,
                    edgecolor="white", linewidth=1.2, zorder=3)
        ax2.hlines(vals.mean(), xi - .27, xi + .27, color=col, lw=3, zorder=2)
        ax2.text(xi + .30, vals.mean(), f"mean\n{vals.mean():.4f}", va="center", ha="left",
                 fontsize=8.6, color=col, fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f"layered\nMB/CX (22–84×)\nn={len(lay)} cells",
                         f"shallow\nAL (1.19×)\nn={len(sha)} cells"], fontsize=9.5)
    ax2.set_ylabel("|change in the verdict|  =  |margin$_{fair}$ − margin$_{shuffle}$|")
    u = stats.mannwhitneyu(lay, sha, alternative="greater")
    rho = stats.spearmanr(np.log10(R.handicap_x), R.verdict_move)
    mb = R[R.region == "MB"].verdict_move.mean(); cx = R[R.region == "CX"].verdict_move.mean()
    ax2.set_title("Verdicts move where the control had shortcuts,\nnot where it didn't",
                  fontsize=11, fontweight="bold")
    ax2.text(.5, .97, f"Mann–Whitney one-sided p = {u.pvalue:.3f}"
                      f"{'  (n.s.)' if u.pvalue >= .05 else ''}\n"
                      f"Spearman(log handicap) ρ = {rho.statistic:+.2f}, p = {rho.pvalue:.3f}\n"
                      f"NOT graded: MB (84×) {mb:.4f} ≈ CX (22×) {cx:.4f}",
             transform=ax2.transAxes, ha="center", va="top", fontsize=8.6,
             bbox=dict(boxstyle="round,pad=0.45", fc="#f4f4f4", ec="#bbb"))
    ax2.set_ylim(bottom=-0.001)
    ax2.margins(x=.35)

    fig.suptitle("Does a shortcut-matched control change what we conclude about the connectome?",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    a.out.parent.mkdir(exist_ok=True)
    fig.savefig(a.out, dpi=145, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
