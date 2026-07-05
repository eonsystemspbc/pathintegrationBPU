#!/usr/bin/env python3
"""Figures for Exp 4 · subrun 01 — the KC-code control (2x2: backbone x readout).

Reads outputs/analysis.json (table_test_acc + comparisons) and renders:
  fig1_kc_code_2x2        — recall per rule x condition (the 2x2 factorial), with chance line
  fig2_which_wiring_matters — Δ(control − connectome) per rule for each scramble, so the
                              READOUT effect (prior) and the KC-CODING effect (new) sit side by side

Defensive: only plots conditions/rules that exist, so it works on partial (smoke) data too.
Consistent color per condition; validated palette (dataviz).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent

# condition -> color (validated CVD-safe; aqua's sub-3:1 contrast is covered by direct labels)
COND_COLOR = {
    "connectome": "#2a78d6",        # real backbone + real readout (biological baseline)
    "readout_matched": "#eb6834",   # scrambled KC->MBON readout  (the PRIOR control)
    "backbone_matched": "#1baf7a",  # scrambled ALPN->KC code     (the NEW control)
    "both_matched": "#4a3aa7",      # full scramble (joint null)
}
COND_LABEL = {
    "connectome": "connectome\n(real)",
    "readout_matched": "readout\nscrambled",
    "backbone_matched": "KC-code\nscrambled",
    "both_matched": "both\nscrambled",
}
CONDITIONS = ("connectome", "readout_matched", "backbone_matched", "both_matched")
RULES = ("hebbian", "delta", "hybrid")
INK, INK2, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
CHANCE = 0.0312

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12, "axes.edgecolor": MUT, "axes.linewidth": 0.9,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK2,
})


def _despine(ax, keep=("bottom", "left")):
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(s in keep)


def _titles(fig, cx, title, sub):
    fig.text(cx, 0.955, title, ha="center", va="top", fontsize=15.5, fontweight="bold", color=INK)
    fig.text(cx, 0.888, sub, ha="center", va="top", fontsize=10.5, color=MUT)


def fig1_2x2(A, figdir):
    table = A.get("table_test_acc", {})
    rules = [r for r in RULES if table.get(r)]
    if not rules:
        return
    conds = [c for c in CONDITIONS if any(c in table[r] for r in rules)]
    import numpy as np
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ngrp, nbar = len(rules), len(conds)
    w = 0.8 / nbar
    for j, cond in enumerate(conds):
        xs, ys, es = [], [], []
        for i, rule in enumerate(rules):
            cell = table[rule].get(cond)
            if not cell:
                continue
            xs.append(i + (j - (nbar - 1) / 2) * w)
            ys.append(cell["mean"]); es.append(cell.get("std", 0.0))
        ax.bar(xs, ys, width=w, color=COND_COLOR[cond], zorder=3,
               yerr=es, error_kw=dict(ecolor=INK2, elinewidth=1.1, capsize=2.5))
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.012, f"{y:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.axhline(CHANCE, ls=(0, (4, 3)), color=MUT, lw=1.2, zorder=2)
    ax.text(ngrp - 0.5, CHANCE + 0.008, "chance", ha="right", va="bottom", fontsize=9.5, color=MUT)
    ax.set_xticks(range(ngrp)); ax.set_xticklabels(rules, fontsize=12)
    ax.set_ylim(0, 1.03); ax.set_ylabel("MQAR recall accuracy")
    _despine(ax, keep=("left",)); ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=COND_COLOR[c], label=COND_LABEL[c].replace("\n", " ")) for c in conds],
              loc="upper center", ncol=len(conds), frameon=False, fontsize=9.5,
              bbox_to_anchor=(0.5, -0.08))
    _titles(fig, 0.55, "Which part of the MB wiring matters: KC code vs readout",
            "2×2 — {backbone real/scrambled} × {KC→MBON readout real/scrambled}, per learning rule")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.20)
    fig.savefig(figdir / "fig1_kc_code_2x2.png", dpi=200)
    plt.close(fig)


def fig2_which(A, figdir):
    """Δ(control − connectome) per rule for each scramble. Positive => connectome worse than
    that control (scramble helps); negative => connectome better (real wiring helps)."""
    comps = A.get("comparisons", {})
    ctrls = ("readout_matched", "backbone_matched", "both_matched")
    rules = [r for r in RULES
             if any(f"{r}_connectome_vs_{c}__test_acc" in comps for c in ctrls)]
    if not rules:
        return
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    nbar = len(ctrls)
    w = 0.8 / nbar
    for j, ctrl in enumerate(ctrls):
        xs, ys, ps = [], [], []
        for i, rule in enumerate(rules):
            comp = comps.get(f"{rule}_connectome_vs_{ctrl}__test_acc")
            if not comp:
                continue
            # empirical_null reports control_mean & connectome_mean; Δ = control − connectome
            d = round(comp.get("control_mean", 0) - comp.get("connectome_mean", 0), 4)
            xs.append(i + (j - (nbar - 1) / 2) * w); ys.append(d)
            ps.append(comp.get("permutation_p_one_sided"))
        ax.bar(xs, ys, width=w, color=COND_COLOR[ctrl], zorder=3)
        for x, y, pp in zip(xs, ys, ps):
            va = "bottom" if y >= 0 else "top"
            off = 0.004 if y >= 0 else -0.004
            lbl = f"{y:+.3f}" + (f"\np={pp:g}" if pp is not None else "")
            ax.text(x, y + off, lbl, ha="center", va=va, fontsize=8, color=INK)
    ax.axhline(0, color=INK2, lw=1.4, zorder=4)
    ax.set_xticks(range(len(rules))); ax.set_xticklabels(rules, fontsize=12)
    ax.set_ylabel("Δ recall   (control − connectome)")
    _despine(ax, keep=("bottom",)); ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    lo, hi = ax.get_ylim()                # pad so the +Δ / −Δ p-labels clear the title and x-ticks
    pad = 0.28 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.legend(handles=[Patch(color=COND_COLOR[c],
                             label={"readout_matched": "readout scrambled (prior question)",
                                    "backbone_matched": "KC-code scrambled (NEW question)",
                                    "both_matched": "both scrambled"}[c]) for c in ctrls],
              loc="upper center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.08))
    _titles(fig, 0.55, "Does the KC-coding wiring help the plastic memory?",
            "Δ vs real connectome — above 0: scramble beats real wiring · below 0: real wiring helps")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.80, bottom=0.20)
    fig.savefig(figdir / "fig2_which_wiring_matters.png", dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    outdir = Path(argv[0]) if argv else (HERE / "outputs")
    figdir = HERE / "figures"
    figdir.mkdir(exist_ok=True)
    aj = outdir / "analysis.json"
    if not aj.exists():
        print(f"no analysis.json under {outdir} — nothing to plot yet.")
        return 0
    A = json.loads(aj.read_text())
    fig1_2x2(A, figdir)
    fig2_which(A, figdir)
    print(f"wrote figures to {figdir} (from {aj})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
