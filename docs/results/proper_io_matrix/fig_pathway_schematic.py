#!/usr/bin/env python3
"""Explainer figure: a degree-preserving shuffle is not a fair control for a LAYERED circuit.

The argument, in three panels:

  (1) The real mushroom-body pathway is layered: ALPN -> Kenyon cells -> MBON. Almost nothing
      goes straight from input to output (32 direct ALPN->MBON edges measured on the operator).
  (2) A degree-preserving shuffle keeps every neuron's in-degree and out-degree exactly, but it
      does NOT keep the paths those degrees sat on. Rewiring manufactures 2,680 direct
      ALPN->MBON edges — an 84x express lane that skips the Kenyon-cell layer entirely. Any
      "the connectome's wiring is special" margin measured against this control is partly just
      the connectome being denied a shortcut its control was handed for free.
  (3) The shortcut-matched control ('degree_sm') repairs that: start from the same shuffle, then
      apply degree-preserving double-edge swaps until the direct input->output count is back at
      the connectome's own 32. Degree sequences are byte-identical to (2); only path structure
      changes.

The take-away printed in the caption strip: degree sequences do not encode path structure, so
matching degrees alone does not match what the network is allowed to compute. The measured
consequence (from shortcut_matched_runs.csv, read at plot time) is that MB x mqar's apparent
connectome advantage shrinks once the express lane is closed while AL x flow — whose shuffle
never gained one (1.19x) — is unmoved. That comparison is DIRECTIONAL ONLY: with n=6 training
seeds on a single connectome per region, MB's shrink (+0.0072) has a 95% CI spanning zero
(p=0.25) and its size swings between 32% and 80% under leave-one-seed-out, so the caption states
that explicitly rather than showing bare point estimates.

No data are plotted here except the direct-edge counts and the caption's summary lines;
everything else is a schematic. The schematic conserves its own edge budget — panel 2 reroutes
three ALPN->KC and three KC->MBON arrows onto the express lane (plus the compensating KC->KC
arrows a double-edge swap produces) rather than adding arrows, so the drawing does not contradict
the claim that degrees are preserved. Run:  python fig_pathway_schematic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "fig_pathway_schematic.png"
RUNS = HERE / "shortcut_matched_runs.csv"

# Measured direct input->output edge counts for the MB pathway operators (see SHORTCUT_MATCHED.md
# / build_shortcut_matched.py). These are counts of edges, not invented illustrative numbers.
N_DIRECT_CONNECTOME = 32
N_DIRECT_DEGREE = 2680
N_DIRECT_MATCHED = 32  # by construction: repaired back to the connectome's own count

# Okabe-Ito colour-blind-safe palette.
C_BLUE = "#0072B2"
C_SKY = "#56B4E9"
C_PURPLE = "#CC79A7"
C_VERM = "#D55E00"
C_GREEN = "#009E73"
C_GREY = "#9a9a9a"

PANELS = [
    dict(
        tag="1",
        title="Real connectome\n(layered pathway)",
        accent=C_BLUE,
        n_direct=N_DIRECT_CONNECTOME,
        express=False,
        arm_label="real wiring",
        note="input and output are separated\nby the Kenyon-cell layer",
    ),
    dict(
        tag="2",
        title="Degree-preserving shuffle\n(the standard control)",
        accent=C_VERM,
        n_direct=N_DIRECT_DEGREE,
        express=True,
        arm_label="degree shuffle",
        note="same degrees, new paths: an express lane\nappears (edges rerouted, none added)",
    ),
    dict(
        tag="3",
        title="Shortcut-matched control\n('degree_sm')",
        accent=C_GREEN,
        n_direct=N_DIRECT_MATCHED,
        express=False,
        arm_label="shortcut-matched",
        note="degrees identical to panel 2,\nexpress lane swapped away",
    ),
]

# Node layout in axes coordinates (schematic only; no scale, no units).
X_IN, X_MID, X_OUT = 0.17, 0.515, 0.86
Y_IN = np.linspace(0.44, 0.74, 4)
Y_MID = np.linspace(0.36, 0.80, 8)
Y_OUT = np.linspace(0.50, 0.68, 3)
R_BIG, R_SMALL = 0.045, 0.030


def _paired(a, b):
    """Seed-paired mean difference with a 95% t interval and a paired-t p-value."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    sem = d.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=sem)
    p = float(stats.ttest_rel(a, b).pvalue)
    return float(d.mean()), float(lo), float(hi), p, n


def _read_effect() -> list[str]:
    """Caption lines grounded in the actual per-run scores (no invented numbers).

    Every margin is reported with its 95% CI, and the *shrink itself* — the quantity the whole
    figure is about — gets its own test plus a leave-one-seed-out range, because with six
    training seeds on a single connectome the point estimates alone would overstate the case.
    """
    if not RUNS.exists():
        sys.exit(f"missing {RUNS} — cannot build the data-grounded caption line")
    df = pd.read_csv(RUNS)
    need = ["connectome", "degree", "degree_sm"]
    bits, mb = [], None
    for region, task, tag in (("MB", "mqar", "84× lane"), ("AL", "flow", "1.19× lane, no handicap")):
        cell = df[(df.region == region) & (df.task == task)]
        wide = cell.pivot_table(index="seed", columns="arm", values="score")
        if not set(need).issubset(wide.columns):
            continue
        wide = wide.dropna(subset=need)
        o = _paired(wide["connectome"], wide["degree"])
        m = _paired(wide["connectome"], wide["degree_sm"])
        bits.append(
            f"{region}×{task} ({tag}):  {o[0]:+.4f} [{o[1]:+.4f}, {o[2]:+.4f}]"
            f"  →  {m[0]:+.4f} [{m[1]:+.4f}, {m[2]:+.4f}]"
        )
        if region == "MB":
            s = _paired(wide["degree_sm"], wide["degree"])
            loo = [
                1 - (w["connectome"] - w["degree_sm"]).mean() / (w["connectome"] - w["degree"]).mean()
                for w in (wide.drop(index=i) for i in wide.index)
            ]
            mb = (s, min(loo), max(loo), 1 - m[0] / o[0])
    n_seeds = int(o[4])
    lines = [
        "Measured consequence — seed-paired connectome margin (task score) against the old control "
        f"→ against the shortcut-matched control;\nmean [95% CI] over n={n_seeds} training seeds, "
        "one connectome per region (seed variability only — no connectome-level replicates):",
        *bits,
    ]
    if mb is not None:
        s, lo_f, hi_f, frac = mb
        lines.append(
            f"Underpowered: MB's shrink is a paired {s[0]:+.4f} [{s[1]:+.4f}, {s[2]:+.4f}], "
            f"p={s[3]:.2f} — the CI spans zero, and the apparent {frac:.0%} shrink\nranges "
            f"{lo_f:.0%}–{hi_f:.0%} when any single seed is dropped. Direction matches the "
            "confound; the size is not established by these six runs."
        )
    return lines


def _arrow(ax, p0, p1, *, color, lw, rad=0.0, alpha=1.0, ls="-", zorder=2, ms=6):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=f"-|>,head_length={ms},head_width={ms * 0.5}",
            mutation_scale=1.0,
            color=color,
            lw=lw,
            alpha=alpha,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
            capstyle="round",
        )
    )


def _node(ax, x, y, r, color, zorder=4):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="white", lw=1.4, zorder=zorder))


def draw_panel(ax, spec, rng):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # --- layered edges: ALPN -> KC -> MBON ------------------------------------------------
    # The same edge budget is drawn in every panel. In panel 2 three ALPN->KC and three
    # KC->MBON edges are *moved* onto the express lane (plus three compensating KC->KC edges),
    # never added, so the arrow count per node stays honest to "degrees are preserved".
    e_in = [(i, int(j)) for i in range(len(Y_IN)) for j in rng.choice(len(Y_MID), 3, replace=False)]
    e_out = [(j, int(rng.integers(len(Y_OUT)))) for j in range(len(Y_MID))]
    n_swap = 3 if spec["express"] else 0
    moved_in, e_in = e_in[:n_swap], e_in[n_swap:]
    moved_out, e_out = e_out[:n_swap], e_out[n_swap:]

    for i, j in e_in:
        _arrow(ax, (X_IN + R_BIG, Y_IN[i]), (X_MID - R_SMALL, Y_MID[j]),
               color=C_GREY, lw=0.9, alpha=0.75, ms=5)
    for j, k in e_out:
        _arrow(ax, (X_MID + R_SMALL, Y_MID[j]), (X_OUT - R_BIG, Y_OUT[k]),
               color=C_GREY, lw=0.9, alpha=0.75, ms=5)
    # compensating within-layer edges created by the same double-edge swaps
    for (_, j_lost), (j_free, _) in zip(moved_in, moved_out):
        _arrow(ax, (X_MID + R_SMALL * 0.8, Y_MID[j_free]), (X_MID + R_SMALL * 0.8, Y_MID[j_lost]),
               color=C_GREY, lw=0.9, alpha=0.75, rad=-0.9, ms=5)

    # --- nodes ----------------------------------------------------------------------------
    for y in Y_IN:
        _node(ax, X_IN, y, R_BIG, C_BLUE)
    for y in Y_MID:
        _node(ax, X_MID, y, R_SMALL, C_SKY)
    for y in Y_OUT:
        _node(ax, X_OUT, y, R_BIG, C_PURPLE)

    # --- the direct input->output edges, routed BELOW the whole layered stack --------------
    p0 = (X_IN, Y_IN[0] - R_BIG)
    p1 = (X_OUT, Y_OUT[0] - R_BIG)
    if spec["express"]:
        # prominent express lane: a thick bundle sweeping under the Kenyon-cell layer
        for rad, lw, a in ((0.42, 5.0, 0.95), (0.52, 3.4, 0.55), (0.32, 3.4, 0.55)):
            _arrow(ax, p0, p1, color=C_VERM, lw=lw, rad=rad, alpha=a, zorder=5, ms=9)
        ax.text(
            0.50,
            0.055,
            "EXPRESS LANE:  2,680 direct ALPN→MBON edges,\nskipping the Kenyon-cell layer",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color=C_VERM,
            fontweight="bold",
            linespacing=1.5,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_VERM, lw=1.2, alpha=0.96),
        )
    else:
        _arrow(ax, p0, p1, color=spec["accent"], lw=1.2, rad=0.42, alpha=0.9,
               ls=(0, (4, 2)), zorder=5, ms=6)
        ax.text(
            0.50,
            0.055,
            f"only {spec['n_direct']} direct ALPN→MBON edges",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color=spec["accent"],
            fontweight="bold",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=spec["accent"], lw=1.1, alpha=0.96),
        )

    # --- layer captions (above the columns) -----------------------------------------------
    ax.text(X_IN, 0.99, "ALPN\ninput", ha="center", va="top", fontsize=9.5, color=C_BLUE,
            fontweight="bold", linespacing=1.35)
    ax.text(X_MID, 0.99, "Kenyon cells\n(expansion layer)", ha="center", va="top", fontsize=9.5,
            color="#1f6f99", fontweight="bold", linespacing=1.35)
    ax.text(X_OUT, 0.99, "MBON\noutput", ha="center", va="top", fontsize=9.5, color=C_PURPLE,
            fontweight="bold", linespacing=1.35)

    ax.text(0.01, 0.99, spec["tag"], ha="left", va="top", fontsize=12, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="circle,pad=0.30", fc=spec["accent"], ec="none"))
    ax.set_title(spec["title"], fontsize=11.5, fontweight="bold", color=spec["accent"], pad=8,
                 linespacing=1.35)


def draw_bar(ax, spec):
    ax.barh([0], [spec["n_direct"]], height=0.5, color=spec["accent"], edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlim(1, 6000)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"], fontsize=8)
    ax.set_xlabel("direct ALPN→MBON edges (count, log scale)", fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(spec["n_direct"] * 1.3, 0, f"{spec['n_direct']:,}", va="center", ha="left",
            fontsize=9, fontweight="bold", color=spec["accent"])
    if spec["express"]:
        # log axis: bar LENGTH is not proportional to count, so state the ratio in words
        ax.text(np.sqrt(spec["n_direct"]), 0,
                f"{spec['n_direct'] / N_DIRECT_CONNECTOME:.0f}× the real count",
                va="center", ha="center", fontsize=8.5, fontweight="bold", color="white")
    ax.text(0.0, 1.20, spec["arm_label"], transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold", color=spec["accent"])
    ax.text(1.0, 1.20, spec["note"], transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2,
            color="#444444", linespacing=1.35, style="italic")


def main():
    effect = _read_effect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    fig = plt.figure(figsize=(12.4, 8.0))
    gs = fig.add_gridspec(
        3, 3, height_ratios=[1.0, 0.16, 1.00], hspace=0.62, wspace=0.14,
        left=0.03, right=0.99, top=0.875, bottom=0.03,
    )

    for c, spec in enumerate(PANELS):
        draw_panel(fig.add_subplot(gs[0, c]), spec, np.random.default_rng(rng.integers(1 << 30)))
        draw_bar(fig.add_subplot(gs[1, c]), spec)

    cap = fig.add_subplot(gs[2, :])
    cap.set_axis_off()
    cap.add_patch(
        plt.Rectangle((0, 0), 1, 1, transform=cap.transAxes, facecolor="#f2f2f2",
                      edgecolor="#cccccc", lw=1.0, zorder=0)
    )
    cap.text(
        0.5,
        0.91,
        "Degree sequences do not encode path structure.",
        ha="center", va="center", transform=cap.transAxes, fontsize=12, fontweight="bold",
        zorder=2,
    )
    cap.text(
        0.5,
        0.755,
        "Panels 2 and 3 have identical in- and out-degree sequences; they differ only in which "
        "paths those degrees sit on. So preserving degrees does not preserve\nwhat the network "
        "is allowed to compute — in a layered circuit it hands the control a one-hop route from "
        "input to output that the real wiring forbids.\nThe fair control is panel 3: same "
        "degrees, and the same direct input→output edge count as the connectome.",
        ha="center", va="center", transform=cap.transAxes, fontsize=9.2, linespacing=1.7,
        color="#222222", zorder=2,
    )
    ys = (0.435, 0.315, 0.235, 0.10)
    weights = ("normal", "bold", "bold", "normal")
    for y, line, weight in zip(ys, effect, weights):
        cap.text(
            0.5, y, line, ha="center", va="center", transform=cap.transAxes, fontsize=8.4,
            color="#333333", style="italic", fontweight=weight, linespacing=1.65, zorder=2,
        )

    fig.suptitle(
        "A degree-preserving shuffle is not a fair control for a layered circuit",
        fontsize=15, fontweight="bold", y=0.985, va="top",
    )

    fig.savefig(OUT, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
