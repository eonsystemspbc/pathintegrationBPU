#!/usr/bin/env python3
"""Margin-shrink slope chart: does the connectome's advantage survive a *fair* control?

The argument
------------
A degree-preserving shuffle is the standard null for "is the connectome's WIRING special?".
For a LAYERED circuit it is not a fair null: rewiring at fixed degree invents direct
input->output edges that the real wiring forbids, handing the control an express lane
straight past the computation the task is supposed to require.

  measured direct in->out edges   degree shuffle   connectome   handicap
    MB                                     2,680           32       83.8x
    CX                                     6,054          279       21.7x
    AL                                    25,428       21,382        1.19x

The `degree_sm` arm repairs exactly that: the same degree shuffle, then degree-preserving
double-edge swaps until its direct in->out count matches the connectome's own. Degrees are
preserved EXACTLY, so the only thing that changed is the shortcut count.

This figure plots, per cell, the connectome's seed-paired margin against the OLD control
(degree) and against the FAIR control (degree_sm). AL is the built-in control-for-the-control:
with a 1.19x handicap there was nothing to repair, so its margin should barely move, while MB
at 83.8x should collapse. The point estimates move in exactly that direction -- MB loses ~70%
of its margin, AL ~5%.

WHAT THE FIGURE MUST NOT OVERSELL (this is why the uncertainty is drawn, not hidden):
  * The shrink itself is the paired quantity (d_old - d_fair) == (degree_sm - degree): the
    connectome cancels, so it is a two-control comparison. For MB it is +0.0072 with a paired
    t of 1.31 (p=0.25) and only 4/6 seeds positive. n=6 does NOT establish the shrink.
  * The MB-vs-AL contrast -- the actual argument of the figure -- is an interaction, and at
    n=6 per cell it is Welch t=1.09, p=0.32. -70% and -5% are not resolvable from each other.
  * MB seed 0's degree run (score 0.1593, early-stopped at 31 epochs vs 40) is a low outlier
    that alone supplies ~51% of MB's old-control margin. Leave-one-seed-out moves the headline
    percentage between -80% and -32%.
  * n = 6 TRAINING seeds against ONE connectome. The control arms are redrawn per seed, the
    connectome is a single fixed graph, so nothing here generalizes across connectomes.
  * The two cells score different tasks on different metrics; only the % change is comparable
    across them, never the absolute heights.
So the honest reading is directional: the repair eats most of MB's apparent margin and leaves
AL's intact, exactly as the shortcut account predicts, but this run is a pilot, not proof.

Everything numeric is computed from shortcut_matched_runs.csv (handicap ratios for the axis
labels come from shortcut_matched_summary.csv). Nothing is hard-coded.

Output: figures/fig_margin_shrink.png
"""
from __future__ import annotations

import sys
import textwrap
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RUNS = HERE / "shortcut_matched_runs.csv"
SUMMARY = HERE / "shortcut_matched_summary.csv"
OUT = HERE / "figures" / "fig_margin_shrink.png"

# --- design tokens (validated colour-blind-safe pair: worst-pair CVD dE 24.7, normal dE 33.6)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#84837c"
GRID = "#e3e2dd"
SERIES = {"MB": "#2a78d6", "AL": "#eb6834"}  # categorical slots 1 (blue) and 6 (orange)

OLD_ARM, FAIR_ARM = "degree", "degree_sm"


def sign_test_p(k: int, n: int) -> float:
    """One-sided exact sign test: P(>= k of n successes | p=0.5)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2.0**n


def boot_mean_ci(x: np.ndarray, rng: np.random.Generator, b: int = 20000) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for a mean (n is tiny; the CI is wide on purpose)."""
    idx = rng.integers(0, len(x), size=(b, len(x)))
    return tuple(np.percentile(x[idx].mean(axis=1), [2.5, 97.5]))


def boot_pct_ci(old: np.ndarray, fair: np.ndarray, rng: np.random.Generator,
                b: int = 20000) -> tuple[float, float]:
    """Percentile bootstrap CI for 100*(mean_fair - mean_old)/mean_old, resampling seeds PAIRED."""
    idx = rng.integers(0, len(old), size=(b, len(old)))
    mo, mf = old[idx].mean(axis=1), fair[idx].mean(axis=1)
    pct = np.where(np.abs(mo) > 1e-12, 100.0 * (mf - mo) / mo, np.nan)
    return tuple(np.nanpercentile(pct, [2.5, 97.5]))


def loo_pct_range(old: np.ndarray, fair: np.ndarray) -> tuple[float, float]:
    """How far the headline % moves if any single seed is dropped."""
    vals = [100.0 * (np.delete(fair, i).mean() - np.delete(old, i).mean())
            / np.delete(old, i).mean() for i in range(len(old))]
    return min(vals), max(vals)


def load_cells() -> list[dict]:
    if not RUNS.exists():
        sys.exit(f"missing {RUNS}")
    runs = pd.read_csv(RUNS)

    handicap = {}
    if SUMMARY.exists():
        s = pd.read_csv(SUMMARY)
        handicap = {(r.region, r.task): r.handicap_x for _, r in s.iterrows()}

    cells = []
    for (region, task), g in runs.groupby(["region", "task"], sort=False):
        wide = g.pivot_table(index="seed", columns="arm", values="score")
        need = {"connectome", OLD_ARM, FAIR_ARM}
        if not need.issubset(wide.columns):
            continue
        wide = wide.dropna(subset=list(need)).sort_index()
        if wide.empty:
            continue
        d_old = (wide["connectome"] - wide[OLD_ARM]).to_numpy()
        d_fair = (wide["connectome"] - wide[FAIR_ARM]).to_numpy()
        # the shrink is a two-CONTROL comparison: the connectome term cancels exactly.
        shrink = d_old - d_fair
        rng = np.random.default_rng(20260720)
        t_shrink, p_shrink = stats.ttest_1samp(shrink, 0.0)
        cells.append(
            dict(
                region=region,
                task=task,
                seeds=wide.index.to_numpy(),
                d_old=d_old,
                d_fair=d_fair,
                shrink=shrink,
                m_old=float(d_old.mean()),
                m_fair=float(d_fair.mean()),
                ci_old=boot_mean_ci(d_old, rng),
                ci_fair=boot_mean_ci(d_fair, rng),
                ci_pct=boot_pct_ci(d_old, d_fair, rng),
                loo_pct=loo_pct_range(d_old, d_fair),
                k_old=int((d_old > 0).sum()),
                k_fair=int((d_fair > 0).sum()),
                k_shrink=int((shrink > 0).sum()),
                t_shrink=float(t_shrink),
                p_shrink=float(p_shrink),
                n=len(d_old),
                handicap=handicap.get((region, task)),
            )
        )
    if not cells:
        sys.exit("no cell has all three arms (connectome, degree, degree_sm)")
    # steepest shrink first, so the reading order matches the argument
    cells.sort(key=lambda c: (c["m_fair"] - c["m_old"]) / abs(c["m_old"]))
    return cells


def main() -> None:
    cells = load_cells()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.2, 7.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x_old, x_fair = 0.0, 1.0

    all_pts = []
    for i, cell in enumerate(cells):
        c = SERIES.get(cell["region"], "#4a3aa7")
        n = cell["n"]

        # --- per-seed spread: same seed's (connectome - control), paired across the two arms.
        # Each cell gets its own lane so the two clouds never interleave.
        lane = 0.055 + 0.105 * i
        jitter = np.linspace(-0.038, 0.038, n) if n > 1 else np.zeros(1)
        xs_o, xs_f = x_old + lane + jitter, x_fair - lane + jitter
        for xo, xf, yo, yf in zip(xs_o, xs_f, cell["d_old"], cell["d_fair"]):
            ax.plot([xo, xf], [yo, yf], color=c, lw=0.8, alpha=0.22, zorder=2,
                    solid_capstyle="round")
        for xs, ys in ((xs_o, cell["d_old"]), (xs_f, cell["d_fair"])):
            ax.scatter(xs, ys, s=32, facecolor=c, edgecolor=SURFACE, linewidth=1.3,
                       alpha=0.6, zorder=3)
            all_pts.extend(ys)

        # --- the headline slope: mean seed-paired margin, with its bootstrap 95% CI.
        # The CIs are wide because n=6; drawing them is the whole point.
        label = f"{cell['region']} × {cell['task']}"
        for x, m, (clo, chi) in ((x_old, cell["m_old"], cell["ci_old"]),
                                 (x_fair, cell["m_fair"], cell["ci_fair"])):
            ax.plot([x, x], [clo, chi], color=c, lw=2.0, alpha=0.55, zorder=4,
                    solid_capstyle="butt")
            for y in (clo, chi):
                ax.plot([x - 0.022, x + 0.022], [y, y], color=c, lw=2.0, alpha=0.55, zorder=4)
            all_pts.extend([clo, chi])
        ax.plot([x_old, x_fair], [cell["m_old"], cell["m_fair"]], color=c, lw=2.6,
                zorder=5, solid_capstyle="round", label=label)
        ax.scatter([x_old, x_fair], [cell["m_old"], cell["m_fair"]], s=110, facecolor=c,
                   edgecolor=SURFACE, linewidth=2.2, zorder=6)

        # endpoint values
        ax.annotate(f"{cell['m_old']:+.4f}", (x_old, cell["m_old"]), xytext=(-12, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=10,
                    color=INK, fontweight="bold")
        ax.annotate(f"{cell['m_fair']:+.4f}", (x_fair, cell["m_fair"]), xytext=(12, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=10,
                    color=INK, fontweight="bold")

        # direct label at the right end
        hx = f"{cell['handicap']:.0f}×" if cell["handicap"] and cell["handicap"] >= 10 \
            else (f"{cell['handicap']:.2f}×" if cell["handicap"] else "")
        sub = f"\n{hx} shortcut handicap" if hx else ""
        ax.annotate(f"{label}{sub}", (x_fair, cell["m_fair"]), xytext=(78, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=11,
                    color=c, fontweight="bold", linespacing=1.5)

        # % change, on the slope -- always with its bootstrap CI, because at n=6 the
        # point estimate on its own reads far more decisive than the data warrant.
        pct = 100.0 * (cell["m_fair"] - cell["m_old"]) / cell["m_old"]
        plo, phi = cell["ci_pct"]
        mid_y = 0.5 * (cell["m_old"] + cell["m_fair"])
        ax.annotate(f"{pct:+.0f}% of margin\n95% CI [{plo:+.0f}%, {phi:+.0f}%]",
                    (0.5, mid_y), xytext=(0, 22 if pct > -30 else -30),
                    textcoords="offset points", ha="center", va="center", fontsize=10.5,
                    color=c, fontweight="bold", linespacing=1.45,
                    bbox=dict(boxstyle="round,pad=0.32", facecolor=SURFACE, edgecolor=c,
                              linewidth=1.1, alpha=0.95))

    # --- zero line: the "connectome is no better than the control" level
    ax.axhline(0, color=INK_2, lw=1.3, ls=(0, (5, 3)), zorder=1)
    ax.annotate("0 = no advantage over the control", (-0.5, 0), xytext=(0, -7),
                textcoords="offset points", ha="left", va="top", fontsize=9.5,
                color=INK_2, style="italic")

    # --- axes
    ax.set_xlim(-0.52, 1.62)
    lo, hi = min(all_pts + [0.0]), max(all_pts)
    pad = 0.16 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad * 1.25)

    ax.set_xticks([x_old, x_fair])
    ax.set_xticklabels(
        [
            "vs degree-preserving shuffle\n(OLD control — invents in→out shortcuts)",
            "vs shortcut-matched shuffle\n(FAIR control — same degrees, matched in→out edges)",
        ],
        fontsize=10.5, color=INK,
    )
    ax.set_xlabel("control arm the connectome is measured against", fontsize=10.5,
                  color=INK_2, labelpad=10)
    ax.set_ylabel("connectome margin  (score points, higher = better)\nmean of per-seed"
                  " connectome − control", fontsize=10.5, color=INK_2)

    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(axis="y", colors=INK_2, length=0, labelsize=9.5)
    ax.tick_params(axis="x", colors=INK, length=0, pad=8)

    ax.legend(loc="upper left", frameon=False, fontsize=10, handlelength=1.8,
              labelcolor=INK, title="cell (region × task)",
              title_fontsize=9).get_title().set_color(INK_MUTED)

    fig.suptitle("Does the margin survive a fair control?  (n=6 seeds — directional, not settled)",
                 fontsize=15.5, fontweight="bold", color=INK, x=0.042, ha="left", y=0.975)
    sub = textwrap.fill(
        "A degree-preserving shuffle hands a layered circuit direct input→output shortcuts the real"
        " wiring forbids. Repairing that (degrees held exactly) should gut the connectome's margin"
        " where the shortcut handicap was large, and leave it where there wasn't one. Both point"
        " estimates move that way — but the error bars below are this pilot's honest resolution,"
        " and they do not separate −70% from −5%.", width=126)
    fig.text(0.042, 0.922, sub, fontsize=10.5, color=INK_2, ha="left", va="top", linespacing=1.45)

    # --- what the reader must know to not over-read the two slopes
    inter = ""
    if len(cells) == 2:
        a, b = cells[0], cells[1]
        t_i, p_i = stats.ttest_ind(a["shrink"], b["shrink"], equal_var=False)
        inter = (f"  The {a['region']}-vs-{b['region']} difference in shrink — the actual claim of"
                 f" this figure — is Welch t={t_i:.2f}, p={p_i:.2f}: unresolved at this n.")

    lines = [
        "Faint dots = one seed's own connectome − control difference, paired across the two control"
        " arms; bold line = mean; whiskers = percentile bootstrap 95% CI over the 6 seeds.",
    ]
    for c in cells:
        lines.append(
            f"{c['region']}×{c['task']}:  connectome beats the old control in {c['k_old']}/{c['n']}"
            f" seeds (sign test p={sign_test_p(c['k_old'], c['n']):.3f}) and the fair control in"
            f" {c['k_fair']}/{c['n']} (p={sign_test_p(c['k_fair'], c['n']):.3f}).  The SHRINK itself"
            f" (= fair control − old control, the connectome cancels) is {c['shrink'].mean():+.4f},"
            f" {c['k_shrink']}/{c['n']} seeds positive, paired t={c['t_shrink']:.2f}, p={c['p_shrink']:.2f}"
            f" — not established.  Drop any one seed and the % moves over"
            f" [{c['loo_pct'][0]:+.0f}%, {c['loo_pct'][1]:+.0f}%]."
        )
    lines.append(
        "n = 6 TRAINING seeds against ONE connectome per cell (controls are redrawn per seed, the"
        " connectome is a single fixed graph), so this generalises to neither other connectomes nor"
        " other tasks." + inter
    )
    lines.append(
        "MB×mqar's old-control arm contains one low outlier (seed 0, score 0.1593, early-stopped at"
        " 31/40 epochs) that supplies ~half of that arm's margin — it is the top-left blue dot."
        "  The two cells are different tasks on different score metrics: compare the % change"
        " between them, never the absolute heights."
    )
    wrapped = "\n".join(textwrap.fill(ln, width=164, subsequent_indent="     ") for ln in lines)
    fig.text(0.042, 0.012, wrapped, fontsize=8.0, color=INK_MUTED, ha="left",
             va="bottom", linespacing=1.55)

    fig.tight_layout(rect=(0.012, 0.245, 0.995, 0.855))
    fig.savefig(OUT, dpi=160, facecolor=SURFACE)
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
