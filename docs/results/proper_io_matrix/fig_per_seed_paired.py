#!/usr/bin/env python3
"""Per-seed paired figure: the connectome's margin is small, and on MB the shortcut repair is
CONSISTENT WITH -- but does not establish -- most of that margin being the control's shortcuts.

WHAT THIS FIGURE ARGUES
-----------------------
A degree-preserving shuffle is the standard control for "is the connectome's WIRING special?".
But rewiring a LAYERED circuit manufactures direct input->output edges the real wiring forbids,
handing the control a one-hop express lane past the computation. Measured (seed 0):
MB 2,680 shuffled vs 32 real direct edges (~84x), AL 25,428 vs 21,382 (~1.19x, i.e. no handicap).
The `degree_sm` arm is the same degree shuffle, then repaired by degree-preserving double-edge
swaps until its direct input->output count matches the connectome's own -- degrees preserved
exactly, shortcuts removed.

Two claims, both meant to be read straight off the ink:

(a) THE EFFECTS ARE SMALL RELATIVE TO SEED-TO-SEED SPREAD. Each thin grey line is one seed carried
    across all three arms. The lines wander over a range several times larger than the separation
    between the bold arm means, and individual lines cross. With n=6 a per-cell sign test bottoms
    out at p=0.03125, so nothing here is strongly powered.

(b) REPAIRING THE SHORTCUTS MOVES MB, NOT AL -- SUGGESTIVELY, NOT SIGNIFICANTLY. On MB the repaired
    control (degree_sm) sits closer to the connectome than the plain degree shuffle does. But the
    control's shift is +0.0072 with a seed-bootstrap 95% CI of [-0.0003, +0.0185] (paired Wilcoxon
    p=0.31, only 4/6 seeds move up), and 'it absorbs 70% of the margin' has a 95% CI of [-0.07,
    0.97]; drop the cell's one outlier run (seed 0, degree = 0.1593) and 70% becomes 32%. On AL,
    which never had a handicap, degree and degree_sm land on top of each other. The direction
    tracks the handicap, which is what a shortcut explanation predicts -- but with n=6 training
    seeds on ONE connectome and two cells, this figure is a motivating observation, not a test.

WHAT THIS FIGURE CANNOT SHOW
----------------------------
* n=6 is training seeds (and shuffle draws) on a SINGLE fixed connectome per region -- there is no
  replication over connectomes, so nothing here generalises beyond these two graphs.
* MB x MQAR and AL x flow differ in region AND task AND handicap. AL is a useful control-for-the-
  control, but it is not a clean one-factor manipulation of the handicap.
* Every early stop is a patience stop with best-val weights restored (run_matrix.py), so a run that
  ended at epoch 31 is a converged/model-selected run, not a truncated one. Early stopping is
  slightly MORE common in the connectome arm (3/6 runs) than in degree (2/6).

Every number plotted or printed is read from shortcut_matched_runs.csv (scores, seeds, epochs) and
operators_pathway/shortcut_match_report.json (direct-edge counts / handicap). Nothing is hardcoded;
p-values are exact paired Wilcoxon over the 6 seeds and CIs are 20k paired seed bootstraps.

Colours are the Okabe-Ito-derived blue/vermillion/green triple, validated colour-blind-safe
(worst adjacent pair deutan dE 11.0, normal dE 25.8) against both light and dark surfaces.

Usage:  python fig_per_seed_paired.py   ->  figures/fig_per_seed_paired.png
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / "shortcut_matched_runs.csv"
REPORT = HERE / "operators_pathway" / "shortcut_match_report.json"
OUT = HERE / "figures" / "fig_per_seed_paired.png"

ARMS = ["connectome", "degree", "degree_sm"]
ARM_LABEL = {
    "connectome": "connectome\n(real wiring)",
    "degree": "degree\n(shuffle)",
    "degree_sm": "degree_sm\n(shuffle + repair)",
}
# colour-blind-safe categorical triple (validated: lightness band, chroma, CVD dE, contrast)
ARM_COLOR = {"connectome": "#0173B2", "degree": "#D55E00", "degree_sm": "#029E73"}

SEED_GREY = "#7a7a7a"

# Metric each task is scored with, for the y-axis unit string. Metadata about the harness
# (run_matrix.py: classification cells report accuracy, regression cells report R^2), not data.
TASK_METRIC = {
    "mqar": "MQAR query-recall accuracy\n(fraction of query steps correct, held-out test)",
    "flow": "optic-flow ego-motion $R^2$\n(fraction of variance explained, held-out test)",
}
TASK_TITLE = {"mqar": "MQAR", "flow": "optic flow"}
# What "0" means on each metric, so a truncated window can't be read as near-floor or near-ceiling.
# From tasks.py (mqar vocab=32 -> 32-way choice) and run_matrix.py (regression cells report R^2).
TASK_SCALE = {
    "mqar": "axis window is {pts:.1f} accuracy points of a 32-way choice: chance = 1/32 = 0.031 and\n"
            "ceiling = 1.0 are both far off-scale. All three arms sit in the same weakly-learned regime.",
    "flow": "axis window is {span:.3f} $R^2$: 0 (predict the mean) and 1 (perfect) are both off-scale.\n"
            "All three arms sit in the same partially-learned regime.",
}

N_BOOT = 20000


def wilcoxon_exact(d: np.ndarray) -> float:
    """Exact two-sided paired Wilcoxon signed-rank p (n<=6, enumerate all 2^n sign flips).

    Same routine as analyze_shortcut_matched.py, so figure and table cannot disagree.
    """
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    r = pd.Series(np.abs(d)).rank().to_numpy()
    obs = float(r[d > 0].sum())
    hits = tot = 0
    for signs in itertools.product([0, 1], repeat=n):
        s = float(r[np.array(signs, bool)].sum())
        tot += 1
        if abs(s - r.sum() / 2) >= abs(obs - r.sum() / 2) - 1e-9:
            hits += 1
    return hits / tot


def boot_ci(S: np.ndarray, fn, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% CI of fn(seed-resampled score matrix). Seeds are resampled as whole rows,
    keeping the pairing -- the only resampling unit this design actually has."""
    rng = np.random.default_rng(seed)
    n = len(S)
    vals = np.array([fn(S[rng.integers(0, n, n)]) for _ in range(N_BOOT)], float)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def load_runs() -> pd.DataFrame:
    if not RUNS.exists():
        sys.exit(f"missing {RUNS}")
    df = pd.read_csv(RUNS)
    return df[df.arm.isin(ARMS)].copy()


def load_handicap() -> dict[str, dict]:
    """Direct input->output edge counts per region, from the shortcut-matching build report."""
    if not REPORT.exists():
        sys.exit(f"missing {REPORT}")
    rep = json.loads(REPORT.read_text())
    out = {}
    for region, d in rep.items():
        # per_seed rows are [direct in->out BEFORE repair, AFTER repair, converged]. Seed 0's
        # "before" count is the figure of record elsewhere in this experiment (analyze_shortcut_
        # matched.py's HANDICAP), so quote it, and carry the seed spread for the caption.
        shuffled = [row[0] for row in d["per_seed"]]
        real = int(d["connectome_direct"])
        out[region] = {
            "real": real,
            "shuffled": int(shuffled[0]),
            "shuffled_lo": int(min(shuffled)),
            "shuffled_hi": int(max(shuffled)),
            "x": shuffled[0] / real if real else float("nan"),
        }
    return out


def fmt_x(x: float) -> str:
    return f"{x:.0f}×" if x >= 10 else f"{x:.2f}×"


def complete_cells(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(task, region) cells that have every arm at every seed -- only those can be paired."""
    cells = []
    for (task, region), g in df.groupby(["task", "region"]):
        seeds = set.intersection(*(set(g[g.arm == a].seed) for a in ARMS)) if all(
            (g.arm == a).any() for a in ARMS) else set()
        if len(seeds) >= 2 and all((g[g.arm == a].seed.isin(seeds)).sum() == len(seeds) for a in ARMS):
            cells.append((task, region, sorted(seeds)))
    # biggest handicap first, so the reader meets the effect before the control-for-the-control
    return cells


def main() -> int:
    df = load_runs()
    hand = load_handicap()
    cells = complete_cells(df)
    if not cells:
        sys.exit("no cell has all three arms at a common set of seeds")
    cells.sort(key=lambda c: -hand.get(c[1], {}).get("x", 0.0))

    # ---- gather per-cell matrices -------------------------------------------------------
    packed = []
    for task, region, seeds in cells:
        g = df[(df.task == task) & (df.region == region)]
        S = np.array([[float(g[(g.arm == a) & (g.seed == s)].score.iloc[0]) for a in ARMS]
                      for s in seeds])                      # [n_seeds, 3]
        E = np.array([[int(g[(g.arm == a) & (g.seed == s)].epochs.iloc[0]) for a in ARMS]
                      for s in seeds])
        packed.append(dict(task=task, region=region, seeds=seeds, S=S, E=E, mean=S.mean(0)))

    # one shared y-SPAN (not shared limits -- the two panels are in different units) so that a
    # given vertical distance means the same number of score units in both panels
    base = max(p["S"].max() - p["S"].min() for p in packed) + 0.024
    BAND = 0.030            # empty strip at the bottom of every panel, reserved for the caveat box
    span = base + BAND      # identical in both panels, so a vertical distance is still comparable

    fig, axes = plt.subplots(1, len(packed), figsize=(13.6, 8.6), dpi=150)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    xs = np.arange(len(ARMS), dtype=float)
    WHITE_BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)

    for ax, p in zip(axes, packed):
        S, seeds, mean = p["S"], p["seeds"], p["mean"]
        n = len(seeds)
        # fixed (not random) per-seed horizontal offset: seed 0 leftmost, so lines stay traceable
        off = np.linspace(-0.085, 0.085, n) if n > 1 else np.zeros(1)

        ax.set_facecolor("white")
        ax.grid(axis="y", color="#e2e2e2", lw=0.8, zorder=0)
        ax.set_axisbelow(True)

        # ---- (a) the seed lines: spread ------------------------------------------------
        for i in range(n):
            ax.plot(xs + off[i], S[i], color=SEED_GREY, lw=1.0, alpha=0.65, zorder=2,
                    solid_capstyle="round")
            ax.plot(xs + off[i], S[i], ls="none", marker="o", ms=4.0, mfc=SEED_GREY,
                    mec="white", mew=0.8, alpha=0.9, zorder=3)

        # ---- (b) the arm means ---------------------------------------------------------
        ax.plot(xs, mean, color="#404040", lw=1.6, ls=(0, (5, 3)), zorder=4)
        for j, a in enumerate(ARMS):
            ax.plot([xs[j]], [mean[j]], marker="D", ms=13, mfc=ARM_COLOR[a], mec="white",
                    mew=2.0, ls="none", zorder=10)

        # connectome mean as a reference rule across the panel
        ax.axhline(mean[0], color=ARM_COLOR["connectome"], lw=1.3, ls=(0, (2, 2)), alpha=0.85,
                   zorder=5)

        # paired margins, drawn as the gap from each control mean up to that rule
        paired = (S[:, [0]] - S)             # connectome - control, per seed
        for j in (1, 2):
            gap = float(paired[:, j].mean())
            wins = int((paired[:, j] > 0).sum())
            pv = wilcoxon_exact(paired[:, j])
            lo, hi = boot_ci(S, lambda B, j=j: float((B[:, 0] - B[:, j]).mean()))
            sig = "" if pv < 0.05 else "  (n.s.)"
            # degree's label goes left of its arrow, degree_sm's right, so the two blocks and the
            # arm-mean labels never overlap
            side = -1.0 if j == 1 else 1.0
            ax.annotate("", xy=(xs[j] + 0.17 * side, mean[0]), xytext=(xs[j] + 0.17 * side, mean[j]),
                        arrowprops=dict(arrowstyle="<->", color=ARM_COLOR[ARMS[j]], lw=1.7,
                                        shrinkA=0, shrinkB=0, mutation_scale=13), zorder=6)
            ax.text(xs[j] + 0.23 * side, (mean[0] + mean[j]) / 2,
                    f"+{gap:.4f}{sig}\nCI [{lo:+.4f}, {hi:+.4f}]\n{wins}/{n} seeds, p={pv:.2f}",
                    color=ARM_COLOR[ARMS[j]], fontsize=8.0,
                    fontweight="bold", va="center", ha="left" if side > 0 else "right",
                    linespacing=1.35, zorder=7, bbox=WHITE_BOX)
            ax.text(xs[j], mean[j] - span * 0.030, f"{mean[j]:.4f}", ha="center", va="top",
                    fontsize=9, color="#333333", zorder=7, bbox=WHITE_BOX)
        ax.text(xs[0], mean[0] + span * 0.030, f"{mean[0]:.4f}", ha="center", va="bottom",
                fontsize=9, color="#333333", zorder=7, bbox=WHITE_BOX)

        # ---- axes, units, handicap ------------------------------------------------------
        h = hand[p["region"]]
        ax.set_xticks(xs)
        ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=10)
        for tick, a in zip(ax.get_xticklabels(), ARMS):
            tick.set_color(ARM_COLOR[a])
            tick.set_fontweight("bold")
        ax.set_xlim(-0.62, len(ARMS) + 0.38)
        mid = (S.max() + S.min()) / 2
        ax.set_ylim(mid - base / 2 - BAND, mid + base / 2)
        ax.set_ylabel(TASK_METRIC[p["task"]], fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_title(
            f"{p['region']} × {TASK_TITLE[p['task']]}   —   handicap {fmt_x(h['x'])}\n"
            f"direct input→output edges: {h['shuffled']:,} shuffled vs {h['real']:,} real",
            fontsize=11.5, fontweight="bold", pad=62)

        # how far the repair moved the CONTROL (the quantity the handicap predicts) -- stated
        # above the axes so it can never collide with the marks. This is the figure's headline
        # quantity, so it carries its own uncertainty: a bare "70%" from 6 seeds is not a result.
        shift = float(S[:, 2].mean() - S[:, 1].mean())
        shift_p = wilcoxon_exact(S[:, 2] - S[:, 1])
        shift_wins = int((S[:, 2] - S[:, 1] > 0).sum())
        s_lo, s_hi = boot_ci(S, lambda B: float((B[:, 2] - B[:, 1]).mean()))
        frac = shift / float(paired[:, 1].mean()) if paired[:, 1].mean() else float("nan")
        f_lo, f_hi = boot_ci(S, lambda B: float((B[:, 2] - B[:, 1]).mean() /
                                                (B[:, 0] - B[:, 1]).mean())
                             if (B[:, 0] - B[:, 1]).mean() != 0 else np.nan)
        ax.text(0.5, 1.02,
                f"shortcut repair moves the control {shift:+.4f}\n"
                f"95% CI [{s_lo:+.4f}, {s_hi:+.4f}];  {shift_wins}/{n} seeds;  p={shift_p:.2f}"
                f"{'' if shift_p < 0.05 else '  (n.s.)'}\n"
                f"= {frac:.0%} of the margin — ratio 95% CI [{f_lo:.0%}, {f_hi:.0%}]",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8.6,
                fontweight="bold", color="#222222", linespacing=1.45,
                bbox=dict(boxstyle="round,pad=0.34", fc="#f2f2f0", ec="#c9c9c9", lw=0.8))

        # what the truncated window is a window ONTO (chance / floor / ceiling all off-scale)
        ax.text(0.5, -0.40, TASK_SCALE[p["task"]].format(span=span, pts=span * 100),
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8.3, color="#666666", linespacing=1.4, style="italic")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#9a9a9a")

        # honest flag: how much of the cell's story rests on its single lowest run. Note this is
        # NOT an "unfinished run" claim -- run_matrix.py early-stops on val patience and restores
        # best-val weights, so a short run is model-selected, not truncated (and early stopping is
        # if anything MORE common in the connectome arm; the per-arm counts are printed below).
        emax = int(p["E"].max())
        i_low, j_low = np.unravel_index(int(np.argmin(S)), S.shape)
        sc = float(S[i_low, j_low])
        ep = int(p["E"][i_low, j_low])
        keep = [i for i in range(n) if i != i_low]
        B = S[keep]
        frac_wo = (float((B[:, 2] - B[:, 1]).mean()) / float((B[:, 0] - B[:, 1]).mean())
                   if (B[:, 0] - B[:, 1]).mean() else float("nan"))
        gap_wo = float((B[:, 0] - B[:, 1]).mean())
        estop = {a: int((p["E"][:, k] < emax).sum()) for k, a in enumerate(ARMS)}
        note = (f"cell's lowest run: seed {seeds[i_low]} {ARMS[j_low]} = {sc:.4f} "
                f"({ep}/{emax} epochs).\n"
                f"Drop that seed and the vs-degree gap falls to {gap_wo:+.4f}\n"
                f"and the repair accounts for {frac_wo:.0%}, not {frac:.0%}.\n"
                f"Short runs are val-patience stops with best-val weights restored,\n"
                f"not truncated budgets: " + ", ".join(f"{a} {estop[a]}/{n}" for a in ARMS) + ".")
        ax.annotate(note,
                    xy=(xs[j_low] + off[i_low], sc), xycoords="data",
                    xytext=(0.015, 0.015), textcoords="axes fraction",
                    fontsize=8.0, color="#555555", ha="left", va="bottom", linespacing=1.35,
                    zorder=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dddddd", lw=0.7),
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.9,
                                    connectionstyle="arc3,rad=-0.2"))

    # ---- shared framing -----------------------------------------------------------------
    handles = [plt.Line2D([], [], color=SEED_GREY, lw=1.0, marker="o", ms=4.0, mec="white",
                          label=f"one training seed, carried across all three arms "
                                f"(n={len(packed[0]['seeds'])}; one connectome)"),
               plt.Line2D([], [], color="#404040", lw=1.6, ls=(0, (5, 3)), marker="D", ms=10,
                          mfc="#bdbdbd", mec="white", mew=1.5,
                          label="arm mean (dashed line joins categories — not a trend)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.215), ncol=2,
               frameon=False, fontsize=9.5)

    fig.suptitle("Seed-paired scores per control arm: the margins are small, and on MB\n"
                 "repairing the control's shortcuts closes most of one — suggestively, not significantly",
                 fontsize=13.0, fontweight="bold", y=0.988)
    fig.text(0.5, 0.925,
             "Each thin grey line is one seed carried across all three arms. `degree_sm` = the same "
             "degree shuffle, then repaired by degree-preserving swaps until its direct input→output\n"
             "edge count matches the connectome's — degrees preserved exactly. AL, whose shuffle was "
             "never handicapped, is the control for the control — but it differs from MB in task and\n"
             "region as well as in handicap, so it is a plausibility check, not a one-factor "
             "manipulation.",
             ha="center", va="top", fontsize=9.5, color="#444444", linespacing=1.5)
    fig.text(0.5, 0.012,
             f"Higher is better. Both panels span an identical {span:.3f} score units, so a vertical "
             f"distance means the same number of score units in either panel; the units themselves\n"
             f"differ (accuracy vs $R^2$), so the axes are not shared numerically and neither starts "
             f"at zero — the italic line under each panel says what is off-scale. Seeds carry a fixed\n"
             f"small horizontal offset (seed 0 leftmost) for legibility. Coloured arrows are "
             f"seed-paired means of (connectome − control); p is an exact paired Wilcoxon and CIs are "
             f"20,000\npaired seed bootstraps — a percentile bootstrap over 6 units is itself optimistic, "
             f"so read the CIs as descriptive and the exact Wilcoxon p as the test.\nNO TREND IS FITTED — the dashed black line just joins "
             f"the three arm means of three unordered categories. STATISTICAL CEILING: n=6 is training "
             f"seeds\n(and shuffle draws) on ONE fixed connectome per region, so p-values describe "
             f"seed noise only, not variation over connectomes; a sign test over 6 seeds bottoms out "
             f"at p=0.031,\nand with 2 cells × 2 comparisons these are nominal, uncorrected p-values. "
             f"Source: shortcut_matched_runs.csv (scores, epochs); direct-edge counts: "
             f"operators_pathway/shortcut_match_report.json.",
             ha="center", va="bottom", fontsize=8.0, color="#666666", linespacing=1.6)

    fig.subplots_adjust(left=0.075, right=0.99, top=0.705, bottom=0.345, wspace=0.24)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor="white")
    plt.close(fig)

    # ---- console echo of everything drawn, so the figure can be checked against the CSV ----
    for p in packed:
        S = p["S"]
        paired = S[:, [0]] - S
        print(f"{p['region']}×{p['task']}  n={len(p['seeds'])}  handicap={hand[p['region']]['x']:.3g}x")
        for j, a in enumerate(ARMS):
            print(f"   {a:11s} mean={S[:,j].mean():.6f} sd={S[:,j].std(ddof=1):.6f}")
        for j, lab in ((1, "vs degree   "), (2, "vs degree_sm")):
            lo, hi = boot_ci(S, lambda B, j=j: float((B[:, 0] - B[:, j]).mean()))
            print(f"   {lab} +{paired[:,j].mean():.6f}  CI[{lo:+.6f},{hi:+.6f}] "
                  f"({int((paired[:,j]>0).sum())}/{len(p['seeds'])} seeds) "
                  f"wilcoxon p={wilcoxon_exact(paired[:,j]):.4f}")
        lo, hi = boot_ci(S, lambda B: float((B[:, 2] - B[:, 1]).mean()))
        flo, fhi = boot_ci(S, lambda B: float((B[:, 2] - B[:, 1]).mean() / (B[:, 0] - B[:, 1]).mean())
                           if (B[:, 0] - B[:, 1]).mean() else np.nan)
        print(f"   ctl shift    {S[:,2].mean()-S[:,1].mean():+.6f}  CI[{lo:+.6f},{hi:+.6f}] "
              f"({int((S[:,2]-S[:,1]>0).sum())}/{len(p['seeds'])}) "
              f"wilcoxon p={wilcoxon_exact(S[:,2]-S[:,1]):.4f}")
        print(f"   frac margin  {(S[:,2].mean()-S[:,1].mean())/paired[:,1].mean():.3f}  "
              f"CI[{flo:.3f},{fhi:.3f}]")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
