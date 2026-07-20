#!/usr/bin/env python3
"""Does removing a control's free shortcuts help it in proportion to how many it had?

THE ARGUMENT. A degree-preserving shuffle is the standard "is the WIRING special?" control, but on a
LAYERED circuit it manufactures direct input->output edges the real wiring forbids — an express lane
past the computation. The size of that handicap differs enormously by region (MB 83.8x, CX 21.7x,
AL 1.19x). If the handicap is what the connectome was really beating, then repairing it (arm
`degree_sm`: the same degree shuffle, then degree-preserving double-edge swaps until the direct
in->out count matches the connectome's own, degrees preserved EXACTLY) should LIFT the control by an
amount that tracks the handicap — a lot for MB, essentially nothing for AL. AL is the built-in
control-for-the-control.

WHAT THE DATA ACTUALLY SUPPORT — and this figure is built to show the weakness, not hide it:
  * AL x flow (1.19x):  control shift +0.0010 +- 0.0016 SEM, n=6, t=0.60, p=0.58.  Consistent with 0.
  * MB x mqar (83.8x):  control shift +0.0072 +- 0.0055 SEM, n=6, t=1.31, p=0.25.  ALSO consistent
    with 0.  Its 95% CI (-0.0070, +0.0214) contains zero, only 4/6 seeds are positive, and ONE seed
    (seed 0, +0.0337) supplies 78% of the mean.  That seed's `degree` run early-stopped at 31 epochs
    versus 40 for its `degree_sm` partner, so its pair is not even budget-matched.  Leave it out and
    the MB shift is +0.0019 — 2.0x the AL shift, not 7.5x.
  * The two shifts do not differ from each other (Welch t=1.09, p=0.32).
So the mean pattern is in the predicted direction and nothing more.  The headline ratio ("7x") and
the "absorbs 70% of the connectome's margin" arithmetic are ratios of quantities that are individually
indistinguishable from zero; both are printed with that caveat attached, or not printed at all.

WHAT THIS FIGURE DELIBERATELY DOES NOT DO. Two points define a line, so no line is drawn, no slope is
fitted and no correlation is quoted. Error bars are 95% t CIs (not SEM) so the reader can see both
intervals cross zero. The full y range is shown — no seed is placed off-scale — so seed 0's leverage
on the MB mean is visible rather than described. CX (21.7x) is an empty placeholder on the x axis: it
is the pending third point, and the only one that can actually test the claim.

CAVEAT OF RECORD: n=6 is six TRAINING seeds on ONE connectome per region. There is no biological
replicate here, so nothing in this figure generalises beyond these two circuits.

All numbers are read from shortcut_matched_runs.csv (per-run scores; higher is better) and
operators_pathway/shortcut_match_report.json (direct in->out edge counts). Nothing is hard-coded.

Usage:  python fig_dose_response.py   ->  figures/fig_dose_response.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RUNS = HERE / "shortcut_matched_runs.csv"
REPORT = HERE / "operators_pathway" / "shortcut_match_report.json"
OUT = HERE / "figures" / "fig_dose_response.png"

# Okabe-Ito, colour-blind safe.
C_MEASURED = {"AL": "#0072B2", "MB": "#D55E00"}  # blue, vermillion
C_PENDING = "#7F7F7F"
INK = "#1a1a1a"
MUTED = "#5c5c5c"


def load_handicaps() -> dict[str, tuple[int, int, float]]:
    """direct in->out edges in a degree shuffle / in the real connectome, per region (seed 0)."""
    rep = json.loads(REPORT.read_text())
    out = {}
    for reg, d in rep.items():
        shuffled = d["per_seed"][0][0]  # before repair, seed 0
        real = d["connectome_direct"]
        out[reg] = (shuffled, real, shuffled / real)
    return out


def load_shifts(runs: pd.DataFrame) -> pd.DataFrame:
    """Seed-paired control shift: degree_sm - degree, per cell, with honest uncertainty."""
    rows = []
    for (task, region), g in runs.groupby(["task", "region"]):
        w = g.pivot_table(index="seed", columns="arm", values="score")
        ep = g.pivot_table(index="seed", columns="arm", values="epochs")
        if not {"degree", "degree_sm", "connectome"} <= set(w.columns):
            continue
        w = w.dropna(subset=["degree", "degree_sm", "connectome"])
        if w.empty:
            continue
        d_ctl = (w["degree_sm"] - w["degree"]).to_numpy()
        d_con = (w["connectome"] - w["degree"]).to_numpy()
        n = len(d_ctl)
        sem = d_ctl.std(ddof=1) / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, n - 1)
        t1 = stats.ttest_1samp(d_ctl, 0.0)
        # leverage of the single most extreme seed
        k = int(np.argmax(np.abs(d_ctl)))
        loo = np.delete(d_ctl, k)
        rows.append(
            dict(
                task=task, region=region, n=n,
                ctl_shift=d_ctl.mean(), ctl_sem=sem, ci=tcrit * sem,
                t=t1.statistic, p=t1.pvalue,
                n_pos=int((d_ctl > 0).sum()),
                per_seed=d_ctl, seeds=w.index.to_numpy(),
                lev_seed=int(w.index[k]), lev_val=d_ctl[k],
                lev_frac=d_ctl[k] / (n * d_ctl.mean()) if d_ctl.mean() else np.nan,
                loo_mean=loo.mean(),
                lev_ep_deg=float(ep["degree"].iloc[k]), lev_ep_sm=float(ep["degree_sm"].iloc[k]),
                vs_degree=d_con.mean(),
                p_vs_degree=stats.ttest_1samp(d_con, 0.0).pvalue,
                vs_degree_sm=(w["connectome"] - w["degree_sm"]).mean(),
                p_vs_degree_sm=stats.ttest_1samp(
                    (w["connectome"] - w["degree_sm"]).to_numpy(), 0.0).pvalue,
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    for f in (RUNS, REPORT):
        if not f.exists():
            sys.exit(f"missing {f}")
    runs = pd.read_csv(RUNS)
    hand = load_handicaps()
    S = load_shifts(runs)
    if S.empty:
        sys.exit("no complete cells (need connectome + degree + degree_sm) in the runs file")
    S["handicap"] = S.region.map(lambda r: hand[r][2])
    S = S.sort_values("handicap").reset_index(drop=True)
    measured = set(S.region)
    pending = [r for r in ("MB", "CX", "AL") if r not in measured and r in hand]

    lo = S.iloc[0]
    hi = S.iloc[-1]
    between = stats.ttest_ind(hi.per_seed, lo.per_seed, equal_var=False)

    fig, ax = plt.subplots(figsize=(10.6, 7.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_xscale("log")
    ax.axhline(0, color="#8a8a8a", lw=1.2, zorder=1)
    ax.grid(True, which="major", axis="y", color="#e6e6e6", lw=0.8, zorder=0)
    ax.grid(True, which="both", axis="x", color="#f2f2f2", lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    # full range of the data: nothing is ever pushed off-scale
    allv = np.concatenate([np.asarray(r.per_seed) for _, r in S.iterrows()])
    locis = np.array([r.ctl_shift - r.ci for _, r in S.iterrows()])
    hicis = np.array([r.ctl_shift + r.ci for _, r in S.iterrows()])
    ymin = min(allv.min(), locis.min()) - 0.0035
    ymax = max(allv.max(), hicis.max()) + 0.0055

    # ---- measured points -----------------------------------------------------------------
    for _, r in S.iterrows():
        col = C_MEASURED.get(r.region, "#009E73")
        jit = r.handicap * np.exp(np.linspace(-0.11, 0.11, r.n))
        ax.scatter(jit, r.per_seed, s=30, facecolor=col, edgecolor="white",
                   linewidth=0.8, alpha=0.45, zorder=3)
        # 95% t CI, not SEM: the reader must be able to see that it spans zero
        ax.errorbar(r.handicap, r.ctl_shift, yerr=r.ci, fmt="none",
                    ecolor=col, elinewidth=2, capsize=6, capthick=2, alpha=0.9, zorder=4)
        ax.scatter([r.handicap], [r.ctl_shift], s=210, facecolor=col,
                   edgecolor="white", linewidth=2.0, zorder=5)
        # leave-one-out mean (drop the single most extreme seed) as a hollow ghost
        ax.scatter([r.handicap * 1.30], [r.loo_mean], s=110, facecolor="white",
                   edgecolor=col, linewidth=1.6, zorder=5)
        ax.annotate("", xy=(r.handicap * 1.30, r.loo_mean), xytext=(r.handicap, r.ctl_shift),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.0, alpha=0.5,
                                    shrinkA=9, shrinkB=7), zorder=4)

    ax.annotate(
        f"{hi.region} × {hi.task}   {hi.handicap:.1f}× handicap\n"
        f"control shift {hi.ctl_shift:+.4f}   95% CI [{hi.ctl_shift - hi.ci:+.4f}, "
        f"{hi.ctl_shift + hi.ci:+.4f}]\n"
        f"n={hi.n} seeds, t={hi.t:.2f}, p={hi.p:.2f} — NOT distinguishable from zero\n"
        f"only {hi.n_pos}/{hi.n} seeds positive; seed {hi.lev_seed} ({hi.lev_val:+.4f}) alone is "
        f"{100 * hi.lev_frac:.0f}% of the mean\n"
        f"and its degree run early-stopped at {hi.lev_ep_deg:.0f} epochs vs "
        f"{hi.lev_ep_sm:.0f} for its pair\n"
        f"drop that one seed (○) and the shift is {hi.loo_mean:+.4f}",
        xy=(hi.handicap, hi.ctl_shift + hi.ci), xytext=(1.03, ymax - 0.0010),
        fontsize=9, color=C_MEASURED.get(hi.region, INK), ha="left", va="top",
        arrowprops=dict(arrowstyle="-", color=C_MEASURED.get(hi.region, INK), lw=1.1,
                        shrinkA=6, shrinkB=6, alpha=0.55),
    )
    ax.annotate(
        f"{lo.region} × {lo.task}   {lo.handicap:.2f}× handicap\n"
        f"control shift {lo.ctl_shift:+.4f}   95% CI [{lo.ctl_shift - lo.ci:+.4f}, "
        f"{lo.ctl_shift + lo.ci:+.4f}]\n"
        f"n={lo.n} seeds, t={lo.t:.2f}, p={lo.p:.2f} — consistent with zero,\n"
        f"as predicted: there were essentially no shortcuts to take away",
        xy=(lo.handicap, lo.ctl_shift + lo.ci), xytext=(1.55, 0.0125),
        fontsize=9, color=C_MEASURED.get(lo.region, INK), ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=C_MEASURED.get(lo.region, INK), lw=1.1,
                        shrinkA=4, shrinkB=10, alpha=0.55),
    )

    # ---- the comparison the figure is really about, stated as null ------------------------
    ax.text(
        1.03, 0.0245,
        f"The two shifts are NOT significantly different from each other\n"
        f"(Welch t={between.statistic:.2f}, p={between.pvalue:.2f}).  The ratio of the means is "
        f"{hi.ctl_shift / lo.ctl_shift:.1f}×,\nbut it is {hi.loo_mean / lo.ctl_shift:.1f}× with "
        f"{hi.region} seed {hi.lev_seed} removed — a ratio of two numbers that\nindividually cannot "
        f"be told from zero is not a measurement.",
        fontsize=8.8, color=INK, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fbfbfb", edgecolor="#d0d0d0", lw=0.9),
    )

    # ---- pending placeholder(s) on the x axis ---------------------------------------------
    for reg in pending:
        hx = hand[reg][2]
        ax.axvline(hx, color=C_PENDING, lw=1.1, ls=(0, (4, 4)), alpha=0.6, zorder=2)
        ax.scatter([hx], [0.0], s=200, facecolor="white", edgecolor=C_PENDING,
                   linewidth=1.8, linestyle="--", zorder=5)
        ax.annotate(
            f"{reg}   {hx:.1f}×   NOT YET RUN\nno y value — this is the third point\n"
            f"that would actually test the trend",
            xy=(hx, 0.0), xytext=(hx * 0.93, -0.0014),
            fontsize=9, color=C_PENDING, ha="right", va="top", style="italic",
        )

    ax.set_xlim(0.95, 300)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(
        "Handicap removed  =  direct input→output edges in the degree shuffle ÷ in the real "
        "connectome\n(dimensionless ratio, log scale; 1× = the shuffle invented no extra shortcuts)",
        fontsize=10, color=INK, labelpad=8,
    )
    ax.set_ylabel(
        "How much the CONTROL moved\nshortcut-matched − degree shuffle  (task score units,\n"
        "seed-paired mean ± 95% CI; higher score = better)",
        fontsize=10, color=INK,
    )
    ax.set_title(
        "Repairing the control moves MB and not AL — in the predicted direction,\n"
        "but 6 seeds on one connectome per region cannot yet tell either shift from zero",
        fontsize=12.5, fontweight="bold", color=INK, pad=26,
    )
    ax.text(
        0.5, 1.012,
        "Two points only — no line is fitted and no correlation is quoted; a monotone pair is "
        "consistent with a dose-response, not evidence of one.",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8.8, color=MUTED,
    )
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}×"))
    ax.minorticks_off()
    ax.tick_params(labelsize=9, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b0b0b0")

    faint = "; ".join(
        f"{r.region}×{r.task} n={r.n} training seeds" for _, r in S.iterrows())
    fig.text(
        0.5, -0.075,
        f"Filled ● = seed-paired mean (bar = 95% t CI); small dots = the individual seed-paired "
        f"differences; hollow ○ = mean after dropping the single most extreme seed;\n"
        f"open grey = pending, no y value. Measured: {faint} — six TRAINING seeds on ONE connectome "
        f"per region, so this is not a biological replicate.\n"
        f"Degrees preserved exactly in both control arms. "
        f"Source: shortcut_matched_runs.csv, operators_pathway/shortcut_match_report.json.",
        ha="center", fontsize=8.2, color=MUTED,
    )

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")
    for _, r in S.iterrows():
        print(f"  {r.region}×{r.task}: handicap {r.handicap:.2f}x  shift {r.ctl_shift:+.6f} "
              f"±{r.ci:.6f} (95% CI)  t={r.t:.2f} p={r.p:.3f}  {r.n_pos}/{r.n} pos  "
              f"loo={r.loo_mean:+.6f} (drop seed {r.lev_seed})  "
              f"conn-degree {r.vs_degree:+.6f} p={r.p_vs_degree:.3f}  "
              f"conn-degree_sm {r.vs_degree_sm:+.6f} p={r.p_vs_degree_sm:.3f}")
    print(f"  between-cell Welch t={between.statistic:.2f} p={between.pvalue:.3f}")
    for reg in pending:
        print(f"  {reg}: handicap {hand[reg][2]:.2f}x  (pending)")


if __name__ == "__main__":
    main()
