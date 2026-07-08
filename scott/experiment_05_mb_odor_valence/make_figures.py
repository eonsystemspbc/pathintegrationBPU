#!/usr/bin/env python3
"""Figures for Experiment 5 -- biological MB I/O on odor->valence (Phase 2).

Reads outputs/analysis.json and renders three figures (validated dataviz palette):
  fig1_paradigms   -- pooled recall per learning paradigm on the connectome (which paradigm
                      solves the ALIGNED task), with chance line.
  fig2_wiring      -- connectome vs degree-matched control per paradigm (THE Phase-2 question:
                      does biological wiring help when the task fits the circuit?), with
                      permutation-p; contrast with Exp 4's "no".
  fig3_reversal    -- initial-recall vs after-reversal accuracy per paradigm (where the
                      error-correcting delta rule should beat plain Hebbian).
  fig4_effect_size -- the honest Q2 read: effect size (connectome-control, in control-graph SD)
                      behind every 'win', since all wins report the same permutation floor
                      p=0.0476. Same p, different stories (backprop loses; hybrid at ceiling;
                      only delta-reversal is a substantial outlier).
  fig5_io_bottleneck -- Q3 (descriptive): biological-port I/O vs generic all-neuron I/O for
                      backprop (not a clean control -- generic has ~1.8x params + query bit).
  fig6_learning_curves -- per-rule connectome vs degree-matched-control val_acc-vs-epoch curves
                      (mean +/-1 SD over 20 units, best-hp per unit by validation): the paradigm
                      story as a trajectory -- backprop connectome sits BELOW control, hebbian/delta
                      tie, hybrid solves near-instantly. Reads per-run curves, not analysis.json.

Defensive: only plots paradigms/metrics that exist, so it works on partial (smoke) data too.
fig1-5 read outputs/analysis.json; fig6 reads outputs/runs/*/result.json (per-epoch curves).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent

# validated palette (same as Exp 4 main + subrun): blue / orange / aqua / violet
PARADIGM_ORDER = ("backprop", "hebbian", "delta", "hybrid")
PARADIGM_COLOR = {"backprop": "#eb6834", "hebbian": "#1baf7a", "delta": "#4a3aa7", "hybrid": "#2a78d6"}
CONN_COLOR, CTRL_COLOR = "#2a78d6", "#eb6834"
INK, INK2, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
CHANCE = 0.5

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


def _chance(ax, xr):
    ax.axhline(CHANCE, ls=(0, (4, 3)), color=MUT, lw=1.2, zorder=2)
    ax.text(xr, CHANCE + 0.008, "chance", ha="right", va="bottom", fontsize=9.5, color=MUT)


def fig1_paradigms(A, figdir):
    table = A.get("paradigm_table_connectome", {})
    pars = [p for p in PARADIGM_ORDER if table.get(p, {}).get("test_acc")]
    if not pars:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    xs = range(len(pars))
    ys = [table[p]["test_acc"]["mean"] for p in pars]
    es = [table[p]["test_acc"].get("std", 0.0) for p in pars]
    ax.bar(xs, ys, width=0.62, color=[PARADIGM_COLOR[p] for p in pars], zorder=3,
           yerr=es, error_kw=dict(ecolor=INK2, elinewidth=1.1, capsize=3))
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.012, f"{y:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
    _chance(ax, len(pars) - 0.55)
    ax.set_xticks(list(xs)); ax.set_xticklabels(pars, fontsize=12)
    ax.set_ylim(0, 1.03); ax.set_ylabel("odor→valence recall accuracy")
    _despine(ax, keep=("left",)); ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    _titles(fig, 0.55, "Which learning paradigm solves odor→valence",
            "connectome wiring + biological ports, best-hp per unit · pooled query recall (chance 0.5)")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.80, bottom=0.12)
    fig.savefig(figdir / "fig1_paradigms.png", dpi=200)
    plt.close(fig)


def fig2_wiring(A, figdir):
    comps = A.get("comparisons", {})
    rows = []
    # backprop
    c = comps.get("bptt_connectome_vs_degree__test_acc")
    if c:
        rows.append(("backprop", c))
    for rule in ("hebbian", "delta", "hybrid"):
        c = comps.get(f"plasticity_{rule}_connectome_vs_degree__test_acc")
        if c:
            rows.append((rule, c))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    n = len(rows); w = 0.38
    xs = np.arange(n)
    conn = [r[1].get("connectome_mean", 0) for r in rows]
    ctrl = [r[1].get("control_mean", 0) for r in rows]
    conn_e = [r[1].get("connectome_std", 0) for r in rows]
    ctrl_e = [r[1].get("control_std", 0) for r in rows]
    ax.bar(xs - w / 2, conn, width=w, color=CONN_COLOR, zorder=3, label="connectome",
           yerr=conn_e, error_kw=dict(ecolor=INK2, elinewidth=1.0, capsize=2.5))
    ax.bar(xs + w / 2, ctrl, width=w, color=CTRL_COLOR, zorder=3, label="degree-matched control",
           yerr=ctrl_e, error_kw=dict(ecolor=INK2, elinewidth=1.0, capsize=2.5))
    for i, (_name, c) in enumerate(rows):
        pp = c.get("permutation_p_one_sided")
        top = max(conn[i], ctrl[i]) + max(conn_e[i], ctrl_e[i])
        if pp is not None:
            ax.text(i, top + 0.03, f"p={pp:g}", ha="center", va="bottom", fontsize=8.5, color=MUT)
    _chance(ax, n - 0.55)
    ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows], fontsize=12)
    ax.set_ylim(0, 1.08); ax.set_ylabel("odor→valence recall accuracy")
    _despine(ax, keep=("left",)); ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.07))
    _titles(fig, 0.55, "Does the biological wiring help on the aligned task?",
            "connectome vs degree-matched control, per paradigm · permutation-rank primary")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.18)
    fig.savefig(figdir / "fig2_wiring.png", dpi=200)
    plt.close(fig)


def fig3_reversal(A, figdir):
    table = A.get("paradigm_table_connectome", {})
    pars = [p for p in PARADIGM_ORDER
            if table.get(p, {}).get("test_initial_acc") and table.get(p, {}).get("test_reversed_acc")]
    if not pars:
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    n = len(pars); w = 0.38
    xs = np.arange(n)
    ini = [table[p]["test_initial_acc"]["mean"] for p in pars]
    rev = [table[p]["test_reversed_acc"]["mean"] for p in pars]
    ini_e = [table[p]["test_initial_acc"].get("std", 0) for p in pars]
    rev_e = [table[p]["test_reversed_acc"].get("std", 0) for p in pars]
    # initial = solid paradigm color; reversal = same color, lighter (hatched) so identity holds
    ax.bar(xs - w / 2, ini, width=w, color=[PARADIGM_COLOR[p] for p in pars], zorder=3,
           yerr=ini_e, error_kw=dict(ecolor=INK2, elinewidth=1.0, capsize=2.5))
    ax.bar(xs + w / 2, rev, width=w, color=[PARADIGM_COLOR[p] for p in pars], zorder=3, alpha=0.5,
           hatch="///", edgecolor="white", yerr=rev_e, error_kw=dict(ecolor=INK2, elinewidth=1.0, capsize=2.5))
    for i in range(n):
        ax.text(i - w / 2, ini[i] + 0.012, f"{ini[i]:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.text(i + w / 2, rev[i] + 0.012, f"{rev[i]:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    _chance(ax, n - 0.55)
    ax.set_xticks(xs); ax.set_xticklabels(pars, fontsize=12)
    ax.set_ylim(0, 1.03); ax.set_ylabel("odor→valence recall accuracy")
    _despine(ax, keep=("left",)); ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=MUT, label="initial recall (all odors)"),
                       Patch(facecolor=MUT, alpha=0.5, hatch="///", edgecolor="white",
                             label="reversed odors (after flip)")],
              loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.07))
    _titles(fig, 0.55, "Recall before vs after valence reversal",
            "reversed-odor recall (an association must be overwritten) is where error-correction should win")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.18)
    fig.savefig(figdir / "fig3_reversal.png", dpi=200)
    plt.close(fig)


def fig4_effect_size(A, figdir):
    """The honest Q2 read: every connectome 'win' reports the SAME permutation p=0.0476
    (the 1/(N+1) floor for N=20 — a rank flag, not an effect size). This plots the actual
    effect: (connectome - control) in units of the control-graph SD (the null spread the
    permutation test ranks against), with the absolute Δ accuracy annotated. Same p, wildly
    different stories: backprop loses, hybrid is at ceiling, only delta-on-reversal is a
    substantial topology outlier."""
    comps = A.get("comparisons", {})
    METRIC_LABEL = {"test_acc": "pooled", "test_initial_acc": "initial", "test_reversed_acc": "reversed"}
    keys = [("backprop", "bptt_connectome_vs_degree__{}")]
    keys += [(r, f"plasticity_{r}_connectome_vs_degree__{{}}") for r in ("hebbian", "delta", "hybrid")]
    rows = []  # (paradigm, metric_label, effect_sd, dabs, at_ceiling)
    for par, tmpl in keys:
        for m in ("test_acc", "test_initial_acc", "test_reversed_acc"):
            c = comps.get(tmpl.format(m))
            if not c:
                continue
            dabs = c["connectome_mean"] - c["control_mean"]
            sd = c.get("control_std", 0.0)
            ceil = (c["connectome_mean"] > 0.99 and c["control_mean"] > 0.99)
            eff = dabs / sd if sd > 1e-6 else 0.0
            rows.append((par, METRIC_LABEL[m], eff, dabs, ceil, sd))
    if not rows:
        return
    rows = rows[::-1]  # barh plots bottom-up; keep backprop at top
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    ys = np.arange(len(rows))
    effs = [r[2] for r in rows]
    colors = [CONN_COLOR if e > 0 else CTRL_COLOR for e in effs]
    bars = ax.barh(ys, effs, height=0.66, color=colors, zorder=3)
    for i, (par, ml, eff, dabs, ceil, sd) in enumerate(rows):
        # bars at ceiling: mute + note (no headroom for topology to express)
        if ceil:
            bars[i].set_alpha(0.32); bars[i].set_hatch("///"); bars[i].set_edgecolor("white")
        off = 0.18 if eff >= 0 else -0.18
        ha = "left" if eff >= 0 else "right"
        lab = f"Δ={dabs:+.4f}" + ("  (ceiling)" if ceil else "")
        ax.text(eff + off, i, lab, ha=ha, va="center", fontsize=8.6, color=INK2)
    ax.axvline(0, color=INK2, lw=1.1, zorder=4)
    # paradigm group labels on the left
    yt, ytl = [], []
    for i, (par, ml, *_ ) in enumerate(rows):
        ytl.append(ml); yt.append(i)
    ax.set_yticks(yt); ax.set_yticklabels(ytl, fontsize=10)
    # paradigm brackets (one label per group of 3)
    seen = {}
    for i, r in enumerate(rows):
        seen.setdefault(r[0], []).append(i)
    for par, idxs in seen.items():
        ax.text(-0.085, np.mean(idxs), par, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=11.5, fontweight="bold",
                color=PARADIGM_COLOR.get(par, INK))
    xmax = max(abs(min(effs)), abs(max(effs))) * 1.25 + 1
    ax.set_xlim(-xmax, xmax)
    ax.set_xlabel("connectome − control   (in control-graph SD)")
    _despine(ax, keep=("bottom",)); ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    # directional hints in the top corners (clear of the x-axis tick labels)
    ytop = len(rows) - 0.4
    ax.text(xmax * 0.55, ytop, "connectome better →", ha="center", va="center", fontsize=9.5, color=CONN_COLOR)
    ax.text(-xmax * 0.55, ytop, "← control better", ha="center", va="center", fontsize=9.5, color=CTRL_COLOR)
    ax.set_ylim(-0.7, len(rows) - 0.05)
    _titles(fig, 0.55, "Same p-value (0.0476), different stories",
            "effect size behind every connectome 'win' · all plasticity wins sit at the permutation floor")
    fig.subplots_adjust(left=0.20, right=0.96, top=0.80, bottom=0.15)
    fig.savefig(figdir / "fig4_effect_size.png", dpi=200)
    plt.close(fig)


def fig5_io_bottleneck(A, figdir):
    """Q3 (descriptive): restricting backprop I/O to the biological ports collapses recall.
    NOT a controlled comparison — generic_io also has ~1.8x trainable params, dense all-neuron
    I/O, and the extra query bit; labelled as such."""
    c = A.get("comparisons", {}).get("bptt_bio_vs_generic__test_acc")
    if not c:
        return
    bio = c.get("bio_connectome_mean"); gen = c.get("generic_io_mean")
    if bio is None or gen is None:
        return
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    xs = [0, 1]; ys = [gen, bio]
    cols = [MUT, PARADIGM_COLOR["backprop"]]
    ax.bar(xs, ys, width=0.58, color=cols, zorder=3)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.014, f"{y:.3f}", ha="center", va="bottom", fontsize=11, color=INK)
    ax.annotate("", xy=(1, bio + 0.05), xytext=(1, gen - 0.01),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.3))
    ax.text(1.06, (bio + gen) / 2, f"−{gen - bio:.2f}", ha="left", va="center", fontsize=10.5, color=INK2)
    _chance(ax, 1.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(["generic all-neuron I/O", "biological ports\n(ALPN in · MBON out)"], fontsize=11)
    ax.set_ylim(0, 1.08); ax.set_ylabel("odor→valence recall accuracy")
    ax.set_xlim(-0.6, 1.7)
    _despine(ax, keep=("left",)); ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    _titles(fig, 0.55, "Biological-port I/O bottlenecks backprop",
            "backprop only · descriptive (generic_io has ~1.8× params + query bit)")
    fig.subplots_adjust(left=0.13, right=0.95, top=0.80, bottom=0.14)
    fig.savefig(figdir / "fig5_io_bottleneck.png", dpi=200)
    plt.close(fig)


def _load_runs(outdir):
    """Every per-run result.json (carries per-epoch 'curve' = val_acc, plus val_acc for hp-select)."""
    rd = outdir / "runs"
    rows = []
    if not rd.exists():
        return rows
    for p in sorted(rd.glob("*/result.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            pass
    return rows


def _rule_of(r):
    """Normalise a run to a paradigm label, or None to skip.
    bptt connectome/degree_matched -> 'backprop' (generic_io excluded: not a degree-matched control);
    plasticity -> its rule (hebbian/delta/hybrid)."""
    if r.get("arm") == "bptt":
        return "backprop" if r.get("condition") in ("connectome", "degree_matched") else None
    if r.get("arm") == "plasticity":
        return r.get("rule")
    return None


def _best_hp_curves(rows, rule, condition, val_key="val_acc"):
    """One curve per unit: the hp with the highest validation acc (mirrors the analysis'
    best-hp-per-unit-by-val test_acc selection). Returns list of curves (lists of val_acc)."""
    best = {}  # unit -> (val, curve)
    for r in rows:
        if _rule_of(r) != rule or r.get("condition") != condition:
            continue
        curve = r.get("curve")
        if not curve:
            continue
        v = r.get(val_key)
        v = v if v is not None else r.get("best_val_acc", -1.0)
        u = int(r.get("unit", -1))
        if u not in best or v > best[u][0]:
            best[u] = (v, curve)
    return [c for _v, c in best.values()]


def _mean_sd_padded(curves, L):
    """Pad each curve forward (hold last value) to length L, then mean/SD over units.
    Forward-hold is correct for converged-stop runs (hybrid stops at ~ep6 already at its plateau)."""
    if not curves:
        return None, None
    A = np.array([c[:L] + [c[min(len(c), L) - 1]] * (L - min(len(c), L)) for c in curves], dtype=float)
    return A.mean(0), A.std(0)


def fig6_learning_curves(rows, figdir):
    """Per-rule connectome vs degree-matched-control learning curves (val_acc vs epoch),
    mean ±1 SD over the 20 units, best-hp-per-unit by validation. Shows the paradigm story as a
    trajectory: backprop's connectome sits BELOW control; hebbian/delta tie; hybrid solves near-
    instantly (converged-stop ~ep6, value held forward). Reads per-run curves, not analysis.json."""
    present = [p for p in PARADIGM_ORDER if _best_hp_curves(rows, p, "connectome")]
    if not present:
        return
    L = max(len(c) for p in present for cond in ("connectome", "degree_matched")
            for c in _best_hp_curves(rows, p, cond)) or 1
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6), squeeze=False)
    for ax, rule in zip(axes.flat, PARADIGM_ORDER):
        finals, medlen = {}, {}
        for cond, col, lab in (("connectome", CONN_COLOR, "connectome"),
                               ("degree_matched", CTRL_COLOR, "degree-matched control")):
            curves = _best_hp_curves(rows, rule, cond)
            if not curves:
                continue
            m, sd = _mean_sd_padded(curves, L)
            x = np.arange(1, L + 1)
            ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.15, lw=0, zorder=1)
            ax.plot(x, m, color=col, lw=2, zorder=3, label=lab)
            finals[cond] = float(m[-1]); medlen[cond] = int(np.median([len(c) for c in curves]))
        _chance(ax, L)
        ax.set_ylim(0.45, 1.03); ax.set_xlim(1, L)
        ax.set_xlabel("epoch", fontsize=10.5); ax.set_ylabel("val accuracy", fontsize=10.5)
        ax.set_title(rule, fontsize=13, color=INK, fontweight="bold", pad=7)
        # direct end-of-line value labels (nudged apart when the two curves finish close)
        cm, km = finals.get("connectome"), finals.get("degree_matched")
        yc, yk = cm, km
        if cm is not None and km is not None and abs(cm - km) < 0.035:
            yc, yk = max(cm, km) + 0.028, min(cm, km) - 0.028
        for val, ypos, col in ((cm, yc, CONN_COLOR), (km, yk, CTRL_COLOR)):
            if val is not None:
                ax.text(L - 3, ypos, f"{val:.3f}", color=col, fontsize=10, ha="right",
                        va="center", fontweight="bold",
                        bbox=dict(fc=SURF, ec="none", alpha=0.75, pad=0.4), zorder=6)
        if rule == "hybrid" and medlen.get("connectome"):
            ax.text(0.5, 0.90, f"converged-stop ~ep{medlen['connectome']} (value held forward)",
                    transform=ax.transAxes, ha="center", va="top", fontsize=9, color=MUT, style="italic")
        _despine(ax); ax.grid(True, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    axes.flat[0].legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9.5)
    fig.suptitle("Learning curves: connectome vs degree-matched control, per learning rule",
                 y=0.975, fontsize=15.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.935, "odor→valence · biological ports · mean ±1 SD over 20 units, "
             "best-hp per unit by validation", ha="center", va="top", fontsize=10.5, color=MUT)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.885, bottom=0.075, hspace=0.30, wspace=0.18)
    fig.savefig(figdir / "fig6_learning_curves.png", dpi=200)
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
    fig1_paradigms(A, figdir)
    fig2_wiring(A, figdir)
    fig3_reversal(A, figdir)
    fig4_effect_size(A, figdir)
    fig5_io_bottleneck(A, figdir)
    fig6_learning_curves(_load_runs(outdir), figdir)
    print(f"wrote figures to {figdir} (from {aj})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
