#!/usr/bin/env python3
"""Experiment 4 — figure generation (clean, communication-first set).

Five figures, one consistent color code throughout:
    BLUE   = biological MB I/O (ALPN in / MBON out — the Exp-4 regime)
    ORANGE = generic all-neuron I/O (the Exp 1-3 regime)

Palette validated with the dataviz skill's validator (CVD ΔE 96.7, all checks pass).

Fig 1  paradigm_comparison            — within Exp 4: how the network learns >> how it's wired
Fig 2  io_bottleneck                  — why backprop fails: bio vs generic I/O on the SAME graph
Fig 3  advantage_across_experiments   — cross-experiment: connectome edge vanishes under bio I/O
Fig 4  accuracy_vs_cost               — fly-like learning is more accurate AND cheaper
Fig 5  learning_curves                — convergence dynamics (log epoch axis)

Fig 1/2/5 read Exp-4 numbers/curves straight from outputs/. Fig 3/4 carry cross-experiment
constants (final concluded values), cited inline with provenance. All best-hp-per-unit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)

# ---- consistent palette (dataviz-validated) ----------------------------------------------
BIO = "#2a78d6"     # biological MB I/O
GEN = "#eb6834"     # generic all-neuron I/O
INK = "#0b0b0b"     # primary text
INK2 = "#52514e"    # secondary text
MUT = "#898781"     # muted axis / reference lines
GRID = "#e1e0d9"    # hairline grid
SURF = "#ffffff"
CHANCE = 0.0312

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12, "axes.edgecolor": MUT, "axes.linewidth": 0.9,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
})


def _despine(ax, keep=("bottom", "left")):
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(s in keep)


def _titles(fig, cx, title, sub):
    """Centered title + muted subtitle in FIGURE coords (cx = center of the plot area),
    so neither can clip on a wide label gutter."""
    fig.text(cx, 0.955, title, ha="center", va="top", fontsize=16, fontweight="bold", color=INK)
    fig.text(cx, 0.887, sub, ha="center", va="top", fontsize=11, color=MUT)


def _analysis():
    return json.load(open(OUT / "analysis.json"))


def _mean_curve(arm, rule, condition, hp, L=300):
    """Mean per-epoch val-acc curve across units for one (arm, rule, condition, hp) group.
    Shorter (early-converged) curves are padded with their final value to length L."""
    curves = []
    for rj in (OUT / "runs").glob("*/result.json"):
        d = json.loads(rj.read_text())
        if (d.get("arm") == arm and d.get("rule") == rule and d.get("condition") == condition
                and d.get("hp") is not None and abs(float(d["hp"]) - hp) < 1e-9):
            c = d.get("curve")
            if c:
                curves.append([float(x) for x in c])
    if not curves:
        return None
    padded = []
    for c in curves:
        cc = c[:L]
        if len(cc) < L:
            cc = cc + [cc[-1]] * (L - len(cc))
        padded.append(cc)
    return np.asarray(padded).mean(axis=0)


# ==========================================================================================
# Fig 1 — paradigm comparison (within Exp 4, biological I/O, connectome substrate)
# ==========================================================================================
def fig1_paradigm():
    a = _analysis()["paradigm_table_connectome_test_acc"]
    rows = [("hybrid", a["hybrid"]), ("delta", a["delta"]),
            ("hebbian", a["hebbian"]), ("backprop", a["backprop"])]
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ys = list(range(len(rows)))[::-1]
    for y, (name, st) in zip(ys, rows):
        val, sd = st["mean"], st["std"]
        ax.barh(y, val, height=0.58, color=BIO, zorder=3,
                xerr=sd, error_kw=dict(ecolor=INK2, elinewidth=1.4, capsize=4, capthick=1.4))
        ax.text(val + sd + 0.016, y, f"{val:.3f}", va="center", ha="left",
                fontsize=12.5, color=INK, fontweight="bold")
        ax.text(-0.02, y, name, va="center", ha="right", fontsize=13.5,
                color=INK, fontweight="bold")

    ax.axvline(CHANCE, ls=(0, (4, 3)), color=MUT, lw=1.3, zorder=2)
    ax.text(CHANCE, -0.55, "chance", ha="center", va="bottom", fontsize=10, color=MUT)
    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.7, len(rows) - 0.25)
    ax.set_yticks([])
    ax.set_xlabel("MQAR recall accuracy")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _despine(ax, keep=("bottom",))
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    _titles(fig, 0.57, "How the network learns beats how it's wired",
            "Four learning rules · same MB circuit + biological ports (ALPN in, MBON out)")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.80, bottom=0.14)
    fig.savefig(FIGDIR / "fig1_paradigm_comparison.png", dpi=200)
    plt.close(fig)


# ==========================================================================================
# Fig 2 — the I/O bottleneck: same connectome graph, backprop, two I/O regimes
# ==========================================================================================
def fig2_bottleneck():
    a = _analysis()["comparisons"]["bptt_bio_vs_generic__test_acc"]
    bio, gen = a["bio_connectome_mean"], a["generic_io_mean"]
    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    ax.bar(0, gen, width=0.58, color=GEN, zorder=3)
    ax.bar(1, bio, width=0.58, color=BIO, zorder=3)
    for x, v in [(0, gen), (1, bio)]:
        ax.text(x, v + 0.018, f"{v:.3f}", ha="center", va="bottom",
                fontsize=13.5, color=INK, fontweight="bold")
    ax.axhline(CHANCE, ls=(0, (4, 3)), color=MUT, lw=1.3, zorder=2)
    ax.text(-0.62, CHANCE + 0.008, "chance", ha="left", va="bottom", fontsize=10, color=MUT)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["generic\nall-neuron I/O", "biological ports\n(ALPN → MBON)"], fontsize=12)
    ax.set_ylim(0, 1.03)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylabel("MQAR recall accuracy")
    _despine(ax, keep=("left",))
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    _titles(fig, 0.55, "The biological I/O bottleneck breaks backprop",
            "Same connectome graph, backprop — only the I/O ports change")
    fig.subplots_adjust(left=0.13, right=0.97, top=0.80, bottom=0.13)
    fig.savefig(FIGDIR / "fig2_io_bottleneck.png", dpi=200)
    plt.close(fig)


# ==========================================================================================
# Fig 3 — connectome advantage across experiments (Δ = connectome − degree-matched control)
# ==========================================================================================
def fig3_across():
    # provenance: Exp1 experiment_01.../subruns/03_full_fleet/outputs/analysis.json test_acc;
    #  Exp2 experiment_02.../outputs/analysis.json core_vs_core_degree.test_acc; Exp4 outputs/analysis.json
    rows = [  # (label, connectome, control, perm_p, regime)
        ("Exp 1 · backprop", 0.9182, 0.7689, 0.048, "gen"),
        ("Exp 2 · backprop (core)", 0.8807, 0.7009, 0.048, "gen"),
        ("Exp 4 · backprop", 0.1776, 0.1672, 0.095, "bio"),
        ("Exp 4 · hybrid", 0.9993, 0.9984, 0.19, "bio"),
        ("Exp 4 · delta", 0.3699, 0.4034, 1.0, "bio"),
        ("Exp 4 · hebbian", 0.3694, 0.4029, 1.0, "bio"),
    ]
    fig, ax = plt.subplots(figsize=(9.8, 5.1))
    ys = list(range(len(rows)))[::-1]
    for y, (label, conn, ctrl, p, reg) in zip(ys, rows):
        d = conn - ctrl
        color = GEN if reg == "gen" else BIO
        ax.barh(y, d, height=0.6, color=color, zorder=3)
        ptxt = f"p={p:g}" + (" *" if p < 0.05 else "")
        if d >= 0:
            ax.text(d + 0.006, y, f"+{d:.3f}   {ptxt}", va="center", ha="left", fontsize=10.5, color=INK)
        else:  # negative bars are short — annotate on the right of zero to clear the row label
            ax.text(0.006, y, f"{d:.3f}   {ptxt}", va="center", ha="left", fontsize=10.5, color=INK)
        ax.text(-0.088, y, label, va="center", ha="right", fontsize=12, color=INK)

    ax.axvline(0, color=INK2, lw=1.5, zorder=4)
    ax.set_xlim(-0.088, 0.285)
    ax.set_ylim(-0.9, len(rows) - 0.25)
    ax.set_yticks([])
    ax.set_xlabel("connectome advantage   (Δ recall  vs  degree-matched control)")
    ax.set_xticks([-0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25])
    _despine(ax, keep=("bottom",))
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.12, -0.78, "→ connectome better", ha="center", va="center", fontsize=9.8, color=MUT)
    ax.text(-0.045, -0.78, "control better ←", ha="center", va="center", fontsize=9.8, color=MUT)
    ax.legend(handles=[Patch(color=GEN, label="generic all-neuron I/O  (Exp 1–3)"),
                       Patch(color=BIO, label="biological ports  (Exp 4)")],
              loc="lower right", frameon=False, fontsize=10.5, handlelength=1.1)
    _titles(fig, 0.61, "The connectome's edge was a feature of generic I/O",
            "Wiring beats matched controls only when read/write can bypass the ports")
    fig.subplots_adjust(left=0.25, right=0.97, top=0.80, bottom=0.15)
    fig.savefig(FIGDIR / "fig3_advantage_across_experiments.png", dpi=200)
    plt.close(fig)


# ==========================================================================================
# Fig 4 — accuracy vs compute cost
# ==========================================================================================
def fig4_cost():
    # wall = median s/run (metrics_by_run.csv / prior runs); acc = connectome best-hp test acc.
    # Exp-4 (biological I/O, BLUE); Exp 1-2 backprop (generic I/O, ORANGE).
    pts = [  # regime carried by color + legend, so point labels stay short
        ("hybrid", 368.0, 0.9993, BIO, (13, 0), "left"),
        ("delta", 30.4, 0.3699, BIO, (14, 12), "left"),
        ("hebbian", 28.9, 0.3694, BIO, (14, -14), "left"),
        ("backprop · Exp 4", 14356.0, 0.1776, BIO, (0, -22), "center"),
        ("backprop · Exp 1", 9896.0, 0.9182, GEN, (0, 19), "center"),
        ("backprop · Exp 2", 4014.0, 0.8807, GEN, (0, -19), "center"),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    for name, wall, acc, col, (dx, dy), ha in pts:
        ax.scatter(wall, acc, s=185, color=col, zorder=3, edgecolor=SURF, linewidth=1.6)
        ax.annotate(name, (wall, acc), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va="center", fontsize=11.5, color=INK, fontweight="bold")
    ax.axhline(CHANCE, ls=(0, (4, 3)), color=MUT, lw=1.3, zorder=1)
    ax.text(18, CHANCE + 0.012, "chance", ha="left", va="bottom", fontsize=10, color=MUT)
    ax.set_xscale("log")
    ax.set_xlim(16, 30000)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("compute per run   (wall-clock, log scale)")
    ax.set_ylabel("MQAR recall accuracy")
    ax.set_xticks([30, 60, 300, 3600, 14400])
    ax.set_xticklabels(["30 s", "1 min", "5 min", "1 hr", "4 hr"])
    ax.minorticks_off()
    _despine(ax)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=10, color=BIO, label="biological ports  (Exp 4)"),
                       Line2D([], [], marker="o", ls="", ms=10, color=GEN, label="generic I/O backprop  (Exp 1–2)")],
              loc="center left", frameon=False, fontsize=10.5)
    _titles(fig, 0.54, "Fly-like learning is more accurate and cheaper",
            "Biological MB I/O (Exp 4) vs generic-I/O backprop (Exp 1–2)")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.80, bottom=0.13)
    fig.savefig(FIGDIR / "fig4_accuracy_vs_cost.png", dpi=200)
    plt.close(fig)


# ==========================================================================================
# Fig 5 — learning curves (connectome, best hp per paradigm; log epoch axis)
# ==========================================================================================
def fig5_curves():
    L = 300
    x = np.arange(1, L + 1)
    # (label, (arm, rule, condition, hp), color, linestyle, lw)
    dyn = [
        ("hybrid\n(biological I/O)", ("plasticity", "hybrid", "connectome", 0.01), BIO, "-", 2.6),
        ("backprop\n(generic I/O)", ("bptt", None, "generic_io", 0.001), GEN, "-", 2.2),
        ("backprop\n(biological I/O)", ("bptt", None, "connectome", 0.0003), BIO, "--", 2.2),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    for label, key, col, ls, lw in dyn:
        m = _mean_curve(*key, L=L)
        if m is None:
            continue
        ax.plot(x, m, color=col, ls=ls, lw=lw, zorder=3)
        ax.annotate(label, (L, m[-1]), textcoords="offset points", xytext=(9, 0),
                    va="center", ha="left", fontsize=10.5, color=col, fontweight="bold")

    # pure local plasticity is one-shot (no training) → flat reference at its final recall
    delta = _analysis()["paradigm_table_connectome_test_acc"]["delta"]["mean"]
    ax.plot([1, L], [delta, delta], color=MUT, ls=(0, (1, 2)), lw=1.8, zorder=2)
    ax.annotate("delta / hebbian\n(one-shot)", (L, delta), textcoords="offset points",
                xytext=(9, 0), va="center", ha="left", fontsize=10, color=MUT)

    ax.axhline(CHANCE, ls=(0, (4, 3)), color=MUT, lw=1.1, zorder=1)
    ax.text(1.1, CHANCE + 0.012, "chance", ha="left", va="bottom", fontsize=10, color=MUT)
    ax.set_xscale("log")
    ax.set_xlim(1, L)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("training epoch  (log scale)")
    ax.set_ylabel("validation recall accuracy")
    ax.set_xticks([1, 3, 10, 30, 100, 300])
    ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    ax.minorticks_off()
    _despine(ax)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    _titles(fig, 0.41, "The fly-like rule converges in a few epochs",
            "Mean validation curve, connectome, best hyperparameter per paradigm")
    fig.subplots_adjust(left=0.10, right=0.72, top=0.80, bottom=0.13)
    fig.savefig(FIGDIR / "fig5_learning_curves.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1_paradigm()
    fig2_bottleneck()
    fig3_across()
    fig4_cost()
    fig5_curves()
    print("wrote:", *(p.name for p in sorted(FIGDIR.glob("*.png"))))
