#!/usr/bin/env python3
"""cx-02 figures. Four panels carrying the audit's findings:
  fig1  the converge-stop censors the primary metric (why the sweep reads "flat")
  fig2  time-to-criterion -- the one uncensored readout
  fig3  the tempo knob moved amplitude, not bandwidth
  fig4  run coverage (the unsigned x norm=ON arm is absent)
Run: uv run python scott/experiment_cx_02_stimulus_spectrum/make_figures.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# validated categorical palette (see dataviz validator, light mode) + fixed marker order
# as the secondary encoding required by the 6-8 band CVD warn.
BLUE, ORANGE, GREEN = "#3b6ea5", "#c4622d", "#2f8f5b"
CRIT = "#b3352b"
INK, MUTED, GRID = "#22221f", "#6b6b64", "#dedbd4"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150,
})

TEMPOS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.15]
CONVERGE = 0.05


def load() -> list[dict]:
    return [json.load(open(f)) for f in glob.glob(str(HERE / "outputs/runs/*/result.json"))]


def _tempo(r):
    return round(float(r.get("tempo", 1.0)), 4)


def _xaxis(ax, label=True):
    ax.set_xscale("log")
    ax.set_xticks(TEMPOS)
    ax.set_xticklabels([str(t) for t in TEMPOS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())  # kill colliding decade ticks
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.invert_xaxis()
    if label:
        ax.set_xlabel("tempo  (run-length scale)          faster target →")
    ax.grid(True, alpha=0.5, lw=0.6)
    ax.set_axisbelow(True)


def fig1(rows):
    """Every successful run stops at the 0.05 threshold -> the metric cannot show gradation."""
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    conn = [r for r in rows if r.get("condition") == "connectome"]
    jit = np.random.default_rng(0)
    for reason, color, marker, label in [
        ("converged", BLUE, "o", "reached 0.05 criterion (training halted)"),
        ("epoch_cap", CRIT, "X", "hit 300-epoch cap (never reached it)"),
    ]:
        sel = [r for r in conn if r.get("stopped_reason") == reason]
        x = np.array([_tempo(r) for r in sel]) * np.exp(jit.normal(0, 0.035, len(sel)))
        y = [r["test_heading_error"] for r in sel]
        ax.scatter(x, y, c=color, marker=marker, s=26, alpha=0.85, lw=0, label=label, zorder=3)
    ax.axhline(CONVERGE, color=INK, lw=1.2, ls="--", zorder=2)
    ax.text(0.16, CONVERGE * 1.06, "converge-stop threshold = 0.05 rad", fontsize=8,
            color=INK, va="bottom", ha="left")
    ax.set_yscale("log")
    ax.set_ylabel("test heading error (rad, lower = better)")
    ax.set_title("The primary metric is a stopping rule, not a performance level",
                 loc="left", weight="bold")
    _xaxis(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_censored_metric.png", bbox_inches="tight")
    plt.close(fig)


def fig2(rows):
    """Two honest readouts. LEFT: epochs-to-criterion, but only for the two arms where
    essentially every run reaches it (a median of survivors is meaningless at 25% coverage,
    so norm=ON is deliberately NOT drawn as a trend line here). RIGHT: for norm=ON the
    interpretable quantity is the reach RATE itself."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))

    a = axes[0]
    for label, color, marker, pred in [
        ("GRU-256 (dense reference)", BLUE, "o", lambda r: r.get("condition") == "gru_ceiling"),
        ("connectome, normalization OFF", ORANGE, "s",
         lambda r: r.get("condition") == "connectome" and not r.get("normalize")),
    ]:
        sel = [r for r in rows if pred(r)]
        xs, ys, lo, hi = [], [], [], []
        for t in TEMPOS:
            reach = [r["best_epoch"] for r in sel
                     if _tempo(r) == t and r.get("stopped_reason") == "converged"]
            if not reach:
                continue
            xs.append(t); ys.append(np.median(reach))
            lo.append(np.percentile(reach, 25)); hi.append(np.percentile(reach, 75))
        a.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)
        a.plot(xs, ys, color=color, marker=marker, lw=2, ms=6, label=label, zorder=3)
    a.set_ylabel("epochs to reach 0.05 rad  (median, IQR band)")
    a.set_title("Both arms slow down together", loc="left", weight="bold")
    _xaxis(a)
    a.legend(frameon=False, fontsize=8, loc="upper left")

    b = axes[1]
    for label, color, marker, nm in [
        ("normalization OFF", ORANGE, "s", False),
        ("normalization ON (contracting)", GREEN, "^", True),
    ]:
        sel = [r for r in rows if r.get("condition") == "connectome"
               and bool(r.get("normalize")) == nm and r["substrate"] == "signed_full"]
        xs, ys, ns = [], [], []
        for t in TEMPOS:
            cell = [r for r in sel if _tempo(r) == t]
            if not cell:
                continue
            xs.append(t)
            ys.append(100 * sum(1 for r in cell if r.get("stopped_reason") == "converged") / len(cell))
            ns.append(len(cell))
        b.plot(xs, ys, color=color, marker=marker, lw=2, ms=6, label=label, zorder=3)
        for x, y, n in zip(xs, ys, ns):
            b.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, -13),
                       ha="center", fontsize=7, color=color)
    b.set_ylim(-8, 112)
    b.set_ylabel("% of runs reaching 0.05 rad within 300 epochs")
    b.set_title("Only the contracting arm fails — signed_full", loc="left", weight="bold")
    _xaxis(b)
    b.legend(frameon=False, fontsize=8, loc="lower left")

    fig.suptitle("Time-to-criterion: the one uncensored readout", x=0.005, ha="left",
                 weight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "fig2_time_to_criterion.png", bbox_inches="tight")
    plt.close(fig)


def fig3():
    """The knob raised amplitude, not bandwidth. Two measures -> two panels, never a dual axis."""
    import spectrum_task as pt
    frac, power, slew, ac = [], [], [], []
    for t in TEMPOS:
        rng = np.random.default_rng(777)
        vs = pt.speed_scale_for(pt.TaskSpec(T=50, tempo=t))
        th = []
        for _ in range(512):
            c = pt.run_turn_controls(50, rng, tempo=t)
            c[:, 0] *= vs
            th.append(pt.integrate_path_state(c)[0])
        th = np.unwrap(np.array(th), axis=1)
        P = np.abs(np.fft.rfft(th - th.mean(1, keepdims=True), axis=1)) ** 2
        f = np.fft.rfftfreq(th.shape[1])
        frac.append(P[:, f > 0.25].sum() / P.sum())
        power.append(P.sum() / th.shape[0])
        slew.append(np.abs(np.diff(th, axis=1)).mean())
        ac.append(pt.stimulus_spectrum_metrics(pt.TaskSpec(T=50, tempo=t))
                  ["heading_autocorr_time_steps"])

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    a = axes[0]
    a.plot(TEMPOS, np.array(frac) * 100, color=BLUE, marker="o", lw=2, ms=6)
    a.set_ylim(0, 5)
    a.set_ylabel("heading power above 0.25 cyc/step  (% of total)")
    a.set_title("Bandwidth: unchanged", loc="left", weight="bold")
    _xaxis(a)
    a.annotate("flat — the target's spectral\nshape never sped up", (0.35, frac[3] * 100),
               textcoords="offset points", xytext=(0, 34), ha="center", fontsize=8, color=BLUE)

    b = axes[1]
    b.plot(TEMPOS, np.array(slew) / slew[0], color=ORANGE, marker="s", lw=2, ms=6,
           label="mean |Δheading| per step")
    b.plot(TEMPOS, np.array(power) / power[0], color=GREEN, marker="^", lw=2, ms=6,
           label="total heading power")
    b.set_ylabel("fold change vs tempo = 1.0")
    b.set_title("Amplitude: rises ~2.5–2.8×", loc="left", weight="bold")
    _xaxis(b)
    b.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("The tempo knob moved amplitude, not bandwidth", x=0.005, ha="left",
                 weight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig3_manipulation_check.png", bbox_inches="tight")
    plt.close(fig)


def fig4(rows):
    """Completed runs per cell -- sequential single hue (magnitude), 6 planned per cell."""
    conn = [r for r in rows if r.get("condition") == "connectome"]
    arms = [("signed_full", False), ("signed_full", True),
            ("unsigned_full", False), ("unsigned_full", True)]
    M = np.zeros((len(arms), len(TEMPOS)))
    for i, (sub, nm) in enumerate(arms):
        for j, t in enumerate(TEMPOS):
            M[i, j] = sum(1 for r in conn if r["substrate"] == sub
                          and bool(r.get("normalize")) == nm and _tempo(r) == t)
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=6, aspect="auto")
    for i in range(len(arms)):
        for j in range(len(TEMPOS)):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > 3.5 else INK, weight="bold")
    ax.set_xticks(range(len(TEMPOS)), [str(t) for t in TEMPOS])
    ax.set_yticks(range(len(arms)),
                  [f"{s.replace('_full','')}, norm {'ON' if n else 'OFF'}" for s, n in arms])
    ax.set_xlabel("tempo          faster target →")
    ax.set_title("Completed runs per cell (6 planned)", loc="left", weight="bold")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, ticks=[0, 3, 6], label="runs landed")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_coverage.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rows = load()
    fig1(rows)
    fig2(rows)
    fig3()
    fig4(rows)
    print(f"wrote 4 figures to {FIG}")
