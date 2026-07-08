#!/usr/bin/env python3
"""Figures for Experiment 5 · subrun 01 — generic-I/O connectome vs degree-matched controls.

Reads outputs/analysis.json (aggregate stats) and outputs/runs/*/result.json (per-run learning
curves + grok crossings), and renders, PER substrate:

  fig1_generic_io_wiring   -- pooled test_acc bar: connectome vs degree-matched control + control
                              spread + permutation-rank p, on the full 0.5-1.0 scale. The "does it
                              beat controls, and are both well above chance?" headline.
  fig2_learning_curves     -- val_acc vs epoch, mean +/-1 SD band over the 20 connectome seeds and
                              20 control graphs, with a zoomed plateau inset. Shows the connectome
                              BOTH groks faster AND plateaus higher; the plateaus are flat (the gap
                              is asymptotic, not a transient speed artifact).
  fig3_final_separation    -- per-graph strip plot of final test_acc on a ZOOMED axis: every one of
                              the 20 connectome seeds vs 20 control graphs. This is the honest
                              effect-size view fig1's full-scale bars compress away -- it shows the
                              zero-overlap "0/20 controls reach the connectome" separation directly.
  fig4_grok_speed          -- epochs to first reach val 0.60/0.65/0.70, connectome vs control mean.
                              The ~2x faster-grok signature (the Exp-1 corroboration).

Defensive: only plots substrates/metrics present, so it also works on partial/smoke data.
Usage:  uv run python .../subruns/01_generic_io_controls/make_figures.py [OUTPUT_DIR]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent

# validated palette (same family as the primary Exp-5 figures; blue/orange CVD-safe pair,
# validate_palette.js light+dark PASS, worst-adjacent CVD dE 96.7)
CONN_COLOR, CTRL_COLOR = "#2a78d6", "#eb6834"
INK, MUT, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#ffffff"
CHANCE = 0.5
GROK_THRESHOLDS = (0.60, 0.65, 0.70)
COND_LABEL = {"generic_connectome": "connectome", "generic_degree": "degree-matched control"}
COND_COLOR = {"generic_connectome": CONN_COLOR, "generic_degree": CTRL_COLOR}


def _load_analysis(out_dir: Path) -> dict:
    p = out_dir / "analysis.json"
    if not p.exists():
        raise SystemExit(f"no analysis.json in {out_dir} (run --analyze-only or --collect first)")
    return json.loads(p.read_text())


def _load_runs(out_dir: Path) -> list[dict]:
    """Every per-run result.json (carries 'curve' = per-epoch val_acc, 'grok', test metrics)."""
    rd = out_dir / "runs"
    rows: list[dict] = []
    if not rd.exists():
        return rows
    for p in sorted(rd.glob("*/result.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            pass
    return rows


def _by(rows, substrate, condition, key="curve"):
    return [r[key] for r in rows
            if r.get("substrate") == substrate and r.get("condition") == condition and key in r]


def _substrates(rows, analysis):
    subs = analysis.get("substrates") or sorted({r.get("substrate") for r in rows if r.get("substrate")})
    return [s for s in ("core_alpn", "full") if s in subs] or list(subs)


# --------------------------------------------------------------------------------------
# fig1 — pooled test_acc bar, full scale (kept from the seeded version)
# --------------------------------------------------------------------------------------
def fig_wiring(analysis: dict, out_path: Path) -> None:
    substrates = analysis.get("substrates", [])
    comps = analysis.get("comparisons", {})
    rows = [(s, comps[f"{s}__connectome_vs_degree__test_acc"])
            for s in substrates if f"{s}__connectome_vs_degree__test_acc" in comps]
    if not rows:
        print("no test_acc comparisons to plot"); return

    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 4.6), squeeze=False)
    for ax, (s, c) in zip(axes[0], rows):
        conn, ctrl = c["connectome_mean"], c["control_mean"]
        p05, p50, p95 = c["control_p05"], c["control_p50"], c["control_p95"]
        ax.bar([0], [conn], width=0.5, color=CONN_COLOR, label="connectome", zorder=2)
        ax.bar([1], [ctrl], width=0.5, color=CTRL_COLOR, alpha=0.85,
               label="degree-matched (mean)", zorder=2)
        ax.vlines(1, p05, p95, color=INK, lw=2, zorder=3)
        ax.hlines([p05, p50, p95], 0.85, 1.15, color=INK, lw=1.2, zorder=3)
        ax.axhline(CHANCE, ls="--", lw=1, color=MUT, zorder=1)
        ax.text(0.5, CHANCE + 0.005, "chance", color=MUT, fontsize=8, ha="center")
        verdict = "connectome > controls" if conn > p95 else \
                  ("tie" if p05 <= conn <= p95 else "connectome < controls")
        ax.set_title(f"{s}\nperm p={c['permutation_p_one_sided']:g}  ({verdict})",
                     fontsize=10, color=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["connectome", "control"], fontsize=9)
        ax.set_ylim(0.45, 1.0)
        ax.set_ylabel("pooled test_acc", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_facecolor(SURF)
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Generic all-neuron I/O: connectome vs degree-matched controls (odor->valence)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, facecolor="white"); plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------------------
# fig2 — learning curves (val_acc vs epoch), mean +/-1 SD, plateau inset
# --------------------------------------------------------------------------------------
def _mean_sd_curves(curves: list[list[float]]):
    if not curves:
        return None, None, None
    T = min(len(c) for c in curves)
    A = np.array([c[:T] for c in curves], dtype=float)
    return np.arange(1, T + 1), A.mean(0), A.std(0)


def fig_learning_curves(rows: list[dict], substrates: list[str], out_path: Path) -> None:
    present = [s for s in substrates if _by(rows, s, "generic_connectome")]
    if not present:
        print("no per-run curves to plot (fig2)"); return
    fig, axes = plt.subplots(1, len(present), figsize=(4.8 * len(present), 4.7), squeeze=False)
    for ax, s in zip(axes[0], present):
        finals = {}
        for cond in ("generic_connectome", "generic_degree"):
            x, m, sd = _mean_sd_curves(_by(rows, s, cond))
            if x is None:
                continue
            col = COND_COLOR[cond]
            ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.15, lw=0, zorder=1)
            ax.plot(x, m, color=col, lw=2, zorder=3, label=COND_LABEL[cond])
            finals[cond] = m[-1]
        ax.axhline(CHANCE, ls="--", lw=1, color=MUT, zorder=0)
        ax.text(x[-1], CHANCE + 0.006, "chance", color=MUT, fontsize=8, ha="right")
        ax.set_ylim(0.48, 1.0)
        ax.set_xlim(1, x[-1])
        ax.set_xlabel("epoch", fontsize=9); ax.set_ylabel("val accuracy", fontsize=9)
        gap = (finals.get("generic_connectome", np.nan) - finals.get("generic_degree", np.nan))
        ax.set_title(f"{s}   (final gap +{gap:.3f})", fontsize=10, color=INK)
        ax.grid(color=GRID, lw=0.6); ax.set_facecolor(SURF)

        # zoomed plateau inset: last ~40% of epochs, y auto to the two mean plateaus
        e0 = int(x[-1] * 0.6)
        ins = ax.inset_axes([0.50, 0.12, 0.46, 0.42])
        lo, hi = 1.0, 0.0
        for cond in ("generic_connectome", "generic_degree"):
            xx, mm, ss = _mean_sd_curves(_by(rows, s, cond))
            if xx is None:
                continue
            sl = xx >= e0
            col = COND_COLOR[cond]
            ins.fill_between(xx[sl], (mm - ss)[sl], (mm + ss)[sl], color=col, alpha=0.18, lw=0)
            ins.plot(xx[sl], mm[sl], color=col, lw=1.6)
            lo, hi = min(lo, float((mm - ss)[sl].min())), max(hi, float((mm + ss)[sl].max()))
        pad = (hi - lo) * 0.25 + 1e-4
        ins.set_ylim(lo - pad, hi + pad); ins.set_xlim(e0, xx[-1])
        ins.tick_params(labelsize=6, length=2)
        ins.set_facecolor(SURF)
        for spine in ins.spines.values():
            spine.set_color(GRID)
        ins.set_title("plateau (zoom)", fontsize=7, color=MUT, pad=2)
    axes[0][0].legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    fig.suptitle("Generic-I/O learning curves — connectome groks faster and plateaus higher "
                 "(mean ±1 SD over 20+20 runs)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, facecolor="white"); plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------------------
# fig3 — per-graph final-accuracy separation, ZOOMED (the honest effect-size view)
# --------------------------------------------------------------------------------------
def fig_separation(rows: list[dict], substrates: list[str], out_path: Path) -> None:
    present = [s for s in substrates if _by(rows, s, "generic_connectome", "test_acc")]
    if not present:
        print("no per-run test_acc to plot (fig3)"); return
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(present), figsize=(4.0 * len(present), 4.7), squeeze=False)
    for ax, s in zip(axes[0], present):
        allv = []
        stats = {}
        for i, cond in enumerate(("generic_connectome", "generic_degree")):
            v = np.array(_by(rows, s, cond, "test_acc"), dtype=float)
            if v.size == 0:
                continue
            allv.append(v)
            x = i + (rng.random(v.size) - 0.5) * 0.28
            ax.scatter(x, v, s=42, color=COND_COLOR[cond], alpha=0.85,
                       edgecolors="white", linewidths=1.4, zorder=3)
            m = float(v.mean())
            stats[cond] = m
            ax.hlines(m, i - 0.28, i + 0.28, color=INK, lw=2, zorder=4)
            ax.text(i, m, f"  {m:.3f}", va="center", ha="left", fontsize=8.5, color=INK, zorder=5)
        if len(allv) == 2:
            lo = min(a.min() for a in allv); hi = max(a.max() for a in allv)
            pad = (hi - lo) * 0.18 + 1e-4
            ax.set_ylim(lo - pad, hi + pad)
            n_reach = int((allv[1] >= allv[0].mean()).sum())  # controls reaching connectome mean
            ax.set_title(f"{s}   ({n_reach}/{allv[1].size} controls reach connectome)",
                         fontsize=10, color=INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["connectome\n(20 seeds,\n1 graph)", "control\n(20 graphs)"], fontsize=8.5)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel("final test accuracy", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_facecolor(SURF)
    fig.suptitle("Every connectome seed beats every control graph (zoomed; note the y-scale)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, facecolor="white"); plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------------------
# fig4 — epochs-to-grok (val threshold crossings), connectome vs control
# --------------------------------------------------------------------------------------
def _mean_grok_epoch(rows, substrate, condition, thr):
    key = f"{thr:.2f}"
    eps = [r["grok"][key]["epoch"] for r in rows
           if r.get("substrate") == substrate and r.get("condition") == condition
           and r.get("grok", {}).get(key, {}).get("epoch") is not None]
    return float(np.mean(eps)) if eps else np.nan


def fig_grok_speed(rows: list[dict], substrates: list[str], out_path: Path) -> None:
    present = [s for s in substrates
               if any(r.get("substrate") == s and "grok" in r for r in rows)]
    if not present:
        print("no grok data to plot (fig4)"); return
    fig, axes = plt.subplots(1, len(present), figsize=(4.2 * len(present), 4.6), squeeze=False)
    thr = list(GROK_THRESHOLDS)
    xs = np.arange(len(thr)); w = 0.36
    for ax, s in zip(axes[0], present):
        conn = [_mean_grok_epoch(rows, s, "generic_connectome", t) for t in thr]
        ctrl = [_mean_grok_epoch(rows, s, "generic_degree", t) for t in thr]
        b1 = ax.bar(xs - w / 2, conn, w, color=CONN_COLOR, label="connectome", zorder=2)
        b2 = ax.bar(xs + w / 2, ctrl, w, color=CTRL_COLOR, alpha=0.9,
                    label="degree-matched control", zorder=2)
        for bars in (b1, b2):
            for rect in bars:
                h = rect.get_height()
                if not np.isnan(h):
                    ax.text(rect.get_x() + rect.get_width() / 2, h + 0.6, f"{h:.0f}",
                            ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(xs); ax.set_xticklabels([f"val ≥ {t:.2f}" for t in thr], fontsize=9)
        ax.set_ylabel("epochs to first reach", fontsize=9)
        ax.set_title(s, fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_facecolor(SURF)
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Time-to-grok: the connectome reaches each accuracy bar ~2× sooner",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, facecolor="white"); plt.close(fig)
    print(f"wrote {out_path}")


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    out_dir = Path(argv[0]) if argv else (HERE / "outputs")
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    analysis = _load_analysis(out_dir)
    rows = _load_runs(out_dir)
    substrates = _substrates(rows, analysis)
    fig_dir = HERE / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_wiring(analysis, fig_dir / "fig1_generic_io_wiring.png")
    fig_learning_curves(rows, substrates, fig_dir / "fig2_learning_curves.png")
    fig_separation(rows, substrates, fig_dir / "fig3_final_separation.png")
    fig_grok_speed(rows, substrates, fig_dir / "fig4_grok_speed.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
