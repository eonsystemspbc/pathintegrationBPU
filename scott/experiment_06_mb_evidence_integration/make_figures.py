#!/usr/bin/env python3
"""Figures for Experiment 6 -- MB evidence integration (generic-I/O connectome vs degree-matched
controls on the odor->evidence temporal-integration task).

Reads outputs/analysis.json (+ optional outputs/verifier_<substrate>.json) and renders:
  fig1_integration_wiring -- THE headline: pooled 3-way test_acc, connectome vs degree-matched
                             control (mean + control spread + connectome point) with the
                             permutation-rank p, per substrate. Does integration-task topology beat
                             controls?  (chance 1/3.)
  fig2_per_category       -- the overloaded secondary: neutral-class vs polar-class recall,
                             connectome vs control, per substrate (per-category difficulty split).
  fig3_integration_curve  -- verifier: pooled accuracy vs K (integration must rise monotonically)
                             with the analytic Bayes ceiling, plus the first-only / shuffled-evidence
                             ablation markers. Rendered only if a verifier_*.json is present.
  fig4_learning_curves    -- per-epoch validation accuracy vs epoch, connectome vs degree control,
                             mean over the 20 runs/arm + across-seed min-max band, one panel per
                             substrate; reads the 300-epoch `curve` field from outputs/runs/*/result.json.
                             Shows the connectome leads throughout and both arms still climb at the cap.

Defensive: only plots substrates/metrics/files present, so it also works on partial / smoke data.
Usage:  uv run python .../experiment_06_mb_evidence_integration/make_figures.py [OUTPUT_DIR]
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

# validated palette (same family as the Exp-5 figures)
CONN_COLOR, CTRL_COLOR = "#2a78d6", "#eb6834"
NEUTRAL_COLOR, POLAR_COLOR = "#4a3aa7", "#1baf7a"
BAYES_COLOR = "#898781"
INK, MUT, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#ffffff"
CHANCE = 1.0 / 3.0


def _load(out_dir: Path) -> dict:
    p = out_dir / "analysis.json"
    if not p.exists():
        raise SystemExit(f"no analysis.json in {out_dir} (run --analyze-only or --collect first)")
    return json.loads(p.read_text())


def fig_wiring(analysis: dict, out_path: Path) -> None:
    substrates = analysis.get("substrates", [])
    comps = analysis.get("comparisons", {})
    rows = [(s, comps[f"{s}__connectome_vs_degree__test_acc"]) for s in substrates
            if f"{s}__connectome_vs_degree__test_acc" in comps]
    if not rows:
        print("no test_acc comparisons to plot")
        return

    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 4.6), squeeze=False)
    for ax, (s, c) in zip(axes[0], rows):
        conn, ctrl = c["connectome_mean"], c["control_mean"]
        p05, p50, p95 = c["control_p05"], c["control_p50"], c["control_p95"]
        pperm = c["permutation_p_one_sided"]
        eff = c.get("effect_size_ctrl_sd")           # (conn_mean - ctrl_mean)/ctrl_std -- lead with this
        ax.bar([0], [conn], width=0.5, color=CONN_COLOR, label="connectome", zorder=2)
        ax.bar([1], [ctrl], width=0.5, color=CTRL_COLOR, alpha=0.85,
               label="degree-matched (mean)", zorder=2)
        ax.vlines(1, p05, p95, color=INK, lw=2, zorder=3)
        ax.hlines([p05, p50, p95], 0.85, 1.15, color=INK, lw=1.2, zorder=3)
        ax.axhline(CHANCE, ls="--", lw=1, color=MUT, zorder=1)
        ax.text(0.5, CHANCE + 0.005, "chance (1/3)", color=MUT, fontsize=8, ha="center")
        verdict = "connectome > controls" if conn > p95 else \
                  ("tie" if p05 <= conn <= p95 else "connectome < controls")
        eff_str = f"d={eff:+.2f} ctrl-SD  " if eff is not None else ""
        ax.set_title(f"{s}\n{eff_str}perm p={pperm:g}  ({verdict})", fontsize=10, color=INK)
        if eff is not None:                          # annotate the connectome-vs-control gap in SD units
            ax.annotate(f"{eff:+.2f} ctrl-SD", xy=(0, conn), xytext=(0, conn + 0.02),
                        ha="center", fontsize=8, color=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["connectome", "control"], fontsize=9)
        ax.set_ylim(0.30, 1.0)
        ax.set_ylabel("pooled 3-way test_acc", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_facecolor(SURF)
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Generic all-neuron I/O: connectome vs degree-matched controls (odor->evidence)\n"
                 "effect size d = (connectome - control mean) / control SD", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, facecolor="white")
    print(f"wrote {out_path}")


def fig_per_category(analysis: dict, out_path: Path) -> None:
    substrates = analysis.get("substrates", [])
    tconn, tctrl = analysis.get("table_connectome", {}), analysis.get("table_control", {})
    rows = [s for s in substrates if s in tconn and s in tctrl]
    if not rows:
        print("no per-category table to plot")
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 4.4), squeeze=False)
    for ax, s in zip(axes[0], rows):
        cats = [("test_initial_acc", "neutral", NEUTRAL_COLOR), ("test_reversed_acc", "polar", POLAR_COLOR)]
        x = np.arange(len(cats))
        conn_v = [tconn[s].get(k, {}).get("mean", np.nan) for k, _, _ in cats]
        ctrl_v = [tctrl[s].get(k, {}).get("mean", np.nan) for k, _, _ in cats]
        ax.bar(x - 0.19, conn_v, width=0.36, color=CONN_COLOR, label="connectome")
        ax.bar(x + 0.19, ctrl_v, width=0.36, color=CTRL_COLOR, alpha=0.85, label="control")
        ax.axhline(CHANCE, ls="--", lw=1, color=MUT)
        ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl, _ in cats], fontsize=9)
        ax.set_ylim(0.30, 1.0); ax.set_ylabel("recall", fontsize=9)
        ax.set_title(s, fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_facecolor(SURF)
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Per-category recall (neutral vs polar), connectome vs control", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor="white")
    print(f"wrote {out_path}")


def fig_integration_curve(out_dir: Path, out_path: Path) -> None:
    vfiles = sorted(out_dir.glob("verifier_*.json"))
    vfiles = [v for v in vfiles if json.loads(v.read_text()).get("integration_curve")]
    if not vfiles:
        print("no verifier integration_curve to plot (run --eval-K-curve)")
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for v in vfiles:
        d = json.loads(v.read_text())
        curve = d["integration_curve"]
        Ks = [c["K"] for c in curve]
        accs = [c["pooled_acc"] for c in curve]
        bays = [c["bayes"] for c in curve]
        ax.plot(Ks, accs, "-o", label=f"{d.get('substrate','?')} (model)", color=CONN_COLOR)
        ax.plot(Ks, bays, "--", label=f"{d.get('substrate','?')} Bayes ceiling", color=BAYES_COLOR)
        if "shuffled_evidence" in d:
            ax.axhline(d["shuffled_evidence"]["pooled_acc"], ls=":", lw=1, color=CTRL_COLOR,
                       label="shuffled-evidence (ablation)")
    ax.axhline(CHANCE, ls="--", lw=1, color=MUT); ax.text(Ks[0], CHANCE + 0.01, "chance", color=MUT, fontsize=8)
    ax.set_xlabel("K (presentations per odor)", fontsize=9)
    ax.set_ylabel("pooled 3-way accuracy", fontsize=9)
    ax.set_title("Integration curve: accuracy rises with K (vs Bayes ceiling)", fontsize=10, color=INK)
    ax.grid(color=GRID, lw=0.6); ax.set_facecolor(SURF); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    print(f"wrote {out_path}")


BAYES_CEIL = 0.895  # analytic thresholded-sample-mean oracle at m=1/sigma=1/K=8
PREFLIGHT_READ = 0.716  # the under-trained pre-flight "plateau" -- real runs climb past it


def _group_curves(out_dir: Path) -> dict:
    """{substrate: {condition: np.ndarray (n_runs, T)}} of per-epoch val-accuracy trajectories."""
    groups: dict = {}
    for rp in sorted((out_dir / "runs").glob("*/result.json")):
        d = json.loads(rp.read_text())
        curve = d.get("curve")
        if not curve:
            continue
        groups.setdefault(d["substrate"], {}).setdefault(d["condition"], []).append(np.asarray(curve, float))
    out: dict = {}
    for s, byc in groups.items():
        out[s] = {}
        for cond, curves in byc.items():
            T = min(len(c) for c in curves)              # defensive: align to shortest
            out[s][cond] = np.vstack([c[:T] for c in curves])
    return out


def fig_learning_curves(out_dir: Path, out_path: Path) -> None:
    """Per-epoch val-accuracy: connectome vs degree control, mean + across-seed min-max band,
    one panel per substrate. Shows the connectome leads throughout, both arms still climbing at the
    300-epoch cap, the ~45-epoch grok latency, and that real runs pass the under-trained pre-flight read."""
    grp = _group_curves(out_dir)
    substrates = [s for s in ("core_alpn", "full") if s in grp] or sorted(grp)
    if not substrates:
        print("no per-epoch curves to plot (no runs/*/result.json with a 'curve')")
        return
    series = [("generic_connectome", "connectome", "connectome", CONN_COLOR),
              ("generic_degree", "degree-matched control", "control", CTRL_COLOR)]
    fig, axes = plt.subplots(1, len(substrates), figsize=(5.0 * len(substrates), 4.6),
                             squeeze=False, sharey=True)
    for ax, s in zip(axes[0], substrates):
        for cond, _legend, endlab, color in series:
            arr = grp[s].get(cond)
            if arr is None:
                continue
            x = np.arange(1, arr.shape[1] + 1)
            mean, lo, hi = arr.mean(0), arr.min(0), arr.max(0)
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0, zorder=2)   # across-seed spread
            ax.plot(x, mean, color=color, lw=2, zorder=4)                          # mean trajectory
            ax.annotate(f"{endlab} {mean[-1]:.2f}", xy=(x[-1], mean[-1]),          # short direct label at line end
                        xytext=(-4, 7 if cond == "generic_connectome" else -13),
                        textcoords="offset points", ha="right", fontsize=8, color=color, weight="bold")
        ax.axhline(CHANCE, ls="--", lw=1, color=MUT, zorder=1)
        ax.text(3, CHANCE + 0.008, "chance (1/3)", color=MUT, fontsize=8)
        ax.axhline(BAYES_CEIL, ls="--", lw=1, color=BAYES_COLOR, zorder=1)
        ax.text(3, BAYES_CEIL - 0.028, "Bayes ceiling 0.895", color=BAYES_COLOR, fontsize=8)
        ax.axhline(PREFLIGHT_READ, ls=":", lw=1, color=MUT, zorder=1)
        ax.text(3, PREFLIGHT_READ + 0.008, "pre-flight read (under-trained)",     # left side: curves are at chance here
                color=MUT, fontsize=7, ha="left")
        ax.set_xlim(1, None); ax.set_ylim(0.30, 0.95)
        ax.set_xlabel("epoch", fontsize=9)
        ax.set_title(s, fontsize=10, color=INK)
        ax.grid(color=GRID, lw=0.6); ax.set_facecolor(SURF)
    axes[0][0].set_ylabel("validation accuracy (pooled 3-way)", fontsize=9)
    axes[0][0].legend(handles=[plt.Line2D([], [], color=c, lw=2, label=l) for _, l, _e, c in series],
                      fontsize=8, loc="lower right")
    fig.suptitle("Learning curves: connectome leads throughout; both arms still rising at the 300-epoch cap\n"
                 "(mean over 20 runs; band = across-seed min-max)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor="white")
    print(f"wrote {out_path}")


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    out_dir = Path(argv[0]) if argv else (HERE / "outputs")
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    analysis = _load(out_dir)
    fig_dir = HERE / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_wiring(analysis, fig_dir / "fig1_integration_wiring.png")
    fig_per_category(analysis, fig_dir / "fig2_per_category.png")
    fig_integration_curve(out_dir, fig_dir / "fig3_integration_curve.png")
    fig_learning_curves(out_dir, fig_dir / "fig4_learning_curves.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
