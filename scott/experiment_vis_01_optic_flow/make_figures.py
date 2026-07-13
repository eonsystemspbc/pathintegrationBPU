#!/usr/bin/env python3
"""Figures for Experiment vis-01 (optic-lobe connectome vs degree-matched controls on optic flow).

Reads outputs/analysis.json (+ optional outputs/verifier_<substrate>.json) and renders:
  fig1_wiring         -- THE headline: mean R² (5-DOF self-motion), connectome vs degree-matched
                         control (mean + control spread + connectome point), permutation-rank p +
                         control-SD effect size, per substrate.
  fig2_per_dof        -- per-DOF R², connectome vs degree control (which DOF the wiring helps).
  fig3_verifier       -- verifier ablations: baseline vs time-shuffle / single-frame / no-objects /
                         no-parallax / naive-baseline (proves the task needs motion/temporal/depth).
                         Rendered only if a verifier_*.json is present.
  fig4_learning_curves-- per-epoch validation R² vs epoch, connectome vs degree control, mean +
                         across-seed band, one panel per substrate.

Defensive: only plots what is present, so it also works on partial / smoke data.
Usage:  uv run python .../experiment_vis_01_optic_flow/make_figures.py [OUTPUT_DIR]
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
CONN_COLOR, CTRL_COLOR = "#2a78d6", "#eb6834"
INK, MUT, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#ffffff"
DOF_NAMES = ("yaw_rate", "roll_rate", "pitch_rate", "forward_v", "lateral_v", "heading_az",
             "ventral_flow")   # 7-channel candidate target (matches optic_flow_task.TARGET_NAMES)


def _load(out_dir: Path) -> dict:
    p = out_dir / "analysis.json"
    if not p.exists():
        raise SystemExit(f"no analysis.json in {out_dir} (run --analyze-only or --collect first)")
    return json.loads(p.read_text())


def fig_wiring(analysis: dict, out_path: Path) -> None:
    substrates = analysis.get("substrates", [])
    comps = analysis.get("comparisons", {})
    rows = [(s, comps[f"{s}__connectome_vs_degree_matched__test_r2"]) for s in substrates
            if f"{s}__connectome_vs_degree_matched__test_r2" in comps]
    if not rows:
        print("no degree_matched comparison to plot"); return
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 4.6), squeeze=False)
    for ax, (s, c) in zip(axes[0], rows):
        conn, ctrl = c["connectome_mean"], c["control_mean"]
        p05, p50, p95 = c["control_p05"], c["control_p50"], c["control_p95"]
        pperm = c["permutation_p_one_sided"]; eff = c.get("effect_size_ctrl_sd")
        ax.bar([0], [conn], width=0.5, color=CONN_COLOR, label="connectome", zorder=2)
        ax.bar([1], [ctrl], width=0.5, color=CTRL_COLOR, alpha=0.85, label="degree-matched (mean)", zorder=2)
        ax.vlines(1, p05, p95, color=INK, lw=2, zorder=3)
        ax.hlines([p05, p50, p95], 0.85, 1.15, color=INK, lw=1.2, zorder=3)
        ax.axhline(0.0, ls="--", lw=1, color=MUT, zorder=1)
        ax.text(0.5, 0.005, "R²=0 (predict mean)", color=MUT, fontsize=8, ha="center")
        verdict = "connectome > controls" if conn > p95 else \
                  ("tie" if p05 <= conn <= p95 else "connectome < controls")
        eff_str = f"d={eff:+.2f} ctrl-SD  " if eff is not None else ""
        ax.set_title(f"{s}\n{eff_str}perm p={pperm:g}  ({verdict})", fontsize=10, color=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["connectome", "control"], fontsize=9)
        ax.set_ylabel("mean R² (5-DOF self-motion)", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_facecolor(SURF)
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Generic all-neuron I/O: optic-lobe connectome vs degree-matched controls (optic flow)\n"
                 "effect size d = (connectome - control mean) / control SD", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor="white"); print(f"wrote {out_path}")


def fig_per_dof(analysis: dict, out_path: Path) -> None:
    per_dof = analysis.get("per_dof", {})
    substrates = [s for s in analysis.get("substrates", []) if s in per_dof]
    if not substrates:
        print("no per-DOF table to plot"); return
    fig, axes = plt.subplots(1, len(substrates), figsize=(4.6 * len(substrates), 4.4), squeeze=False)
    for ax, s in zip(axes[0], substrates):
        x = np.arange(len(DOF_NAMES))
        conn = [per_dof[s].get(d, {}).get("connectome") for d in DOF_NAMES]
        ctrl = [per_dof[s].get(d, {}).get("degree_matched") for d in DOF_NAMES]
        conn = [np.nan if v is None else v for v in conn]
        ctrl = [np.nan if v is None else v for v in ctrl]
        ax.bar(x - 0.19, conn, width=0.36, color=CONN_COLOR, label="connectome")
        ax.bar(x + 0.19, ctrl, width=0.36, color=CTRL_COLOR, alpha=0.85, label="degree control")
        ax.axhline(0.0, ls="--", lw=1, color=MUT)
        ax.set_xticks(x); ax.set_xticklabels(DOF_NAMES, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("R²", fontsize=9); ax.set_title(s, fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_facecolor(SURF)
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Per-DOF R²: connectome vs degree control", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, facecolor="white"); print(f"wrote {out_path}")


def fig_verifier(out_dir: Path, out_path: Path) -> None:
    vfiles = sorted(out_dir.glob("verifier_*.json"))
    if not vfiles:
        print("no verifier_*.json to plot (run --verifier)"); return
    fig, axes = plt.subplots(1, len(vfiles), figsize=(5.0 * len(vfiles), 4.4), squeeze=False)
    modes = [("baseline", "baseline"), ("time_shuffle", "time-shuffle"), ("single_frame", "single-frame"),
             ("no_objects", "no-objects"), ("no_parallax", "no-parallax"), ("naive_baseline", "naive")]
    for ax, vf in zip(axes[0], vfiles):
        d = json.loads(vf.read_text())
        labels, vals, colors = [], [], []
        for key, lab in modes:
            if key in d:
                labels.append(lab); vals.append(d[key]["mean_r2"])
                colors.append(CONN_COLOR if key == "baseline" else
                              (MUT if key == "naive_baseline" else CTRL_COLOR))
        ax.bar(range(len(vals)), vals, color=colors)
        ax.axhline(0.0, ls="--", lw=1, color=MUT)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("mean R²", fontsize=9); ax.set_title(d.get("substrate", "?"), fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_facecolor(SURF)
    fig.suptitle("Verifier ablations: task needs motion (time-shuffle/single-frame collapse) + depth "
                 "(no-parallax hits translation)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, facecolor="white"); print(f"wrote {out_path}")


def _group_curves(out_dir: Path) -> dict:
    groups: dict = {}
    for rp in sorted((out_dir / "runs").glob("*/result.json")):
        d = json.loads(rp.read_text())
        curve = d.get("curve")
        if curve:
            groups.setdefault(d["substrate"], {}).setdefault(d["condition"], []).append(np.asarray(curve, float))
    out = {}
    for s, byc in groups.items():
        out[s] = {}
        for cond, curves in byc.items():
            T = min(len(c) for c in curves)
            out[s][cond] = np.vstack([c[:T] for c in curves])
    return out


def fig_learning_curves(out_dir: Path, out_path: Path) -> None:
    grp = _group_curves(out_dir)
    substrates = sorted(grp)
    if not substrates:
        print("no per-epoch curves to plot"); return
    series = [("connectome", CONN_COLOR), ("degree_matched", CTRL_COLOR)]
    fig, axes = plt.subplots(1, len(substrates), figsize=(5.0 * len(substrates), 4.6),
                             squeeze=False, sharey=True)
    for ax, s in zip(axes[0], substrates):
        for cond, color in series:
            arr = grp[s].get(cond)
            if arr is None:
                continue
            x = np.arange(1, arr.shape[1] + 1)
            ax.fill_between(x, arr.min(0), arr.max(0), color=color, alpha=0.15, lw=0)
            ax.plot(x, arr.mean(0), color=color, lw=2, label=cond)
        ax.axhline(0.0, ls="--", lw=1, color=MUT)
        ax.set_xlabel("epoch", fontsize=9); ax.set_title(s, fontsize=10, color=INK)
        ax.grid(color=GRID, lw=0.6); ax.set_facecolor(SURF)
    axes[0][0].set_ylabel("validation mean R²", fontsize=9); axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Learning curves: connectome vs degree control (mean over seeds; band = across-seed min-max)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor="white"); print(f"wrote {out_path}")


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    out_dir = Path(argv[0]) if argv else (HERE / "outputs")
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    analysis = _load(out_dir)
    fig_dir = HERE / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig_wiring(analysis, fig_dir / "fig1_wiring.png")
    fig_per_dof(analysis, fig_dir / "fig2_per_dof.png")
    fig_verifier(out_dir, fig_dir / "fig3_verifier.png")
    fig_learning_curves(out_dir, fig_dir / "fig4_learning_curves.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
