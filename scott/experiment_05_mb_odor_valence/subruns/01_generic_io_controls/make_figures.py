#!/usr/bin/env python3
"""Figures for Experiment 5 · subrun 01 — generic-I/O connectome vs degree-matched controls.

Reads outputs/analysis.json and renders, PER substrate:
  fig1_generic_io_wiring -- pooled test_acc: connectome vs degree-matched control (mean + control
                            spread + connectome point), with the permutation-rank p, per substrate.
                            THE headline: does generic-I/O topology beat controls on odor->valence?

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

# validated palette (same family as the primary Exp-5 figures)
CONN_COLOR, CTRL_COLOR = "#2a78d6", "#eb6834"
INK, MUT, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#ffffff"
CHANCE = 0.5


def _load(out_dir: Path) -> dict:
    p = out_dir / "analysis.json"
    if not p.exists():
        raise SystemExit(f"no analysis.json in {out_dir} (run --analyze-only or --collect first)")
    return json.loads(p.read_text())


def fig_wiring(analysis: dict, out_path: Path) -> None:
    substrates = analysis.get("substrates", [])
    comps = analysis.get("comparisons", {})
    rows = []
    for s in substrates:
        key = f"{s}__connectome_vs_degree__test_acc"
        if key in comps:
            rows.append((s, comps[key]))
    if not rows:
        print("no test_acc comparisons to plot")
        return

    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 4.6), squeeze=False)
    for ax, (s, c) in zip(axes[0], rows):
        conn = c["connectome_mean"]
        ctrl = c["control_mean"]
        p05, p50, p95 = c["control_p05"], c["control_p50"], c["control_p95"]
        pperm = c["permutation_p_one_sided"]
        # control distribution as a vertical whisker at x=1; connectome as a point at x=0
        ax.bar([0], [conn], width=0.5, color=CONN_COLOR, label="connectome", zorder=2)
        ax.bar([1], [ctrl], width=0.5, color=CTRL_COLOR, alpha=0.85,
               label="degree-matched (mean)", zorder=2)
        ax.vlines(1, p05, p95, color=INK, lw=2, zorder=3)
        ax.hlines([p05, p50, p95], 0.85, 1.15, color=INK, lw=1.2, zorder=3)
        ax.axhline(CHANCE, ls="--", lw=1, color=MUT, zorder=1)
        ax.text(0.5, CHANCE + 0.005, "chance", color=MUT, fontsize=8, ha="center")
        verdict = "connectome > controls" if conn > p95 else \
                  ("tie" if p05 <= conn <= p95 else "connectome < controls")
        ax.set_title(f"{s}\nperm p={pperm:g}  ({verdict})", fontsize=10, color=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["connectome", "control"], fontsize=9)
        ax.set_ylim(0.45, 1.0)
        ax.set_ylabel("pooled test_acc", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_facecolor(SURF)
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Generic all-neuron I/O: connectome vs degree-matched controls (odor->valence)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
    fig_wiring(analysis, fig_dir / "fig1_generic_io_wiring.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
