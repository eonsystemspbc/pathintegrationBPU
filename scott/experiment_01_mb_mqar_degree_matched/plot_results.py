#!/usr/bin/env python3
"""Figure for Experiment 1: connectome vs degree-matched controls on MQAR (spectral-radius matched).

Panel A: per-epoch validation-accuracy learning curves (each run faint, arm-mean bold).
Panel B: final test-accuracy by arm (each run a point; mean +/- std overlaid).

Shared across sub-runs. Reads <output-dir>/runs/*/result.json and writes the figure into
that sub-run's sibling figures/ dir (<output-dir>/../figures/).

Pick the sub-run's output dir with the EXP01_OUTPUT_DIR env var (repo-relative or absolute),
or pass it as the first CLI arg. Defaults to the first-pass sub-run. Examples (from repo root):
  EXP01_OUTPUT_DIR=scott/.../subruns/03_full_fleet/outputs uv run python .../plot_results.py
  uv run python .../plot_results.py scott/.../subruns/01_first_pass/outputs
Re-runnable after extending epochs / growing the null.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "subruns" / "01_first_pass" / "outputs"
# Output dir: CLI arg > EXP01_OUTPUT_DIR > first-pass default. Relative paths resolve
# against the repo root first (matches how run.py passes them), else against this file.
_sel = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EXP01_OUTPUT_DIR", str(DEFAULT_OUTPUT))
OUTDIR = Path(_sel)
if not OUTDIR.is_absolute():
    OUTDIR = (REPO_ROOT / _sel) if (REPO_ROOT / _sel).exists() else (HERE / _sel)
RUNS = sorted(glob.glob(str(OUTDIR / "runs" / "*" / "result.json")))
ARMS = {"connectome": "#1f77b4", "control": "#888888"}
ARM_LABEL = {"connectome": "MB connectome", "control": "degree-matched null"}


def load():
    rows = [json.load(open(p)) for p in RUNS]
    if not rows:
        raise SystemExit(f"no result.json found under {OUTDIR/'runs'}")
    return rows


def mean_curve(curves):
    if not curves:
        return np.array([]), np.array([])
    maxlen = max(len(c) for c in curves)
    arr = np.full((len(curves), maxlen), np.nan)
    for i, c in enumerate(curves):
        arr[i, : len(c)] = c
    return np.arange(1, maxlen + 1), np.nanmean(arr, axis=0)


def main():
    rows = load()
    target_rho = None
    man = OUTDIR / "manifest.json"
    if man.exists():
        target_rho = json.load(open(man)).get("target_rho")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel A: learning curves
    for arm, color in ARMS.items():
        curves = [r["curve"] for r in rows if r["arm"] == arm and r["curve"]]
        for c in curves:
            axA.plot(range(1, len(c) + 1), c, color=color, alpha=0.18, lw=0.9)
        x, m = mean_curve(curves)
        if x.size:
            axA.plot(x, m, color=color, lw=2.6, label=f"{ARM_LABEL[arm]} (n={len(curves)})")
    axA.axhline(1 / 32, color="k", ls=":", lw=1, label="chance (1/32)")
    axA.set_xlabel("epoch")
    axA.set_ylabel("validation recall accuracy")
    rho_txt = f"  (ρ matched = {target_rho:.2f})" if target_rho else ""
    axA.set_title(f"Learning curves{rho_txt}")
    axA.set_ylim(0, 1)
    axA.legend(loc="upper left", fontsize=8, frameon=False)

    # Panel B: final test accuracy by arm
    rng = np.random.default_rng(0)
    for i, (arm, color) in enumerate(ARMS.items()):
        vals = np.array([r["test_acc"] for r in rows if r["arm"] == arm])
        if vals.size == 0:
            continue
        x = i + (rng.random(vals.size) - 0.5) * 0.18
        axB.scatter(x, vals, color=color, alpha=0.75, s=34, edgecolor="white", linewidth=0.5, zorder=3)
        axB.errorbar(i, vals.mean(), yerr=vals.std(), fmt="o", color="black",
                     ms=7, capsize=5, zorder=4)
        axB.text(i, 0.04, f"{vals.mean():.3f}\n±{vals.std():.3f}",
                 ha="center", va="bottom", fontsize=9)
    axB.axhline(1 / 32, color="k", ls=":", lw=1)
    axB.set_xticks(range(len(ARMS)))
    axB.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=9)
    axB.set_ylabel("final test recall accuracy")
    axB.set_title("Final accuracy (each run a point)")
    axB.set_ylim(0, 1)

    fig.suptitle("Exp 1 — FlyWire MB connectome vs degree-matched controls on MQAR "
                 "(spectral radius matched; first pass, 100-epoch cap)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    outdir = OUTDIR.parent / "figures"
    outdir.mkdir(exist_ok=True)
    out = outdir / "exp01_connectome_vs_degree_matched.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
