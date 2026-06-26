#!/usr/bin/env python3
"""Figures for Experiment 3 - dense parameter-matched controls vs the connectome on MQAR.

Reads <output_dir>/metrics_by_run.csv (+ analysis.json) and writes figures/ :
  fig1_final_acc.png   final test accuracy, connectome vs C1/C2/C3, core arm | full arm
  fig2_param_budget.png final accuracy vs trainable-param count (the budget view)

Robust to partially-present data (controls not yet trained -> only what's on disk is plotted).
Point it at outputs/:  uv run python make_figures.py <output_dir>
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"

# display order + labels per substrate arm
ARMS = {
    "core": [("core", "connectome\n(sparse)"), ("dense_c1_core", "C1 dense\nceiling"),
             ("dense_c2_core", "C2 dense\nreservoir"), ("dense_c3_core", "C3 dense\nparam-matched")],
    "full": [("full", "connectome\n(sparse)"), ("dense_c1_full", "C1 dense\nceiling"),
             ("dense_c2_full", "C2 dense\nreservoir"), ("dense_c3_full", "C3 dense\nparam-matched")],
}
COLORS = {"connectome": "#1f77b4", "C1": "#d62728", "C2": "#2ca02c", "C3": "#9467bd"}


def load(out_dir: Path):
    rows = []
    csv_path = out_dir / "metrics_by_run.csv"
    if csv_path.exists():
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
    else:  # fall back to per-run result.json
        for rp in sorted((out_dir / "runs").glob("*/result.json")):
            rows.append(json.loads(rp.read_text()))
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    return by_cond


def _fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fig_final_acc(by_cond, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (sub, items) in zip(axes, ARMS.items()):
        labels, data, colors = [], [], []
        for cond, lab in items:
            accs = [_fl(r["test_acc"]) for r in by_cond.get(cond, [])]
            accs = [a for a in accs if a is not None]
            if not accs:
                continue
            labels.append(f"{lab}\n(n={len(accs)})")
            data.append(accs)
            colors.append(COLORS["connectome" if cond in ("core", "full") else lab.split()[0]])
        if not data:
            ax.set_title(f"{sub} arm — no data yet")
            continue
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        for i, (d, c) in enumerate(zip(data, colors), start=1):
            ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(d)), d, s=14, color=c, zorder=3)
        ax.axhline(1 / 32, ls=":", color="grey", lw=1, label="chance")
        ax.set_title(f"{sub} arm")
        ax.set_ylabel("final test accuracy" if sub == "core" else "")
        ax.grid(axis="y", ls=":", alpha=0.4)
    fig.suptitle("Experiment 3 — connectome vs dense parameter-matched controls (MQAR, lr=1e-3)")
    fig.tight_layout()
    fig.savefig(out_dir.parent / "figures" / "fig1_final_acc.png", dpi=130)
    plt.close(fig)


def fig_param_budget(by_cond, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for sub, marker in (("core", "o"), ("full", "s")):
        for cond, lab in ARMS[sub]:
            rs = by_cond.get(cond, [])
            accs = [_fl(r["test_acc"]) for r in rs if _fl(r["test_acc"]) is not None]
            pars = [_fl(r.get("trainable_params")) for r in rs if _fl(r.get("trainable_params")) is not None]
            if not accs or not pars:
                continue
            key = "connectome" if cond in ("core", "full") else lab.split()[0]
            ax.scatter(np.mean(pars), np.mean(accs), s=80, marker=marker, color=COLORS[key],
                       edgecolor="k", zorder=3,
                       label=f"{sub}:{key}" if sub == "core" or key == "connectome" else None)
            ax.annotate(f"{key}", (np.mean(pars), np.mean(accs)), textcoords="offset points",
                        xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log)")
    ax.set_ylabel("final test accuracy")
    ax.set_title("Experiment 3 — accuracy vs trainable-parameter budget\n(circles core, squares full)")
    ax.grid(ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir.parent / "figures" / "fig2_param_budget.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    out_dir = Path(argv[0]) if argv else (HERE / "outputs")
    (out_dir.parent / "figures").mkdir(parents=True, exist_ok=True)
    by_cond = load(out_dir)
    if not by_cond:
        print(f"no runs found under {out_dir}")
        return 1
    fig_final_acc(by_cond, out_dir)
    fig_param_budget(by_cond, out_dir)
    print(f"wrote figures to {HERE/'figures'} from {sum(len(v) for v in by_cond.values())} runs "
          f"({', '.join(f'{k}:{len(v)}' for k, v in sorted(by_cond.items()))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
