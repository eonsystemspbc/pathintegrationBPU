#!/usr/bin/env python3
"""make_figures.py -- figures for Experiment dyn-01 from outputs/analysis.json + outputs/curves.npz.

Two figure types (both fall out of the same probe):
  1. fig_lambda_summary.png   -- final lambda per condition: connectome dot (+/- sem) against the
     degree-matched control distribution (strip). The headline "expand vs contract, and does the
     connectome differ from its shuffle" plot. Zero line = the contract/expand boundary.
  2. fig_convergence_<substrate>.png -- running lambda vs step: connectome (bold) vs control band
     (min-max across graphs), one panel per (normalize, drive) cell. Shows convergence AND the
     non-normal transient (an early bump up then decay when sigma_max >> rho).

Usage: uv run python make_figures.py [OUTPUT_DIR]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "outputs"
FIG = Path(__file__).resolve().parent / "figures"


def _cells(res_sub: dict):
    """yield (rho_key, cond_key, entry) for a substrate's results."""
    for rho_key, conds in res_sub.items():
        for cond_key, entry in conds.items():
            yield rho_key, cond_key, entry


def summary(analysis: dict):
    subs = list(analysis["results"])
    # collect a flat list of (label, substrate, conn_lambda, sem, control_lambdas)
    def _label(rho_key, cond_key):
        pretty = (cond_key.replace("norm0", "norm:off").replace("norm1", "norm:on")
                  .replace("autonomous_warm", "auton").replace("|", " · "))
        multi_rho = len(analysis.get("config", {}).get("rho_grid", [1])) > 1
        return f"{rho_key} {pretty}" if multi_rho else pretty

    rows = []
    for sub in subs:
        for rho_key, cond_key, e in _cells(analysis["results"][sub]):
            rows.append((f"{sub}\n{_label(rho_key, cond_key)}", sub,
                         e["connectome"]["lambda_mean"], e["connectome"]["lambda_sem"],
                         e["control"]["lambdas"]))
    if not rows:
        return
    n = len(rows)
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * n), 5.2))
    for i, (_, _, cl, sem, ctrl) in enumerate(rows):
        ax.scatter(np.full(len(ctrl), i) + np.random.uniform(-0.08, 0.08, len(ctrl)), ctrl,
                   s=16, color="#9aa7b3", alpha=0.7, zorder=2,
                   label="degree-matched controls" if i == 0 else None)
        ax.errorbar(i, cl, yerr=sem, fmt="o", ms=9, color="#d1495b", capsize=4, zorder=3,
                    label="connectome" if i == 0 else None)
    ax.axhline(0.0, color="k", lw=1.0, ls="--", alpha=0.6)
    ax.text(0.005, 0.0, " contract | expand", transform=ax.get_yaxis_transform(),
            va="center", fontsize=8, color="k", alpha=0.7)
    ax.set_xticks(range(n))
    ax.set_xticklabels([r[0] for r in rows], fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("largest Lyapunov exponent  λ  (per step)")
    ax.set_title("dyn-01: global expansion/contraction — connectome vs degree-matched control")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    p = FIG / "fig_lambda_summary.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"[fig] wrote {p}")


def convergence(analysis: dict, curves: dict):
    for sub in analysis["results"]:
        cells = list(_cells(analysis["results"][sub]))
        if not cells:
            continue
        ncol = min(len(cells), 4)
        nrow = int(np.ceil(len(cells) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for k, (rho_key, cond_key, e) in enumerate(cells):
            ax = axes.flat[k]
            ax.set_visible(True)
            key = f"{sub}|{rho_key}|{cond_key}"   # matches run.py's f"{sub}|rho..|norm..|drive"
            conn = curves.get(f"{key}|conn")
            lo = curves.get(f"{key}|ctrl_lo")
            hi = curves.get(f"{key}|ctrl_hi")
            cmean = curves.get(f"{key}|ctrl_mean")
            steps = np.arange(len(conn)) if conn is not None else None
            if lo is not None and hi is not None:
                ax.fill_between(steps, lo, hi, color="#9aa7b3", alpha=0.35, label="control range")
            if cmean is not None:
                ax.plot(steps, cmean, color="#6b7885", lw=1.2, label="control mean")
            if conn is not None:
                ax.plot(steps, conn, color="#d1495b", lw=2.0, label="connectome")
            ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.5)
            ax.set_title(f"{rho_key}  {cond_key}", fontsize=9)
            ax.set_xlabel("step"); ax.set_ylabel("running λ")
            if k == 0:
                ax.legend(fontsize=7.5, loc="best")
        fig.suptitle(f"dyn-01: running Lyapunov exponent — {sub}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = FIG / f"fig_convergence_{sub}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print(f"[fig] wrote {p}")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    analysis = json.loads((OUT / "analysis.json").read_text())
    curves = dict(np.load(OUT / "curves.npz")) if (OUT / "curves.npz").exists() else {}
    np.random.seed(0)
    summary(analysis)
    convergence(analysis, curves)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
