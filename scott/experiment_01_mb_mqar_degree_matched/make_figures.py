#!/usr/bin/env python3
"""Publication-style figures for a learning-rate-swept sub-run of Experiment 1
(MB connectome vs degree-matched controls on MQAR, spectral radius matched).

Shared across sub-runs. Reads <output-dir>/runs/*/result.json and writes a small set of
clean figures into that sub-run's sibling figures/ dir. Adapts to however many learning
rates are present. Style follows subrun 01's figure (blue = connectome, grey = control,
dotted chance line, minimal on-figure text).

Figures:
  fig1_learning_curves_by_lr   mean val-accuracy curve per lr (band = +/-1 SD); 2 panels (arms)
  fig2_best_lr_curves          best-lr mean curve, connectome vs control, one panel
  fig3_final_acc_by_lr         grouped bars of final accuracy per lr (+/-1 SD) + within-lr test
  fig4_best_lr_final_acc       best-lr final accuracy, box + per-run dots + test
  fig5_grok_speed              epochs to reach 80% accuracy at best lr, box + dots

Usage (from repo root): pass the sub-run's output dir as arg or via EXP01_OUTPUT_DIR.
  uv run python .../make_figures.py scott/.../subruns/03_full_fleet/outputs
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.titlesize": 11,
    "legend.frameon": False, "legend.fontsize": 8.5,
})

CONN, CTRL = "#1f77b4", "#7f7f7f"
ARM_LABEL = {"connectome": "MB connectome", "control": "degree-matched control"}
CHANCE = 1 / 32  # vocab = 32

HERE = Path(__file__).resolve().parent          # .../scott/experiment_01_mb_mqar_degree_matched
REPO_ROOT = HERE.parents[1]                      # repo root (scott/<exp>/ is two levels down)


def resolve_outdir() -> Path:
    sel = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "EXP01_OUTPUT_DIR", str(HERE / "subruns" / "03_full_fleet" / "outputs"))
    p = Path(sel)
    if not p.is_absolute():
        p = (REPO_ROOT / sel) if (REPO_ROOT / sel).exists() else (HERE / sel)
    return p


def fmt_lr(lr: float) -> str:
    m, e = f"{lr:.0e}".split("e")
    return f"{int(float(m))}e{int(e)}"


def load(outdir: Path):
    rows = [json.load(open(p)) for p in glob.glob(str(outdir / "runs" / "*" / "result.json"))]
    if not rows:
        raise SystemExit(f"no result.json under {outdir/'runs'}")
    return rows


def cell(rows, arm, lr=None, key="test_acc"):
    return np.array([r[key] for r in rows if r["arm"] == arm and (lr is None or r["lr"] == lr)])


def curves(rows, arm, lr):
    return [r["curve"] for r in rows if r["arm"] == arm and r["lr"] == lr and r["curve"]]


def mean_band(curve_list, L):
    """Forward-fill each curve to length L (early-stop = plateau) then mean +/- SD."""
    arr = np.full((len(curve_list), L), np.nan)
    for i, c in enumerate(curve_list):
        c = np.asarray(c, float)
        n = min(len(c), L)
        arr[i, :n] = c[:n]
        if n < L:
            arr[i, n:] = c[-1]
    return np.nanmean(arr, 0), np.nanstd(arr, 0)


def best_lr(rows, arm, lrs):
    return max(lrs, key=lambda lr: cell(rows, arm, lr, "best_val_acc").mean())


def perm_p(conn, ctrl):
    """One-sided empirical-null: fraction of control graphs >= connectome mean (+1 smoothing)."""
    return (np.sum(np.asarray(ctrl) >= np.mean(conn)) + 1) / (len(ctrl) + 1)


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def lr_colors(lrs):
    return {lr: c for lr, c in zip(lrs, plt.cm.viridis(np.linspace(0.05, 0.85, len(lrs))))}


# ---------------------------------------------------------------------------- figures
def fig1_curves_by_lr(rows, lrs, arms, rho, figdir):
    cols = lr_colors(lrs)
    L = min(300, max(r["epochs_ran"] for r in rows))
    fig, axes = plt.subplots(1, len(arms), figsize=(4.7 * len(arms), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, arm in zip(axes, arms):
        for lr in lrs:
            cl = curves(rows, arm, lr)
            if not cl:
                continue
            m, sd = mean_band(cl, L)
            x = np.arange(1, L + 1)
            ax.plot(x, m, color=cols[lr], lw=2, label=fmt_lr(lr))
            ax.fill_between(x, m - sd, m + sd, color=cols[lr], alpha=0.13, lw=0)
        ax.axhline(CHANCE, color="k", ls=":", lw=1)
        ax.set_title(ARM_LABEL[arm])
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("recall accuracy")
    axes[0].legend(title="learning rate", loc="upper left")
    axes[-1].text(L, CHANCE + 0.01, "chance", ha="right", va="bottom", fontsize=7, color="k")
    fig.tight_layout()
    _save(fig, figdir, "fig1_learning_curves_by_lr")


def fig2_best_curves(rows, lrs, rho, figdir):
    L = min(300, max(r["epochs_ran"] for r in rows))
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for arm, color in (("connectome", CONN), ("control", CTRL)):
        blr = best_lr(rows, arm, lrs)
        m, sd = mean_band(curves(rows, arm, blr), L)
        x = np.arange(1, L + 1)
        ax.plot(x, m, color=color, lw=2.4, label=f"{ARM_LABEL[arm]} (lr {fmt_lr(blr)})")
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15, lw=0)
    ax.axhline(CHANCE, color="k", ls=":", lw=1)
    ax.text(L, CHANCE + 0.01, "chance", ha="right", va="bottom", fontsize=7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("recall accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Best learning rate per arm")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, figdir, "fig2_best_lr_curves")


def fig3_bars_by_lr(rows, lrs, figdir):
    x = np.arange(len(lrs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.4 * len(lrs) + 2.2, 4.2))
    for off, arm, color in ((-w / 2, "connectome", CONN), (w / 2, "control", CTRL)):
        means = [cell(rows, arm, lr).mean() for lr in lrs]
        sds = [cell(rows, arm, lr).std() for lr in lrs]
        ax.bar(x + off, means, w, yerr=sds, color=color, capsize=3,
               error_kw=dict(lw=1), label=ARM_LABEL[arm])
    # within-lr connectome vs control test (Mann-Whitney, two-sided)
    for i, lr in enumerate(lrs):
        c, k = cell(rows, "connectome", lr), cell(rows, "control", lr)
        p = mannwhitneyu(c, k, alternative="two-sided").pvalue
        y = max(c.mean() + c.std(), k.mean() + k.std()) + 0.04
        ax.text(i, y, stars(p), ha="center", va="bottom", fontsize=9)
    ax.axhline(CHANCE, color="k", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([fmt_lr(lr) for lr in lrs])
    ax.set_xlabel("learning rate")
    ax.set_ylabel("final recall accuracy")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, figdir, "fig3_final_acc_by_lr")


def fig4_best_box(rows, lrs, figdir):
    arms = ["connectome", "control"]
    blr = {a: best_lr(rows, a, lrs) for a in arms}
    data = [cell(rows, a, blr[a]) for a in arms]
    colors = [CONN, CTRL]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=1.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set(facecolor=c, alpha=0.25, edgecolor=c, lw=1.3)
    rng = np.random.default_rng(0)
    for i, (vals, c) in enumerate(zip(data, colors), start=1):
        ax.scatter(i + (rng.random(len(vals)) - 0.5) * 0.22, vals, s=28, color=c,
                   edgecolor="white", linewidth=0.5, zorder=3)
    p = perm_p(data[0], data[1])
    pr = mannwhitneyu(data[0], data[1], alternative="two-sided").pvalue
    ax.axhline(CHANCE, color="k", ls=":", lw=1)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{ARM_LABEL[a]}\n(lr {fmt_lr(blr[a])}, n={len(d)})"
                        for a, d in zip(arms, data)])
    ax.set_ylabel("final recall accuracy")
    ax.set_ylim(0, 1)
    top = max(np.max(data[0]), np.max(data[1]))
    ax.plot([1, 2], [top + 0.05] * 2, color="k", lw=1)
    ax.text(1.5, top + 0.06, f"permutation p = {p:.3f}  ({stars(p)})",
            ha="center", va="bottom", fontsize=8.5)
    ax.set_title("Final accuracy at best learning rate")
    fig.tight_layout()
    _save(fig, figdir, "fig4_best_lr_final_acc")
    return p, pr, blr


def fig5_grok(rows, lrs, figdir, thr="0.80"):
    arms = ["connectome", "control"]
    blr = {a: best_lr(rows, a, lrs) for a in arms}
    data, reached = [], []
    for a in arms:
        ep = [r["grok"][thr]["epoch"] for r in rows
              if r["arm"] == a and r["lr"] == blr[a] and r["grok"][thr]["epoch"] is not None]
        tot = sum(1 for r in rows if r["arm"] == a and r["lr"] == blr[a])
        data.append(np.array(ep, float))
        reached.append((len(ep), tot))
    if not any(len(d) for d in data):
        return  # nobody reached the threshold; skip
    colors = [CONN, CTRL]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    pos = [i for i, d in enumerate(data, 1) if len(d)]
    bp = ax.boxplot([d for d in data if len(d)], positions=pos, widths=0.55,
                    patch_artist=True, showfliers=False, medianprops=dict(color="k", lw=1.4))
    for patch, c in zip(bp["boxes"], [colors[i - 1] for i in pos]):
        patch.set(facecolor=c, alpha=0.25, edgecolor=c, lw=1.3)
    rng = np.random.default_rng(1)
    for i, (vals, c) in enumerate(zip(data, colors), start=1):
        if len(vals):
            ax.scatter(i + (rng.random(len(vals)) - 0.5) * 0.22, vals, s=28, color=c,
                       edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{ARM_LABEL[a]}\n({r}/{t} reached)" for a, (r, t) in zip(arms, reached)])
    ax.set_ylabel(f"epochs to {int(float(thr)*100)}% accuracy")
    ax.set_title("Learning speed at best learning rate")
    fig.tight_layout()
    _save(fig, figdir, "fig5_grok_speed")


def _save(fig, figdir, name):
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main():
    outdir = resolve_outdir()
    figdir = outdir.parent / "figures"
    rows = load(outdir)
    lrs = sorted(set(r["lr"] for r in rows))
    arms = [a for a in ("connectome", "control") if any(r["arm"] == a for r in rows)]
    rho = None
    for f in (outdir / "analysis.json", outdir / "manifest.json"):
        if f.exists():
            rho = json.load(open(f)).get("target_rho", rho)
    print(f"figures for {outdir.relative_to(REPO_ROOT)}  (lrs={[fmt_lr(l) for l in lrs]})")
    fig1_curves_by_lr(rows, lrs, arms, rho, figdir)
    fig2_best_curves(rows, lrs, rho, figdir)
    fig3_bars_by_lr(rows, lrs, figdir)
    p, pr, blr = fig4_best_box(rows, lrs, figdir)
    fig5_grok(rows, lrs, figdir)
    bc = cell(rows, "connectome", blr["connectome"])
    kc = cell(rows, "control", blr["control"])
    print(f"  best lr: connectome {fmt_lr(blr['connectome'])} ({bc.mean():.3f}+/-{bc.std():.3f}), "
          f"control {fmt_lr(blr['control'])} ({kc.mean():.3f}+/-{kc.std():.3f})")
    print(f"  best-lr test: permutation p={p:.4f}, Mann-Whitney p={pr:.2e}")


if __name__ == "__main__":
    main()
