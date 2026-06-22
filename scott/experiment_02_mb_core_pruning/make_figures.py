#!/usr/bin/env python3
"""Publication-style figures for Experiment 2 (MB-core pruning vs full 14k + matched controls
on MQAR, spectral radius matched across all conditions).

Reads <output-dir>/runs/*/result.json and writes figures into the sibling figures/ dir.
Each unit's best learning rate is chosen by validation accuracy (never test), matching the
engine's analysis. Style follows Exp 1: dotted chance line, minimal on-figure text.

Figures:
  fig1_curves_best_lr     best-lr mean val-accuracy curve per condition (band = +/-1 SD)   [headline]
  fig2_final_acc          final test accuracy per condition (box + dots) + the two perm-p's
  fig3_grok_epochs        epochs to 80% accuracy per condition (learning speed)
  fig4_wallclock          total training wall-clock per condition (the pruning speed-up)
  fig5_acc_by_lr          final accuracy per lr: core vs core_degree (Exp 1 replication) + core vs full

Usage (from repo root): pass the sub-run's output dir as arg.
  uv run python scott/experiment_02_mb_core_pruning/make_figures.py scott/experiment_02_mb_core_pruning/outputs
"""
from __future__ import annotations

import glob
import json
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

CONDS = ["core", "full", "full_degree", "core_degree", "random_subset"]
COLOR = {"core": "#1f77b4", "full": "#111111", "full_degree": "#9467bd",
         "core_degree": "#7f7f7f", "random_subset": "#ff7f0e"}
LABEL = {"core": "MB core (5.6k)", "full": "full (14k)",
         "full_degree": "degree-matched 14k", "core_degree": "degree-matched core",
         "random_subset": "random 5.6k subset"}
CHANCE = 1 / 32

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def resolve_outdir() -> Path:
    sel = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "outputs")
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


def cell(rows, cond, lr=None, key="test_acc"):
    return np.array([r[key] for r in rows if r["condition"] == cond and (lr is None or r["lr"] == lr)], float)


def best_lr(rows, cond, lrs):
    return max(lrs, key=lambda lr: cell(rows, cond, lr, "best_val_acc").mean()
               if cell(rows, cond, lr, "best_val_acc").size else -1)


def best_rows(rows, cond, lrs):
    blr = best_lr(rows, cond, lrs)
    return [r for r in rows if r["condition"] == cond and r["lr"] == blr], blr


def grok_epochs(rows, cond, lrs, thr="0.80"):
    rs, _ = best_rows(rows, cond, lrs)
    return np.array([r["grok"][thr]["epoch"] for r in rs if r["grok"][thr]["epoch"] is not None], float), len(rs)


def perm_p(conn, ctrl):
    if not len(conn) or not len(ctrl):
        return float("nan")
    return (np.sum(np.asarray(ctrl) >= np.mean(conn)) + 1) / (len(ctrl) + 1)


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def mean_band(curve_list, L):
    arr = np.full((len(curve_list), L), np.nan)
    for i, c in enumerate(curve_list):
        c = np.asarray(c, float)
        n = min(len(c), L)
        arr[i, :n] = c[:n]
        if n < L:
            arr[i, n:] = c[-1]
    return np.nanmean(arr, 0), np.nanstd(arr, 0)


def _save(fig, figdir, name):
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def fig1_curves(rows, lrs, figdir):
    L = min(300, max(r["epochs_ran"] for r in rows))
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    for cond in CONDS:
        rs, blr = best_rows(rows, cond, lrs)
        cl = [r["curve"] for r in rs if r["curve"]]
        if not cl:
            continue
        m, sd = mean_band(cl, L)
        x = np.arange(1, L + 1)
        ax.plot(x, m, color=COLOR[cond], lw=2.2, label=f"{LABEL[cond]} (lr {fmt_lr(blr)})")
        ax.fill_between(x, m - sd, m + sd, color=COLOR[cond], alpha=0.12, lw=0)
    ax.axhline(CHANCE, color="k", ls=":", lw=1)
    ax.text(L, CHANCE + 0.01, "chance", ha="right", va="bottom", fontsize=7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("recall accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("MQAR learning curves at best learning rate")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, figdir, "fig1_curves_best_lr")


def fig2_final_acc(rows, lrs, figdir):
    data = [cell(rows, c, best_lr(rows, c, lrs)) for c in CONDS]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=1.4))
    for patch, c in zip(bp["boxes"], CONDS):
        patch.set(facecolor=COLOR[c], alpha=0.25, edgecolor=COLOR[c], lw=1.3)
    rng = np.random.default_rng(0)
    for i, (vals, c) in enumerate(zip(data, CONDS), start=1):
        if len(vals):
            ax.scatter(i + (rng.random(len(vals)) - 0.5) * 0.22, vals, s=26, color=COLOR[c],
                       edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(CHANCE, color="k", ls=":", lw=1)
    ax.set_xticks(range(1, len(CONDS) + 1))
    ax.set_xticklabels([f"{LABEL[c]}\n(n={len(d)})" for c, d in zip(CONDS, data)], fontsize=8)
    ax.set_ylabel("final recall accuracy")
    ax.set_ylim(0, 1)
    # permutation p annotations: core vs each control
    core = cell(rows, "core", best_lr(rows, "core", lrs))
    notes = []
    for ctrl in ("core_degree", "random_subset", "full_degree"):
        if ctrl not in CONDS:
            continue
        p = perm_p(core, cell(rows, ctrl, best_lr(rows, ctrl, lrs)))
        notes.append(f"core vs {ctrl.replace('_',' ')}: perm p={p:.3f} ({stars(p)})")
    ax.set_title("Final accuracy by condition\n" + "   |   ".join(notes), fontsize=9)
    fig.tight_layout()
    _save(fig, figdir, "fig2_final_acc")


def fig3_grok(rows, lrs, figdir, thr="0.80"):
    data, reached = [], []
    for c in CONDS:
        ep, tot = grok_epochs(rows, c, lrs, thr)
        data.append(ep)
        reached.append((len(ep), tot))
    if not any(len(d) for d in data):
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    pos = [i for i, d in enumerate(data, 1) if len(d)]
    bp = ax.boxplot([d for d in data if len(d)], positions=pos, widths=0.6,
                    patch_artist=True, showfliers=False, medianprops=dict(color="k", lw=1.4))
    for patch, i in zip(bp["boxes"], pos):
        c = CONDS[i - 1]
        patch.set(facecolor=COLOR[c], alpha=0.25, edgecolor=COLOR[c], lw=1.3)
    rng = np.random.default_rng(1)
    for i, (vals, c) in enumerate(zip(data, CONDS), start=1):
        if len(vals):
            ax.scatter(i + (rng.random(len(vals)) - 0.5) * 0.22, vals, s=26, color=COLOR[c],
                       edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(range(1, len(CONDS) + 1))
    ax.set_xticklabels([f"{LABEL[c]}\n({r}/{t})" for c, (r, t) in zip(CONDS, reached)], fontsize=8)
    ax.set_ylabel(f"epochs to {int(float(thr)*100)}% accuracy")
    ax.set_title("Learning speed (epochs to grok) at best learning rate")
    fig.tight_layout()
    _save(fig, figdir, "fig3_grok_epochs")


def fig4_wallclock(rows, lrs, figdir):
    data = [cell(rows, c, best_lr(rows, c, lrs), "total_wall_s") for c in CONDS]
    if not any(len(d) for d in data):
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=1.4))
    for patch, c in zip(bp["boxes"], CONDS):
        patch.set(facecolor=COLOR[c], alpha=0.25, edgecolor=COLOR[c], lw=1.3)
    rng = np.random.default_rng(2)
    for i, (vals, c) in enumerate(zip(data, CONDS), start=1):
        if len(vals):
            ax.scatter(i + (rng.random(len(vals)) - 0.5) * 0.22, vals, s=26, color=COLOR[c],
                       edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(range(1, len(CONDS) + 1))
    ax.set_xticklabels([LABEL[c] for c in CONDS], fontsize=8)
    ax.set_ylabel("total training wall-clock (s)")
    ax.set_title("Training wall-clock by condition (one run per GPU)")
    fig.tight_layout()
    _save(fig, figdir, "fig4_wallclock")


def fig5_acc_by_lr(rows, lrs, figdir):
    # core vs core_degree (Exp 1 replication at core scale), core vs full, and (if ported)
    # core vs the 14k degree-matched control, grouped bars per lr
    pairs = [("core", "core_degree"), ("core", "full")]
    if "full_degree" in CONDS:
        pairs.append(("core", "full_degree"))
    fig, axes = plt.subplots(1, 2, figsize=(2.0 * len(lrs) + 4.0, 4.2), sharey=True)
    x = np.arange(len(lrs))
    w = 0.38
    for ax, (a, b) in zip(np.atleast_1d(axes), pairs):
        for off, cond in ((-w / 2, a), (w / 2, b)):
            means = [cell(rows, cond, lr).mean() if cell(rows, cond, lr).size else np.nan for lr in lrs]
            sds = [cell(rows, cond, lr).std() if cell(rows, cond, lr).size else 0 for lr in lrs]
            ax.bar(x + off, means, w, yerr=sds, color=COLOR[cond], capsize=3,
                   error_kw=dict(lw=1), label=LABEL[cond])
        for i, lr in enumerate(lrs):
            ca, cb = cell(rows, a, lr), cell(rows, b, lr)
            if ca.size and cb.size:
                p = mannwhitneyu(ca, cb, alternative="two-sided").pvalue
                y = max(np.nanmax([ca.mean() + ca.std(), cb.mean() + cb.std()]), 0) + 0.03
                ax.text(i, y, stars(p), ha="center", va="bottom", fontsize=8)
        ax.axhline(CHANCE, color="k", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([fmt_lr(lr) for lr in lrs])
        ax.set_xlabel("learning rate")
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{LABEL[a]} vs {LABEL[b]}")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("final recall accuracy")
    fig.tight_layout()
    _save(fig, figdir, "fig5_acc_by_lr")


def main():
    outdir = resolve_outdir()
    figdir = outdir.parent / "figures"
    rows = load(outdir)
    lrs = sorted(set(r["lr"] for r in rows))
    # only plot conditions that actually have runs (e.g. full_degree only after porting)
    global CONDS
    CONDS = [c for c in CONDS if any(r["condition"] == c for r in rows)]
    print(f"figures for {outdir.relative_to(REPO_ROOT)}  (lrs={[fmt_lr(l) for l in lrs]}, "
          f"{len(rows)} runs)")
    fig1_curves(rows, lrs, figdir)
    fig2_final_acc(rows, lrs, figdir)
    fig3_grok(rows, lrs, figdir)
    fig4_wallclock(rows, lrs, figdir)
    fig5_acc_by_lr(rows, lrs, figdir)
    for c in CONDS:
        d = cell(rows, c, best_lr(rows, c, lrs))
        if d.size:
            print(f"  {c:14s} best-lr final acc {d.mean():.3f}±{d.std():.3f}")


if __name__ == "__main__":
    main()
