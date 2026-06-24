#!/usr/bin/env python3
"""Focused figures for the Experiment 2 *dense eigenvector-structure* follow-up.

These isolate the one question the eigvec controls were built to answer: is the connectome's
MQAR advantage in its specific SPARSE WIRING, or would any dense substrate that merely shares the
connectome's eigen-DIRECTIONS (same orthogonal Schur basis, same trainable-param budget) do as
well? So every panel shows ONLY the connectome vs its two dense surrogates, split by arm:

  core arm : core            vs  eigvec_matched_core   vs  eigvec_shuffle_core
  full arm : full            vs  eigvec_matched_full   vs  eigvec_shuffle_full

The sparse degree/random/full-degree controls (in make_figures.py) are deliberately left out here
so the eigvec comparison stands on its own. Same conventions as make_figures.py: best learning
rate chosen per condition by validation accuracy (never test); fig cohort = completed runs at that
lr (plateau-cut runs excluded, as in the main figures); dotted chance line; minimal on-figure text.

Figures (written into the sibling figures/ dir, eigvec_ prefix so they don't collide with fig1-5):
  eigvec_fig1_acc_by_lr   final accuracy per lr, connectome vs both dense surrogates  [raw diagnostic, all runs]
  eigvec_fig2_final_acc   final test accuracy at each condition's best lr (box + dots) + connectome-vs-surrogate p
  eigvec_fig3_curves      best-lr mean val-accuracy curve per condition (band = +/-1 SD) — late-grok check
  eigvec_fig4_wallclock   total training wall-clock per condition (the dense-substrate cost)

Usage (from repo root):
  uv run python scott/experiment_02_mb_core_pruning/make_figures_eigvec.py scott/experiment_02_mb_core_pruning/outputs
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

# Each arm: (title, connectome condition, [dense surrogate conditions]). The connectome is the
# real sparse substrate; the surrogates are dense Schur-basis scaffolds + nnz(connectome) random
# trainable edges, so trainable-param count matches the connectome exactly.
ARMS = [
    ("core arm — 5.6k MB core", "core", ["eigvec_matched_core", "eigvec_shuffle_core"]),
    ("full arm — 14k substrate", "full", ["eigvec_matched_full", "eigvec_shuffle_full"]),
]

COLOR = {
    "core": "#1f77b4", "full": "#111111",
    "eigvec_matched_core": "#d62728", "eigvec_matched_full": "#d62728",
    "eigvec_shuffle_core": "#2ca02c", "eigvec_shuffle_full": "#2ca02c",
}
LABEL = {
    "core": "MB core (sparse)", "full": "full 14k (sparse)",
    "eigvec_matched_core": "eigvec-matched\n(directions only)",
    "eigvec_matched_full": "eigvec-matched\n(directions only)",
    "eigvec_shuffle_core": "eigvec-shuffle\n(+ real spectrum)",
    "eigvec_shuffle_full": "eigvec-shuffle\n(+ real spectrum)",
}
# Short tags for inline stat notes (full meaning is on the x-axis labels + caption: both dense
# surrogates keep the connectome's directions; matched gives them random eigenvalues, shuffle keeps
# the real eigenvalue spectrum but mis-pairs which direction gets which).
SHORT = {
    "eigvec_matched_core": "matched", "eigvec_matched_full": "matched",
    "eigvec_shuffle_core": "shuffle", "eigvec_shuffle_full": "shuffle",
}
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


def present(rows, cond) -> bool:
    return any(r["condition"] == cond for r in rows)


def cell(rows, cond, lr=None, key="test_acc"):
    return np.array([r[key] for r in rows if r["condition"] == cond and (lr is None or r["lr"] == lr)], float)


def best_lr(rows, cond, lrs):
    """Learning rate with the highest mean validation accuracy for this condition (never test)."""
    return max(lrs, key=lambda lr: cell(rows, cond, lr, "best_val_acc").mean()
               if cell(rows, cond, lr, "best_val_acc").size else -1)


# Cohort for the box/curve/wall-clock figs: each condition at its OWN best-val lr (the surrogates
# peak at different lrs than the connectome — e.g. eigvec_matched_core at 3e-3, the connectome at
# 1e-3 — so a single shared lr would understate whichever condition isn't at its optimum). Within
# that lr, drop plateau-cut runs, exactly as make_figures.py does, so a patience=40 truncation on
# the (original) connectome runs isn't scored as a fair finish. The eigvec runs ran patience-off so
# none are plateau-cut.
def comp_at_best(rows, cond, lrs):
    blr = best_lr(rows, cond, lrs)
    rs = [r for r in rows if r["condition"] == cond and r["lr"] == blr
          and r.get("stopped_reason") != "plateau"]
    return rs, blr


def comp_vals(rows, cond, lrs, key="test_acc"):
    rs, blr = comp_at_best(rows, cond, lrs)
    return np.array([r[key] for r in rs], float), blr


def grok_curve(rows, cond, lrs):
    rs, blr = comp_at_best(rows, cond, lrs)
    return [r["curve"] for r in rs if r.get("curve")], blr


def mwu(a, b):
    """Two-sided Mann-Whitney U p (the surrogate can land either side of the connectome)."""
    if len(a) < 1 or len(b) < 1:
        return float("nan")
    try:
        return mannwhitneyu(a, b, alternative="two-sided").pvalue
    except ValueError:  # all-equal inputs
        return float("nan")


def stars(p):
    if np.isnan(p):
        return "n/a"
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


def arm_conds(rows, conn, surrs):
    """conn + whichever surrogates have landed (so this works while --collect is still running)."""
    return [conn] + [s for s in surrs if present(rows, s)]


def eigvec_fig1_acc_by_lr(rows, lrs, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(2.0 * len(lrs) + 4.0, 4.4), sharey=True)
    x = np.arange(len(lrs))
    for ax, (title, conn, surrs) in zip(np.atleast_1d(axes), ARMS):
        conds = arm_conds(rows, conn, surrs)
        w = 0.8 / len(conds)
        for j, cond in enumerate(conds):
            off = (j - (len(conds) - 1) / 2) * w
            means = [cell(rows, cond, lr).mean() if cell(rows, cond, lr).size else np.nan for lr in lrs]
            sds = [cell(rows, cond, lr).std() if cell(rows, cond, lr).size else 0 for lr in lrs]
            ax.bar(x + off, means, w, yerr=sds, color=COLOR[cond], capsize=2.5,
                   error_kw=dict(lw=0.9), label=LABEL[cond])
        ax.axhline(CHANCE, color="k", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([fmt_lr(lr) for lr in lrs])
        ax.set_xlabel("learning rate")
        ax.set_ylim(0, 1.08)
        ax.set_title(title)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("final recall accuracy")
    fig.suptitle("Sparse connectome vs dense eigen-direction surrogates, by learning rate (all runs)",
                 fontsize=10.5)
    fig.tight_layout()
    _save(fig, figdir, "eigvec_fig1_acc_by_lr")


def eigvec_fig2_final_acc(rows, lrs, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True)
    for ax, (title, conn, surrs) in zip(np.atleast_1d(axes), ARMS):
        conds = arm_conds(rows, conn, surrs)
        data, blrs = [], []
        for c in conds:
            v, blr = comp_vals(rows, c, lrs)
            data.append(v)
            blrs.append(blr)
        pos = [i for i, d in enumerate(data, 1) if len(d)]
        if pos:
            bp = ax.boxplot([d for d in data if len(d)], positions=pos, widths=0.6,
                            patch_artist=True, showfliers=False, medianprops=dict(color="k", lw=1.4))
            for patch, i in zip(bp["boxes"], pos):
                c = conds[i - 1]
                patch.set(facecolor=COLOR[c], alpha=0.25, edgecolor=COLOR[c], lw=1.3)
        rng = np.random.default_rng(0)
        for i, (vals, c) in enumerate(zip(data, conds), start=1):
            if len(vals):
                ax.scatter(i + (rng.random(len(vals)) - 0.5) * 0.22, vals, s=26, color=COLOR[c],
                           edgecolor="white", linewidth=0.5, zorder=3)
        ax.axhline(CHANCE, color="k", ls=":", lw=1)
        ax.set_xticks(range(1, len(conds) + 1))
        ax.set_xticklabels([f"{LABEL[c]}\nlr {fmt_lr(b)} (n={len(d)})"
                            for c, b, d in zip(conds, blrs, data)], fontsize=7.6)
        # connectome vs each surrogate, two-sided MWU; note the direction of the difference
        conn_v = data[0]
        notes = []
        for c, v in zip(conds[1:], data[1:]):
            p = mwu(conn_v, v)
            d = (np.mean(v) - np.mean(conn_v)) if len(v) and len(conn_v) else float("nan")
            arrow = "↑" if d > 0 else "↓"  # surrogate above / below the connectome
            notes.append(f"vs {SHORT[c]}: {arrow}{abs(d):.3f} ({stars(p)})")
        ax.set_title(f"{title}\nconnectome " + ",  ".join(notes), fontsize=8.6)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("final recall accuracy")
    fig.suptitle("Final accuracy at each condition's best lr (completed runs)", fontsize=10.5)
    fig.tight_layout()
    _save(fig, figdir, "eigvec_fig2_final_acc")


def eigvec_fig3_curves(rows, lrs, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    for ax, (title, conn, surrs) in zip(np.atleast_1d(axes), ARMS):
        conds = arm_conds(rows, conn, surrs)
        curves = {c: grok_curve(rows, c, lrs) for c in conds}
        present_curves = [cl for c in conds for (cl, _) in [curves[c]] if cl]
        if not present_curves:
            continue
        L = min(300, max(len(c) for cl in present_curves for c in cl))
        x = np.arange(1, L + 1)
        for c in conds:
            cl, blr = curves[c]
            if not cl:
                continue
            m, sd = mean_band(cl, L)
            ax.plot(x, m, color=COLOR[c], lw=2.2, label=f"{LABEL[c]} (lr {fmt_lr(blr)})")
            ax.fill_between(x, m - sd, m + sd, color=COLOR[c], alpha=0.12, lw=0)
        ax.axhline(CHANCE, color="k", ls=":", lw=1)
        ax.text(L, CHANCE + 0.01, "chance", ha="right", va="bottom", fontsize=7)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("recall accuracy")
    fig.suptitle("MQAR learning curves at best lr (completed runs)", fontsize=10.5)
    fig.tight_layout()
    _save(fig, figdir, "eigvec_fig3_curves")


def eigvec_fig4_wallclock(rows, lrs, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True)
    for ax, (title, conn, surrs) in zip(np.atleast_1d(axes), ARMS):
        conds = arm_conds(rows, conn, surrs)
        data, blrs = [], []
        for c in conds:
            v, blr = comp_vals(rows, c, lrs, "total_wall_s")
            data.append(v)
            blrs.append(blr)
        pos = [i for i, d in enumerate(data, 1) if len(d)]
        if pos:
            bp = ax.boxplot([d for d in data if len(d)], positions=pos, widths=0.6,
                            patch_artist=True, showfliers=False, medianprops=dict(color="k", lw=1.4))
            for patch, i in zip(bp["boxes"], pos):
                c = conds[i - 1]
                patch.set(facecolor=COLOR[c], alpha=0.25, edgecolor=COLOR[c], lw=1.3)
        rng = np.random.default_rng(2)
        for i, (vals, c) in enumerate(zip(data, conds), start=1):
            if len(vals):
                ax.scatter(i + (rng.random(len(vals)) - 0.5) * 0.22, vals, s=26, color=COLOR[c],
                           edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xticks(range(1, len(conds) + 1))
        ax.set_xticklabels([f"{LABEL[c]}\nlr {fmt_lr(b)} (n={len(d)})"
                            for c, b, d in zip(conds, blrs, data)], fontsize=7.6)
        ax.set_title(title)
    axes[0].set_ylabel("total training wall-clock (s)")
    fig.suptitle("Training wall-clock: sparse connectome vs dense surrogates (one run per GPU)\n"
                 "dense recurrence is denser GEMM — the practical cost of matching params with a dense scaffold",
                 fontsize=9.2)
    fig.tight_layout()
    _save(fig, figdir, "eigvec_fig4_wallclock")


def main():
    outdir = resolve_outdir()
    figdir = outdir.parent / "figures"
    rows = load(outdir)
    lrs = sorted(set(r["lr"] for r in rows))
    # keep only rows in the eigvec comparison (connectome arms + their surrogates)
    keep = {c for _, conn, surrs in ARMS for c in [conn, *surrs]}
    rows = [r for r in rows if r["condition"] in keep]
    have = sorted({r["condition"] for r in rows})
    print(f"eigvec figures for {outdir.relative_to(REPO_ROOT)}  "
          f"(lrs={[fmt_lr(l) for l in lrs]}, {len(rows)} runs; conditions present: {have})")
    eigvec_fig1_acc_by_lr(rows, lrs, figdir)
    eigvec_fig2_final_acc(rows, lrs, figdir)
    eigvec_fig3_curves(rows, lrs, figdir)
    eigvec_fig4_wallclock(rows, lrs, figdir)
    print("\n  best-lr summary (completed runs):")
    for _, conn, surrs in ARMS:
        for c in arm_conds(rows, conn, surrs):
            v, blr = comp_vals(rows, c, lrs)
            if v.size:
                tag = "" if c == conn else f"  (vs {conn}: p={mwu(comp_vals(rows, conn, lrs)[0], v):.3f})"
                print(f"    {c:22s} lr{fmt_lr(blr)}: acc {v.mean():.3f}±{v.std():.3f} (n={v.size}){tag}")


if __name__ == "__main__":
    main()
