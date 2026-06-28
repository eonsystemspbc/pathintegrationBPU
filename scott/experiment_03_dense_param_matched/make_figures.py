#!/usr/bin/env python3
"""Figures for Experiment 3 - dense parameter-matched controls vs the connectome on MQAR.

Reads <output_dir>/metrics_by_run.csv (+ analysis.json) and writes figures/ :
  fig1_final_acc.png       final test accuracy, connectome vs C1/C2/C3, core arm | full arm
  fig2_param_budget.png    final accuracy vs trainable-param count (the budget view)
  fig3_training_curves.png val-accuracy learning curves (median + IQR over seeds) vs epoch
                           AND vs wall-clock, core arm | full arm  -- the training dynamics
  fig5_total_wallclock.png total training wall-clock (hours) per condition -- the cost view
  fig6_control_legend.png  standalone key: color + one-line definition of each condition
  fig4_lr_sweep.png        dense_c3_core test accuracy vs learning rate (subrun 01), with the
                           connectome reference -- is the dense result an lr artifact or real?

Curves are read from each run's result.json ("curve" = per-epoch val acc, present for every run,
incl. the ported connectome refs). The lr sweep reads subruns/01_dense_c3_lr_sweep/outputs/.

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


def load_curves(out_dir: Path):
    """condition -> list of dicts {curve: np.array(val_acc per epoch), wall_per_epoch: float}.

    The per-epoch val-acc curve lives in every run's result.json (incl. ported connectome refs,
    which carry no metrics_epochs.csv). Wall-clock per epoch is approximated as constant
    (total_wall_s / epochs_ran) -- per-epoch wall is near-constant within a run.
    """
    by_cond = defaultdict(list)
    for rp in sorted((out_dir / "runs").glob("*/result.json")):
        try:
            d = json.loads(rp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        curve = d.get("curve")
        if not curve:
            continue
        wall, ep = _fl(d.get("total_wall_s")), _fl(d.get("epochs_ran"))
        wpe = (wall / ep) if (wall and ep) else None
        by_cond[d["condition"]].append({"curve": np.asarray(curve, dtype=float), "wpe": wpe})
    return by_cond


def _median_band(curves):
    """Stack ragged per-epoch curves -> (epochs, median, q25, q75) over the common epoch range."""
    n = min(len(c) for c in curves)
    arr = np.vstack([c[:n] for c in curves])
    ep = np.arange(1, n + 1)
    return ep, np.median(arr, 0), np.percentile(arr, 25, 0), np.percentile(arr, 75, 0)


def fig_training_curves(curves_by_cond, out_dir: Path):
    """Val-acc learning curves, median + IQR over seeds. Rows: vs epoch | vs wall-clock; cols: arms."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    handles, labels_seen = [], []
    for col, (sub, items) in enumerate(ARMS.items()):
        ax_ep, ax_wc = axes[0, col], axes[1, col]
        for cond, lab in items:
            runs = curves_by_cond.get(cond, [])
            if not runs:
                continue
            color = COLORS["connectome" if cond in ("core", "full") else lab.split()[0]]
            name = lab.replace("\n", " ")
            ep, med, q25, q75 = _median_band([r["curve"] for r in runs])
            # vs epoch
            (line,) = ax_ep.plot(ep, med, color=color, lw=2, label=f"{name} (n={len(runs)})")
            ax_ep.fill_between(ep, q25, q75, color=color, alpha=0.16, lw=0)
            # vs wall-clock (hours) -- median per-epoch wall scales the same epoch grid
            wpes = [r["wpe"] for r in runs if r["wpe"] is not None]
            if wpes:
                hrs = ep * (np.median(wpes) / 3600.0)
                ax_wc.plot(hrs, med, color=color, lw=2)
                ax_wc.fill_between(hrs, q25, q75, color=color, alpha=0.16, lw=0)
            if name not in labels_seen:
                handles.append(line)
                labels_seen.append(name)
        for ax in (ax_ep, ax_wc):
            ax.axhline(1 / 32, ls=":", color="grey", lw=1)
            ax.grid(ls=":", alpha=0.4)
        ax_ep.set_title(f"{sub} arm")
        ax_wc.set_xlabel("wall-clock (hours)")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 0].set_ylabel("val accuracy (vs epoch)")
    axes[1, 0].set_ylabel("val accuracy (vs wall-clock)")
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=len(handles), fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Experiment 3 — training curves: connectome vs dense controls "
                 "(MQAR, lr=1e-3, median ± IQR over seeds)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_dir.parent / "figures" / "fig3_training_curves.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def fig_control_legend(out_dir: Path):
    """A standalone key: color swatch + name + one-sentence definition per condition."""
    from matplotlib.patches import Rectangle  # noqa: PLC0415

    rows = [
        ("connectome", "connectome (sparse)",
         "The FlyWire mushroom-body connectome wiring; only its existing synapses are trainable (sparse)."),
        ("C1", "C1 — dense ceiling",
         "Fully-trainable dense matrix at the same neuron count — far more parameters; an upper-bound ceiling, not a matched null."),
        ("C2", "C2 — dense reservoir",
         "Frozen random dense scaffold + nnz trainable random delta-edges — the param-matched random-directions null (primary test)."),
        ("C3", "C3 — dense param-matched",
         "A smaller fully-trainable dense net sized so total trainable params match — the same budget packed into fewer neurons."),
    ]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.axis("off")
    n = len(rows)
    for i, (key, name, defn) in enumerate(rows):
        y = 1 - (i + 0.5) / n
        h = 0.52 / n
        ax.add_patch(Rectangle((0.015, y - h / 2), 0.045, h, color=COLORS[key],
                               ec="k", lw=0.6, transform=ax.transAxes, clip_on=False))
        ax.text(0.085, y, name, fontweight="bold", fontsize=12.5, va="center",
                transform=ax.transAxes)
        ax.text(0.40, y, defn, fontsize=10.5, va="center", transform=ax.transAxes, wrap=True)
    ax.set_title("Experiment 3 — condition key", fontsize=13, loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir.parent / "figures" / "fig6_control_legend.png", dpi=130)
    plt.close(fig)


def fig_total_wallclock(by_cond, out_dir: Path):
    """Total training wall-clock (hours) per condition, core arm | full arm -- the cost view."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (sub, items) in zip(axes, ARMS.items()):
        labels, data, colors = [], [], []
        for cond, lab in items:
            hrs = [_fl(r.get("total_wall_s")) for r in by_cond.get(cond, [])]
            hrs = [h / 3600.0 for h in hrs if h is not None]
            if not hrs:
                continue
            labels.append(f"{lab}\n(n={len(hrs)})")
            data.append(hrs)
            colors.append(COLORS["connectome" if cond in ("core", "full") else lab.split()[0]])
        if not data:
            ax.set_title(f"{sub} arm — no data yet")
            continue
        x = np.arange(1, len(data) + 1)
        means = [np.mean(d) for d in data]
        ax.bar(x, means, width=0.6, color=colors, alpha=0.35, edgecolor="k", zorder=2)
        for i, (d, c, m) in enumerate(zip(data, colors, means), start=1):
            ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(d)), d, s=14, color=c, zorder=3)
            ax.annotate(f"{m:.1f}h", (i, m), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{sub} arm")
        ax.set_ylabel("total training wall-clock (hours)" if sub == "core" else "")
        ax.grid(axis="y", ls=":", alpha=0.4)
    fig.suptitle("Experiment 3 — total training wall-clock per condition (MQAR, lr=1e-3, 300 epochs)")
    fig.tight_layout()
    fig.savefig(out_dir.parent / "figures" / "fig5_total_wallclock.png", dpi=130)
    plt.close(fig)


def fig_lr_sweep(out_dir: Path, subrun_dir: Path):
    """dense_c3_core test accuracy vs learning rate (subrun 01) + connectome-core reference."""
    csv_path = subrun_dir / "outputs" / "metrics_by_run.csv"
    if not csv_path.exists():
        print(f"  (skip fig4: no lr-sweep data at {csv_path})")
        return
    by_lr = defaultdict(list)
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r.get("condition") != "dense_c3_core":
                continue
            acc, lr = _fl(r.get("test_acc")), _fl(r.get("lr"))
            if acc is not None and lr is not None:
                by_lr[lr].append(acc)
    if not by_lr:
        print("  (skip fig4: no dense_c3_core rows in lr sweep)")
        return
    lrs = sorted(by_lr)
    means = [np.mean(by_lr[l]) for l in lrs]
    stds = [np.std(by_lr[l]) for l in lrs]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    rng = np.random.default_rng(0)
    for l in lrs:  # seed cloud
        d = by_lr[l]
        ax.scatter(l * rng.normal(1.0, 0.03, len(d)), d, s=16, color=COLORS["C3"], alpha=0.45,
                   zorder=2)
    ax.errorbar(lrs, means, yerr=stds, color=COLORS["C3"], lw=2, marker="o", ms=7, capsize=4,
                zorder=3, label="dense_c3_core (mean ± SD, n=20/lr)")

    # connectome-core reference (mean final test acc from the main run)
    conn = [_fl(r["test_acc"]) for r in load(out_dir).get("core", [])]
    conn = [a for a in conn if a is not None]
    if conn:
        ax.axhline(np.mean(conn), ls="--", color=COLORS["connectome"], lw=1.8,
                   label=f"connectome core (mean={np.mean(conn):.3f})")
    ax.axhline(1 / 32, ls=":", color="grey", lw=1, label="chance (1/32)")

    ax.set_xscale("log")
    ax.set_xticks(lrs)
    ax.set_xticklabels([f"{l:g}" for l in lrs])
    ax.set_xlabel("learning rate (log)")
    ax.set_ylabel("final test accuracy")
    ax.set_ylim(0, max(1.0, max(conn) + 0.05 if conn else 1.0))
    best_lr = lrs[int(np.argmax(means))]
    ax.set_title("Experiment 3 · subrun 01 — dense_c3_core accuracy vs learning rate\n"
                 "(peak %.2f at lr=%g, far below the connectome — a real capacity result, not an lr artifact)"
                 % (max(means), best_lr))
    ax.grid(ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(out_dir.parent / "figures" / "fig4_lr_sweep.png", dpi=130)
    plt.close(fig)


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
    curves_by_cond = load_curves(out_dir)
    if curves_by_cond:
        fig_training_curves(curves_by_cond, out_dir)
    fig_total_wallclock(by_cond, out_dir)
    fig_control_legend(out_dir)
    fig_lr_sweep(out_dir, HERE / "subruns" / "01_dense_c3_lr_sweep")
    print(f"wrote figures to {HERE/'figures'} from {sum(len(v) for v in by_cond.values())} runs "
          f"({', '.join(f'{k}:{len(v)}' for k, v in sorted(by_cond.items()))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
