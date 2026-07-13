#!/usr/bin/env python3
"""Figures for Experiment 7 (Antennal Lobe x turbulent gas detection).

Reads <output-dir>/metrics_by_run.csv (concatenated fleet shards) and writes a multi-panel
summary + the committed CSV/JSON into docs/results/antennal_lobe_gas/. The discriminating story
is sample-efficiency + low-concentration detection at a fixed false-alarm rate + detection
latency -- NOT the saturated full-data AUPRC.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
LAT_LABELS = ["pre", "0-5s", "5-10s", "10-30s", "30-60s", ">60s"]
ARM_COLORS = {"connectome": "#c0392b", "degree": "#2980b9", "random": "#27ae60",
              "spectrum": "#8e44ad", "dense": "#e67e22"}
ARM_LABEL = {"connectome": "connectome", "degree": "degree-matched", "random": "ER-random",
             "spectrum": "spectrum-matched", "dense": "dense-Gaussian"}


def agg(df, io, arm, variant, metric):
    """mean, sd, x=fraction for an arm."""
    s = df[(df.io == io) & (df.arm == arm) & (df.variant == variant)]
    g = s.groupby("fraction")[metric].agg(["mean", "std", "count"]).reset_index()
    return g


def panel_sample_efficiency(ax, df, metric, title, ylabel, io="bio"):
    for arm in ("connectome", "degree", "random", "spectrum", "dense"):
        g = agg(df, io, arm, "standard", metric)
        if g.empty:
            continue
        sd = g["std"].fillna(0).to_numpy()
        ax.errorbar(g["fraction"], g["mean"], yerr=sd, marker="o", ms=4, capsize=2,
                    color=ARM_COLORS[arm], lw=2 if arm == "connectome" else 1.3,
                    label=ARM_LABEL[arm], zorder=5 if arm == "connectome" else 3)
    ad = agg(df, "bio", "adapter_only", "standard", metric)
    if not ad.empty:
        ax.axhline(float(ad["mean"].iloc[-1]), ls=":", color="gray", lw=1, label="adapter-only floor")
    ax.set_xscale("log"); ax.set_xticks([5, 10, 25, 50, 100]); ax.set_xticklabels([5, 10, 25, 50, 100])
    ax.set_xlabel("training data (%)"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)


def panel_bar_arms(ax, df, metric, frac, title, ylabel, io="bio"):
    arms = ["connectome", "degree", "random", "spectrum", "dense"]
    means, sds = [], []
    for arm in arms:
        s = df[(df.io == io) & (df.arm == arm) & (df.variant == "standard") & (df.fraction == frac)][metric]
        means.append(s.mean()); sds.append(s.std() if len(s) > 1 else 0.0)
    x = np.arange(len(arms))
    ax.bar(x, means, yerr=sds, capsize=3, color=[ARM_COLORS[a] for a in arms], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.25)


def panel_latency(ax, df, io="bio", frac=100):
    for arm in ("connectome", "degree", "random"):
        rows = df[(df.io == io) & (df.arm == arm) & (df.variant == "standard") & (df.fraction == frac)]
        curves = []
        for _, r in rows.iterrows():
            try:
                d = json.loads(r["low_latency"])
                curves.append([d.get(k) if d.get(k) is not None else np.nan for k in LAT_LABELS])
            except Exception:
                pass
        if not curves:
            continue
        m = np.nanmean(np.array(curves, float), axis=0)
        ax.plot(range(len(LAT_LABELS)), m, marker="o", ms=4, color=ARM_COLORS[arm],
                lw=2 if arm == "connectome" else 1.3, label=ARM_LABEL[arm])
    ax.set_xticks(range(len(LAT_LABELS))); ax.set_xticklabels(LAT_LABELS, fontsize=8)
    ax.set_xlabel("time after plume release"); ax.set_ylabel("detection rate @10% FA")
    ax.set_title("Detection latency (low conc.)", fontsize=10); ax.grid(alpha=0.25)


def panel_bio_vs_generic(ax, df, metric):
    for io, color, ls in [("bio", "#c0392b", "-"), ("generic", "#7f8c8d", "--")]:
        g = agg(df, io, "connectome", "standard", metric)
        if g.empty:
            continue
        ax.errorbar(g["fraction"], g["mean"], yerr=g["std"].fillna(0), marker="s", ms=4,
                    color=color, ls=ls, capsize=2, label=f"{io} I/O")
    ax.set_xscale("log"); ax.set_xticks([5, 10, 25, 50, 100]); ax.set_xticklabels([5, 10, 25, 50, 100])
    ax.set_xlabel("training data (%)"); ax.set_ylabel("low-conc recall @10% FA")
    ax.set_title("Biological vs free I/O (connectome)", fontsize=10); ax.grid(alpha=0.25)


def panel_effect_size(ax, analysis, io="bio"):
    keys = [(f"{io}::test_low_recall_at_fpr10::f{f}", f"{f}%") for f in (5, 10, 25, 50, 100)]
    ctrls = ["degree", "random", "spectrum", "dense"]
    mat = np.full((len(ctrls), len(keys)), np.nan)
    labs = []
    for j, (k, lab) in enumerate(keys):
        labs.append(lab)
        row = analysis.get(k, {})
        for i, c in enumerate(ctrls):
            mat[i, j] = row.get(f"d_vs_{c}", np.nan)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=8)
    ax.set_yticks(range(len(ctrls))); ax.set_yticklabels([ARM_LABEL[c] for c in ctrls], fontsize=8)
    for i in range(len(ctrls)):
        for j in range(len(labs)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Effect size d: connectome − control\n(low-conc recall @10% FA)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _fmt(df, io, arm, variant, frac, metric):
    s = df[(df.io == io) & (df.arm == arm) & (df.variant == variant) & (df.fraction == frac)][metric]
    if not len(s):
        return "—"
    return f"{s.mean():.3f}±{(s.std() if len(s) > 1 else 0):.3f}"


def write_results_readme(df, analysis):
    """Replace the <!-- RESULTS --> marker in README.md with a data-driven results section."""
    readme = HERE / "README.md"
    if not readme.exists():
        return
    arms = ["connectome", "degree", "random", "spectrum", "dense"]
    lines = ["## Results\n",
             f"*{int(len(df))} runs (3 seeds × {df.fraction.nunique()} data fractions × arms × I/O).*\n",
             "Full-data (100%) **biological I/O**, low-concentration held-out test "
             "(train med/high ethylene → test LOW). AUPRC saturates at the window level, so the "
             "discriminating metric is **recall at a fixed 10% false-alarm rate**.\n",
             "| arm | low-conc recall@10%FA | low-conc AUROC | low-conc AUPRC |",
             "|---|---|---|---|"]
    for arm in arms:
        lines.append(f"| {ARM_LABEL[arm]} | {_fmt(df,'bio',arm,'standard',100,'test_low_recall_at_fpr10')} "
                     f"| {_fmt(df,'bio',arm,'standard',100,'test_low_auroc')} "
                     f"| {_fmt(df,'bio',arm,'standard',100,'test_low_auprc')} |")
    lines.append(f"| _adapter-only floor_ | {_fmt(df,'bio','adapter_only','standard',100,'test_low_recall_at_fpr10')} "
                 f"| {_fmt(df,'bio','adapter_only','standard',100,'test_low_auroc')} "
                 f"| {_fmt(df,'bio','adapter_only','standard',100,'test_low_auprc')} |")
    # headline effect sizes (connectome − control), recall@10%FA, bio
    d100 = analysis.get("bio::test_low_recall_at_fpr10::f100", {})
    d10 = analysis.get("bio::test_low_recall_at_fpr10::f10", {})
    def ds(a):
        return ", ".join(f"{c} d={a.get(f'd_vs_{c}')}" for c in ("degree", "random", "spectrum", "dense")
                         if a.get(f"d_vs_{c}") is not None)
    lines += ["", "**Connectome vs matched controls** (Cohen's *d*, connectome − control, low-conc recall@10%FA):",
              f"- full data (100%): {ds(d100)}", f"- low data (10%): {ds(d10)}", ""]
    # bio vs generic
    bio100 = _fmt(df, "bio", "connectome", "standard", 100, "test_low_recall_at_fpr10")
    gen100 = _fmt(df, "generic", "connectome", "standard", 100, "test_low_recall_at_fpr10")
    lines += [f"**Biological vs free I/O** (connectome, 100%, low-conc recall@10%FA): bio {bio100} · generic {gen100}.",
              "", "![summary](figures/fig_antennal_lobe_gas_summary.png)",
              "", "See `metrics_by_run.csv`, `analysis.json`, and `figures/` for the full grid, "
              "sample-efficiency curves, detection-latency curves, and worst-interferent breakdown.", ""]
    txt = readme.read_text()
    marker = "<!-- RESULTS -->"
    head = txt.split("## Results")[0] if "## Results" in txt else txt.split(marker)[0]
    head = re.sub(r"(\s*\n---\s*)+$", "", head.rstrip()).rstrip()   # drop trailing horizontal rule(s)
    readme.write_text(head + "\n\n---\n\n" + "\n".join(lines) + "\n" + marker + "\n")
    print(f"updated {readme} with results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", type=Path)
    a = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    csv = a.output_dir / "metrics_by_run.csv"
    df = pd.read_csv(csv)
    analysis = {}
    aj = a.output_dir / "analysis.json"
    if aj.exists():
        analysis = json.loads(aj.read_text())
    # copy the headline results next to the code so they are committed
    shutil.copy(csv, HERE / "metrics_by_run.csv")
    if aj.exists():
        shutil.copy(aj, HERE / "analysis.json")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panel_sample_efficiency(axes[0, 0], df, "test_low_recall_at_fpr10",
                            "Sample efficiency: low-conc detection", "low-conc recall @10% FA")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    panel_sample_efficiency(axes[0, 1], df, "test_low_auroc",
                            "Sample efficiency: low-conc AUROC", "low-conc AUROC")
    panel_bar_arms(axes[0, 2], df, "test_low_recall_at_fpr10", 100,
                   "Full-data low-conc detection", "recall @10% FA")
    panel_latency(axes[1, 0], df)
    axes[1, 0].legend(fontsize=7)
    panel_bio_vs_generic(axes[1, 1], df, "test_low_recall_at_fpr10")
    axes[1, 1].legend(fontsize=8)
    if analysis:
        panel_effect_size(axes[1, 2], analysis)
    else:
        axes[1, 2].axis("off")
    fig.suptitle("Antennal Lobe connectome vs matched controls — turbulent ethylene detection",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGS / "fig_antennal_lobe_gas_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")

    # standalone headline: sample-efficiency low-conc recall
    f2, ax = plt.subplots(figsize=(7, 5))
    panel_sample_efficiency(ax, df, "test_low_recall_at_fpr10",
                            "AL connectome: low-concentration ethylene detection",
                            "low-conc recall @10% false-alarm")
    ax.legend(fontsize=9)
    f2.tight_layout(); f2.savefig(FIGS / "fig_headline_sample_efficiency.png", dpi=140)
    print(f"wrote {FIGS/'fig_headline_sample_efficiency.png'}")

    if analysis:
        write_results_readme(df, analysis)


if __name__ == "__main__":
    main()
