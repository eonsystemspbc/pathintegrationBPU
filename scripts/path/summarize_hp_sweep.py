#!/usr/bin/env python3
"""Summarize the HP + spectrum-matched-control sweep (run_hp_spectrum_sweep.py).

Answers, from the merged shard CSVs:
  (1) Not a convenient regime: at EACH model's OWN best learning rate (and across the LR range),
      does the connectome still beat its controls? -> LR-robustness curves + best-per-model table.
  (2) How much of the connectome's advantage is DYNAMICAL? -> how close the spectrum-matched
      controls (spectrum_full / spectrum_topk) get to the connectome vs a plain random control,
      reported as the fraction of the random->connectome gap that matching the spectrum closes.

Writes a markdown report + CSVs + plots under --out (default docs/results/hp_spectrum_sweep/).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRIC = "test_heading_bump_angular_error"  # primary, lower = better
CENTER_LR = 1e-3
MODEL_ORDER = ["connectome_bpu", "spectrum_full", "spectrum_topk",
               "degree_shuffle", "weight_shuffle", "random", "no_recurrence"]
NICE = {"connectome_bpu": "connectome", "spectrum_full": "spectrum-full",
        "spectrum_topk": "spectrum-topk", "degree_shuffle": "degree-shuffle",
        "weight_shuffle": "weight-shuffle", "random": "random", "no_recurrence": "no-recurrence"}


def load(results_glob):
    files = sorted(glob.glob(results_glob))
    if not files:
        raise SystemExit(f"no result CSVs match {results_glob}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df[METRIC].notna()].copy()
    print(f"loaded {len(df)} cells from {len(files)} shard files; models={sorted(df.model.unique())}")
    return df


def agg_seeds(sub):
    """mean/std/n of the primary metric over seeds for a group of identical-HP cells."""
    g = sub.groupby([c for c in ["model", "axis", "lr", "rho", "wd", "K"] if c in sub])[METRIC]
    return g.agg(["mean", "std", "count"]).reset_index()


def best_per_model(df):
    """Each model's best HP cell (lowest seed-mean metric over ALL its cells)."""
    a = df.groupby(["model", "lr", "rho", "wd", "K"])[METRIC].mean().reset_index()
    rows = []
    for m, sub in a.groupby("model"):
        r = sub.loc[sub[METRIC].idxmin()]
        rows.append(dict(model=m, best_metric=float(r[METRIC]), lr=float(r.lr),
                         rho=float(r.rho), wd=float(r.wd), K=int(r.K)))
    return pd.DataFrame(rows).set_index("model")


def plot_lr_robustness(df, out):
    lr = df[df.axis == "lr"]
    if lr.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in [x for x in MODEL_ORDER if x in lr.model.unique()]:
        s = agg_seeds(lr[lr.model == m]).sort_values("lr")
        ax.plot(s.lr, s["mean"], marker="o", label=NICE.get(m, m),
                lw=2.2 if m == "connectome_bpu" else 1.3,
                zorder=5 if m == "connectome_bpu" else 2)
        ax.fill_between(s.lr, s["mean"] - s["std"].fillna(0), s["mean"] + s["std"].fillna(0), alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate"); ax.set_ylabel(METRIC.replace("test_", "test ") + " (rad, lower=better)")
    ax.set_title("LR robustness — connectome vs controls (CX → path)")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(out / "lr_robustness.png", dpi=140); plt.close(fig)


def plot_best_bar(best, out):
    if "random" not in best.index:
        return
    rand = best.loc["random", "best_metric"]
    models = [m for m in MODEL_ORDER if m in best.index and m != "random"]
    adv = [100.0 * (rand - best.loc[m, "best_metric"]) / rand for m in models]  # +% = better than random
    colors = ["#1f77b4" if m == "connectome_bpu" else
              "#9467bd" if m.startswith("spectrum") else "#7f7f7f" for m in models]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([NICE.get(m, m) for m in models], adv, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("% better than random's best\n(at each model's own best HP)")
    ax.set_title("Best-per-model advantage over the random control (CX → path)")
    plt.xticks(rotation=20, ha="right"); fig.tight_layout()
    fig.savefig(out / "best_per_model_advantage.png", dpi=140); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="outputs/runs/hp_sweep/path/results_shard*.csv")
    p.add_argument("--out", default="docs/results/hp_spectrum_sweep")
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df = load(a.results)
    best = best_per_model(df)
    best.to_csv(out / "best_per_model.csv")
    agg_seeds(df).to_csv(out / "all_cells_seedmean.csv", index=False)
    plot_lr_robustness(df, out)
    plot_best_bar(best, out)

    L = ["# HP + spectrum-matched-control sweep — CX → path integration\n",
         f"Primary metric: `{METRIC}` (radians, **lower = better**). "
         f"{len(df)} completed cells; seeds aggregated by mean.\n",
         "## Best per model (each model at its OWN best hyperparameters)\n",
         "| model | best metric | lr | rho | wd | K | % better than random |",
         "|---|---|---|---|---|---|---|"]
    rand = best.loc["random", "best_metric"] if "random" in best.index else np.nan
    for m in [x for x in MODEL_ORDER if x in best.index]:
        r = best.loc[m]
        adv = 100.0 * (rand - r.best_metric) / rand if np.isfinite(rand) else np.nan
        L.append(f"| {NICE.get(m, m)} | {r.best_metric:.4f} | {r.lr:.0e} | {r.rho:g} | "
                 f"{r.wd:.0e} | {int(r.K)} | {adv:+.1f}% |")
    # spectrum-capture fraction: how much of random->connectome gap is closed by spectrum matching
    if {"connectome_bpu", "random", "spectrum_full"}.issubset(best.index):
        c = best.loc["connectome_bpu", "best_metric"]; rnd = best.loc["random", "best_metric"]
        sf = best.loc["spectrum_full", "best_metric"]
        gap = rnd - c
        cap = 100.0 * (rnd - sf) / gap if abs(gap) > 1e-9 else float("nan")
        L += ["\n## How much of the connectome advantage is dynamical (spectral)?\n",
              f"- connectome best: **{c:.4f}**, random best: **{rnd:.4f}**, "
              f"spectrum-full best: **{sf:.4f}** (rad).",
              f"- random→connectome gap = {gap:+.4f}; matching the FULL eigenvalue spectrum "
              f"(random eigenvectors) closes **{cap:.0f}%** of it.",
              "- Interpretation: a large fraction ⇒ the advantage is substantially the connectome's "
              "*dynamics* (spectrum), capturable by a dynamically-matched surrogate; a small fraction "
              "⇒ the advantage lives in the eigenvectors (specific wiring), beyond the spectrum.\n"]
    # robustness verdict
    if {"connectome_bpu", "random"}.issubset(best.index):
        lr = df[df.axis == "lr"]
        per_lr = lr.groupby(["model", "lr"])[METRIC].mean().unstack("model")
        if {"connectome_bpu", "random"}.issubset(per_lr.columns):
            wins = (per_lr["connectome_bpu"] < per_lr["random"]).sum()
            tot = per_lr[["connectome_bpu", "random"]].dropna().shape[0]
            L += ["## Not a convenient LR regime\n",
                  f"- Connectome beats random at **{wins}/{tot}** of the swept learning rates "
                  f"(matched LR, mean over seeds).",
                  f"- At each model's OWN best LR, connectome={best.loc['connectome_bpu','best_metric']:.4f} "
                  f"vs random={rand:.4f} → **{100*(rand-best.loc['connectome_bpu','best_metric'])/rand:+.1f}%**.\n"]
    L += ["## Figures\n", "![LR robustness](lr_robustness.png)\n",
          "![Best per model](best_per_model_advantage.png)\n"]
    (out / "README.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[wrote] {out}/README.md + CSVs + plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
