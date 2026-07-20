#!/usr/bin/env python3
"""al-01 figures. Regenerated from outputs/ by `run.py --collect`; never hand-edited.

  fig1_learning_curves.png  -- val loss AND val detection rate vs epoch, EVERY condition
                               (connectome / degree-matched / GRU ceiling) x every fraction
  fig2_permutation_null.png -- the primary test, drawn: connectome mean against the
                               distribution of 30 independent control graphs
  fig3_sample_efficiency.png-- primary metric vs training-data fraction, both arms
  fig4_censoring_check.png  -- epochs-to-best + stopped_reason, per arm (is the cap binding?)

Usage:  uv run python scott/experiment_al_01_turbulent_gas/make_figures.py [outputs_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

C_CONN = "#2a6fb0"    # connectome (blue)   -- same palette as cx-01
C_CTRL = "#e07b1a"    # degree-matched, GLOBAL rewire (orange)
C_BLK  = "#8e44ad"    # block-restricted rewire (purple) -- al-02's second null
C_GATE = "#6b6b6b"    # GRU ceiling (grey)
COLORS = {"connectome": C_CONN, "degree_matched": C_CTRL,
          "block_matched": C_BLK, "gru_ceiling": C_GATE}
LABELS = {"connectome": "connectome", "degree_matched": "degree-matched (global)",
          "block_matched": "block-restricted", "gru_ceiling": "GRU ceiling"}
CONDS = ("connectome", "degree_matched", "block_matched", "gru_ceiling")
CONTROLS = ("degree_matched", "block_matched")

# al-02 primary is AUROC, not recall@FAR. Measured on al-01's landed grid, recall@10%FA has
# CV 0.32 vs AUROC's 0.025 -- ~13x noisier -- because its threshold rests on only 6 negative
# trials. al-01 rejected accuracy/AUPRC because the split is 89% positive; that argument does
# NOT apply to AUROC, which is prevalence-independent. Recall@FAR stays as a secondary because
# it is the collaborator's headline metric and we need comparability.
PRIMARY = "test_low_auroc"
SECONDARY = "test_low_recall_at_fpr10"


def graph_means(df, condition, metric, frac):
    """Control runs averaged WITHIN graph before they enter the null.

    al-02 runs SEEDS_PER_GRAPH training seeds per control graph precisely so the null reflects
    graph sampling rather than training noise. Permuting over individual runs would throw that
    variance reduction away, so every control distribution in these figures is over graph means.
    The connectome arm is pseudo-replicated (one graph, N training seeds) and is NOT averaged --
    its spread IS training noise, which is the asymmetry the permutation rank exists to handle.
    """
    sub = df[(df.condition == condition) & (df.fraction == frac)]
    if sub.empty:
        return np.array([])
    if condition in CONTROLS and "unit" in sub.columns:
        return sub.groupby("unit")[metric].mean().to_numpy()
    return sub[metric].to_numpy()


def median_band(curves: list[np.ndarray], maxep: int):
    """Cohort median + interquartile band at each epoch.

    Runs that stopped early (converged / diverged) FORWARD-FILL their final value to maxep.
    Without this the late-epoch median is a survivorship average over only the slowest runs --
    the cx-01 lesson. Padding keeps every run in the cohort at every epoch.
    """
    padded = []
    for c in curves:
        c = np.asarray(c, dtype=float)
        if len(c) == 0:
            continue
        if len(c) < maxep:
            c = np.concatenate([c, np.full(maxep - len(c), c[-1])])
        padded.append(c[:maxep])
    if not padded:
        return None, None, None, None
    M = np.vstack(padded)
    ep = np.arange(1, maxep + 1)
    return ep, np.median(M, axis=0), np.percentile(M, 25, axis=0), np.percentile(M, 75, axis=0)


def fig_learning_curves(hist: pd.DataFrame, fractions, outpath: Path):
    """EVERY condition's learning curve, both the loss being optimized and the metric reported."""
    panels = [("val_loss", "validation loss (BCE)", True),
              ("val_recall_at_fpr10", "validation detection rate @10% FA", False)]
    fig, axes = plt.subplots(len(panels), len(fractions),
                             figsize=(6.2 * len(fractions), 4.6 * len(panels)), squeeze=False)
    maxep = int(hist.epoch.max())
    for r, (col, ylab, lower_better) in enumerate(panels):
        for c, frac in enumerate(fractions):
            ax = axes[r][c]
            sub = hist[hist.fraction == frac]
            for cond in CONDS:
                cc = sub[sub.condition == cond]
                if cc.empty:
                    continue
                curves = [g.sort_values("epoch")[col].to_numpy()
                          for _, g in cc.groupby("run_id")]
                ep, med, lo, hi = median_band(curves, maxep)
                if ep is None:
                    continue
                n = cc.run_id.nunique()
                ax.plot(ep, med, color=COLORS.get(cond, "#999999"), lw=2,
                        label=f"{LABELS[cond]} (n={n})", zorder=3)
                ax.fill_between(ep, lo, hi, color=COLORS.get(cond, "#999999"), alpha=0.16, lw=0, zorder=2)
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylab)
            ax.set_title(f"{frac}% of training data", fontsize=11)
            ax.grid(alpha=0.25, zorder=0)
            if lower_better:
                ax.set_yscale("log")
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.suptitle("al-01 learning curves — median across units, IQR band\n"
                 "(early-stopped runs forward-filled)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"wrote {outpath}")


def fig_permutation_null(df: pd.DataFrame, analysis: dict, fractions, outpath: Path):
    """The primary test as a picture: where does the connectome mean fall among control GRAPHS?

    al-02 has TWO nulls, drawn as stacked rows:
      degree_matched -- global degree-preserving rewire (al-01's control). Scrambles wiring AND
                        gross routing between cell classes: it hands the control ~1.23x more direct
                        RN->PN drive while destroying ~30% of the LN stage, so a win against it
                        conflates "labelled line destroyed" with "circuit rerouted".
      block_matched  -- block-restricted rewire, preserving the 4x4 RN/LN/PN/halo edge matrix
                        EXACTLY while scrambling within each cell. Isolates the wiring itself.
    Control distributions are over GRAPH MEANS (seeds averaged within graph), matching analysis.json.
    """
    nrow = len(CONTROLS)
    fig, axes = plt.subplots(nrow, len(fractions),
                             figsize=(6.0 * len(fractions), 4.4 * nrow), squeeze=False)
    for r, ctrl_name in enumerate(CONTROLS):
        for c, frac in enumerate(fractions):
            ax = axes[r][c]
            conn = graph_means(df, "connectome", PRIMARY, frac)
            ctrl = graph_means(df, ctrl_name, PRIMARY, frac)
            if not len(conn) or not len(ctrl):
                ax.set_visible(False)
                continue
            col = COLORS[ctrl_name]
            ax.hist(ctrl, bins=12, color=col, alpha=0.65,
                    label=f"{LABELS[ctrl_name]} graphs (n={len(ctrl)})")
            ax.axvline(conn.mean(), color=C_CONN, lw=2.5,
                       label=f"connectome mean = {conn.mean():.3f}")
            ax.axvspan(conn.min(), conn.max(), color=C_CONN, alpha=0.14, lw=0,
                       label=f"connectome seeds (n={len(conn)})")
            gate = df[(df.fraction == frac) & (df.condition == "gru_ceiling")][PRIMARY].dropna()
            if len(gate):
                ax.axvline(gate.mean(), color=C_GATE, lw=1.8, ls="--",
                           label=f"GRU ceiling = {gate.mean():.3f}")
            stats = (analysis.get("results", {}).get(f"fraction_{frac}", {})
                     .get(PRIMARY, {}).get(ctrl_name, {}))
            if stats:
                ax.set_title(f"{frac}% data vs {LABELS[ctrl_name]} — perm p = "
                             f"{stats.get('p_perm')} (floor {stats.get('perm_floor')}), "
                             f"{stats.get('effect_size_control_sd')} ctrl-SD", fontsize=9.5)
            ax.set_xlabel("held-out low-conc AUROC")
            ax.set_ylabel("control graphs")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(alpha=0.25)
    fig.suptitle("al-02 primary test — connectome vs two empirical nulls "
                 "(30 independent graphs each, seeds averaged within graph)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"wrote {outpath}")


def fig_sample_efficiency(df: pd.DataFrame, fractions, outpath: Path):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for cond in CONDS:
        means, los, his, xs = [], [], [], []
        for frac in fractions:
            v = df[(df.fraction == frac) & (df.condition == cond)][PRIMARY].dropna().to_numpy()
            if not len(v):
                continue
            xs.append(frac)
            means.append(v.mean())
            los.append(v.mean() - v.std(ddof=1) if len(v) > 1 else v.mean())
            his.append(v.mean() + v.std(ddof=1) if len(v) > 1 else v.mean())
        if not xs:
            continue
        ax.plot(xs, means, "o-", color=COLORS.get(cond, "#999999"), lw=2, label=LABELS[cond])
        ax.fill_between(xs, los, his, color=COLORS.get(cond, "#999999"), alpha=0.16, lw=0)
    base = df["test_low_pos_rate_baseline"].dropna()
    if len(base):
        ax.axhline(0.10, color="k", ls=":", lw=1,
                   label="always-yes detector @10% FA")
    ax.set_xscale("log")
    ax.set_xticks(list(fractions))
    ax.set_xticklabels([f"{f}%" for f in fractions])
    ax.set_xlabel("training data used")
    ax.set_ylabel("held-out low-conc detection rate @10% FA")
    ax.set_title("al-01 sample efficiency (mean ± 1 SD across units)", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"wrote {outpath}")


def fig_censoring(df: pd.DataFrame, fractions, outpath: Path):
    """Is the 150-epoch cap binding, and is it binding EQUALLY on both arms?

    This is the guard against the failure this experiment exists to fix. If one arm hits the cap
    far more than the other, the comparison is censored and the cap must be raised in a subrun.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    conds = [c for c in ("connectome", "degree_matched", "gru_ceiling")
             if c in set(df.condition)]
    width = 0.8 / max(len(conds), 1)
    for i, cond in enumerate(conds):
        vals = [df[(df.fraction == f) & (df.condition == cond)]["best_epoch"].dropna().mean()
                for f in fractions]
        ax.bar(np.arange(len(fractions)) + i * width, vals, width,
               color=COLORS.get(cond, "#999999"), label=LABELS[cond])
    ax.set_xticks(np.arange(len(fractions)) + width)
    ax.set_xticklabels([f"{f}%" for f in fractions])
    ax.set_ylabel("mean best epoch")
    ax.set_title("epochs to best validation loss", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    tab = (df.groupby(["condition", "stopped_reason"]).size()
             .unstack(fill_value=0))
    tab = tab.div(tab.sum(axis=1), axis=0)
    bottom = np.zeros(len(tab))
    for reason in tab.columns:
        ax.bar(tab.index, tab[reason], bottom=bottom, label=reason)
        bottom += tab[reason].to_numpy()
    ax.set_ylabel("fraction of runs")
    ax.set_title("why each run stopped (cap binding = censored)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"wrote {outpath}")


def main(argv=None) -> int:
    out_dir = Path(argv[0]) if argv else HERE / "outputs"
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    metrics = out_dir / "metrics_by_run.csv"
    if not metrics.exists():
        print(f"no metrics at {metrics}; run `run.py --collect` first")
        return 1
    df = pd.read_csv(metrics)
    analysis = {}
    if (out_dir / "analysis.json").exists():
        analysis = json.loads((out_dir / "analysis.json").read_text())
    fractions = sorted(df.fraction.unique())
    OUT.mkdir(parents=True, exist_ok=True)

    hist_path = out_dir / "loss_history.csv"
    if hist_path.exists():
        fig_learning_curves(pd.read_csv(hist_path), fractions, OUT / "fig1_learning_curves.png")
    else:
        print(f"WARNING: {hist_path} missing -- no learning curves. "
              "(analyze() concatenates history_shard*.csv; check the shards were collected.)")

    fig_permutation_null(df, analysis, fractions, OUT / "fig2_permutation_null.png")
    fig_sample_efficiency(df, fractions, OUT / "fig3_sample_efficiency.png")
    fig_censoring(df, fractions, OUT / "fig4_censoring_check.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
