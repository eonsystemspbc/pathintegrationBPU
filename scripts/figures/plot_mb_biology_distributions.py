#!/usr/bin/env python3
"""Underlying ||W_in|| distributions (biological vs non-biological input cells), init -> final,
plus a diagnostic of WHY some mean AUC lines sit slightly below 0.5.

For each condition (FlyWire/hemibrain x connectome/random), pooled over 20 seeds:
  (top)  distribution of per-neuron input drive ||W_in[i]|| split by bio vs non-bio, INIT and FINAL.
  (diag) per-seed init AUC and final AUC -> is the sub-0.5 a real systematic dip or chance scatter?

Reads outputs/runs/mb_biology_assoc_20seed/*.npz. Writes docs/results/mb_biology_convergence/.
"""
from __future__ import annotations
import glob, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/results/mb_biology_convergence")
ORDER = ["flywire_connectome", "flywire_random", "hemibrain_connectome", "hemibrain_random"]
COL = {"flywire_connectome": "#1f77b4", "flywire_random": "#7f7f7f",
       "hemibrain_connectome": "#2ca02c", "hemibrain_random": "#bcbd22"}


def bio_mask(d):
    ty = d["coarse_type"].astype(str)
    return (ty == "PN") if (ty == "PN").sum() > 10 else d["is_sensory"].astype(bool)


def norms(d, snap):  # per-neuron ||W_in|| at a snapshot index
    return np.linalg.norm(d["win_snapshots"][snap], axis=1)


def main():
    by = defaultdict(list)
    for f in sorted(glob.glob("outputs/runs/mb_biology_assoc_20seed/*.npz")):
        cond = re.sub(r"_s\d+$", "", Path(f).stem)
        s = int(re.search(r"_s(\d+)$", Path(f).stem).group(1))
        by[cond].append((s, np.load(f, allow_pickle=True)))
    if not by:
        print("no runs"); return 1
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- diagnostic: per-seed init & final AUC, with SE, one-sample test vs 0.5 ----
    print(f"{'condition':<22}{'init AUC mean±SE':>22}{'final AUC mean±SE':>22}{'init vs .5 p':>14}")
    diag = {}
    for c in ORDER:
        if c not in by: continue
        ia, fa = [], []
        for s, d in by[c]:
            b = bio_mask(d)
            if not (0 < b.sum() < len(b)): continue
            ia.append(roc_auc_score(b, norms(d, 0)))
            fa.append(roc_auc_score(b, norms(d, -1)))
        ia, fa = np.array(ia), np.array(fa)
        diag[c] = (ia, fa)
        se_i, se_f = ia.std(ddof=1)/np.sqrt(len(ia)), fa.std(ddof=1)/np.sqrt(len(fa))
        p_init = stats.ttest_1samp(ia, 0.5).pvalue
        print(f"{c:<22}{ia.mean():>14.4f}±{se_i:.4f}{fa.mean():>14.4f}±{se_f:.4f}{p_init:>14.3f}")

    # ---- plot: distributions (init vs final) + per-seed AUC diagnostic ----
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, len(ORDER), height_ratios=[2, 1.1], hspace=0.42, wspace=0.28)

    for j, c in enumerate(ORDER):
        if c not in by: continue
        # pool ||W_in|| across seeds, normalised per-seed so seeds are comparable
        bio_i, non_i, bio_f, non_f = [], [], [], []
        for s, d in by[c]:
            b = bio_mask(d)
            n0, n1 = norms(d, 0), norms(d, -1)
            # z-score within seed so the *shape* (bio shifted vs non-bio) is what shows, not seed scale
            n0z = (n0 - n0.mean()) / (n0.std() + 1e-9)
            n1z = (n1 - n1.mean()) / (n1.std() + 1e-9)
            bio_i.append(n0z[b]); non_i.append(n0z[~b])
            bio_f.append(n1z[b]); non_f.append(n1z[~b])
        bio_i, non_i = np.concatenate(bio_i), np.concatenate(non_i)
        bio_f, non_f = np.concatenate(bio_f), np.concatenate(non_f)

        ax = fig.add_subplot(gs[0, j])
        bins = np.linspace(-3, 4, 60)
        ax.hist(non_i, bins=bins, density=True, histtype="step", color="#bbb", lw=1.3, ls="--", label="non-bio · init")
        ax.hist(bio_i, bins=bins, density=True, histtype="step", color=COL[c], lw=1.3, ls="--", label="bio · init")
        ax.hist(non_f, bins=bins, density=True, histtype="stepfilled", color="#bbb", alpha=.35, label="non-bio · final")
        ax.hist(bio_f, bins=bins, density=True, histtype="stepfilled", color=COL[c], alpha=.45, label="bio · final")
        ax.axvline(bio_i.mean(), color=COL[c], ls=":", lw=1)
        ax.axvline(bio_f.mean(), color=COL[c], ls="-", lw=1.5)
        ax.set_title(c.replace("_", " "), fontsize=10)
        ax.set_xlabel("‖W_in[i]‖  (z within seed)", fontsize=8.5)
        if j == 0: ax.set_ylabel("density")
        ax.legend(fontsize=6.5, loc="upper right")

        # diagnostic row: per-seed init/final AUC
        axd = fig.add_subplot(gs[1, j])
        ia, fa = diag[c]
        axd.axhline(0.5, color="k", ls=":", lw=1)
        rng = np.random.default_rng(j)
        axd.scatter(rng.normal(0, 0.04, len(ia)), ia, s=16, color="#999", label="init")
        axd.scatter(rng.normal(1, 0.04, len(fa)), fa, s=16, color=COL[c], label="final")
        axd.errorbar([0, 1], [ia.mean(), fa.mean()], yerr=[ia.std(ddof=1)/np.sqrt(len(ia)), fa.std(ddof=1)/np.sqrt(len(fa))],
                     fmt="o-", color="k", lw=1.5, capsize=4, zorder=5)
        axd.set_xticks([0, 1]); axd.set_xticklabels(["init", "final"]); axd.set_xlim(-0.4, 1.4)
        axd.set_ylim(0.40, 0.75)
        if j == 0: axd.set_ylabel("per-seed AUC")
        axd.legend(fontsize=6.5, loc="upper left")

    fig.suptitle("Input drive ‖W_in‖ distribution: biological vs non-biological cells (init → final, 20 seeds pooled)\n"
                 "bottom: per-seed AUC (dot=seed, black=mean±SE) — is sub-0.5 a real dip or chance scatter around 0.5?",
                 fontsize=11)
    fig.savefig(OUT / "biology_distributions.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {OUT/'biology_distributions.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
