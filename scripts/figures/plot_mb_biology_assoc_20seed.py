#!/usr/bin/env python3
"""20-seed version of the MB native-task convergence figure (replaces the 2-seed
assoc_biology_convergence.png). Mean +/- 95% CI bands over 20 seeds.

(A) reversal-probe accuracy vs epoch (the speed signal) + epochs-to-0.9.
(B) input-layer AUC (||W_in|| -> biological input cell) vs epoch, with chance line and an
    honest init-baseline annotation (E[init AUC]=0.5; per-seed scatter ~ null SE).

Reads outputs/runs/mb_biology_assoc_20seed/*.npz. Writes docs/results/mb_biology_convergence/.
"""
from __future__ import annotations
import glob, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/results/mb_biology_convergence")
ORDER = ["flywire_connectome", "flywire_random", "hemibrain_connectome", "hemibrain_random"]
COL = {"flywire_connectome": "#1f77b4", "flywire_random": "#7f7f7f",
       "hemibrain_connectome": "#2ca02c", "hemibrain_random": "#bcbd22"}
LBL = {"flywire_connectome": "FlyWire · connectome", "flywire_random": "FlyWire · random",
       "hemibrain_connectome": "hemibrain · connectome", "hemibrain_random": "hemibrain · random"}


def bio_mask(d):
    ty = d["coarse_type"].astype(str)
    return (ty == "PN") if (ty == "PN").sum() > 10 else d["is_sensory"].astype(bool)


def main():
    by = defaultdict(list)
    for f in sorted(glob.glob("outputs/runs/mb_biology_assoc_20seed/*.npz")):
        by[re.sub(r"_s\d+$", "", Path(f).stem)].append(np.load(f, allow_pickle=True))
    if not by:
        print("no runs"); return 1
    OUT.mkdir(parents=True, exist_ok=True)
    order = [c for c in ORDER if c in by]
    eps = by[order[0]][0]["snapshot_epochs"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    def mean_ci(M):                                  # M: [nseed, nep]
        m = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(M.shape[0]); return m, 1.96 * se

    print(f"{'condition':<24}{'n':>3}{'init AUC':>10}{'final AUC':>12}{'epochs->0.9':>13}")
    for c in order:
        ds = by[c]; col = COL[c]
        # (A) reversal accuracy
        R = np.array([d["reversal_acc"] for d in ds])
        m, ci = mean_ci(R)
        axA.plot(eps, m, color=col, lw=2.2, label=f"{LBL[c]} (n={len(ds)})")
        axA.fill_between(eps, m - ci, m + ci, color=col, alpha=.18)
        e09 = next((int(eps[i]) for i in range(len(eps)) if m[i] >= 0.9), None)
        # (B) input AUC trajectory (load each snapshot array once per seed)
        def auc_traj(d):
            nrm = np.linalg.norm(d["win_snapshots"], axis=2); b = bio_mask(d)
            return [roc_auc_score(b, nrm[e]) for e in range(len(eps))]
        A = np.array([auc_traj(d) for d in ds])
        am, aci = mean_ci(A)
        axB.plot(eps, am, color=col, lw=2.2, label=f"{LBL[c]}: {am[0]:.2f}→{am[-1]:.2f}")
        axB.fill_between(eps, am - aci, am + aci, color=col, alpha=.18)
        print(f"{c:<24}{len(ds):>3}{am[0]:>10.3f}{am[-1]:>12.3f}{(str(e09) if e09 else 'never'):>13}")

    axA.axhline(0.9, color="k", ls=":", lw=1)
    axA.set_xlabel("epoch"); axA.set_ylabel("reversal-probe accuracy")
    axA.set_title("(A) how fast it learns the reversal (mean ± 95% CI, n=20)")
    axA.legend(fontsize=8, loc="lower right"); axA.grid(alpha=.25)

    axB.axhline(0.5, color="k", ls=":", lw=1)
    axB.annotate("chance (E[init AUC]=0.5; per-seed scatter ~ null SE,\n"
                 "larger for hemibrain's small 168-PN class)",
                 xy=(eps[len(eps)//3], 0.5), xytext=(eps[len(eps)//3], 0.455),
                 fontsize=7, color="#444", ha="left")
    axB.set_xlabel("epoch"); axB.set_ylabel("AUC: ||W_in|| → biological input cell")
    axB.set_title("(B) does the input layer converge to biology? (mean ± 95% CI, n=20)")
    axB.legend(fontsize=8, loc="upper left"); axB.grid(alpha=.25)

    fig.tight_layout(); fig.savefig(OUT / "assoc_biology_convergence.png", dpi=150); plt.close(fig)
    print(f"\nwrote {OUT/'assoc_biology_convergence.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
