#!/usr/bin/env python3
"""20-seed robustness of the input-layer convergence-to-biology effect (native odor task).

Per seed: final input-layer AUC (||W_in|| predicts biological input cell) + epochs-to-0.9 reversal.
Reports mean +/- std per condition, a PAIRED test (connectome vs random, same seeds), and the
fraction of seeds where connectome beats random / clears chance. Box/strip plot.
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


def final_auc(d):
    nrm = np.linalg.norm(d["win_snapshots"], axis=2); ty = d["coarse_type"].astype(str)
    bio = (ty == "PN") if (ty == "PN").sum() > 10 else d["is_sensory"].astype(bool)
    if not (0 < bio.sum() < len(bio)):
        return np.nan
    return roc_auc_score(bio, nrm[-1])


def e09(d):
    rev = d["reversal_acc"]; eps = d["snapshot_epochs"]
    hit = np.where(rev >= 0.9)[0]
    return int(eps[hit[0]]) if len(hit) else np.nan


def main():
    by = defaultdict(list)
    for f in sorted(glob.glob("outputs/runs/mb_biology_assoc_20seed/*.npz")):
        cond = re.sub(r"_s\d+$", "", Path(f).stem)
        s = int(re.search(r"_s(\d+)$", Path(f).stem).group(1))
        by[cond].append((s, np.load(f, allow_pickle=True)))
    if not by:
        print("no runs yet"); return 1
    OUT.mkdir(parents=True, exist_ok=True)
    auc = {c: {s: final_auc(d) for s, d in v} for c, v in by.items()}
    spd = {c: {s: e09(d) for s, d in v} for c, v in by.items()}

    print(f"{'condition':<24}{'n':>4}{'AUC mean±std':>18}{'frac>0.5':>10}{'epochs→0.9 (med)':>18}")
    for c in ["flywire_connectome", "flywire_random", "hemibrain_connectome", "hemibrain_random"]:
        if c not in auc: continue
        a = np.array([v for v in auc[c].values() if np.isfinite(v)])
        sp = [v for v in spd[c].values() if np.isfinite(v)]
        med = f"{int(np.median(sp))}" if sp else "never"
        print(f"{c:<24}{len(a):>4}{a.mean():>11.3f}±{a.std():.3f}{np.mean(a>0.5):>10.2f}{med:>18}")

    print("\n=== PAIRED connectome vs random (same seeds), final input-layer AUC ===")
    for stem in ["flywire", "hemibrain"]:
        ck, rk = f"{stem}_connectome", f"{stem}_random"
        if ck not in auc or rk not in auc: continue
        seeds = sorted(set(auc[ck]) & set(auc[rk]))
        ca = np.array([auc[ck][s] for s in seeds]); ra = np.array([auc[rk][s] for s in seeds])
        ok = np.isfinite(ca) & np.isfinite(ra); ca, ra = ca[ok], ra[ok]
        t, p = stats.ttest_rel(ca, ra); w = stats.wilcoxon(ca, ra).pvalue if len(ca) > 1 else np.nan
        print(f"  {stem}: connectome {ca.mean():.3f} vs random {ra.mean():.3f} | Δ={ca.mean()-ra.mean():+.3f} "
              f"| conn>rand {int((ca>ra).sum())}/{len(ca)} | paired t p={p:.1e} | wilcoxon p={w:.1e}")

    # box + strip plot
    order = [c for c in ["flywire_connectome", "flywire_random", "hemibrain_connectome", "hemibrain_random"] if c in auc]
    data = [[v for v in auc[c].values() if np.isfinite(v)] for c in order]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True)
    cols = {"flywire_connectome": "#1f77b4", "flywire_random": "#7f7f7f",
            "hemibrain_connectome": "#2ca02c", "hemibrain_random": "#bcbd22"}
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(cols[c]); patch.set_alpha(.5)
    for i, c in enumerate(order, 1):
        y = data[i - 1]; ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(y)), y, s=14, color=cols[c], zorder=3)
    ax.axhline(0.5, color="k", ls=":", lw=1, label="chance")
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels([c.replace("_", "\n") for c in order], fontsize=8.5)
    ax.set_ylabel("final input-layer AUC (||W_in|| → biological input cell)")
    ax.set_title(f"20-seed robustness: does the input layer converge to biology? (native odor task)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "biology_20seed.png", dpi=150); plt.close(fig)
    print(f"\nwrote {OUT/'biology_20seed.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
