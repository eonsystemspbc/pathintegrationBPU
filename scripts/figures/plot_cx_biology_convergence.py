#!/usr/bin/env python3
"""CX biology-convergence analysis (central complex + path integration).

Reads outputs/runs/cx_biology_convergence/{connectome,random}_s*.npz (one per model x seed).
Mean +/- 95% CI over seeds of:
  (A) path-integration R^2 learning curve,
  (B) INPUT  convergence: AUC(||W_in||   -> CX sensory pool) over training,
  (C) OUTPUT convergence: AUC(||readout|| -> CX output pool)  over training.
Plus paired connectome-vs-random final-AUC stats. Writes docs/results/cx_biology_convergence/.
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

OUT = Path("docs/results/cx_biology_convergence")
RUNS = "outputs/runs/cx_biology_convergence"
COL = {"connectome": "#9467bd", "random": "#bcbd22"}


def auc_traj(d, target_key, norm_key):
    tgt = d[target_key].astype(bool); norms = d[norm_key]
    return np.array([roc_auc_score(tgt, norms[e]) for e in range(norms.shape[0])])


def mean_ci(M):
    m = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(M.shape[0]) if M.shape[0] > 1 else np.zeros_like(m)
    return m, 1.96 * se


def main():
    by = defaultdict(list)
    for f in sorted(glob.glob(f"{RUNS}/*.npz")):
        by[re.sub(r"_s\d+$", "", Path(f).stem)].append(np.load(f, allow_pickle=True))
    if not by:
        print(f"no runs in {RUNS}"); return 1
    OUT.mkdir(parents=True, exist_ok=True)
    eps = by[next(iter(by))][0]["snapshot_epochs"]
    n_min = min(len(v) for v in by.values())

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(18, 5.2))
    finals = {"input": defaultdict(dict), "output": defaultdict(dict)}
    print(f"{'condition':<14}{'n':>3}{'final R2':>10}{'input AUC i->f':>18}{'output AUC i->f':>18}")
    for cond in ("connectome", "random"):
        if cond not in by: continue
        ds = by[cond]; col = COL[cond]
        R2 = np.array([d["r2_curve"] for d in ds]); rm, rci = mean_ci(R2)
        axA.plot(range(len(rm)), rm, color=col, lw=2.2, label=f"{cond} (n={len(ds)})")
        axA.fill_between(range(len(rm)), rm - rci, rm + rci, color=col, alpha=.18)
        IN = np.array([auc_traj(d, "is_input", "win_norm_snapshots") for d in ds])
        OUTp = np.array([auc_traj(d, "is_output", "readout_norm_snapshots") for d in ds])
        for ax, M in ((axB, IN), (axC, OUTp)):
            m, ci = mean_ci(M)
            ax.plot(eps, m, color=col, lw=2.2, label=f"{cond}: {m[0]:.2f}→{m[-1]:.2f}")
            ax.fill_between(eps, m - ci, m + ci, color=col, alpha=.18)
        for i, d in enumerate(ds):
            finals["input"][cond][int(d["seed"])] = IN[i, -1]
            finals["output"][cond][int(d["seed"])] = OUTp[i, -1]
        print(f"{cond:<14}{len(ds):>3}{rm[-1]:>10.3f}{IN.mean(0)[0]:>9.3f}→{IN.mean(0)[-1]:.3f}"
              f"{OUTp.mean(0)[0]:>9.3f}→{OUTp.mean(0)[-1]:.3f}")

    tag = f"(n={n_min}/cond, PRELIMINARY)" if n_min < 16 else f"(n={n_min}/cond)"
    axA.set_xlabel("epoch"); axA.set_ylabel("path-integration R²"); axA.set_title(f"(A) does it learn path integration? {tag}")
    axA.legend(fontsize=8); axA.grid(alpha=.25)
    axB.axhline(0.5, color="k", ls=":", lw=1); axB.set_xlabel("epoch")
    axB.set_ylabel("AUC: ||W_in|| → CX sensory pool"); axB.set_title("(B) input → biological input cells?")
    axB.legend(fontsize=8); axB.grid(alpha=.25)
    axC.axhline(0.5, color="k", ls=":", lw=1); axC.set_xlabel("epoch")
    axC.set_ylabel("AUC: ||readout|| → CX output pool"); axC.set_title("(C) readout → biological output cells?")
    axC.legend(fontsize=8); axC.grid(alpha=.25)
    fig.suptitle(f"Central complex + path integration: convergence to biology {tag}", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "cx_biology_convergence.png", dpi=150); plt.close(fig)

    print("\n=== paired connectome vs random (same seeds), final AUC ===")
    for kind in ("input", "output"):
        c, r = finals[kind]["connectome"], finals[kind]["random"]
        seeds = sorted(set(c) & set(r))
        if len(seeds) < 2:
            print(f"  {kind}: <2 paired seeds"); continue
        ca = np.array([c[s] for s in seeds]); ra = np.array([r[s] for s in seeds])
        t = stats.ttest_rel(ca, ra).pvalue; w = stats.wilcoxon(ca, ra).pvalue
        print(f"  {kind:<7}: connectome {ca.mean():.3f} vs random {ra.mean():.3f} | Δ={ca.mean()-ra.mean():+.3f} "
              f"| conn>rand {int((ca>ra).sum())}/{len(seeds)} | paired t p={t:.1e} | wilcoxon p={w:.1e}")
    print(f"\nwrote {OUT/'cx_biology_convergence.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
