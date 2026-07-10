#!/usr/bin/env python3
"""MB native task: CHANNEL-RESOLVED convergence — does the input SPLIT across stimulus categories?

The lumped metric (plot_mb_biology_assoc.py, panel B) collapses the whole input vector into one
per-neuron scalar ||W_in[i]|| and asks only whether TOTAL input weight lands on biological input
cells. That cannot tell you whether the odor (CS/"key") channel specifically routes to the odor
pathway while the reward/punishment (US/"value") channel routes to the reinforcement pathway.

This script slices W_in by stimulus category and scores each channel against its OWN biological cell
class (ROC-AUC, scale-free, per snapshot):
  - odor / CS  ("key")      = W_in cols 0..odor_dim-1        -> should converge onto PN   (odor input pathway)
  - reward+punishment / US  = W_in cols odor_dim, odor_dim+1 -> should converge onto DAN  (reinforcement input)
  - readout (output)        = |readout_w|                    -> should land on MBON        (output neurons)
Specificity: each input channel is ALSO scored against the OTHER class (odor->DAN, value->PN). A real
split means the diagonal (odor->PN, value->DAN) beats the off-diagonal. Connectome vs degree-matched
random control, 20 seeds, hemibrain only (FlyWire 'type' is empty -> no cell-type ground truth).

Reads the saved win_snapshots in outputs/runs/mb_biology_assoc*/ -- NO retraining. Writes
docs/results/mb_biology_convergence/channel_resolved_convergence.png (+ a printed summary table).
"""
from __future__ import annotations
import glob, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/results/mb_biology_convergence"
RUN_DIRS = ["outputs/runs/mb_biology_assoc_20seed", "outputs/runs/mb_biology_assoc"]
ODOR_DIM = 64  # input layout: cols 0..63 odor (CS); 64 reward, 65 punishment (US/value); 66 query

CHANNELS = {                                   # W_in column slice for each stimulus category
    "odor(CS/key)": slice(0, ODOR_DIM),
    "value(US)":    slice(ODOR_DIM, ODOR_DIM + 2),
    "query":        slice(ODOR_DIM + 2, ODOR_DIM + 3),
}
DIAG = {"odor(CS/key)": "PN", "value(US)": "DAN"}   # correct biological routing per channel
OFF  = {"odor(CS/key)": "DAN", "value(US)": "PN"}   # specificity control (the "wrong" class)
COND_COL = {"hemibrain_connectome": "#2ca02c", "hemibrain_random": "#bcbd22"}


def cond_of(stem: str) -> str:
    return re.sub(r"_s\d+$", "", stem)


def load_runs():
    run_dir = next((d for d in RUN_DIRS if sorted(glob.glob(f"{ROOT/d}/*.npz"))), None)
    if run_dir is None:
        print("no assoc runs found in", RUN_DIRS); sys.exit(1)
    by = defaultdict(list)
    for f in sorted(glob.glob(f"{ROOT/run_dir}/*.npz")):
        by[cond_of(Path(f).stem)].append(np.load(f, allow_pickle=True))
    print(f"reading {run_dir}")
    return by


def chan_auc_curve(d, sl, target_class):
    """Per-snapshot ROC-AUC that ||W_in[:, chan]|| predicts membership in target_class."""
    ty = d["coarse_type"].astype(str)
    lab = (ty == target_class)
    scores = np.linalg.norm(d["win_snapshots"][:, :, sl], axis=2)  # [n_snap, N]
    if not (0 < lab.sum() < len(lab)):
        return np.full(scores.shape[0], 0.5)
    return np.array([roc_auc_score(lab, scores[s]) for s in range(scores.shape[0])])


def readout_auc(d, target_class="MBON"):
    ty = d["coarse_type"].astype(str); lab = (ty == target_class)
    s = np.abs(np.asarray(d["readout_w"]).reshape(-1))
    return roc_auc_score(lab, s) if 0 < lab.sum() < len(lab) else 0.5


def band(curves):
    a = np.stack(curves); m = a.mean(0)
    ci = 1.96 * a.std(0, ddof=1) / np.sqrt(a.shape[0]) if a.shape[0] > 1 else np.zeros_like(m)
    return m, ci


def paired_stats(conn_final, rand_final):
    conn_final, rand_final = np.asarray(conn_final), np.asarray(rand_final)
    n = min(len(conn_final), len(rand_final))
    c, r = conn_final[:n], rand_final[:n]
    won = int((c > r).sum())
    try:
        p = wilcoxon(c, r).pvalue if n > 1 and np.any(c != r) else float("nan")
    except ValueError:
        p = float("nan")
    return won, n, float(np.mean(c - r)), p


def main():
    by = load_runs()
    conds = [c for c in ["hemibrain_connectome", "hemibrain_random"] if c in by]
    if "hemibrain_connectome" not in by:
        print("need hemibrain runs (cell types); have:", list(by)); return 1
    eps = by[conds[0]][0]["snapshot_epochs"]
    OUT.mkdir(parents=True, exist_ok=True)

    # curves[cond][(channel,target)] = list-over-seeds of AUC curves
    curves = {c: defaultdict(list) for c in conds}
    ro = {c: [] for c in conds}
    for c in conds:
        for i, d in enumerate(by[c]):
            win = np.asarray(d["win_snapshots"])                 # decompress ONCE per file
            ty = d["coarse_type"].astype(str)
            lab = {cls: (ty == cls) for cls in ("PN", "DAN", "MBON")}
            norms = {ch: np.linalg.norm(win[:, :, sl], axis=2) for ch, sl in CHANNELS.items()}
            for ch in DIAG:
                for tgt in (DIAG[ch], OFF[ch]):
                    L = lab[tgt]; sc = norms[ch]
                    curve = (np.array([roc_auc_score(L, sc[s]) for s in range(win.shape[0])])
                             if 0 < L.sum() < len(L) else np.full(win.shape[0], 0.5))
                    curves[c][(ch, tgt)].append(curve)
            rw = np.abs(np.asarray(d["readout_w"]).reshape(-1))
            ro[c].append(roc_auc_score(lab["MBON"], rw) if 0 < lab["MBON"].sum() < len(lab["MBON"]) else 0.5)
            print(f"  {c} seed {i + 1}/{len(by[c])} done", flush=True)

    # ---- figure: two channel panels + a final-AUC summary bar ----
    fig, (axO, axV, axS) = plt.subplots(1, 3, figsize=(18, 5.4))
    panels = [("odor(CS/key)", "PN", "DAN", axO, "(A) odor/CS channel  ->  PN (odor pathway)"),
              ("value(US)", "DAN", "PN", axV, "(B) reward+punishment/US channel  ->  DAN (reinforcement)")]
    for ch, diag, off, ax, title in panels:
        for c in conds:
            m, ci = band(curves[c][(ch, diag)]); col = COND_COL[c]
            ax.plot(eps, m, color=col, lw=2.4, marker="o", ms=3,
                    label=f"{c.split('_')[1]} -> {diag}  {m[0]:.2f}->{m[-1]:.2f}")
            ax.fill_between(eps, m - ci, m + ci, color=col, alpha=0.18)
        # specificity: connectome scored against the WRONG class
        mo, _ = band(curves["hemibrain_connectome"][(ch, off)])
        ax.plot(eps, mo, color="#d62728", lw=1.6, ls="--", marker="x", ms=3,
                label=f"connectome -> {off} (specificity)  {mo[0]:.2f}->{mo[-1]:.2f}")
        ax.axhline(0.5, color="k", ls=":", lw=1); ax.set_ylim(0.40, 1.0)
        ax.set_xlabel("epoch"); ax.set_ylabel(f"ROC-AUC: ||W_in[{ch}]|| -> cell class")
        ax.set_title(title); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.25)

    # summary bars: final AUC per (channel->target) + readout->MBON, connectome vs random
    labels = ["odor->PN", "value->DAN", "readout->MBON"]
    conn_vals = [band(curves["hemibrain_connectome"][("odor(CS/key)", "PN")])[0][-1],
                 band(curves["hemibrain_connectome"][("value(US)", "DAN")])[0][-1],
                 float(np.mean(ro["hemibrain_connectome"]))]
    rand_vals = [band(curves["hemibrain_random"][("odor(CS/key)", "PN")])[0][-1],
                 band(curves["hemibrain_random"][("value(US)", "DAN")])[0][-1],
                 float(np.mean(ro["hemibrain_random"]))]
    x = np.arange(len(labels)); w = 0.38
    axS.bar(x - w/2, conn_vals, w, color="#2ca02c", label="connectome")
    axS.bar(x + w/2, rand_vals, w, color="#bcbd22", label="random")
    axS.axhline(0.5, color="k", ls=":", lw=1); axS.set_ylim(0.40, 1.0)
    axS.set_xticks(x); axS.set_xticklabels(labels, fontsize=9)
    axS.set_ylabel("final ROC-AUC"); axS.set_title("(C) final routing by channel"); axS.legend(fontsize=8)
    fig.suptitle("MB native task: does the input projection SPLIT across stimulus categories?  "
                 "(odor->PN, value->DAN, output->MBON; hemibrain, 20 seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = OUT / "channel_resolved_convergence.png"
    fig.savefig(out_png, dpi=150); plt.close(fig)

    # ---- printed summary ----
    print(f"\n{'channel -> class':<26}{'conn i->f':>16}{'rand i->f':>16}{'Δfinal':>9}{'seeds>':>8}{'wilcoxon p':>13}")
    rows = [("odor(CS/key)", "PN"), ("value(US)", "DAN")]
    for ch, tgt in rows:
        cm, _ = band(curves["hemibrain_connectome"][(ch, tgt)])
        rm, _ = band(curves["hemibrain_random"][(ch, tgt)])
        cf = [cc[-1] for cc in curves["hemibrain_connectome"][(ch, tgt)]]
        rf = [cc[-1] for cc in curves["hemibrain_random"][(ch, tgt)]]
        won, n, dmean, p = paired_stats(cf, rf)
        print(f"{ch+' -> '+tgt:<26}{cm[0]:>7.3f}->{cm[-1]:.3f}{rm[0]:>8.3f}->{rm[-1]:.3f}"
              f"{dmean:>9.3f}{f'{won}/{n}':>8}{p:>13.2e}")
    # specificity (connectome only): diagonal minus off-diagonal at final
    for ch in ("odor(CS/key)", "value(US)"):
        cd = band(curves["hemibrain_connectome"][(ch, DIAG[ch])])[0][-1]
        co = band(curves["hemibrain_connectome"][(ch, OFF[ch])])[0][-1]
        print(f"  specificity {ch:<14} {DIAG[ch]}={cd:.3f}  vs  {OFF[ch]}={co:.3f}   (diag-off = {cd-co:+.3f})")
    # readout
    won, n, dmean, p = paired_stats(ro["hemibrain_connectome"], ro["hemibrain_random"])
    print(f"{'readout -> MBON':<26}{'':>16}{'':>16}{dmean:>9.3f}{f'{won}/{n}':>8}{p:>13.2e}"
          f"   (conn {np.mean(ro['hemibrain_connectome']):.3f} vs rand {np.mean(ro['hemibrain_random']):.3f})")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    raise SystemExit(main())
