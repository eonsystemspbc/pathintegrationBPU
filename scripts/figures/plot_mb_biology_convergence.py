#!/usr/bin/env python3
"""Does the MQAR input layer W_in converge onto the MB's biological input neurons during training?

For each run (connectome- or random-init), we have W_in snapshots [n_snap, N, input_dim]. Per neuron
we take ||W_in[i]|| = how much task input it receives. The biology test: how well does ||W_in||
predict membership in the biological INPUT set (sensory pool for FlyWire; projection neurons for
hemibrain)? We score it with ROC-AUC (scale-free) at each snapshot:
  AUC = 0.5  -> input weight is unrelated to biology (the random-init expectation)
  AUC -> 1   -> input weight piles onto exactly the biological input cells
If the connectome run's AUC RISES from ~0.5 toward 1 over training while the random control stays at
0.5, the network "converged to biology" -- and it's the connectome's structure that did it.

Reads outputs/runs/mb_biology/*.npz. Writes docs/results/mb_biology_convergence/.
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

OUT = Path("docs/results/mb_biology_convergence")
COL = {"flywire_connectome": "#1f77b4", "flywire_random": "#7f7f7f",
       "hemibrain_connectome": "#2ca02c", "hemibrain_random": "#bcbd22",
       "flywire_connectome_pruned": "#1f77b4", "hemibrain_connectome_pruned": "#2ca02c"}


def winnorm(d):  # [n_snap, N]
    return np.linalg.norm(d["win_snapshots"], axis=2)


def cos(u, v):  # MEAN-CENTERED cosine (so a uniform baseline doesn't dominate; captures the
    u = np.asarray(u, np.float64) - np.mean(u)   # biological *deviation* in the per-neuron profile)
    v = np.asarray(v, np.float64) - np.mean(v)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="outputs/runs/mb_biology/*.npz")
    a = p.parse_args()
    files = sorted(glob.glob(a.runs))
    if not files:
        print("no runs yet"); return 1
    runs = {Path(f).stem.replace("_s0", ""): np.load(f, allow_pickle=True) for f in files}
    OUT.mkdir(parents=True, exist_ok=True)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    rows = []; final_profiles = {}
    for name, d in runs.items():
        nrm = winnorm(d); eps = d["snapshot_epochs"]; sens = d["is_sensory"].astype(bool)
        # biological INPUT target: sensory pool (FlyWire) OR projection neurons (hemibrain, if typed)
        types = d["coarse_type"].astype(str)
        has_pn = (types == "PN").sum() > 10
        bio = (types == "PN") if has_pn else sens
        auc = [roc_auc_score(bio, nrm[s]) if bio.sum() > 0 and bio.sum() < len(bio) else 0.5
               for s in range(len(eps))]
        bio_f = bio.astype(np.float64)
        cos_bio = [cos(nrm[s], bio_f) for s in range(len(eps))]        # dot-product align to biology
        cos_str_final = cos(nrm[-1], d["in_strength"])                 # align to biological centrality
        rho_final = spearmanr(nrm[-1], d["in_strength"]).correlation
        final_profiles[name] = (nrm[-1], int(d["N"]))
        c = COL.get(name, "#888888"); ls = "--" if "pruned" in name else "-"
        axL.plot(eps, auc, color=c, ls=ls, lw=2, marker="o", ms=3,
                 label=f"{name}  ({'PN' if has_pn else 'sensory'})  {auc[0]:.2f}→{auc[-1]:.2f}")
        rows.append((name, "PN" if has_pn else "sensory", auc[0], auc[-1], cos_bio[0], cos_bio[-1],
                     cos_str_final, rho_final, float(d["val_acc"][-1]), int(d["N"]), int(bio.sum())))
    axL.axhline(0.5, color="k", ls=":", lw=1, label="chance (no biology)")
    axL.set_xlabel("epoch"); axL.set_ylabel("ROC-AUC: ||W_in|| predicts biological input neuron")
    axL.set_title("Does the input layer converge onto biological input cells?")
    axL.legend(fontsize=7.5, loc="upper left"); axL.grid(alpha=.25)

    # RIGHT: hemibrain final ||W_in|| by cell type (do PNs — the odor input — win?)
    hb = next((d for n, d in runs.items() if n.startswith("hemibrain_connectome") and "pruned" not in n), None)
    if hb is not None:
        nrm = winnorm(hb); types = hb["coarse_type"].astype(str)
        classes = ["PN", "DAN", "KC", "MBON", "other", "untyped"]
        for lbl, idx in [("init", 0), ("final", -1)]:
            vals = [nrm[idx][types == cl].mean() if (types == cl).sum() else np.nan for cl in classes]
            axR.plot(classes, vals, marker="o", lw=2, label=lbl)
        axR.set_ylabel("mean ||W_in||"); axR.set_title("hemibrain: input weight by cell type\n(PN = odor input pathway)")
        axR.legend(); axR.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "win_biology_convergence.png", dpi=150); plt.close(fig)

    print(f"\n{'run':<32}{'tgt':>8}{'AUC i→f':>13}{'cos_bio i→f':>16}{'cos_instr':>11}{'acc':>7}")
    for r in sorted(rows):
        print(f"{r[0]:<32}{r[1]:>8}{r[2]:>6.2f}→{r[3]:.2f}{r[4]:>11.2f}→{r[5]:.2f}{r[6]:>11.2f}{r[8]:>7.2f}")
    # cross-model: do connectome- and random-init converge to the SAME input solution?
    print("\n=== cross-init cosine of final ||W_in|| profiles (do they reach the same input layer?) ===")
    for stem in ["flywire", "hemibrain"]:
        ck = f"{stem}_connectome"; rk = f"{stem}_random"
        if ck in final_profiles and rk in final_profiles and final_profiles[ck][1] == final_profiles[rk][1]:
            print(f"  {stem}: cos(connectome_Win, random_Win) = {cos(final_profiles[ck][0], final_profiles[rk][0]):.3f}")
    print(f"\nwrote {OUT/'win_biology_convergence.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
