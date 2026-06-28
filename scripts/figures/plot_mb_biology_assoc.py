#!/usr/bin/env python3
"""MB native task (odor->valence reversal): HOW does the connectome learn faster, and does the
faster learning come with convergence to biology?

Three things, connectome vs random (FlyWire + hemibrain, mean over seeds):
  (A) Learning curves: reversal-probe accuracy vs epoch + epochs-to-0.9 (the speed signal).
  (B) Input layer: does ||W_in|| converge onto biological input cells (AUC), init->final?
  (C) Recurrent: weight preservation (corr |final|,|init|) + functional fingerprint
      (per-neuron activation under ASSOCIATIVE inputs vs biological hub-strength).
Reads outputs/runs/mb_biology_assoc/*.npz. Writes docs/results/mb_biology_convergence/.
"""
from __future__ import annotations
import glob, sys, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
for s in ["", "scripts/mqar", "scripts/associative"]:
    sys.path.insert(0, str(ROOT / s))
import run_mb_associative_learning as mb  # noqa: E402
OUT = Path("docs/results/mb_biology_convergence")
COND_COL = {"flywire_connectome": "#1f77b4", "flywire_random": "#7f7f7f",
            "hemibrain_connectome": "#2ca02c", "hemibrain_random": "#bcbd22"}


@torch.no_grad()
def activation_rms(d, device, nb=8):
    N = int(d["N"]); ei = torch.as_tensor(d["edge_indices"], dtype=torch.long, device=device)
    W = torch.sparse_coo_tensor(ei, torch.as_tensor(d["final_W_rec_values"], device=device), (N, N)).coalesce()
    Win = torch.as_tensor(d["win_snapshots"][-1], device=device); brec = torch.as_tensor(d["b_rec"], device=device)
    spec = mb.EpisodeSpec(64, 64, 6, 3, 1, 0.20, 0.03); bank = mb.make_odor_bank(spec, 0)
    rng = np.random.default_rng(5); sq = torch.zeros(N, device=device); cnt = 0
    for _ in range(nb):
        x = torch.from_numpy(mb.generate_batch(bank, spec, 64, rng).inputs).to(device)
        h = x.new_zeros((x.shape[0], N))
        for t in range(x.shape[1]):
            h = torch.relu(torch.sparse.mm(W, h.t()).t() + x[:, t, :] @ Win.t() + brec)
            sq += (h ** 2).sum(0); cnt += h.shape[0]
    return (sq / cnt).sqrt().cpu().numpy()


def cond_of(stem):
    return re.sub(r"_s\d+$", "", stem)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    by = defaultdict(list)
    for f in sorted(glob.glob("outputs/runs/mb_biology_assoc/*.npz")):
        by[cond_of(Path(f).stem)].append(np.load(f, allow_pickle=True))
    if not by:
        print("no assoc runs"); return 1
    OUT.mkdir(parents=True, exist_ok=True)
    order = [c for c in ["flywire_connectome", "flywire_random", "hemibrain_connectome", "hemibrain_random"] if c in by]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))
    print(f"\n{'condition':<22}{'epochs→0.9 rev':>15}{'final rev_acc':>14}{'AUC Win i→f':>14}{'corr|f,i|':>11}{'act~hub':>9}")
    for c in order:
        ds = by[c]; col = COND_COL[c]
        eps = ds[0]["snapshot_epochs"]
        rev = np.mean([d["reversal_acc"] for d in ds], axis=0)
        axA.plot(eps, rev, color=col, lw=2.2, marker="o", ms=3, label=f"{c} (n={len(ds)})")
        e09 = next((int(eps[i]) for i in range(len(eps)) if rev[i] >= 0.9), -1)
        # (B) input-layer AUC init->final (mean over seeds)
        def auc_run(d):
            nrm = np.linalg.norm(d["win_snapshots"], axis=2); ty = d["coarse_type"].astype(str)
            bio = (ty == "PN") if (ty == "PN").sum() > 10 else d["is_sensory"].astype(bool)
            return ([roc_auc_score(bio, nrm[s]) for s in range(len(d["snapshot_epochs"]))]) if 0 < bio.sum() < len(bio) else [0.5] * len(d["snapshot_epochs"])
        aucs = np.mean([auc_run(d) for d in ds], axis=0)
        axB.plot(eps, aucs, color=col, lw=2, marker="o", ms=3, label=f"{c} {aucs[0]:.2f}→{aucs[-1]:.2f}")
        # (C) recurrent: weight preservation + activation~hub
        wp = np.mean([spearmanr(np.abs(d["init_W_rec_values"]), np.abs(d["final_W_rec_values"])).correlation for d in ds])
        ah = np.mean([spearmanr(activation_rms(d, device), d["in_strength"]).correlation for d in ds])
        print(f"{c:<22}{e09:>15}{rev[-1]:>14.3f}{aucs[0]:>8.2f}→{aucs[-1]:.2f}{wp:>11.2f}{ah:>9.2f}")
    axA.axhline(0.9, color="k", ls=":", lw=1); axA.set_xlabel("epoch"); axA.set_ylabel("reversal-probe accuracy")
    axA.set_title("(A) how fast does it learn the reversal? (the speed signal)"); axA.legend(fontsize=8); axA.grid(alpha=.25)
    axB.axhline(0.5, color="k", ls=":", lw=1); axB.set_xlabel("epoch"); axB.set_ylabel("AUC: ||W_in|| → biological input cell")
    axB.set_title("(B) does the input layer converge to biology?"); axB.legend(fontsize=8); axB.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "assoc_biology_convergence.png", dpi=150); plt.close(fig)
    print(f"\nwrote {OUT/'assoc_biology_convergence.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
