#!/usr/bin/env python3
"""Does an MQAR network converge to biological DYNAMICS in its recurrent layer?

#1 Weight preservation: after training, does |W_rec_trained| stay correlated with |W_rec_connectome|
   (do the strong biological edges remain strong), and does the change strengthen strong edges
   (Hebbian-like preservation) or regress to the mean?
#2 Functional fingerprint: reconstruct each trained net, push identical MQAR inputs through, and take
   the per-neuron activation-RMS = "dynamical importance". Does it concentrate on biologically-central
   neurons (high connectome in/out strength)? Do connectome- vs random-init nets COMPUTE alike?

Reads outputs/runs/mb_biology/*.npz (with init_W_rec_values + b_rec + readout). Read-only.
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "mqar")); sys.path.insert(0, str(ROOT / "scripts" / "associative"))
import run_mqar_associative_recall as mqar  # noqa: E402
OUT = Path("docs/results/mb_biology_convergence")


def cosc(u, v):  # mean-centered cosine
    u = np.asarray(u, np.float64) - np.mean(u); v = np.asarray(v, np.float64) - np.mean(v)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu and nv else 0.0


@torch.no_grad()
def activation_rms(d, device, n_batches=6, batch=64, vocab=32, pairs=8, queries=8):
    """Reconstruct the trained recurrence and return per-neuron RMS activation over MQAR inputs."""
    N = int(d["N"]); ei = torch.as_tensor(d["edge_indices"], dtype=torch.long, device=device)
    W = torch.sparse_coo_tensor(ei, torch.as_tensor(d["final_W_rec_values"], device=device), (N, N)).coalesce()
    Win = torch.as_tensor(d["win_snapshots"][-1], device=device)        # [N, input_dim]
    brec = torch.as_tensor(d["b_rec"], device=device)
    rng = np.random.default_rng(123)
    sq = torch.zeros(N, device=device); cnt = 0
    for _ in range(n_batches):
        b = mqar.to_torch(mqar.make_batch(rng, batch, vocab, pairs, queries, 0), device)
        x = b[0]; T = x.shape[1]; h = x.new_zeros((x.shape[0], N))
        for t in range(T):
            rec = torch.sparse.mm(W, h.t()).t()
            h = torch.relu(rec + x[:, t, :] @ Win.t() + brec)
            sq += (h ** 2).sum(0); cnt += h.shape[0]
    return (sq / cnt).sqrt().cpu().numpy()                              # [N] RMS activation per neuron


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    runs = {Path(f).stem.replace("_s0", ""): np.load(f, allow_pickle=True)
            for f in sorted(glob.glob("outputs/runs/mb_biology/*.npz"))}
    runs = {k: d for k, d in runs.items() if "init_W_rec_values" in d.files and "pruned" not in k}
    if not runs:
        print("no full-model runs yet"); return 1
    OUT.mkdir(parents=True, exist_ok=True)

    print("=== #1 WEIGHT PRESERVATION (did the biological weight structure survive training?) ===")
    print(f"{'run':<24}{'corr(|fin|,|init|)':>20}{'strengthen-strong':>20}{'neuron out-str corr':>21}")
    for name, d in runs.items():
        wi, wf = np.abs(d["init_W_rec_values"]), np.abs(d["final_W_rec_values"])
        edge_corr = spearmanr(wi, wf).correlation
        # does the change strengthen strong edges (>0) or regress to the mean (<0)?
        strengthen = spearmanr(wi, wf - wi).correlation
        # per-neuron OUT-strength: trained vs initial (biological)
        ei = d["edge_indices"]; N = int(d["N"])
        out_i = np.bincount(ei[1], weights=wi, minlength=N); out_f = np.bincount(ei[1], weights=wf, minlength=N)
        neuron_corr = spearmanr(out_i, out_f).correlation
        print(f"{name:<24}{edge_corr:>20.3f}{strengthen:>20.3f}{neuron_corr:>21.3f}")

    print("\n=== #2 FUNCTIONAL FINGERPRINT (does dynamical activity concentrate on biological hubs?) ===")
    print(f"{'run':<24}{'corr(act, in_str)':>18}{'corr(act, out_str)':>19}{'val_acc':>9}")
    acts = {}
    for name, d in runs.items():
        a = activation_rms(d, device); acts[name] = a
        ci = spearmanr(a, d["in_strength"]).correlation; co = spearmanr(a, d["out_strength"]).correlation
        print(f"{name:<24}{ci:>18.3f}{co:>19.3f}{float(d['val_acc'][-1]):>9.3f}")
    print("\n=== do connectome- and random-init nets COMPUTE alike? (centered cosine of activation) ===")
    for stem in ["flywire", "hemibrain"]:
        ck, rk = f"{stem}_connectome", f"{stem}_random"
        if ck in acts and rk in acts and len(acts[ck]) == len(acts[rk]):
            print(f"  {stem}: cos(act_connectome, act_random) = {cosc(acts[ck], acts[rk]):.3f}")

    # figure: per-neuron activation vs biological in-strength (connectome vs random), + hemibrain by type
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    for name, d in runs.items():
        if not name.endswith(("connectome", "random")):
            continue
        c = {"flywire_connectome": "#1f77b4", "flywire_random": "#7f7f7f",
             "hemibrain_connectome": "#2ca02c", "hemibrain_random": "#bcbd22"}.get(name, "#888")
        ins = d["in_strength"]; a = acts[name]
        order = np.argsort(ins); k = max(len(ins)//40, 1)
        binned = [(ins[order][i:i+k].mean(), a[order][i:i+k].mean()) for i in range(0, len(ins), k)]
        bx, by = zip(*binned)
        axL.plot(bx, by, color=c, lw=1.6, label=f"{name} (rho={spearmanr(a,ins).correlation:+.2f})")
    axL.set_xlabel("biological in-strength (connectome)"); axL.set_ylabel("trained activation-RMS (dynamical importance)")
    axL.set_xscale("log"); axL.set_title("#2 does dynamical activity track biological hubs?"); axL.legend(fontsize=8); axL.grid(alpha=.25)
    hb = runs.get("hemibrain_connectome")
    if hb is not None:
        a = acts["hemibrain_connectome"]; types = hb["coarse_type"].astype(str)
        classes = ["PN", "DAN", "KC", "MBON", "other", "untyped"]
        vals = [a[types == cl].mean() if (types == cl).sum() else np.nan for cl in classes]
        axR.bar(classes, vals, color="#2ca02c"); axR.axhline(a.mean(), color="k", ls="--", lw=1, label="global mean")
        axR.set_ylabel("mean activation-RMS"); axR.set_title("hemibrain: dynamical activity by cell type"); axR.legend()
    fig.tight_layout(); fig.savefig(OUT / "recurrent_fingerprint.png", dpi=150); plt.close(fig)
    print(f"\nwrote {OUT/'recurrent_fingerprint.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
