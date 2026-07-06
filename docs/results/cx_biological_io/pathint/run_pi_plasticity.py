#!/usr/bin/env python3
"""Result #2 for CX + path integration: BIOLOGICAL LEARNING RULES on the frozen connectome.

Scott's MB Result #2 swapped backprop for fly-like learning rules (hebbian / delta / hybrid) and
asked: do they solve the task, and do degree-matched controls also solve it (wiring irrelevant)?
This is the CX + path-integration analogue.

Key biological point: the CX's plastic site for a learned readout is the OUTPUT projection
(hidden -> PFL/PFR), and the *integration* is done by the recurrent ring-attractor + FB network,
which is FROZEN at the connectome. So the biological learning rules train ONLY the readout, by a
LOCAL rule (no backprop through time):
  * hebbian : W_out  = correlational   (W ∝ Σ_t  target_t ⊗ h_out_t)          -- 0 backprop
  * delta   : W_out  = local error/LMS (the delta rule Δw ∝ (target-pred)⊗h_out; its converged
              fixed point is the ridge readout, computed exactly here)          -- 0 backprop
  * hybrid  : delta readout (inner) + BPTT-meta-learned input encoder W_in (outer)
The input encoder W_in is a FIXED random projection for the pure rules (biological: the encoding is
anatomically set), and is meta-learned only in hybrid.

Because the readout is linear and the backbone is frozen, performance is governed by how well the
FROZEN backbone integrates self-motion -- so this directly tests whether the CONNECTOME's dynamics
beat a degree-matched rewiring under purely-local learning (the prediction the alignment hypothesis
makes for the native task, and the OPPOSITE of the MQAR finding where wiring didn't matter).

Reference: backprop (#1, run_pi.py) trains W_in+W_out end-to-end (bio_connectome ~0.391).
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, torch

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO))
from src.connectome import degree_preserving_shuffle_matrix, power_iteration_radius

CXDIR = REPO / "connectomes/cx_polar_bump_seed0"
SEQ = CXDIR / "sequences/cx_polar_bump_bins32"
HERE = Path(__file__).resolve().parent
HB = 32  # heading bins
ANG = torch.linspace(-np.pi, np.pi, HB + 1)[:-1]


def build_ports():
    t = pd.read_csv(CXDIR / "neurons.csv")["type"].fillna("NA").astype(str).values
    rx = lambda p: np.where([bool(re.match(p, x, re.I)) for x in t])[0]
    return np.sort(rx(r"^(PFN|PEN|LNO|LCNO|GLNO)")), np.sort(rx(r"^(PFL|PFR)")), len(t)


def load_split(name):
    d = np.load(SEQ / f"{name}.npz")
    return (torch.from_numpy(d["inputs"][:, :, :2].astype(np.float32)),
            torch.from_numpy(d["targets"].astype(np.float32)))


def backbone_sparse(cond, seed, A_csr, device):
    if cond == "degree_matched":
        M = degree_preserving_shuffle_matrix(A_csr, seed=20000 + seed)
        M = (M * np.float32(0.95 / max(power_iteration_radius(M, iters=120), 1e-8))).tocoo()
    else:
        M = A_csr.tocoo()
    idx = torch.from_numpy(np.vstack([M.row, M.col]).astype(np.int64))
    val = torch.from_numpy(M.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, size=M.shape, device=device).coalesce()


@torch.no_grad()
def collect_states(W, W_in, sens, out, X, N, K, device, bs=512):
    """Run the FROZEN recurrence (fixed random W_in); return output-port activity H [n_seq,T,n_out]."""
    n, T, _ = X.shape
    Hs = []
    for i in range(0, n, bs):
        xb = X[i:i+bs].to(device); B = xb.shape[0]
        h = xb.new_zeros((B, N)); seq = []
        for t in range(T):
            inj = xb[:, t, :] @ W_in.t()                         # [B, n_sens]
            drive = xb.new_zeros((B, N)).index_add(1, sens, inj)
            for m in range(K):
                nxt = torch.sparse.mm(W, h.t()).t()
                if m == 0:
                    nxt = nxt + drive
                h = torch.relu(nxt)
            seq.append(h[:, out])
        Hs.append(torch.stack(seq, 1).cpu())
    return torch.cat(Hs)                                          # [n, T, n_out]


def fit_readout(H, Y, rule, lam=1.0):
    """Local readout learning on the frozen reservoir states. H:[M,d] (bias-augmented), Y:[M,35].
    delta = LMS/delta-rule fixed point (ridge); hebbian = correlational (no whitening)."""
    d = H.shape[1]
    if rule == "delta":
        A = H.t() @ H + lam * torch.eye(d, device=H.device)
        Wt = torch.linalg.solve(A, H.t() @ Y)                    # [d,35]  (delta fixed point)
    elif rule == "hebbian":
        Wt = (H.t() @ Y) / H.shape[0]                            # [d,35]  correlational
        # scale each output to best least-squares gain on the hebbian direction (bias only)
    else:
        raise ValueError(rule)
    return Wt


def _cm(bump):  # circular-mean heading decode (matches src.train._decode_bump_angle)
    return torch.atan2((bump * torch.sin(ANG)).sum(-1), (bump * torch.cos(ANG)).sum(-1))


def metrics(pred, Y):
    mse = torch.mean((pred - Y) ** 2).item()
    # heading: sigmoid(pred bump) vs raw target bump, circular-mean, wrapped abs error (deg)
    d = _cm(torch.sigmoid(pred[..., :HB])) - _cm(Y[..., :HB])
    herr = (torch.atan2(torch.sin(d), torch.cos(d)).abs().mean() * 180 / np.pi).item()
    # home vector: (cos*dist, sin*dist) predicted vs true; RMSE
    def hv(x):
        return torch.stack([x[..., HB] * x[..., HB + 2], x[..., HB + 1] * x[..., HB + 2]], -1)
    prmse = torch.sqrt(torch.mean((hv(pred) - hv(Y)) ** 2)).item()
    return round(mse, 4), round(herr, 2), round(prmse, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["connectome", "degree_matched"])
    ap.add_argument("--rules", nargs="+", default=["hebbian", "delta"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default=str(HERE / "results_plasticity.json"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = sp.load_npz(CXDIR / "adjacency_unsigned.npz").astype(np.float32).tocsr()
    sens_np, out_np, N = build_ports()
    sens = torch.from_numpy(sens_np).to(dev); out = torch.from_numpy(out_np).to(dev)
    n_sens, n_out = len(sens_np), len(out_np)
    print(f"ports: self-motion in={n_sens} PFL/PFR out={n_out} N={N} dev={dev}", flush=True)
    Xtr, Ytr = load_split("train_T50"); Xte, Yte = load_split("test_T50")
    Ytr_f = Ytr.reshape(-1, 35).to(dev); Yte_f = Yte.reshape(-1, 35)

    results = []
    for cond in a.conditions:
        for s in range(a.seeds):
            torch.manual_seed(s)
            W = backbone_sparse(cond, s, A, dev)
            W_in = (torch.randn(n_sens, 2, generator=torch.Generator().manual_seed(s)) /
                    np.sqrt(2)).to(dev)
            t0 = time.time()
            Htr = collect_states(W, W_in, sens, out, Xtr, N, 3, dev).reshape(-1, n_out).to(dev)
            Hte = collect_states(W, W_in, sens, out, Xte, N, 3, dev).reshape(-1, n_out).to(dev)
            # bias-augment
            Htr = torch.cat([Htr, torch.ones(Htr.shape[0], 1, device=dev)], 1)
            Hte = torch.cat([Hte, torch.ones(Hte.shape[0], 1, device=dev)], 1)
            for rule in a.rules:
                Wt = fit_readout(Htr, Ytr_f, rule, a.lam)
                mse, herr, prmse = metrics((Hte @ Wt).cpu(), Yte_f)
                r = dict(condition=cond, rule=rule, seed=s, test_mse=mse, heading_err_deg=herr,
                         position_rmse=prmse, n_input=n_sens, n_output=n_out, wall_s=round(time.time()-t0,1))
                results.append(r); print(f"DONE {r}", flush=True)
                Path(a.out).write_text(json.dumps(results, indent=2))
    print("\n=== #2 CX path integration under BIOLOGICAL learning rules (test MSE; lower=better) ===")
    for rule in a.rules:
        for cond in a.conditions:
            v = [x["test_mse"] for x in results if x["condition"] == cond and x["rule"] == rule]
            h = [x["heading_err_deg"] for x in results if x["condition"] == cond and x["rule"] == rule]
            if v:
                print(f"  {rule:8s} {cond:15s} mse={np.mean(v):.4f}±{np.std(v):.4f} heading={np.mean(h):.2f}deg (n={len(v)})")
    for rule in a.rules:
        c = [x["test_mse"] for x in results if x["condition"] == "connectome" and x["rule"] == rule]
        d = [x["test_mse"] for x in results if x["condition"] == "degree_matched" and x["rule"] == rule]
        if c and d:
            print(f"  -> {rule}: connectome − degree_matched = {np.mean(c)-np.mean(d):+.4f} "
                  f"({'connectome better' if np.mean(c)<np.mean(d) else 'control better/tie'})")


if __name__ == "__main__":
    raise SystemExit(main())
