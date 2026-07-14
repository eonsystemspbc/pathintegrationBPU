"""Generate ALL reproducible data behind the README figures, for one arm.
Datasets (per arm): signal (out_tstd), decode (frozen ridge R2), robustness (out_tstd vs rho x microsteps),
curve (bio_HSVS training), levers (best val mean-R2 per training method).
Usage: generate_data.py --arm connectome --device 0   /   --arm control --device 1
"""
import sys, argparse, csv, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "scripts").is_dir() and (p / "connectomes").is_dir())
for s in (ROOT / "scripts").iterdir():
    if s.is_dir():
        sys.path.insert(0, str(s))
sys.path.insert(0, str(HERE))
import numpy as np, scipy.sparse as sp, torch
import run_optic_flow_benchmark as ofb
import run_bio_data_efficiency as R
import run_mb_associative_learning as mb
from bio_model import BioFlowRNN

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["connectome", "control"])
ap.add_argument("--device", type=int, default=0)
args = ap.parse_args()
dev = torch.device(f"cuda:{args.device}")
spec = ofb.OpticFlowSpec(hex_rings=4, timesteps=16, sensor_noise_std=0.07)
ports = R.load_ports()
A = sp.load_npz(HERE / "substrate/ol_left_unsigned.npz").tocsr().astype(np.float32)
base = A.tocoo() if args.arm == "connectome" else mb.degree_preserving_random_like(A.tocoo(), seed=0, swaps_per_edge=2.0)


def write(name, rows):
    with open(HERE / f"data_{name}_{args.arm}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote data_{name}_{args.arm}.csv ({len(rows)} rows)", flush=True)


def build(op, out_pool, micro=1, **kw):
    return BioFlowRNN(op, spec.input_dim, spec.output_dim, ports["in_R16"], ports[out_pool],
                      microsteps=micro, state_clip=5.0, seed=0, **kw).to(dev)


@torch.no_grad()
def prep(op, out_pool, micro=1):
    m = build(op, out_pool, micro)
    W = torch.sparse_coo_tensor(m.edge_indices, m.W_rec_values, size=(m.N, m.N), device=dev).coalesce()
    return m, W


@torch.no_grad()
def run_seq(m, W, X, micro=1):
    h = X.new_zeros((X.shape[0], m.N)); seq = []
    for t in range(spec.timesteps):
        inj = X[:, t, :] @ m.W_in.t()
        for _ in range(micro):
            h = torch.clamp(torch.relu(torch.sparse.mm(W, h.t()).t() + m.b_rec).index_add(1, m.input_indices, inj), max=5.0)
        seq.append(h.index_select(1, m.output_indices))
    return torch.stack(seq, 1)


def out_seq(op, out_pool, X, micro=1):  # build + run (single batch; for signal/robustness)
    m, W = prep(op, out_pool, micro)
    return run_seq(m, W, X, micro)


Xd = torch.from_numpy(ofb.generate_optic_flow_batch(spec, 64, np.random.default_rng(1)).inputs).to(dev)

# --- 1. signal: out_tstd at init, HSVS & T4T5 (rho 0.95) --------------------------------------
op95 = R.rescale_to_rho(base, 0.95)
sig = []
for pool in ("out_HSVS", "out_T4T5"):
    o = out_seq(op95, pool, Xd)
    sig.append({"arm": args.arm, "pool": pool[4:], "n_out": len(ports[pool]), "out_tstd": o.std(1).mean().item()})
write("signal", sig)

# --- 2. robustness: out_tstd vs rho x microsteps (HSVS) ---------------------------------------
rob = []
for rho in (0.5, 0.7, 0.9, 0.95, 1.05):
    opr = R.rescale_to_rho(base, rho)
    for micro in (1, 3):
        o = out_seq(opr, "out_HSVS", Xd, micro)
        rob.append({"arm": args.arm, "rho": rho, "microsteps": micro, "out_tstd": o.std(1).mean().item()})
write("robustness", rob)

# --- 3. decode: frozen-feature ridge yaw-R2, HSVS & T4T5 --------------------------------------
Xtr, Ytr = R.generate_pool(spec, 1500, seed=12345)
Xte, Yte = R.generate_pool(spec, 500, seed=33000)
ytr, yte = Ytr[:, 0, 0], Yte[:, 0, 0]


def feats(op, pool, X, micro=1):
    m, W = prep(op, pool, micro)  # build ONCE, reuse across chunks
    out = []
    for s in range(0, len(X), 128):
        out.append(run_seq(m, W, torch.from_numpy(X[s:s+128]).to(dev), micro).reshape(min(128, len(X)-s), -1).cpu().numpy())
    return np.concatenate(out, 0)


def ridge_r2(Ftr, Fte, lam=1.0):
    # DUAL (kernel) ridge: solve in sample space [n_tr, n_tr], so feature dim (up to 16*6146) is irrelevant.
    Ftr = Ftr.astype(np.float32); Fte = Fte.astype(np.float32)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    yc = (ytr - ytr.mean()).astype(np.float32)
    Ktr = Ftr @ Ftr.T                                   # [n_tr, n_tr]
    alpha = np.linalg.solve(Ktr + lam * np.eye(Ktr.shape[0], dtype=np.float32), yc)
    pred = (Fte @ Ftr.T) @ alpha + ytr.mean()           # [n_te]
    return 1.0 - np.mean((pred - yte) ** 2) / (np.var(yte) + 1e-12)


dec = []
for pool in ("out_HSVS", "out_T4T5"):
    dec.append({"arm": args.arm, "pool": pool[4:], "n_out": len(ports[pool]),
                "ridge_yaw_r2": float(ridge_r2(feats(op95, pool, Xtr), feats(op95, pool, Xte)))})
write("decode", dec)


# --- 4 & 5. training curve (bio_HSVS) + levers (best mean-R2 per method) -----------------------
def r2(pred, tgt):
    err = pred - tgt; tv = np.var(tgt.reshape(-1, 3), 0) + 1e-8
    return 1.0 - np.mean(err.reshape(-1, 3) ** 2, 0) / tv


Xtr2, Ytr2 = R.generate_pool(spec, 3000, seed=12345)
Xva2, Yva2 = R.generate_pool(spec, 600, seed=22000)


def train(cfg, epochs, record_curve=False):
    torch.manual_seed(0); np.random.seed(0)
    m = build(op95, "out_HSVS", cfg.get("microsteps", 1),
              readout_norm=cfg.get("readout_norm", False), state_norm=cfg.get("state_norm", "none"),
              leak=cfg.get("leak", 1.0))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    rng = np.random.default_rng(0); best = -9.0; curve = []
    for ep in range(1, epochs + 1):
        m.train(); order = rng.permutation(len(Xtr2))
        for s in range(0, len(Xtr2), 64):
            idx = order[s:s+64]
            x = torch.from_numpy(Xtr2[idx]).to(dev); y = torch.from_numpy(Ytr2[idx]).to(dev)
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((m(x) - y) ** 2); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            preds = np.concatenate([m(torch.from_numpy(Xva2[s:s+128]).to(dev)).cpu().numpy() for s in range(0, len(Xva2), 128)], 0)
        rr = r2(preds, Yva2); mr = float(rr.mean())
        if np.isfinite(mr):
            best = max(best, mr)
        if record_curve:
            curve.append({"arm": args.arm, "epoch": ep, "val_mean_r2": mr, "val_yaw_r2": float(rr[0]),
                          "val_fwd_r2": float(rr[1]), "val_lat_r2": float(rr[2])})
    return best, curve


# curve (plain baseline)
_, curve = train({}, 20, record_curve=True)
write("curve", curve)

# levers: connectome gets the full sweep; control just the baseline (reference that learns)
LEVERS = {"baseline": {}, "readout-norm": {"readout_norm": True}, "microsteps=3": {"microsteps": 3},
          "leaky": {"leak": 0.3}, "per-neuron norm": {"state_norm": "per_neuron"}}
lev = []
todo = LEVERS if args.arm == "connectome" else {"baseline": {}}
for name, cfg in todo.items():
    t0 = time.monotonic(); best, _ = train(cfg, 12)
    lev.append({"arm": args.arm, "lever": name, "best_mean_r2": round(best, 4)})
    print(f"[lever {args.arm}/{name}] best_mean_r2={best:.4f} ({time.monotonic()-t0:.0f}s)", flush=True)
write("levers", lev)
print(f"DONE {args.arm}", flush=True)
