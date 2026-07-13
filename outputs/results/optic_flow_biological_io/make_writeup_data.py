"""Generate the reproducible data behind the writeup figure, for ONE arm:
  * signal diagnostic  -- temporal-std of the output-pool activations at init (untrained recurrence)
  * training curve     -- per-epoch val mean-R2 / yaw-R2 for bio_HSVS (the key stall-vs-learn result)
Usage: make_writeup_data.py --arm connectome --device 0   (and --arm control --device 1)
"""
import sys, argparse, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "scripts").is_dir() and (p / "connectomes").is_dir())
for s in (ROOT / "scripts").iterdir():
    if s.is_dir():
        sys.path.insert(0, str(s))
sys.path.insert(0, str(HERE))
import numpy as np, scipy.sparse as sp, torch, csv
import run_optic_flow_benchmark as ofb
import run_bio_data_efficiency as R
import run_mb_associative_learning as mb
from bio_model import BioFlowRNN

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["connectome", "control"])
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--rho", type=float, default=0.95)
args = ap.parse_args()
dev = torch.device(f"cuda:{args.device}")

A = sp.load_npz(HERE / "substrate/ol_left_unsigned.npz").tocsr().astype(np.float32)
base = A.tocoo() if args.arm == "connectome" else mb.degree_preserving_random_like(A.tocoo(), seed=0, swaps_per_edge=2.0)
op = R.rescale_to_rho(base, args.rho)
ports = R.load_ports()
spec = ofb.OpticFlowSpec(hex_rings=4, timesteps=16, sensor_noise_std=0.07)

# --- signal diagnostic: output-pool temporal std at init (untrained) -------------------------
Xd = torch.from_numpy(ofb.generate_optic_flow_batch(spec, 64, np.random.default_rng(1)).inputs).to(dev)
sig_rows = []
for pool in ("out_HSVS", "out_T4T5"):
    m = BioFlowRNN(op, spec.input_dim, spec.output_dim, ports["in_R16"], ports[pool], seed=0, state_clip=5.0).to(dev).eval()
    with torch.no_grad():
        W = torch.sparse_coo_tensor(m.edge_indices, m.W_rec_values, size=(m.N, m.N), device=dev).coalesce()
        h = Xd.new_zeros((Xd.shape[0], m.N)); seq = []
        for t in range(spec.timesteps):
            inj = Xd[:, t, :] @ m.W_in.t()
            rec = torch.sparse.mm(W, h.t()).t() + m.b_rec
            h = torch.clamp(torch.relu(rec.index_add(1, m.input_indices, inj)), max=5.0)
            seq.append(h.index_select(1, m.output_indices))
        outs = torch.stack(seq, 1)
        sig_rows.append({"arm": args.arm, "pool": pool[4:], "n_out": len(ports[pool]),
                         "out_tstd": outs.std(dim=1).mean().item()})
with open(HERE / f"data_signal_{args.arm}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sig_rows[0])); w.writeheader(); w.writerows(sig_rows)

# --- training curve: bio_HSVS, plain model (the setup that stalls for connectome) -------------
Xtr, Ytr = R.generate_pool(spec, 3000, seed=12345)
Xva, Yva = R.generate_pool(spec, 600, seed=22000)


def r2(pred, tgt):
    err = pred - tgt; tv = np.var(tgt.reshape(-1, 3), axis=0) + 1e-8
    return 1.0 - np.mean(err.reshape(-1, 3) ** 2, axis=0) / tv


torch.manual_seed(0); np.random.seed(0)
model = BioFlowRNN(op, spec.input_dim, spec.output_dim, ports["in_R16"], ports["out_HSVS"], seed=0, state_clip=5.0).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
rng = np.random.default_rng(0)
rows = []
for ep in range(1, args.epochs + 1):
    model.train(); order = rng.permutation(len(Xtr)); losses = []
    for s in range(0, len(Xtr), 64):
        idx = order[s:s + 64]
        x = torch.from_numpy(Xtr[idx]).to(dev); y = torch.from_numpy(Ytr[idx]).to(dev)
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((model(x) - y) ** 2); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        preds = np.concatenate([model(torch.from_numpy(Xva[s:s+128]).to(dev)).cpu().numpy() for s in range(0, len(Xva), 128)], 0)
    rr = r2(preds, Yva)
    rows.append({"arm": args.arm, "epoch": ep, "train_loss": float(np.mean(losses)),
                 "val_mean_r2": float(rr.mean()), "val_yaw_r2": float(rr[0]),
                 "val_fwd_r2": float(rr[1]), "val_lat_r2": float(rr[2])})
    print(f"[{args.arm}] ep{ep} mean_r2={rr.mean():+.4f} yaw_r2={rr[0]:+.4f} loss={np.mean(losses):.5f}", flush=True)
with open(HERE / f"data_curve_{args.arm}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"wrote data_signal_{args.arm}.csv and data_curve_{args.arm}.csv", flush=True)
