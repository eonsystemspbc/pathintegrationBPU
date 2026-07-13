"""Quick search for a model config that lets the FULL-OL connectome learn the deep bio readout
(R1-6 -> HS/VS). Trains a few BioFlowRNN configs ~15 epochs and prints val yaw R2 trajectory.
Usage: test_fix.py --device 0 --configs baseline ro ro_ms3"""
import sys, argparse, time
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

CONFIGS = {
    "baseline":       dict(microsteps=1, readout_norm=False, state_norm="none", input_gain=1.0),
    "ro":             dict(microsteps=1, readout_norm=True,  state_norm="none", input_gain=1.0),
    "ro_ms3":         dict(microsteps=3, readout_norm=True,  state_norm="none", input_gain=1.0),
    "ro_ms3_sn":      dict(microsteps=3, readout_norm=True,  state_norm="global_rms", input_gain=1.0),
    "ro_ms5_sn":      dict(microsteps=5, readout_norm=True,  state_norm="global_rms", input_gain=1.0),
    "ro_ms3_g8":      dict(microsteps=3, readout_norm=True,  state_norm="none", input_gain=8.0),
    "leak_only":      dict(microsteps=1, readout_norm=False, state_norm="none", leak=0.3),
    "leak_ro":        dict(microsteps=1, readout_norm=True,  state_norm="none", leak=0.3),
    "leak_ro_ms3":    dict(microsteps=3, readout_norm=True,  state_norm="none", leak=0.3),
    "leak_ro_sn":     dict(microsteps=1, readout_norm=True,  state_norm="global_rms", leak=0.3),
    "pn":             dict(microsteps=1, readout_norm=False, state_norm="per_neuron", leak=1.0),
    "pn_leak":        dict(microsteps=1, readout_norm=False, state_norm="per_neuron", leak=0.3),
    "pn_leak_ms3":    dict(microsteps=3, readout_norm=False, state_norm="per_neuron", leak=0.3),
}


def r2(pred, tgt):
    err = pred - tgt
    tv = np.var(tgt.reshape(-1, 3), axis=0) + 1e-8
    return 1.0 - np.mean(err.reshape(-1, 3) ** 2, axis=0) / tv  # [yaw, fwd, lat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--arm", default="connectome", choices=["connectome", "control"])
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n-train", type=int, default=1600)
    ap.add_argument("--out-pool", default="out_HSVS")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wrec-lr-mult", type=float, default=1.0, help="lr multiplier on W_rec_values")
    ap.add_argument("--freeze-rec", action="store_true", help="freeze W_rec (reservoir + trained I/O)")
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.device}")

    A = sp.load_npz(HERE / "substrate/ol_left_unsigned.npz").tocsr().astype(np.float32)
    base = A.tocoo() if args.arm == "connectome" else mb.degree_preserving_random_like(A.tocoo(), seed=0, swaps_per_edge=2.0)
    op = R.rescale_to_rho(base, args.rho)
    ports = R.load_ports()
    spec = ofb.OpticFlowSpec(hex_rings=4, timesteps=16, sensor_noise_std=0.07)
    Xtr, Ytr = R.generate_pool(spec, args.n_train, seed=12345)
    Xva, Yva = R.generate_pool(spec, 400, seed=22000)
    in_idx, out_idx = ports["in_R16"], ports[args.out_pool]

    for name in args.configs:
        cfg = CONFIGS[name]
        torch.manual_seed(0); np.random.seed(0)
        model = BioFlowRNN(op, spec.input_dim, spec.output_dim, in_idx, out_idx, seed=0, state_clip=5.0, **cfg).to(dev)
        if args.freeze_rec:
            model.W_rec_values.requires_grad_(False)
        wrec = [model.W_rec_values]
        rest = [p for n, p in model.named_parameters() if n != "W_rec_values" and p.requires_grad]
        groups = [{"params": rest, "lr": args.lr}]
        if model.W_rec_values.requires_grad:
            groups.append({"params": wrec, "lr": args.lr * args.wrec_lr_mult})
        opt = torch.optim.Adam(groups)
        rng = np.random.default_rng(0)
        t0 = time.monotonic()
        best = -9
        for ep in range(1, args.epochs + 1):
            model.train(); order = rng.permutation(len(Xtr))
            for s in range(0, len(Xtr), 64):
                idx = order[s:s + 64]
                x = torch.from_numpy(Xtr[idx]).to(dev); y = torch.from_numpy(Ytr[idx]).to(dev)
                opt.zero_grad(set_to_none=True)
                loss = torch.mean((model(x) - y) ** 2); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if ep % 3 == 0 or ep == 1:
                model.eval()
                with torch.no_grad():
                    preds = np.concatenate([model(torch.from_numpy(Xva[s:s+128]).to(dev)).cpu().numpy()
                                            for s in range(0, len(Xva), 128)], 0)
                rr = r2(preds, Yva); best = max(best, rr[0])
                print(f"[{args.arm}/{name}] ep{ep:2d} yaw_r2={rr[0]:+.4f} mean_r2={rr.mean():+.4f} "
                      f"loss={loss.item():.5f} ({time.monotonic()-t0:.0f}s)", flush=True)
        print(f"==> {args.arm}/{name}: BEST yaw_r2={best:+.4f}", flush=True)


if __name__ == "__main__":
    main()
