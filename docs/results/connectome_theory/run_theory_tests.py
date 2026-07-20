#!/usr/bin/env python3
"""Causal tests of the CONTRAST-ENHANCEMENT theory of why connectomes help.

THEORY: a connectome is not a gain stage, it is a contrast enhancer — it suppresses BACKGROUND
harder than SIGNAL at its own native readout. Therefore it helps exactly when the task's difficulty
is NUISANCE VARIATION (level/concentration that must be normalised away), and hurts when the task
needs faithful amplitude or throughput.

Measured support so far (untrained, docs/results/region_task_4x4/operators_bioio):
  change-SNR  AL 0.160 > deg 0.118 > rnd 0.063   (AL WINS gas  +5.2%)
              MB 0.131 ~ deg 0.127 ~ rnd 0.137   (MB TIES gas  -0.5%)
              CX 0.057 < deg 0.144 < rnd 0.171   (CX LOSES gas -0.7%)
  transfer gain 0.10-0.46x controls; readout activity 2.5-10x lower; reciprocity 2-20x higher.

MODES
  gain      Input-gain sweep. Scott (vis-01 subrun 07) found that REMOVING contraction (norm off +
            boosted drive) unlocked optic flow but then the connectome TIED its control. If
            contraction is the AL's ASSET on gas, driving the network off its contractive set point
            must SHRINK the connectome's advantage. Falsifies the theory if the advantage grows.
  nuisance  Dose-response. Multiply every input window by a random level g ~ LogNormal(0, sigma),
            drawn independently per window at train and test: a pure, controllable CONCENTRATION
            NUISANCE. Theory predicts the connectome's advantage GROWS with sigma.
  snr       Many-graph correlation. Build a spectrum of graphs spanning a range of change-SNR,
            measure SNR (cheap, closed form) and train each, to test SNR->performance as a
            continuous law rather than an n=3 ordering.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, torch

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
AL = ROOT / "docs/results/antennal_lobe_gas"
OPS = ROOT / "docs/results/region_task_4x4/operators_bioio"
for p in (ROOT, HERE, AL, ROOT / "docs/results/region_task_4x4"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import gas_task as GT, common as CM                     # noqa: E402
from bio_al_model import BioALRNN                        # noqa: E402

NG_O, NG_T = 53, 8


def broadcast_for(region, N):
    if region == "AL":
        p = json.loads((AL / "substrate" / "ports.json").read_text())
        B = np.zeros((N, NG_O + NG_T), np.float32)
        for c, g in enumerate(sorted(p["orn_by_glom"])):
            for i in p["orn_by_glom"][g]:
                if i < N: B[i, c] = 1.0
        for c, g in enumerate(sorted(p["thr_by_glom"])):
            for i in p["thr_by_glom"][g]:
                if i < N: B[i, NG_O + c] = 1.0
        return B
    ports = json.loads((OPS / region / "ports.json").read_text())
    B = np.zeros((N, NG_O + NG_T), np.float32)
    inp = np.asarray(ports["input"], int)
    rng = np.random.default_rng(1234)
    B[inp, rng.integers(0, NG_O + NG_T, size=len(inp))] = 1.0
    return B


def load_W(region, arm, seed):
    f = "connectome.npz" if arm == "connectome" else f"{arm}_s{seed}.npz"
    return sp.load_npz(OPS / region / f).tocsr().astype(np.float32)


def change_snr(W, inp, out, steps=60, batch=32, seed=0):
    """Untrained relative change signal at the readout: ||h(sig)-h(base)|| / ||h(base)||."""
    N = W.shape[0]; coo = W.tocoo()
    Wt = torch.sparse_coo_tensor(torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long),
                                 torch.tensor(coo.data), (N, N)).coalesce()
    g = torch.Generator().manual_seed(seed)
    drive = torch.rand(len(inp), batch, generator=g)
    delta = torch.rand(len(inp), batch, generator=g) * 0.5
    ii = torch.tensor(inp, dtype=torch.long); oi = torch.tensor(out, dtype=torch.long)
    def settle(d):
        h = torch.zeros(batch, N); inj = torch.zeros(batch, N); inj[:, ii] = d.t()
        for _ in range(steps):
            h = 0.7 * h + 0.3 * torch.tanh(torch.sparse.mm(Wt, h.t()).t() + inj)
        return h[:, oi]
    hb, hp = settle(drive), settle(drive + delta)
    base = float(hb.norm(dim=1).mean())
    return float((hp - hb).norm(dim=1).mean() / max(base, 1e-9)), base


def apply_nuisance(X, sigma, seed):
    """Multiply each window by a random level g ~ LogNormal(0, sigma): concentration nuisance."""
    if sigma <= 0: return X
    rng = np.random.default_rng(seed)
    g = rng.lognormal(0.0, sigma, size=(len(X), 1, 1)).astype(np.float32)
    return (X * g).astype(np.float32)


@torch.no_grad()
def predict(m, X, dev, bs=256):
    m.eval(); o = []
    for i in range(0, len(X), bs):
        o.append(torch.sigmoid(m(torch.from_numpy(X[i:i + bs]).to(dev))).cpu().numpy())
    return np.concatenate(o)


def train_eval(W, region, splits, dev, args, seed, gain=1.0, sigma=0.0):
    N = W.shape[0]
    ports = json.loads((OPS / region / "ports.json").read_text())
    out = np.asarray(ports["output"], int)
    torch.manual_seed(7000 + seed); np.random.seed(7000 + seed)
    m = BioALRNN(recurrent=W, input_dim=10, n_sensor=8, pn_indices=out,
                 broadcast=broadcast_for(region, N), n_glom_olf=NG_O, n_glom_thr=NG_T,
                 bio_io=True, leak=0.3, readout_norm=True, output_dim=1, seed=7000 + seed).to(dev)
    m.in_gain.data.fill_(gain); m.in_gain.requires_grad_(False)   # FROZEN operating point
    tr, va, te = splits["train"], splits["val"], splits["test_low"]
    Xtr = apply_nuisance(tr["X"], sigma, 100 + seed)
    Xva = apply_nuisance(va["X"], sigma, 200 + seed)
    Xte = apply_nuisance(te["X"], sigma, 300 + seed)
    pos = float(tr["y"].mean()); pw = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=dev)
    lf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    rng = np.random.default_rng(1234 + seed); best, st, wait = 1e9, None, 0
    for ep in range(1, args.epochs + 1):
        m.train(); order = rng.permutation(len(tr["y"]))
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            xb = torch.from_numpy(Xtr[idx]).to(dev); yb = torch.from_numpy(tr["y"][idx]).to(dev)
            opt.zero_grad(set_to_none=True); loss = lf(m(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        vp = predict(m, Xva, dev)
        vl = float(torch.nn.functional.binary_cross_entropy(
            torch.from_numpy(vp).clamp(1e-6, 1 - 1e-6), torch.from_numpy(va["y"])))
        if vl < best - 1e-6: best, wait, st = vl, 0, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else: wait += 1
        if wait >= args.patience: break
    if st: m.load_state_dict(st)
    sc = predict(m, Xte, dev)
    return CM.detection_metrics(sc, te["y"])


def jobs_for(args):
    J = []
    if args.mode == "gain":
        for gn in args.gains:
            for arm in ("connectome", "degree", "random"):
                for s in args.seeds: J.append(dict(mode="gain", region="AL", arm=arm, seed=s, gain=gn, sigma=0.0))
    elif args.mode == "nuisance":
        for sg in args.sigmas:
            for arm in ("connectome", "degree", "random"):
                for s in args.seeds: J.append(dict(mode="nuisance", region="AL", arm=arm, seed=s, gain=1.0, sigma=sg))
    elif args.mode == "recipsweep":
        # RECIPROCITY DOSE-RESPONSE: does performance track reciprocal-loop density?
        arms = ["random", "recipsweep0.0", "recipsweep0.25", "recipsweep0.5",
                "recipsweep0.75", "recipsweep1.0", "connectome"]
        for arm in arms:
            for s in args.seeds:
                J.append(dict(mode="recipsweep", region="AL", arm=arm, seed=s, gain=1.0, sigma=0.0))
    elif args.mode == "snr":
        for reg in args.regions:
            for arm in ("connectome", "degree", "random"):
                for s in args.seeds: J.append(dict(mode="snr", region=reg, arm=arm, seed=s, gain=1.0, sigma=0.0))
    return J


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("gain", "nuisance", "snr", "recipsweep"))
    ap.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--gains", nargs="+", type=float, default=[0.25, 0.5, 1, 2, 4, 8, 16, 32])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--regions", nargs="+", default=["AL", "MB", "CX"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--shard", type=int, default=None); ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--device", default="auto"); ap.add_argument("--print-shard-run-ids", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.print_shard_run_ids: return 0
    if a.analyze_only:
        df = pd.concat([pd.read_csv(p) for p in sorted(a.output_dir.glob(f"{a.mode}_shard*.csv"))], ignore_index=True)
        df.to_csv(a.output_dir / f"{a.mode}_metrics.csv", index=False); print(f"{len(df)} runs"); return 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, _ = GT.load_cache(AL / "substrate" / "task_cache.npz")
    J = jobs_for(a)
    if a.shard is not None: J = J[a.shard::a.num_shards]
    print(f"theory[{a.mode}]: {len(J)} jobs on {dev}", flush=True)
    rows = []
    for j in J:
        t0 = time.monotonic()
        W = load_W(j["region"], j["arm"], j["seed"])
        ports = json.loads((OPS / j["region"] / "ports.json").read_text())
        snr, base = change_snr(W, np.asarray(ports["input"], int), np.asarray(ports["output"], int))
        met = train_eval(W, j["region"], splits, dev, a, j["seed"], gain=j["gain"], sigma=j["sigma"])
        rows.append({**j, "change_snr": round(snr, 5), "readout_base": round(base, 4),
                     "recall_at_fpr10": met["recall_at_fpr10"], "auroc": met["auroc"],
                     "auprc": met["auprc"], "wall_s": round(time.monotonic() - t0, 1)})
        print(f"done {j['region']} {j['arm']} s{j['seed']} gain={j['gain']} sig={j['sigma']} "
              f"snr={snr:.4f} recall={met['recall_at_fpr10']:.4f} auroc={met['auroc']:.4f}", flush=True)
    tag = f"_shard{a.shard}" if a.shard is not None else "_all"
    pd.DataFrame(rows).to_csv(a.output_dir / f"{a.mode}{tag}.csv", index=False)
    if a.shard is not None:
        (a.output_dir / f"result_shard{a.shard}.json").write_text(json.dumps({"shard": a.shard, "n": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
