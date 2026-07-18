#!/usr/bin/env python3
"""Gas column of the region x task matrix: every brain region on the AL's native task.

Runs the turbulent ethylene-detection task (docs/results/antennal_lobe_gas) on AL / MB / CX / OL,
all capped to a COMMON N=3499 so size is not a confound, with GENERIC all-neuron I/O (only the AL
has glomeruli, so the biological adapter is not defined for the other regions -- generic I/O is the
only fair shared interface). Each region is compared against its OWN degree-preserving and
edge-random controls at rho=0.95.
"""
from __future__ import annotations
import argparse, json, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, torch

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
AL = ROOT / "docs/results/antennal_lobe_gas"
for p in (ROOT, HERE, AL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import gas_task as GT           # noqa: E402
import common as CM             # noqa: E402
from bio_al_model import BioALRNN  # noqa: E402

REGIONS = ("AL", "MB", "CX", "OL")
ARMS = ("connectome", "degree", "random")


@dataclass
class Job:
    region: str
    arm: str
    seed: int


def load_op(region, arm, seed):
    d = HERE / "operators" / region
    f = d / ("connectome.npz" if arm == "connectome" else f"{arm}_s{seed}.npz")
    return sp.load_npz(f).tocsr().astype(np.float32)


@torch.no_grad()
def predict(model, X, device, bs=256):
    model.eval(); o = []
    for i in range(0, len(X), bs):
        o.append(torch.sigmoid(model(torch.from_numpy(X[i:i+bs]).to(device))).cpu().numpy())
    return np.concatenate(o)


def train_job(job, splits, args, device):
    torch.manual_seed(7000+job.seed); np.random.seed(7000+job.seed)
    op = load_op(job.region, job.arm, job.seed)
    model = BioALRNN(recurrent=op, input_dim=10, n_sensor=8, pn_indices=np.arange(op.shape[0]),
                     broadcast=np.zeros((op.shape[0], 1), np.float32), n_glom_olf=1, n_glom_thr=0,
                     bio_io=False, leak=0.3, readout_norm=True, output_dim=1,
                     seed=7000+job.seed).to(device)
    tr, va = splits["train"], splits["val"]
    pos = float(tr["y"].mean()); pw = torch.tensor([(1-pos)/max(pos,1e-6)], device=device)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(1234+job.seed)
    best, state, wait = 1e9, None, 0
    t0 = time.monotonic()
    print(f"job-start {job.region} {job.arm} s{job.seed} N={op.shape[0]} nnz={op.nnz}", flush=True)
    for ep in range(1, args.epochs+1):
        model.train(); order = rng.permutation(len(tr["y"]))
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i+args.batch_size]
            xb = torch.from_numpy(tr["X"][idx]).to(device); yb = torch.from_numpy(tr["y"][idx]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        vp = predict(model, va["X"], device)
        vl = float(torch.nn.functional.binary_cross_entropy(
            torch.from_numpy(vp).clamp(1e-6,1-1e-6), torch.from_numpy(va["y"])))
        if vl < best - 1e-6:
            best, wait = vl, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if args.patience and wait >= args.patience:
            break
    if state: model.load_state_dict(state)
    row = {**asdict(job), "N": int(op.shape[0]), "edges": int(op.nnz),
           "wall_s": round(time.monotonic()-t0, 1)}
    for sp_name in ("test_low", "test_iid"):
        d = splits[sp_name]; sc = predict(model, d["X"], device)
        for k, v in CM.detection_metrics(sc, d["y"]).items():
            row[f"{sp_name}_{k}"] = round(v, 5) if isinstance(v, float) else v
    print(f"job-done {job.region} {job.arm} s{job.seed} low_recall@fpr10={row['test_low_recall_at_fpr10']:.4f} "
          f"low_auroc={row['test_low_auroc']:.4f} wall={row['wall_s']}s", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=HERE/"gas_column_outputs")
    ap.add_argument("--regions", nargs="+", default=list(REGIONS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--shard", type=int, default=None); ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--device", default="auto"); ap.add_argument("--device-ids", nargs="+", type=int, default=None)
    ap.add_argument("--print-shard-run-ids", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.print_shard_run_ids: return 0
    if a.analyze_only:
        df = pd.concat([pd.read_csv(p) for p in sorted(a.output_dir.glob("metrics_shard*.csv"))], ignore_index=True)
        df.to_csv(a.output_dir/"metrics_by_run.csv", index=False); print(f"{len(df)} runs"); return 0
    dev = torch.device(f"cuda:{a.device_ids[0]}" if a.device_ids else ("cuda" if torch.cuda.is_available() else "cpu"))
    splits, _ = GT.load_cache(AL/"substrate"/"task_cache.npz")
    jobs = [Job(r, arm, s) for r in a.regions for arm in a.arms for s in a.seeds]
    if a.shard is not None: jobs = jobs[a.shard::a.num_shards]
    print(f"gas-column: {len(jobs)} jobs on {dev}", flush=True)
    rows = [train_job(j, splits, a, dev) for j in jobs]
    tag = f"_shard{a.shard}" if a.shard is not None else ""
    pd.DataFrame(rows).to_csv(a.output_dir/f"metrics{tag or '_by_run'}.csv", index=False)
    if a.shard is not None:
        (a.output_dir/f"result_shard{a.shard}.json").write_text(json.dumps({"shard":a.shard,"n":len(rows)}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
