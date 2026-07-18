#!/usr/bin/env python3
"""REGRESSION benchmark: can a connectome-derived RNN TRACK a continuously changing target?

Tests the standing "fixed-point collapse" hypothesis — that connectome nets do well when they can
settle into a stable state and emit a discrete answer, but fail when the correct answer changes
every timestep. Here the model must output the target-gas concentration c(t) at EVERY step of an
intermittent turbulent plume (see plume_task.py). A collapsed network can only emit the mean, which
scores R^2 = 0 exactly.

Arms: the antennal-lobe connectome vs its matched controls (degree / edge-random / spectrum-matched
/ dense-Gaussian, all rho=0.95), under biological I/O (sensor->glomerulus adapter -> receptors, read
from projection neurons) and free all-neuron I/O. Plus two references:
  * adapter_only : no recurrence at all (memoryless) — can the task be done without dynamics?
  * gru          : a standard GRU — the engineered-recurrence anchor.

Fleet-ready: --shard/--num-shards (one run per GPU), same contract as the other experiments.
"""
from __future__ import annotations

import argparse, json, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / "pyproject.toml").exists()), HERE.parents[-1])
AL = ROOT / "docs" / "results" / "antennal_lobe_gas"
for _p in (ROOT, HERE, AL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import plume_task as PT                      # noqa: E402
import common as CM                          # noqa: E402
from bio_al_model import BioALRNN            # noqa: E402

ARMS = ("connectome", "degree", "random", "spectrum", "dense")
REFS = ("adapter_only", "gru")


class GRURef(nn.Module):
    """Engineered-recurrence anchor: a plain GRU with a per-step linear readout."""
    def __init__(self, hidden=128, input_dim=PT.INPUT_DIM, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, 1)

    def trainable_parameter_count(self):
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x, return_sequence=True):
        h, _ = self.gru(x)
        return self.readout(h).squeeze(-1)


class AdapterOnlyReg(nn.Module):
    """Memoryless floor: nonnegative sensor->glomerulus adapter + per-step linear readout."""
    def __init__(self, n_glom, n_sensor=PT.N_SENSOR, input_dim=PT.INPUT_DIM, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n_sensor = n_sensor
        self.A = nn.Parameter(torch.empty(n_glom, n_sensor).uniform_(-2.0, -0.5, generator=g))
        self.readout = nn.Linear(n_glom, 1)

    def trainable_parameter_count(self):
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x, return_sequence=True):
        drive = torch.nn.functional.softplus(x[:, :, :self.n_sensor] @
                                             torch.nn.functional.softplus(self.A).t())
        return self.readout(drive).squeeze(-1)


@dataclass
class Job:
    io: str
    arm: str
    seed: int


def enumerate_jobs(ios, arms, seeds, refs=True):
    jobs = [Job(io, a, s) for io in ios for a in arms for s in seeds]
    if refs:
        jobs += [Job("bio", r, s) for r in REFS for s in seeds]
    return jobs


def build_model(job, io_pieces, device, seed_off=7000):
    s = seed_off + job.seed
    if job.arm == "gru":
        return GRURef(seed=s).to(device)
    if job.arm == "adapter_only":
        return AdapterOnlyReg(n_glom=io_pieces["n_glom_olf"], seed=s).to(device)
    op = CM.load_operator(job.arm, job.seed)
    # readout_norm MUST be off for per-timestep regression: it rescales by the batch RMS at each
    # step, and early in a sequence the state is ~0 -> tiny divisor -> the output explodes (we
    # measured output std 2.5-3.6x the target and R^2 < 0). It was designed for a single
    # final-step classification readout.
    return BioALRNN(recurrent=op, input_dim=PT.INPUT_DIM, n_sensor=PT.N_SENSOR,
                    pn_indices=io_pieces["pn_idx"], broadcast=io_pieces["broadcast"],
                    n_glom_olf=io_pieces["n_glom_olf"], n_glom_thr=io_pieces["n_glom_thr"],
                    bio_io=(job.io == "bio"), leak=0.3, readout_norm=False,
                    output_dim=1, seed=s).to(device)


@torch.no_grad()
def predict(model, X, device, bs=128):
    model.eval(); outs = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        outs.append(model(xb, return_sequence=True).cpu().numpy())
    return np.concatenate(outs)


def train_job(job, splits, io_pieces, args, device):
    torch.manual_seed(7000 + job.seed); np.random.seed(7000 + job.seed)
    model = build_model(job, io_pieces, device)
    Xtr, Ytr = splits["train"]; Xva, Yva = splits["val"]; Xte, Yte = splits["test"]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.MSELoss()
    rng = np.random.default_rng(1234 + job.seed)
    best, best_state, wait, hist = -1e9, None, 0, []
    t0 = time.monotonic()
    print(f"job-start io={job.io} arm={job.arm} seed={job.seed} params={model.trainable_parameter_count()}",
          flush=True)
    for ep in range(1, args.epochs + 1):
        model.train(); order = rng.permutation(len(Xtr)); losses = []
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            xb = torch.from_numpy(Xtr[idx]).to(device); yb = torch.from_numpy(Ytr[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb, return_sequence=True), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(float(loss.detach().cpu()))
        vr2 = PT.r2_score(predict(model, Xva, device), Yva)
        hist.append({"io": job.io, "arm": job.arm, "seed": job.seed, "epoch": ep,
                     "train_mse": float(np.mean(losses)), "val_r2": round(vr2, 4)})
        if vr2 > best + 1e-5:
            best, wait = vr2, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if args.log_every and (ep % args.log_every == 0 or ep == 1):
            print(f"  io={job.io} arm={job.arm} ep={ep} val_r2={vr2:.4f} mse={np.mean(losses):.5f}", flush=True)
        if args.patience > 0 and wait >= args.patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    m = PT.metrics(predict(model, Xte, device), Yte)
    row = {**asdict(job), "params": model.trainable_parameter_count(),
           "epochs_ran": len(hist), "best_val_r2": round(best, 4),
           **{f"test_{k}": round(v, 5) for k, v in m.items()},
           "wall_s": round(time.monotonic() - t0, 1)}
    print(f"job-done io={job.io} arm={job.arm} seed={job.seed} test_r2={row['test_r2']:.4f} "
          f"rmse={row['test_rmse']:.4f} out/target_std={row['test_output_std_ratio']:.3f} "
          f"wall={row['wall_s']}s", flush=True)
    return row, hist


def run_jobs(jobs, args, device):
    splits = PT.build_splits(T=args.timesteps, n_train=args.n_train, noise=args.noise)
    io_pieces = CM.build_io(CM.load_ports())
    print(f"plume-regression: train={len(splits['train'][0])} T={args.timesteps} "
          f"N={io_pieces['N']} jobs={len(jobs)}", flush=True)
    ms, hs = [], []
    for j in jobs:
        m, h = train_job(j, splits, io_pieces, args, device)
        ms.append(m); hs.extend(h)
    return ms, hs


def analyze(out: Path):
    parts = sorted(out.glob("metrics_shard*.csv"))
    if not parts:
        print("no shards"); return 1
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.to_csv(out / "metrics_by_run.csv", index=False)
    summ = {}
    for io in df.io.unique():
        con = df[(df.io == io) & (df.arm == "connectome")]["test_r2"]
        if not len(con):
            continue
        row = {"connectome_r2": round(float(con.mean()), 4)}
        for c in ARMS[1:]:
            v = df[(df.io == io) & (df.arm == c)]["test_r2"]
            if len(v):
                row[f"{c}_r2"] = round(float(v.mean()), 4)
                row[f"beats_{c}_graphs"] = f"{int((con.mean() > v).sum())}/{len(v)}"
        summ[io] = row
    for r in REFS:
        v = df[df.arm == r]["test_r2"]
        if len(v):
            summ[f"ref_{r}_r2"] = round(float(v.mean()), 4)
    (out / "analysis.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=HERE / "fleet_outputs")
    p.add_argument("--ios", nargs="+", default=["bio", "generic"])
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--timesteps", type=int, default=80)
    p.add_argument("--n-train", type=int, default=1500)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--device-ids", nargs="+", type=int, default=None)
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--print-shard-run-ids", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    dev = torch.device("cuda" if (a.device != "cpu" and torch.cuda.is_available()) else "cpu")
    if a.device_ids:
        dev = torch.device(f"cuda:{a.device_ids[0]}")
    if a.analyze_only:
        return analyze(a.output_dir)
    if a.print_shard_run_ids:
        return 0
    if a.smoke:
        a.epochs, a.patience, a.n_train = 3, 3, 400
        jobs = [Job("bio", "connectome", 0), Job("bio", "dense", 0), Job("generic", "connectome", 0),
                Job("bio", "adapter_only", 0), Job("bio", "gru", 0)]
        ms, _ = run_jobs(jobs, a, dev)
        print("\nSMOKE:")
        for r in ms:
            print(f"  {r['io']:7s} {r['arm']:12s} test_r2={r['test_r2']:.3f} "
                  f"out/target_std={r['test_output_std_ratio']:.2f}")
        return 0
    alljobs = enumerate_jobs(a.ios, a.arms, a.seeds)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.shard is not None:
        jobs = alljobs[a.shard::a.num_shards]
        print(f"shard {a.shard}/{a.num_shards} jobs={len(jobs)}/{len(alljobs)}", flush=True)
        ms, hs = run_jobs(jobs, a, dev)
        pd.DataFrame(ms).to_csv(a.output_dir / f"metrics_shard{a.shard}.csv", index=False)
        pd.DataFrame(hs).to_csv(a.output_dir / f"history_shard{a.shard}.csv", index=False)
        (a.output_dir / f"result_shard{a.shard}.json").write_text(json.dumps({"shard": a.shard, "n": len(ms)}))
        return 0
    ms, hs = run_jobs(alljobs, a, dev)
    pd.DataFrame(ms).to_csv(a.output_dir / "metrics_by_run.csv", index=False)
    pd.DataFrame(hs).to_csv(a.output_dir / "loss_history.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
