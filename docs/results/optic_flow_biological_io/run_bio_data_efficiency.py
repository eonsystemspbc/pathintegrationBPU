#!/usr/bin/env python3
"""vis-02: sample-efficiency of the FULL left optic-lobe connectome vs a degree-matched control
on the optic-flow task, under BIOLOGICALLY-CORRECT I/O ports.

Same task + sample-efficiency protocol as scripts/flow/run_optic_flow_data_efficiency.py (the +12%
generic-I/O result), with three rigor upgrades:

  1. BIOLOGICAL I/O  -- input injected ONLY into the R1-6 photoreceptor pool (or L1-3 lamina);
     readout taken ONLY from the HS/VS lobula-plate tangential cells (or LPTC-wide / T4-T5).
     Ports come from the FlyWire 783 cell-type join (build_bio_substrate.py), not connectivity.
  2. DEGREE-MATCHED control  -- degree-preserving rewire (mb.degree_preserving_random_like),
     node identity preserved so the SAME port indices stay valid -- replacing flow's weak
     Gaussian-random control. Both arms rescaled to the same target rho (fair conditioning).
  3. FULL OL  -- N=48,749, no top-activity cap (sparse recurrence, so no dense OOM).

Axes:  io_condition x arm{connectome, control} x seed x data-fraction.
Model  = ofb.SparseOpticFlowRNN (port-gated sparse trainable recurrence, ReLU, state-clip).

Outputs (into --output-dir): metrics_by_run.csv, loss_history.csv, run_config.json.
Multi-GPU via --device-ids (round-robin worker processes, same pattern as the original runner).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "scripts").is_dir() and (p / "connectomes").is_dir())
for _sub in (ROOT / "scripts").iterdir():
    if _sub.is_dir() and str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_optic_flow_benchmark as ofb           # noqa: E402  task + SparseOpticFlowRNN
import run_mb_associative_learning as mb          # noqa: E402  degree_preserving_random_like

SUB = HERE / "substrate"

# --- I/O conditions: (input pool, output pool). None,None -> generic all-N I/O. -----------------
IO_CONDITIONS = {
    "bio_HSVS":      ("in_R16",  "out_HSVS"),      # PRIMARY: photoreceptors -> HS/VS LPTCs (textbook)
    "bio_LPTCwide":  ("in_R16",  "out_LPTCwide"),  # + H2/CH/VST tangential cells (20 neurons)
    "bio_T4T5":      ("in_R16",  "out_T4T5"),      # dense-flow readout bracket (motion detectors)
    "bio_L123_HSVS": ("in_L123", "out_HSVS"),      # input-fallback robustness (lamina -> HS/VS)
    "generic":       (None,      None),            # reference: free all-N I/O (the +12% regime)
}


# --- operator building (cached to disk so fleet workers share) -----------------------------------
def power_iteration_rho(matrix: sp.spmatrix, iters: int = 150, seed: int = 0) -> float:
    A = sp.csr_matrix(np.abs(matrix.astype(np.float64)))
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[0]); v /= np.linalg.norm(v) + 1e-12
    lam = 0.0
    for _ in range(iters):
        w = A @ v
        nw = np.linalg.norm(w)
        if nw < 1e-30:
            return 0.0
        v = w / nw; lam = nw
    return float(lam)


def rescale_to_rho(matrix: sp.spmatrix, target_rho: float) -> sp.coo_matrix:
    coo = matrix.tocoo().astype(np.float32)
    rho = power_iteration_rho(coo)
    if rho > 1e-12:
        coo = coo.copy()
        coo.data = (coo.data * (target_rho / rho)).astype(np.float32)
    return coo


def get_operator(arm: str, seed: int, target_rho: float, cache: Path, swaps: float = 2.0) -> sp.coo_matrix:
    """Connectome (fixed) or a degree-matched control graph for `seed`, rescaled to target_rho.
    Cached as npz keyed by (arm, seed, rho) so the pre-flight and every fleet worker reuse it."""
    cache.mkdir(parents=True, exist_ok=True)
    tag = f"{arm}_s{seed}_rho{target_rho}".replace(".", "p")
    fp = cache / f"op_{tag}.npz"
    if fp.exists():
        return sp.load_npz(fp).tocoo()
    A = sp.load_npz(SUB / "ol_left_unsigned.npz").tocsr().astype(np.float32)
    if arm == "connectome":
        base = A.tocoo()
    elif arm == "control":
        base = mb.degree_preserving_random_like(A.tocoo(), seed=int(seed), swaps_per_edge=float(swaps))
    else:
        raise ValueError(arm)
    op = rescale_to_rho(base, target_rho)
    sp.save_npz(fp, op.tocsr())
    return op


def load_ports() -> dict[str, np.ndarray]:
    raw = json.loads((SUB / "ports.json").read_text())
    return {k: np.asarray(v, dtype=np.int64) for k, v in raw.items()}


# --- data pool (fixed, shared across all arms/fractions/seeds) -----------------------------------
def generate_pool(spec, n_episodes: int, seed: int, chunk: int = 256):
    rng = np.random.default_rng(seed)
    xs, ys, remaining = [], [], int(n_episodes)
    while remaining > 0:
        b = min(chunk, remaining)
        batch = ofb.generate_optic_flow_batch(spec, b, rng)
        xs.append(batch.inputs); ys.append(batch.targets); remaining -= b
    return np.concatenate(xs, 0), np.concatenate(ys, 0)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    err = pred - target
    tv = np.var(target.reshape(-1, 3), axis=0) + 1e-8
    r2 = 1.0 - np.mean(err.reshape(-1, 3) ** 2, axis=0) / tv
    return {
        "loss": float(np.mean(err ** 2)),
        "overall_rmse": float(np.sqrt(np.mean(err ** 2))),
        "yaw_r2": float(r2[0]), "forward_r2": float(r2[1]), "lateral_r2": float(r2[2]),
        "mean_r2": float(np.mean(r2)),
    }


@torch.no_grad()
def evaluate(model, X, Y, device, bs) -> dict:
    model.eval(); preds = []
    for s in range(0, X.shape[0], bs):
        preds.append(model(torch.from_numpy(X[s:s + bs]).to(device)).cpu().numpy())
    return compute_metrics(np.concatenate(preds, 0), Y)


@dataclass
class Job:
    io: str
    arm: str
    seed: int
    fraction: int


def enumerate_jobs(ios, arms, con_seeds, ctrl_seeds, fractions) -> list[Job]:
    jobs = []
    for io in ios:
        for arm in arms:
            seeds = con_seeds if arm == "connectome" else ctrl_seeds
            for s in seeds:
                for f in fractions:
                    jobs.append(Job(io, arm, s, f))
    return jobs


def train_job(job: Job, spec, pools, ports, args, device) -> tuple[dict, list]:
    torch.manual_seed(args.init_seed + job.seed)
    np.random.seed(args.init_seed + job.seed)
    op = get_operator(job.arm, job.seed, args.target_rho, Path(args.cache_dir), swaps=args.control_swaps)
    in_pool, out_pool = IO_CONDITIONS[job.io]
    in_idx = ports[in_pool] if in_pool else None
    out_idx = ports[out_pool] if out_pool else None
    model = ofb.SparseOpticFlowRNN(
        recurrent=op, input_dim=spec.input_dim, output_dim=spec.output_dim,
        state_clip=args.state_clip, seed=args.init_seed + job.seed,
        input_indices=in_idx, output_indices=out_idx,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_x, train_y = pools["train"]
    n_full = train_x.shape[0]
    n_use = min(max(args.batch_size, int(round(n_full * job.fraction / 100.0))), n_full)
    sub_x, sub_y = train_x[:n_use], train_y[:n_use]
    val_x, val_y = pools["val"]; test_x, test_y = pools["test"]
    n_in = int(in_idx.size) if in_idx is not None else model.N
    n_out = int(out_idx.size) if out_idx is not None else model.N
    print(f"job-start io={job.io} arm={job.arm} seed={job.seed} frac={job.fraction}% "
          f"n_train={n_use} N={model.N} edges={op.nnz} n_in={n_in} n_out={n_out} "
          f"params={model.trainable_parameter_count()}", flush=True)

    rng = np.random.default_rng(args.data_seed + job.seed)
    best_val, best_state, wait, hist = float("inf"), None, 0, []
    t0 = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train(); order = rng.permutation(n_use); losses = []
        for s in range(0, n_use, args.batch_size):
            idx = order[s:s + args.batch_size]
            x = torch.from_numpy(sub_x[idx]).to(device); y = torch.from_numpy(sub_y[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((model(x) - y) ** 2)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); losses.append(float(loss.detach().cpu()))
        val = evaluate(model, val_x, val_y, device, args.batch_size)
        if val["loss"] < best_val - 1e-9:
            best_val = float(val["loss"]); wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        hist.append({"io": job.io, "arm": job.arm, "seed": job.seed, "fraction": job.fraction,
                     "epoch": epoch, "train_loss": float(np.mean(losses)),
                     "val_loss": float(val["loss"]), "val_mean_r2": val["mean_r2"],
                     "val_yaw_r2": val["yaw_r2"], "best_val_loss": best_val})
        if args.log_every and (epoch % args.log_every == 0 or epoch == 1):
            print(f"  epoch io={job.io} arm={job.arm} f={job.fraction}% ep={epoch} "
                  f"val_yaw_r2={val['yaw_r2']:.4f} val_mean_r2={val['mean_r2']:.4f} "
                  f"train_loss={np.mean(losses):.5f}", flush=True)
        if args.patience > 0 and wait >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, test_x, test_y, device, args.batch_size)
    metrics = {"io": job.io, "arm": job.arm, "seed": job.seed, "fraction": job.fraction,
               "n_train": n_use, "N": model.N, "edges": int(op.nnz), "n_in": n_in, "n_out": n_out,
               "epochs_ran": len(hist), "best_val_loss": best_val,
               "test_loss": test["loss"], "test_overall_rmse": test["overall_rmse"],
               "test_yaw_r2": test["yaw_r2"], "test_forward_r2": test["forward_r2"],
               "test_lateral_r2": test["lateral_r2"], "test_mean_r2": test["mean_r2"],
               "wall_s": round(time.monotonic() - t0, 1)}
    print(f"job-done io={job.io} arm={job.arm} seed={job.seed} frac={job.fraction}% "
          f"test_mean_r2={test['mean_r2']:.4f} test_yaw_r2={test['yaw_r2']:.4f} "
          f"rmse={test['overall_rmse']:.5f} wall={metrics['wall_s']}s", flush=True)
    return metrics, hist


def build_pools(args):
    spec = ofb.OpticFlowSpec(hex_rings=args.hex_rings, timesteps=args.timesteps,
                             sensor_noise_std=args.sensor_noise_std)
    pools = {
        "train": generate_pool(spec, args.full_train_episodes, seed=args.data_seed),
        "val": generate_pool(spec, args.val_episodes, seed=args.val_seed),
        "test": generate_pool(spec, args.test_episodes, seed=args.test_seed),
    }
    return spec, pools


def run_jobs(jobs, args, device):
    spec, pools = build_pools(args)
    ports = load_ports()
    print(f"prepared N-pool train={pools['train'][0].shape[0]} val={pools['val'][0].shape[0]} "
          f"test={pools['test'][0].shape[0]} input_dim={spec.input_dim} jobs={len(jobs)}", flush=True)
    m_rows, h_rows = [], []
    for job in jobs:
        m, h = train_job(job, spec, pools, ports, args, device)
        m_rows.append(m); h_rows.extend(h)
    return m_rows, h_rows


def dispatch_multi_gpu(jobs, args):
    device_ids = args.device_ids
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts = {d: [] for d in device_ids}
    for ji in range(len(jobs)):
        parts[device_ids[ji % len(device_ids)]].append(ji)
    procs, part_files = [], []
    for dev, jids in parts.items():
        if not jids:
            continue
        out = args.output_dir / f"_worker_dev{dev}.json"; part_files.append(out)
        cmd = [sys.executable, str(Path(__file__).resolve())] + _worker_argv(args, dev, jids, out)
        log = (args.output_dir / f"_worker_dev{dev}.log").open("w")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log, dev))
    failed = []
    for proc, log, dev in procs:
        rc = proc.wait(); log.close()
        if rc != 0:
            failed.append(dev)
    if failed:
        raise RuntimeError(f"worker(s) on device(s) {failed} failed; see _worker_dev*.log")
    m_rows, h_rows = [], []
    for pf in part_files:
        payload = json.loads(pf.read_text())
        m_rows.extend(payload["metrics"]); h_rows.extend(payload["history"])
    return pd.DataFrame(m_rows), pd.DataFrame(h_rows)


def _worker_argv(args, dev, jids, out):
    return [
        "--output-dir", str(args.output_dir), "--cache-dir", str(args.cache_dir),
        "--io-conditions", *args.io_conditions, "--arms", *args.arms,
        "--con-seeds", *[str(s) for s in args.con_seeds],
        "--ctrl-seeds", *[str(s) for s in args.ctrl_seeds],
        "--fractions", *[str(f) for f in args.fractions],
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--batch-size", str(args.batch_size), "--lr", str(args.lr),
        "--grad-clip", str(args.grad_clip), "--state-clip", str(args.state_clip),
        "--target-rho", str(args.target_rho), "--control-swaps", str(args.control_swaps),
        "--log-every", str(args.log_every),
        "--full-train-episodes", str(args.full_train_episodes),
        "--val-episodes", str(args.val_episodes), "--test-episodes", str(args.test_episodes),
        "--hex-rings", str(args.hex_rings), "--timesteps", str(args.timesteps),
        "--sensor-noise-std", str(args.sensor_noise_std),
        "--data-seed", str(args.data_seed), "--init-seed", str(args.init_seed),
        "--val-seed", str(args.val_seed), "--test-seed", str(args.test_seed),
        "--_worker-device", str(dev), "--_worker-out", str(out),
        "--_worker-job-ids", *[str(j) for j in jids],
    ]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    p.add_argument("--cache-dir", type=Path, default=SUB / "operators")
    p.add_argument("--io-conditions", nargs="+", default=list(IO_CONDITIONS), choices=list(IO_CONDITIONS))
    p.add_argument("--arms", nargs="+", default=["connectome", "control"], choices=["connectome", "control"])
    p.add_argument("--con-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--ctrl-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--fractions", nargs="+", type=int, default=[5, 10, 20, 50, 100])
    p.add_argument("--device-ids", nargs="+", type=int, default=None)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--state-clip", type=float, default=5.0)
    p.add_argument("--target-rho", type=float, default=0.9)
    p.add_argument("--control-swaps", type=float, default=2.0, help="degree-preserving swaps per edge")
    p.add_argument("--log-every", type=int, default=5, help="print val R2 every N epochs (0=off)")
    p.add_argument("--full-train-episodes", type=int, default=4000)
    p.add_argument("--val-episodes", type=int, default=600)
    p.add_argument("--test-episodes", type=int, default=1200)
    p.add_argument("--hex-rings", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=16)
    p.add_argument("--sensor-noise-std", type=float, default=0.07)
    p.add_argument("--data-seed", type=int, default=12345)
    p.add_argument("--init-seed", type=int, default=7000)
    p.add_argument("--val-seed", type=int, default=22000)
    p.add_argument("--test-seed", type=int, default=33000)
    # fleet entry: run all_jobs[shard::num_shards] on this instance's single GPU
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--_worker-device", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-job-ids", nargs="*", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def resolve_device(args, explicit_id):
    if explicit_id is not None:
        return torch.device(f"cuda:{explicit_id}")
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if args.device == "cuda":
        raise RuntimeError("--device cuda requested but CUDA unavailable")
    return torch.device("cpu")


def main(argv=None):
    args = parse_args(argv)
    all_jobs = enumerate_jobs(args.io_conditions, args.arms, args.con_seeds, args.ctrl_seeds, args.fractions)
    if args._worker_job_ids is not None:
        device = resolve_device(args, args._worker_device)
        jobs = [all_jobs[i] for i in args._worker_job_ids]
        m, h = run_jobs(jobs, args, device)
        args._worker_out.write_text(json.dumps({"metrics": m, "history": h}))
        return 0
    # fleet shard mode: this instance runs its slice of the plan on a single GPU
    if args.shard is not None and args.num_shards is not None:
        device = resolve_device(args, None)
        jobs = all_jobs[args.shard::args.num_shards]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"shard {args.shard}/{args.num_shards} device={device} jobs={len(jobs)}/{len(all_jobs)}", flush=True)
        m, h = run_jobs(jobs, args, device)
        tag = f"shard{args.shard}"
        pd.DataFrame(m).to_csv(args.output_dir / f"metrics_{tag}.csv", index=False)
        pd.DataFrame(h).to_csv(args.output_dir / f"history_{tag}.csv", index=False)
        (args.output_dir / f"result_{tag}.json").write_text(json.dumps({"metrics": m, "shard": args.shard}))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if args.device_ids and len(args.device_ids) > 1:
        print(f"dispatch multi-gpu device_ids={args.device_ids} jobs={len(all_jobs)}", flush=True)
        metrics, history = dispatch_multi_gpu(all_jobs, args)
    else:
        device = resolve_device(args, args.device_ids[0] if args.device_ids else None)
        print(f"single-device device={device} jobs={len(all_jobs)}", flush=True)
        m, h = run_jobs(all_jobs, args, device)
        metrics, history = pd.DataFrame(m), pd.DataFrame(h)
    metrics.to_csv(args.output_dir / "metrics_by_run.csv", index=False)
    history.to_csv(args.output_dir / "loss_history.csv", index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                    if not k.startswith("_worker")}, indent=2, sort_keys=True))
    print(f"complete jobs={len(all_jobs)} metrics={args.output_dir/'metrics_by_run.csv'} "
          f"elapsed={ofb._format_seconds(time.monotonic()-t0)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
