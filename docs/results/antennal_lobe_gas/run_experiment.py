#!/usr/bin/env python3
"""Experiment 7 -- Antennal Lobe x turbulent target-gas detection.

Does the FlyWire antennal-lobe connectome, wired as a leaky-tanh RNN, detect ethylene in a
turbulent 8-MOX-sensor stream MORE sample-efficiently / more robustly to low concentration than
matched control graphs -- under biologically-correct I/O (sensor->glomerulus adapter -> ORN
input, projection-neuron readout)?

Matrix of runs:  io {bio, generic} x arm {connectome, degree, random, spectrum, dense}
                 x variant {standard, graded_ln*} x seed {0,1,2} x fraction {25,50,100}
                 (+ adapter_only floor).  * graded_ln only for bio + connectome/degree.

  connectome  -- the AL graph (one graph; seed = training replicate)     [sparse, param-matched]
  degree      -- degree-preserving rewire (per-seed independent graph)    [sparse, param-matched]
  random      -- Erdos-Renyi, same edge count + weights (per-seed)        [sparse, param-matched]
  spectrum    -- eigenvalue-spectrum-matched dense surrogate (per-seed)   [dense, dynamics ref]
  dense       -- dense Gaussian init (per-seed)                            [dense, density ref]

Trains on medium/high ethylene, tests on held-out LOW concentration (primary) + IID reference.
Metrics per test split: AUPRC, AUROC, low-conc recall, best-F1, balanced-acc; plus a
detection-latency curve (recall vs time-after-release) and worst-interferent recall.

Fleet: sharded (--shard/--num-shards, one GPU each) OR local multi-GPU (--device-ids). Same
contract as scott/experiment_vis_02.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / "pyproject.toml").exists()), HERE.parents[-1])
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import common as CM                      # noqa: E402
import gas_task as GT                    # noqa: E402
from bio_al_model import BioALRNN, AdapterOnly  # noqa: E402

SUB = HERE / "substrate"
IO_CONDITIONS = ("bio", "generic")
SPARSE_ARMS = ("connectome", "degree", "random")
DENSE_ARMS = ("spectrum", "dense")
ALL_ARMS = SPARSE_ARMS + DENSE_ARMS
GRADED_ARMS = ("connectome", "degree")   # graded-LN robustness only on these (bio only)

# latency bins (seconds after detected release) for the detection-latency curve
LAT_BINS = [(-1e9, 0), (0, 5), (5, 10), (10, 30), (30, 60), (60, 1e9)]
LAT_LABELS = ["pre", "0-5s", "5-10s", "10-30s", "30-60s", ">60s"]


@dataclass
class Job:
    io: str
    arm: str
    variant: str
    seed: int
    fraction: int


def enumerate_jobs(ios, arms, variants, seeds, fractions, include_adapter=True) -> list[Job]:
    jobs: list[Job] = []
    for io in ios:
        for arm in arms:
            for variant in variants:
                if variant == "graded_ln" and (io != "bio" or arm not in GRADED_ARMS):
                    continue
                for s in seeds:
                    for f in fractions:
                        jobs.append(Job(io, arm, variant, s, f))
    if include_adapter and "bio" in ios:
        for s in seeds:
            for f in fractions:
                jobs.append(Job("bio", "adapter_only", "standard", s, f))
    return jobs


def build_model(job: Job, io_pieces: dict, args, device):
    if job.arm == "adapter_only":
        m = AdapterOnly(input_dim=10, n_glom_olf=io_pieces["n_glom_olf"],
                        n_glom_thr=io_pieces["n_glom_thr"], n_sensor=8, seed=args.init_seed + job.seed)
        return m.to(device)
    op = CM.load_operator(job.arm, job.seed)
    bio = (job.io == "bio")
    m = BioALRNN(
        recurrent=op, input_dim=10, n_sensor=8,
        pn_indices=io_pieces["pn_idx"], broadcast=io_pieces["broadcast"],
        n_glom_olf=io_pieces["n_glom_olf"], n_glom_thr=io_pieces["n_glom_thr"],
        bio_io=bio, leak=args.leak, readout_norm=True,
        graded_ln=(job.variant == "graded_ln"), ln_indices=io_pieces["ln_idx"],
        seed=args.init_seed + job.seed,
    )
    return m.to(device)


@torch.no_grad()
def predict(model, X, device, bs=256) -> np.ndarray:
    model.eval(); outs = []
    for s in range(0, len(X), bs):
        xb = torch.from_numpy(X[s:s + bs]).to(device)
        outs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros(0, np.float32)


def latency_curve(scores, y, onset, release, thr) -> dict:
    """Positive-window detection rate (recall) binned by time-after-release, at a fixed
    false-alarm-rate threshold `thr`."""
    rel = onset - release
    out = {}
    for (lo, hi), lab in zip(LAT_BINS, LAT_LABELS):
        m = (rel >= lo) & (rel < hi) & (y == 1)
        out[lab] = round(float((scores[m] > thr).mean()), 4) if m.sum() else None
    return out


def worst_interferent(scores, y, intc) -> dict:
    """Low-conc recall split by interferent (0=methane, 1=CO)."""
    out = {}
    for code, name in [(0, "methane"), (1, "CO")]:
        m = (y == 1) & (intc == code)
        out[name] = round(float((scores[m] >= 0.5).mean()), 4) if m.sum() else None
    vals = [v for v in out.values() if v is not None]
    out["worst"] = round(min(vals), 4) if vals else None
    return out


def train_job(job: Job, splits: dict, io_pieces: dict, args, device) -> tuple[dict, list]:
    torch.manual_seed(args.init_seed + job.seed)
    np.random.seed(args.init_seed + job.seed)
    model = build_model(job, io_pieces, args, device)

    tr = splits["train"]; va = splits["val"]
    n_full = len(tr["y"])
    rng = np.random.default_rng(args.data_seed + job.seed)
    perm = rng.permutation(n_full)
    n_use = min(max(args.batch_size, int(round(n_full * job.fraction / 100.0))), n_full)
    sub = perm[:n_use]
    Xtr, ytr = tr["X"][sub], tr["y"][sub]
    pos = float(ytr.mean()); pos_weight = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=device)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"job-start io={job.io} arm={job.arm} var={job.variant} seed={job.seed} f={job.fraction}% "
          f"n_train={n_use} params={model.trainable_parameter_count()}", flush=True)
    best_val, best_state, wait, hist = -1.0, None, 0, []
    t0 = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train(); order = rng.permutation(n_use); losses = []
        for s in range(0, n_use, args.batch_size):
            idx = order[s:s + args.batch_size]
            xb = torch.from_numpy(Xtr[idx]).to(device); yb = torch.from_numpy(ytr[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); losses.append(float(loss.detach().cpu()))
        vp = predict(model, va["X"], device)
        vauprc = CM.average_precision(vp, va["y"])
        # model selection on val LOSS (AUPRC saturates near ceiling -> noisy selection); minimise
        vloss = float(torch.nn.functional.binary_cross_entropy(
            torch.from_numpy(vp).clamp(1e-6, 1 - 1e-6), torch.from_numpy(va["y"])))
        hist.append({"io": job.io, "arm": job.arm, "variant": job.variant, "seed": job.seed,
                     "fraction": job.fraction, "epoch": epoch, "train_loss": float(np.mean(losses)),
                     "val_auprc": round(float(vauprc), 4), "val_loss": round(vloss, 4)})
        if -vloss > best_val + 1e-6:
            best_val = float(-vloss); wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if args.log_every and (epoch % args.log_every == 0 or epoch == 1):
            print(f"  io={job.io} arm={job.arm} f={job.fraction}% ep={epoch} "
                  f"val_auprc={vauprc:.4f} val_loss={vloss:.4f} loss={np.mean(losses):.4f}", flush=True)
        if args.patience > 0 and wait >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    row = {**asdict(job), "n_train": n_use, "params": model.trainable_parameter_count(),
           "epochs_ran": len(hist), "best_val_loss": round(-best_val, 4),
           "wall_s": round(time.monotonic() - t0, 1)}
    for split in ("test_iid", "test_low"):
        d = splits[split]
        sc = predict(model, d["X"], device)
        met = CM.detection_metrics(sc, d["y"])
        for k, v in met.items():
            row[f"{split}_{k}"] = round(v, 5) if isinstance(v, float) else v
        if split == "test_low":
            thr = CM.threshold_at_fpr(sc, d["y"], 0.10)
            row["low_latency"] = json.dumps(latency_curve(sc, d["y"], d["onset"], d["rel"], thr))
            row["low_worst_interferent"] = json.dumps(worst_interferent(sc, d["y"], d["intc"]))
    print(f"job-done io={job.io} arm={job.arm} var={job.variant} seed={job.seed} f={job.fraction}% "
          f"low_auprc={row['test_low_auprc']:.4f} low_recall={row['test_low_recall']:.4f} "
          f"iid_auprc={row['test_iid_auprc']:.4f} wall={row['wall_s']}s", flush=True)
    return row, hist


def load_pools():
    splits, meta = GT.load_cache(SUB / "task_cache.npz")
    io_pieces = CM.build_io(CM.load_ports())
    return splits, io_pieces, meta


def run_jobs(jobs, args, device):
    splits, io_pieces, meta = load_pools()
    print(f"pools: " + " ".join(f"{k}={len(v['y'])}" for k, v in splits.items())
          + f" | N={io_pieces['N']} pn={len(io_pieces['pn_idx'])} glom={io_pieces['n_glom_olf']}"
          + f"+{io_pieces['n_glom_thr']} jobs={len(jobs)}", flush=True)
    m_rows, h_rows = [], []
    for job in jobs:
        m, h = train_job(job, splits, io_pieces, args, device)
        m_rows.append(m); h_rows.extend(h)
    return m_rows, h_rows


# --------------------------------------------------------------------------- dispatch
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
        "--output-dir", str(args.output_dir), "--io-conditions", *args.io_conditions,
        "--arms", *args.arms, "--variants", *args.variants,
        "--seeds", *[str(s) for s in args.seeds], "--fractions", *[str(f) for f in args.fractions],
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--batch-size", str(args.batch_size), "--lr", str(args.lr), "--leak", str(args.leak),
        "--grad-clip", str(args.grad_clip), "--log-every", str(args.log_every),
        "--data-seed", str(args.data_seed), "--init-seed", str(args.init_seed),
        "--no-adapter" if not args.include_adapter else "--adapter",
        "--_worker-device", str(dev), "--_worker-out", str(out),
        "--_worker-job-ids", *[str(j) for j in jids],
    ]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    p.add_argument("--io-conditions", nargs="+", default=list(IO_CONDITIONS), choices=list(IO_CONDITIONS))
    p.add_argument("--arms", nargs="+", default=list(ALL_ARMS), choices=list(ALL_ARMS))
    p.add_argument("--variants", nargs="+", default=["standard", "graded_ln"],
                   choices=["standard", "graded_ln"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--fractions", nargs="+", type=int, default=[5, 10, 25, 50, 100])
    p.add_argument("--adapter", dest="include_adapter", action="store_true", default=True)
    p.add_argument("--no-adapter", dest="include_adapter", action="store_false")
    p.add_argument("--device-ids", nargs="+", type=int, default=None)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--leak", type=float, default=0.3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--data-seed", type=int, default=1234)
    p.add_argument("--init-seed", type=int, default=7000)
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="tiny CPU/GPU pipeline check")
    p.add_argument("--analyze-only", action="store_true", help="concatenate shard CSVs -> summary")
    p.add_argument("--print-shard-run-ids", action="store_true")
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
    return torch.device("cpu")


def all_jobs_for(args) -> list[Job]:
    return enumerate_jobs(args.io_conditions, args.arms, args.variants, args.seeds,
                          args.fractions, include_adapter=args.include_adapter)


def analyze(output_dir: Path):
    parts = sorted(output_dir.glob("metrics_shard*.csv"))
    if not parts:
        print(f"no metrics_shard*.csv in {output_dir}"); return 1
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.to_csv(output_dir / "metrics_by_run.csv", index=False)
    # connectome-vs-control effect size (Cohen's d across seeds), bio, standard, per fraction,
    # on the discriminating metrics (AUPRC saturates -> lean on recall-at-fixed-false-alarm + latency).
    summ = {}
    for io in df["io"].unique():
        for metric in ("test_low_recall_at_fpr10", "test_low_auroc", "test_low_auprc",
                       "test_iid_recall_at_fpr10"):
            for frac in sorted(df.fraction.unique()):
                sub = df[(df.io == io) & (df.variant == "standard") & (df.fraction == frac)]
                con = sub[sub.arm == "connectome"][metric].dropna()
                if not len(con):
                    continue
                row = {"connectome_mean": round(float(con.mean()), 4), "n_seed": int(len(con))}
                for ctrl in ("degree", "random", "spectrum", "dense"):
                    cv = sub[sub.arm == ctrl][metric].dropna()
                    if len(cv) and len(con):
                        pooled = np.sqrt((con.var(ddof=1) + cv.var(ddof=1)) / 2 + 1e-12)
                        row[f"{ctrl}_mean"] = round(float(cv.mean()), 4)
                        row[f"d_vs_{ctrl}"] = round(float((con.mean() - cv.mean()) / (pooled + 1e-9)), 3)
                summ[f"{io}::{metric}::f{frac}"] = row
    (output_dir / "analysis.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    print(f"wrote {output_dir/'metrics_by_run.csv'} ({len(df)} runs)")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        return smoke(args)
    if args.analyze_only:
        return analyze(args.output_dir)
    all_jobs = all_jobs_for(args)
    if args.print_shard_run_ids:
        return 0  # no per-run resume; shards are cheap + idempotent
    if args._worker_job_ids is not None:
        device = resolve_device(args, args._worker_device)
        jobs = [all_jobs[i] for i in args._worker_job_ids]
        m, h = run_jobs(jobs, args, device)
        args._worker_out.write_text(json.dumps({"metrics": m, "history": h}))
        return 0
    if args.shard is not None and args.num_shards is not None:
        device = resolve_device(args, None)
        jobs = all_jobs[args.shard::args.num_shards]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"shard {args.shard}/{args.num_shards} device={device} jobs={len(jobs)}/{len(all_jobs)}", flush=True)
        m, h = run_jobs(jobs, args, device)
        pd.DataFrame(m).to_csv(args.output_dir / f"metrics_shard{args.shard}.csv", index=False)
        pd.DataFrame(h).to_csv(args.output_dir / f"history_shard{args.shard}.csv", index=False)
        (args.output_dir / f"result_shard{args.shard}.json").write_text(
            json.dumps({"metrics": m, "shard": args.shard}))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if args.device_ids and len(args.device_ids) > 1:
        print(f"dispatch multi-gpu {args.device_ids} jobs={len(all_jobs)}", flush=True)
        metrics, history = dispatch_multi_gpu(all_jobs, args)
    else:
        device = resolve_device(args, args.device_ids[0] if args.device_ids else None)
        print(f"single-device {device} jobs={len(all_jobs)}", flush=True)
        m, h = run_jobs(all_jobs, args, device)
        metrics, history = pd.DataFrame(m), pd.DataFrame(h)
    metrics.to_csv(args.output_dir / "metrics_by_run.csv", index=False)
    history.to_csv(args.output_dir / "loss_history.csv", index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                    if not k.startswith("_worker")}, indent=2, sort_keys=True))
    print(f"complete jobs={len(all_jobs)} elapsed={round(time.monotonic()-t0,1)}s", flush=True)
    return 0


def smoke(args):
    """Fast pipeline check: 1 job per io, 2 epochs, tiny fraction."""
    device = resolve_device(args, args.device_ids[0] if args.device_ids else None)
    args.epochs = 2; args.patience = 2
    jobs = [Job("bio", "connectome", "standard", 0, 100),
            Job("bio", "degree", "standard", 0, 100),
            Job("bio", "spectrum", "standard", 0, 100),
            Job("generic", "connectome", "standard", 0, 100),
            Job("bio", "adapter_only", "standard", 0, 100),
            Job("bio", "connectome", "graded_ln", 0, 100)]
    m, _ = run_jobs(jobs, args, device)
    print("\nSMOKE OK:")
    for r in m:
        print(f"  {r['io']:7s} {r['arm']:12s} {r['variant']:9s} low_auprc={r['test_low_auprc']:.3f} "
              f"low_recall={r['test_low_recall']:.3f} iid_auprc={r['test_iid_auprc']:.3f} wall={r['wall_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
