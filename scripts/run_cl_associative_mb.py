#!/usr/bin/env python3
"""Experiment #1 — continual ASSOCIATIVE learning on the mushroom body: the CL
analog of the central-complex ring-attractor test, on the MB's NATIVE modality.

Motivation. Our Split-CIFAR continual-learning experiments found the MB connectome
no better than random — but Split-CIFAR is object recognition, a NON-native modality
for the MB. The committed associative-learning results tell the opposite story: on
odor→valence learning (the MB's native task) the connectome decisively beats matched
random/degree controls (pruned: 97.6% vs 90.5%; reversal-probe 0.967 vs 0.89), and the
topology-preserving weight_shuffle tracks the connectome — so the WIRING carries the
advantage there. The thesis: the connectome helps when the task matches the computation
its topology implements. The CX test probes that for ring-attractor working memory; THIS
probes it for continual learning, by swapping Split-CIFAR's pixels for the MB's native
odor→valence associations.

Task. "Split-Odor": K sequential binary tasks, each a disjoint set of sparse odor
prototypes with a fixed appetitive/aversive valence. Domain-incremental, single shared
2-logit valence head (no task IDs) — the same fairness protocol as run_continual_learning.
Catastrophic forgetting is measured by the R[a][b] accuracy matrix and ACC/BWT/Forgetting.

Model. odor → (trainable W_in) sensory/PN pool → MB recurrent core (T microsteps) →
(trainable readout) output/MBON pool → valence. The recurrent core is the FlyWire MB
connectome, run FROZEN (reservoir) or TRAINABLE (one weight per observed edge), and
compared against degree/weight-matched RANDOM and weight-SHUFFLED cores — the direct
frozen-vs-trainable × connectome-vs-random matrix of the CX structure test.

Prediction. Unlike Split-CIFAR, here the connectome should resist forgetting better than
random — this is where the MB wiring "earns its keep" on continual learning.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (ROOT, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_bpu_image_classification as bpu  # noqa: E402
import run_continual_learning as cl  # noqa: E402  (train_one_task, eval_acc_loss, cl_metrics)
import run_mb_associative_learning as mb  # noqa: E402  (matrix_for_model)
import run_optic_flow_data_efficiency as de  # noqa: E402  (matrix/pool loaders)

PRUNE = cl.PRUNE
cl_metrics = cl.cl_metrics

# model -> (variant, recurrent_trainable). variant: connectome | random | weight_shuffle
ASSOC_MODEL_SPECS = {
    "connectome_frozen":      ("connectome",     False),
    "connectome_trainable":   ("connectome",     True),
    "random_frozen":          ("random",         False),
    "random_trainable":       ("random",         True),
    "weight_shuffle_frozen":  ("weight_shuffle", False),
    "weight_shuffle_trainable": ("weight_shuffle", True),
}
DEFAULT_MODELS = tuple(ASSOC_MODEL_SPECS)
MODEL_COLORS = {
    "connectome_frozen": "#1f77b4", "connectome_trainable": "#17becf",
    "random_frozen": "#ff7f0e", "random_trainable": "#ffbb78",
    "weight_shuffle_frozen": "#2ca02c", "weight_shuffle_trainable": "#98df8a",
}


# --- data: Split-Odor continual valence tasks -------------------------------
def _odor_bank(num_odors, odor_dim, sparsity, rng):
    bank = rng.normal(0.0, 1.0, size=(num_odors, odor_dim)).astype(np.float32)
    bank *= (rng.random(bank.shape) < sparsity).astype(np.float32)
    norms = np.linalg.norm(bank, axis=1, keepdims=True)
    empty = norms.squeeze(-1) == 0
    if np.any(empty):  # guarantee no all-zero odor
        bank[empty, rng.integers(0, odor_dim, size=int(empty.sum()))] = 1.0
        norms = np.linalg.norm(bank, axis=1, keepdims=True)
    return (bank / np.maximum(norms, 1e-6)).astype(np.float32)


def build_assoc_tasks(args):
    """K binary odor->valence tasks over disjoint odor sets; balanced valence."""
    K = args.num_tasks
    m = args.odors_per_task
    rng = np.random.default_rng(args.data_seed)
    bank = _odor_bank(K * m, args.odor_dim, args.odor_sparsity, rng)
    tasks = []
    for t in range(K):
        ids = np.arange(t * m, (t + 1) * m)
        valence = np.zeros(m, dtype=np.int64)
        valence[: m // 2] = 1
        valence = rng.permutation(valence)

        def split(n_per_odor, sseed):
            r = np.random.default_rng(sseed)
            X, Y = [], []
            for i, oid in enumerate(ids):
                noise = r.normal(0.0, args.odor_noise_std, size=(n_per_odor, args.odor_dim)).astype(np.float32)
                X.append(bank[oid][None, :] + noise)
                Y.append(np.full(n_per_odor, valence[i], dtype=np.int64))
            return np.concatenate(X), np.concatenate(Y)

        tr_x, tr_y = split(args.train_per_odor, args.data_seed + 1000 * t + 1)
        va_x, va_y = split(args.val_per_odor, args.data_seed + 1000 * t + 2)
        te_x, te_y = split(args.test_per_odor, args.data_seed + 1000 * t + 3)
        tasks.append({"task_id": t, "train_x": tr_x, "train_y": tr_y,
                      "val_x": va_x, "val_y": va_y, "test_x": te_x, "test_y": te_y})
    # one global z-score from all training odors (task-agnostic), like the CIFAR harness
    allx = np.concatenate([t["train_x"] for t in tasks])
    mu, sd = allx.mean(0, keepdims=True), allx.std(0, keepdims=True) + 1e-6
    for t in tasks:
        for k in ("train_x", "val_x", "test_x"):
            t[k] = ((t[k] - mu) / sd).astype(np.float32)
    return tasks, args.odor_dim


def seed_order(seed, K):
    rng = np.random.default_rng(10_000 + seed)
    return rng.permutation(K).tolist()


# --- model ------------------------------------------------------------------
def build_assoc_model(name, base_cap, pools_cap, input_dim, args, seed):
    variant, trainable = ASSOC_MODEL_SPECS[name]
    n = base_cap.shape[0]
    sensory = PRUNE.pool_indices(pools_cap, n, "sensory")
    output = PRUNE.pool_indices(pools_cap, n, "output")
    if variant == "connectome":
        rec = base_cap.tocoo()
    elif variant == "random":
        rec = mb.matrix_for_model(base_cap.tocoo(), mb.MODEL_RANDOM, seed)
    elif variant == "weight_shuffle":
        rec = mb.matrix_for_model(base_cap.tocoo(), mb.MODEL_WEIGHT_SHUFFLE, seed)
    else:
        raise ValueError(variant)
    model = bpu.BPUClassifier(rec, input_dim, 2, sensory, output, runtime="sparse",
                              recurrent_trainable=trainable, timesteps=args.timesteps,
                              state_clip=args.state_clip, seed=seed)
    return model, trainable


def _rec_values(model):
    return getattr(model, "W_rec_values", None)


# --- one (model, seed) stream ----------------------------------------------
@dataclass
class Stream:
    model: str
    seed: int


def run_stream(stream, base_cap, pools_cap, tasks_all, input_dim, args, device):
    variant, trainable = ASSOC_MODEL_SPECS[stream.model]
    K = len(tasks_all)
    order = seed_order(stream.seed, K)
    ordered = [tasks_all[t] for t in order]
    torch.manual_seed(stream.seed); np.random.seed(stream.seed)
    model, _ = build_assoc_model(stream.model, base_cap, pools_cap, input_dim, args,
                                 seed=args.init_seed + stream.seed)
    model = model.to(device)
    rv = _rec_values(model)
    rec0 = rv.detach().cpu().clone() if rv is not None else None

    t0 = time.monotonic()
    R = np.full((K, K), np.nan)
    diag_epochs = [0] * K
    for p in range(K):
        rng = np.random.default_rng(args.init_seed + 100 * stream.seed + p)
        opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=args.lr)
        ep, _ = cl.train_one_task(model, ordered[p], opt, args, device, rng)
        diag_epochs[p] = ep
        for a in range(K):
            acc, _ = cl.eval_acc_loss(model, ordered[a]["test_x"], ordered[a]["test_y"], device, args.batch_size)
            R[a][p] = acc

    w_rec_drift = 0.0
    rv = _rec_values(model)
    if rv is not None and rec0 is not None:
        w_rec_drift = float(torch.linalg.vector_norm(rv.detach().cpu() - rec0).item())
    if not trainable and rv is not None:
        assert w_rec_drift == 0.0, f"FROZEN VIOLATION: {stream.model} drifted {w_rec_drift}"

    cm = cl_metrics(R)
    diag = cm["diag"]
    elapsed = time.monotonic() - t0
    rec = {
        "model": stream.model, "seed": stream.seed, "variant": variant,
        "recurrent_trainable": int(bool(trainable)), "order": json.dumps(order),
        "N": int(getattr(model, "N", 0)), "trainable_params": model.trainable_parameter_count(),
        "acc_final": cm["acc_final"], "bwt": cm["bwt"], "forgetting": cm["forgetting"], "fwt": cm["fwt"],
        "learning_acc_mean": float(np.mean(diag)), "w_rec_drift": w_rec_drift,
        "R": json.dumps(np.round(R, 5).tolist()), "diag": json.dumps([round(d, 5) for d in diag]),
        "epochs": json.dumps(diag_epochs), "wall_seconds": round(elapsed, 1),
    }
    print(f"stream-done model={stream.model:24s} seed={stream.seed} ACC_final={cm['acc_final']:.4f} "
          f"BWT={cm['bwt']:+.4f} F={cm['forgetting']:.4f} learn={np.mean(diag):.3f} "
          f"w_rec_drift={w_rec_drift:.2e} wall={elapsed:.1f}s", flush=True)
    return rec


# --- orchestration ----------------------------------------------------------
def prepare_inputs(args):
    base = de.ofb.load_matrix(args.matrix)
    pools_full = de.load_pools_aligned(args.pool_assignments, base.shape[0])
    base_cap, keep = bpu.pool_aware_truncate(base, pools_full, args.max_neurons)
    pools_cap = de.remap_pools(pools_full, keep)
    tasks, input_dim = build_assoc_tasks(args)
    return base_cap, pools_cap, tasks, input_dim


def run_streams(streams, args, device):
    base_cap, pools_cap, tasks, input_dim = prepare_inputs(args)
    ns = PRUNE.pool_indices(pools_cap, base_cap.shape[0], "sensory").size
    no = PRUNE.pool_indices(pools_cap, base_cap.shape[0], "output").size
    print(f"prepared base_cap N={base_cap.shape[0]} edges={base_cap.nnz} sensory={ns} output={no} "
          f"odor_dim={input_dim} tasks={len(tasks)} streams={len(streams)}", flush=True)
    return [run_stream(s, base_cap, pools_cap, tasks, input_dim, args, device) for s in streams]


def enumerate_streams(models, seeds):
    return [Stream(m, s) for m in models for s in seeds]


def _worker_argv(args, dev, ids, out):
    return [
        "--matrix", str(args.matrix), "--pool-assignments", str(args.pool_assignments),
        "--output-dir", str(args.output_dir), "--max-neurons", str(args.max_neurons),
        "--models", *args.models, "--seeds", *[str(s) for s in args.seeds],
        "--num-tasks", str(args.num_tasks), "--odors-per-task", str(args.odors_per_task),
        "--odor-dim", str(args.odor_dim), "--odor-sparsity", str(args.odor_sparsity),
        "--odor-noise-std", str(args.odor_noise_std), "--train-per-odor", str(args.train_per_odor),
        "--val-per-odor", str(args.val_per_odor), "--test-per-odor", str(args.test_per_odor),
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--batch-size", str(args.batch_size), "--lr", str(args.lr), "--grad-clip", str(args.grad_clip),
        "--state-clip", str(args.state_clip), "--timesteps", str(args.timesteps),
        "--data-seed", str(args.data_seed), "--init-seed", str(args.init_seed),
        "--_worker-device", str(dev), "--_worker-out", str(out),
        "--_worker-stream-ids", *[str(i) for i in ids],
    ]


def dispatch_multi_gpu(streams, args):
    device_ids = args.device_ids
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts = {d: [] for d in device_ids}
    for i in range(len(streams)):
        parts[device_ids[i % len(device_ids)]].append(i)
    procs, files = [], []
    for dev, ids in parts.items():
        if not ids:
            continue
        out = args.output_dir / f"_worker_dev{dev}.json"
        files.append(out)
        cmd = [sys.executable, str(Path(__file__).resolve())] + _worker_argv(args, dev, ids, out)
        log = (args.output_dir / f"_worker_dev{dev}.log").open("w")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log, dev))
    failed = []
    for proc, log, dev in procs:
        rc = proc.wait(); log.close()
        if rc != 0:
            failed.append(dev)
    if failed:
        raise RuntimeError(f"worker(s) on device(s) {failed} failed; see _worker_dev*.log")
    rows = []
    for f in files:
        rows.extend(json.loads(f.read_text()))
    return pd.DataFrame(rows)


# --- outputs ----------------------------------------------------------------
def write_outputs(output_dir, df, total_seconds, args):
    df.to_csv(output_dir / "metrics_by_stream.csv", index=False)
    agg = df.groupby("model").agg(
        variant=("variant", "first"), recurrent_trainable=("recurrent_trainable", "first"),
        acc_final=("acc_final", "mean"), acc_final_se=("acc_final", "sem"),
        forgetting=("forgetting", "mean"), forgetting_se=("forgetting", "sem"),
        bwt=("bwt", "mean"), learning_acc=("learning_acc_mean", "mean"),
        w_rec_drift=("w_rec_drift", "mean"), trainable_params=("trainable_params", "first"),
    ).reset_index().sort_values(["recurrent_trainable", "forgetting"])
    agg.to_csv(output_dir / "cl_associative_summary.csv", index=False)

    # paired connectome-vs-control at each train mode
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4), dpi=150)
    order = list(agg["model"])
    colors = [MODEL_COLORS.get(m, "#333") for m in order]
    x = np.arange(len(order))
    a1.bar(x, agg["forgetting"], yerr=agg["forgetting_se"].fillna(0), color=colors, capsize=3)
    a1.set_xticks(x); a1.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    a1.set_ylabel("Forgetting F (lower=better)"); a1.set_title("Split-Odor continual valence learning")
    a1.grid(True, axis="y", alpha=0.25)
    for m in order:
        r = agg[agg.model == m].iloc[0]
        a2.errorbar(r["acc_final"], r["forgetting"], xerr=r["acc_final_se"], yerr=r["forgetting_se"],
                    fmt="o" if r["recurrent_trainable"] == 0 else "s",
                    color=MODEL_COLORS.get(m, "#333"), capsize=2)
        a2.annotate(m, (r["acc_final"], r["forgetting"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    a2.set_xlabel("Final avg accuracy"); a2.set_ylabel("Forgetting F")
    a2.set_title("Acc vs forgetting (○ frozen, □ trainable)"); a2.grid(True, alpha=0.25)
    fig.suptitle("MB native-modality continual learning: connectome vs matched controls", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "cl_associative_mb.png")
    plt.close(fig)

    lines = [
        "# Split-Odor continual associative learning on the MB",
        "",
        f"K={args.num_tasks} sequential binary odor->valence tasks, shared 2-logit head, "
        f"cap {args.max_neurons}, T={args.timesteps}, seeds {args.seeds}. "
        f"Wall {total_seconds/60:.1f} min.",
        "", "```", agg.round(4).to_string(index=False), "```", "",
        "Connectome vs degree/weight-matched random and weight-shuffle cores, each FROZEN",
        "and TRAINABLE. Prediction: unlike Split-CIFAR, the connectome resists forgetting",
        "better than random on this native odor->valence modality.",
        "",
    ]
    (output_dir / "cl_associative_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--pool-assignments", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/cl_associative_mb"))
    p.add_argument("--max-neurons", type=int, default=5000)
    p.add_argument("--models", nargs="+", choices=list(ASSOC_MODEL_SPECS), default=list(DEFAULT_MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device-ids", nargs="+", type=int, default=None)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    # task
    p.add_argument("--num-tasks", type=int, default=5)
    p.add_argument("--odors-per-task", type=int, default=20)
    p.add_argument("--odor-dim", type=int, default=100)
    p.add_argument("--odor-sparsity", type=float, default=0.2)
    p.add_argument("--odor-noise-std", type=float, default=0.1)
    p.add_argument("--train-per-odor", type=int, default=200)
    p.add_argument("--val-per-odor", type=int, default=50)
    p.add_argument("--test-per-odor", type=int, default=50)
    # train
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--state-clip", type=float, default=5.0)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--data-seed", type=int, default=12345)
    p.add_argument("--init-seed", type=int, default=7000)
    p.add_argument("--dense-lr", type=float, default=1e-4, help=argparse.SUPPRESS)  # cl.train_one_task compat
    p.add_argument("--_worker-device", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-stream-ids", nargs="*", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    streams = enumerate_streams(args.models, args.seeds)
    if args._worker_stream_ids is not None:
        device = torch.device(f"cuda:{args._worker_device}") if args._worker_device is not None \
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        rows = run_streams([streams[i] for i in args._worker_stream_ids], args, device)
        args._worker_out.write_text(json.dumps(rows))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if args.device_ids and len(args.device_ids) > 1:
        print(f"dispatch multi-gpu device_ids={args.device_ids} streams={len(streams)}", flush=True)
        df = dispatch_multi_gpu(streams, args)
    else:
        device = torch.device("cpu") if (args.device == "cpu" or not torch.cuda.is_available()) \
            else (torch.device(f"cuda:{args.device_ids[0]}") if args.device_ids else torch.device("cuda"))
        print(f"single-device run device={device} streams={len(streams)}", flush=True)
        df = pd.DataFrame(run_streams(streams, args, device))

    total = time.monotonic() - t0
    write_outputs(args.output_dir, df, total, args)
    (args.output_dir / "run_config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
         if not k.startswith("_worker")}, indent=2, sort_keys=True))
    print(f"complete streams={len(streams)} total_wall={total/60:.1f}min "
          f"figure={args.output_dir / 'cl_associative_mb.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
