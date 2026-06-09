#!/usr/bin/env python3
"""Split-CIFAR-10 domain-incremental continual learning: does a connectome / pruned
recurrent backbone resist catastrophic forgetting better than a dense matrix or MLP?

Protocol (locked by a design panel; see docs/continual_learning.md):

  * DOMAIN-INCREMENTAL, SINGLE SHARED HEAD (the fairness fix). 5 sequential binary
    tasks over CIFAR-10 class pairs T0={0,1} ... T4={8,9}; the label is within-pair
    parity (first class -> 0, second -> 1). All tasks share ONE 2-logit head and one
    label space; there are no per-task heads and no task IDs. A frozen backbone can
    therefore still forget (through the shared W_in + head) and gets no free pass.
  * Frozen-vs-trainable trunk axis: connectome and pruned-connectome are each run with
    the recurrent matrix frozen AND trainable, separating "frozen forgets less" from
    "connectome forgets less".
  * Per task: fresh Adam (no momentum carryover), per-task RNG reseed, best-val
    checkpoint restored before the next task, early stop (patience), no replay, no
    regularizer (plain lower bound). One global z-score from the full CIFAR-10 train
    set (task-agnostic) is the only normalization. No BatchNorm.
  * After training each stream position p, evaluate every task's test set to fill the
    accuracy matrix R[a][b] = test_acc(task at position a, model after position b).

Metrics: ACC_final, BWT, Forgetting (F), FWT, the R matrix, per-task learning accuracy
(diagonal), W_rec drift (freeze check), and rep_drift (attribution: is a frozen "win"
genuine representational stability or just a static, non-learning representation?).

Substrate: FlyWire optic-lobe matrix capped via --max-neurons (force-keeps sensory).
Inputs: prepared --matrix + --pool-assignments. Job unit for multi-GPU dispatch is a
full (model, seed) 5-task stream (sequential internally).
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
from scipy import sparse
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (ROOT, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_bpu_image_classification as bpu  # noqa: E402
import run_mb_associative_learning as mb  # noqa: E402
import run_optic_flow_data_efficiency as de  # noqa: E402

PRUNE = de.prune_mod

# model -> (variant, runtime, recurrent_trainable). variant: connectome|pruned|random|weight_shuffle|mlp
CL_MODEL_SPECS = {
    "connectome_frozen": ("connectome", "sparse", False),
    "connectome_trainable": ("connectome", "sparse", True),
    "connectome_pruned_frozen": ("pruned", "sparse", False),
    "connectome_pruned_trainable": ("pruned", "sparse", True),
    "dense_trainable": ("connectome", "dense", True),
    "mlp": ("mlp", "mlp", True),
    "random_sparse_frozen": ("random", "sparse", False),
    "weight_shuffle_frozen": ("weight_shuffle", "sparse", False),
}
DEFAULT_MODELS = tuple(CL_MODEL_SPECS)
TASK_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
# Per-seed task orders (positions index into TASK_PAIRS); canonical for seed 0.
SEED_ORDERS = {
    0: [0, 1, 2, 3, 4],
    1: [3, 0, 4, 1, 2],
    2: [1, 4, 2, 0, 3],
}
MODEL_COLORS = {
    "connectome_frozen": "#1f77b4", "connectome_trainable": "#17becf",
    "connectome_pruned_frozen": "#9467bd", "connectome_pruned_trainable": "#c5b0d5",
    "dense_trainable": "#d62728", "mlp": "#7f7f7f",
    "random_sparse_frozen": "#ff7f0e", "weight_shuffle_frozen": "#2ca02c",
}


# --- data -------------------------------------------------------------------
def build_task_datasets(args) -> tuple[list[dict], np.ndarray, dict]:
    """Per-task disjoint, class-balanced splits over CIFAR-10 pairs (shared across runs)."""
    train_x, train_y, test_x, test_y = bpu.load_task("cifar10", args.data_dir)
    rng = np.random.default_rng(args.data_seed)
    tasks = []
    label_maps = {}
    per_class_train = args.per_task_train // 2
    per_class_val = args.per_task_val // 2
    for ti, (c0, c1) in enumerate(TASK_PAIRS):
        label_maps[ti] = {int(c0): 0, int(c1): 1}
        tr_idx, va_idx = [], []
        for cls in (c0, c1):
            pool = rng.permutation(np.where(train_y == cls)[0])
            need = per_class_train + per_class_val
            assert pool.size >= need, f"task {ti} class {cls}: only {pool.size} train imgs"
            tr_idx.append(pool[:per_class_train])
            va_idx.append(pool[per_class_train:per_class_train + per_class_val])
        tr = np.concatenate(tr_idx)
        va = np.concatenate(va_idx)
        te = np.where((test_y == c0) | (test_y == c1))[0]
        # disjointness guards
        assert len(set(tr.tolist()) & set(va.tolist())) == 0, "train/val overlap"

        def parity(idx, ysrc):
            return (ysrc[idx] == c1).astype(np.int64)

        tasks.append({
            "task_id": ti, "pair": (int(c0), int(c1)),
            "train_x": train_x[tr], "train_y": parity(tr, train_y),
            "val_x": train_x[va], "val_y": parity(va, train_y),
            "test_x": test_x[te], "test_y": parity(te, test_y),
        })
    # global normalization is already applied inside bpu.load_task (train stats only)
    input_dim = train_x.shape[1]
    return tasks, np.array([input_dim]), label_maps


# --- model construction -----------------------------------------------------
def _remap_io_to_pruned(keep_indices: np.ndarray, sensory: np.ndarray, output: np.ndarray):
    pos = {int(o): i for i, o in enumerate(keep_indices.tolist())}
    s = np.array([pos[int(o)] for o in sensory if int(o) in pos], dtype=np.int64)
    o = np.array([pos[int(o)] for o in output if int(o) in pos], dtype=np.int64)
    return s, o


def build_cl_model(model: str, base_cap: sparse.csr_matrix, pools_cap: pd.DataFrame,
                   input_dim: int, args, seed: int):
    variant, runtime, rec_trainable = CL_MODEL_SPECS[model]
    n = base_cap.shape[0]
    sensory = PRUNE.pool_indices(pools_cap, n, "sensory")
    output = PRUNE.pool_indices(pools_cap, n, "output")

    if runtime == "mlp":
        ref = bpu.BPUClassifier(base_cap.tocoo(), input_dim, 2, sensory, output,
                                runtime="sparse", recurrent_trainable=False,
                                timesteps=args.timesteps, state_clip=args.state_clip, seed=seed)
        hidden = bpu.matched_hidden(ref.trainable_parameter_count(), input_dim, 2)
        return bpu.MLPBaseline(input_dim, hidden, 2, seed), "mlp", rec_trainable

    if variant == "pruned":
        pr = PRUNE.prune_recurrent_matrix(base_cap.tocoo(), pools_cap,
                                          max_hops=args.prune_max_hops,
                                          max_internal_nodes=args.prune_max_internal_nodes)
        assert pr.matrix.nnz > 0, "pruned matrix is empty"
        rec = pr.matrix.tocoo()
        s_idx, o_idx = _remap_io_to_pruned(pr.keep_indices, sensory, output)
        assert s_idx.size > 0 and o_idx.size > 0, "pruned matrix lost sensory/output pool"
    else:
        if variant == "connectome":
            rec = base_cap.tocoo()
        elif variant == "random":
            rec = mb.matrix_for_model(base_cap.tocoo(), mb.MODEL_RANDOM, seed)
        elif variant == "weight_shuffle":
            rec = mb.matrix_for_model(base_cap.tocoo(), mb.MODEL_WEIGHT_SHUFFLE, seed)
        else:
            raise ValueError(variant)
        s_idx, o_idx = sensory, output

    m = bpu.BPUClassifier(rec, input_dim, 2, s_idx, o_idx, runtime=runtime,
                          recurrent_trainable=rec_trainable, timesteps=args.timesteps,
                          state_clip=args.state_clip, seed=seed)
    return m, runtime, rec_trainable


def _rec_param(model):
    return model.W_rec if getattr(model, "runtime", None) == "dense" else getattr(model, "W_rec_values", None)


def _to_device_state(state, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in state.items()}


def cl_metrics(R) -> dict:
    """Pure continual-learning metrics from the position-indexed accuracy matrix R.

    R[a][b] = test acc on the task at stream-position a, after training position b.
    BWT<0 and F>0 both mean forgetting; both are averaged over positions 0..K-2
    (the last position has no later position).
    """
    R = np.asarray(R, dtype=float)
    K = R.shape[0]
    diag = [float(R[i][i]) for i in range(K)]
    acc_final = float(np.mean([R[i][K - 1] for i in range(K)]))
    if K > 1:
        bwt = float(np.mean([R[i][K - 1] - R[i][i] for i in range(K - 1)]))
        forgetting = float(np.mean([max(R[i][i:K]) - R[i][K - 1] for i in range(K - 1)]))
        fwt = float(np.mean([R[i][i - 1] - 0.5 for i in range(1, K)]))
    else:
        bwt = forgetting = fwt = 0.0
    return {"diag": diag, "acc_final": acc_final, "bwt": bwt, "forgetting": forgetting, "fwt": fwt}


# --- one task train + eval --------------------------------------------------
def train_one_task(model, task, optimizer, args, device, rng):
    ce = nn.CrossEntropyLoss()
    tr_x, tr_y = task["train_x"], task["train_y"]
    n = tr_x.shape[0]
    best_val = float("inf")
    best_state = None
    wait = 0
    epochs_run = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(n)
        for s in range(0, n, args.batch_size):
            idx = order[s:s + args.batch_size]
            xb = torch.from_numpy(tr_x[idx]).to(device)
            yb = torch.from_numpy(tr_y[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = ce(model(xb), yb)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        _, val_loss = eval_acc_loss(model, task["val_x"], task["val_y"], device, args.batch_size)
        epochs_run = epoch
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if args.patience > 0 and wait >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return epochs_run, best_state


@torch.no_grad()
def eval_acc_loss(model, x, y, device, batch_size):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    correct, loss_sum = 0, 0.0
    for s in range(0, x.shape[0], batch_size):
        xb = torch.from_numpy(x[s:s + batch_size]).to(device)
        yb = torch.from_numpy(y[s:s + batch_size]).to(device)
        logits = model(xb)
        loss_sum += float(ce(logits, yb).item())
        correct += int((logits.argmax(1) == yb).sum().item())
    return correct / x.shape[0], loss_sum / x.shape[0]


@torch.no_grad()
def hidden_repr(model, x, device, batch_size):
    model.eval()
    out = []
    for s in range(0, x.shape[0], batch_size):
        xb = torch.from_numpy(x[s:s + batch_size]).to(device)
        out.append(model.hidden(xb).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


# --- one (model, seed) stream -----------------------------------------------
@dataclass
class Stream:
    model: str
    seed: int


def run_stream(stream: Stream, base_cap, pools_cap, tasks, input_dim, args, device) -> dict:
    order = SEED_ORDERS.get(stream.seed, list(range(len(TASK_PAIRS))))
    ordered = [tasks[t] for t in order]
    K = len(ordered)
    torch.manual_seed(stream.seed)
    np.random.seed(stream.seed)
    model, runtime, rec_trainable = build_cl_model(
        stream.model, base_cap, pools_cap, input_dim, args, seed=args.init_seed + stream.seed)
    model = model.to(device)
    lr = args.dense_lr if runtime == "dense" else args.lr

    # freeze guard: snapshot recurrent weights to verify *_frozen never change
    rec0 = None
    rp = _rec_param(model)
    if rp is not None:
        rec0 = rp.detach().cpu().clone()

    t0 = time.monotonic()
    R = np.full((K, K), np.nan, dtype=np.float64)
    diag_epochs = [0] * K
    task_ckpts: list[dict] = []
    print(f"stream-start model={stream.model} seed={stream.seed} order={order} runtime={runtime} "
          f"N={getattr(model,'N','-')} trainable={model.trainable_parameter_count()} "
          f"recurrent={model.recurrent_parameter_count()} lr={lr}", flush=True)

    for p in range(K):
        # per-task RNG reseed (independent epoch orders across tasks)
        rng = np.random.default_rng(args.init_seed + 100 * stream.seed + p)
        optimizer = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=lr)
        epochs_run, _ = train_one_task(model, ordered[p], optimizer, args, device, rng)
        diag_epochs[p] = epochs_run
        task_ckpts.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        # evaluate every task's test set after training position p
        for a in range(K):
            acc, _ = eval_acc_loss(model, ordered[a]["test_x"], ordered[a]["test_y"], device, args.batch_size)
            R[a][p] = acc

    # --- correctness: frozen recurrent must be bit-identical end-to-end ---
    w_rec_drift = 0.0
    rp = _rec_param(model)
    if rp is not None and rec0 is not None:
        w_rec_drift = float(torch.linalg.vector_norm(rp.detach().cpu() - rec0).item())
    if not rec_trainable and rp is not None:
        assert w_rec_drift == 0.0, f"FROZEN VIOLATION: {stream.model} W_rec drifted by {w_rec_drift}"

    # --- rep_drift attribution: representation change on task a between after-a and final ---
    final_state = _to_device_state(task_ckpts[-1], device)
    rep_drift = []
    for a in range(K):
        model.load_state_dict(_to_device_state(task_ckpts[a], device))
        h_then = hidden_repr(model, ordered[a]["test_x"], device, args.batch_size)
        model.load_state_dict(final_state)
        h_final = hidden_repr(model, ordered[a]["test_x"], device, args.batch_size)
        rep_drift.append(float(np.mean(np.linalg.norm(h_then - h_final, axis=1))))
    model.load_state_dict(final_state)

    # --- metrics ---
    cm = cl_metrics(R)
    diag = cm["diag"]
    acc_final, bwt, forgetting, fwt = cm["acc_final"], cm["bwt"], cm["forgetting"], cm["fwt"]
    elapsed = time.monotonic() - t0

    rec = {
        "model": stream.model, "seed": stream.seed, "runtime": runtime,
        "recurrent_trainable": int(bool(rec_trainable)), "order": json.dumps(order),
        "N": int(getattr(model, "N", 0)),
        "trainable_params": model.trainable_parameter_count(),
        "recurrent_params": model.recurrent_parameter_count(),
        "lr": lr, "acc_final": acc_final, "bwt": bwt, "forgetting": forgetting, "fwt": fwt,
        "learning_acc_mean": float(np.mean(diag)), "w_rec_drift": w_rec_drift,
        "rep_drift_mean": float(np.mean(rep_drift)),
        "R": json.dumps(np.round(R, 5).tolist()),
        "diag": json.dumps([round(d, 5) for d in diag]),
        "rep_drift": json.dumps([round(d, 5) for d in rep_drift]),
        "epochs": json.dumps(diag_epochs), "wall_seconds": round(elapsed, 1),
    }
    lo = min(diag)
    flag = "  [WARN: a task <0.7 learning acc]" if lo < 0.7 else ""
    print(f"stream-done model={stream.model} seed={stream.seed} ACC_final={acc_final:.4f} "
          f"BWT={bwt:+.4f} F={forgetting:.4f} learn={np.mean(diag):.3f} w_rec_drift={w_rec_drift:.2e} "
          f"wall={elapsed:.1f}s{flag}", flush=True)
    return rec


# --- orchestration ----------------------------------------------------------
def prepare_inputs(args):
    base = de.ofb.load_matrix(args.matrix)
    pools_full = de.load_pools_aligned(args.pool_assignments, base.shape[0])
    base_cap, keep = bpu.pool_aware_truncate(base, pools_full, args.max_neurons)
    pools_cap = de.remap_pools(pools_full, keep)
    tasks, dim, label_maps = build_task_datasets(args)
    return base_cap, pools_cap, tasks, int(dim[0]), label_maps


def run_streams(streams, args, device):
    base_cap, pools_cap, tasks, input_dim, label_maps = prepare_inputs(args)
    ns = PRUNE.pool_indices(pools_cap, base_cap.shape[0], "sensory").size
    no = PRUNE.pool_indices(pools_cap, base_cap.shape[0], "output").size
    print(f"prepared base_cap N={base_cap.shape[0]} edges={base_cap.nnz} sensory={ns} output={no} "
          f"tasks={[t['pair'] for t in tasks]} label_maps={label_maps} streams={len(streams)}", flush=True)
    return [run_stream(s, base_cap, pools_cap, tasks, input_dim, args, device) for s in streams]


def enumerate_streams(models, seeds):
    return [Stream(m, s) for m in models for s in seeds]


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
        rc = proc.wait()
        log.close()
        if rc != 0:
            failed.append(dev)
    if failed:
        raise RuntimeError(f"worker(s) on device(s) {failed} failed; see _worker_dev*.log")
    rows = []
    for f in files:
        rows.extend(json.loads(f.read_text()))
    return pd.DataFrame(rows)


def _worker_argv(args, dev, ids, out):
    return [
        "--matrix", str(args.matrix), "--pool-assignments", str(args.pool_assignments),
        "--output-dir", str(args.output_dir), "--data-dir", str(args.data_dir),
        "--max-neurons", str(args.max_neurons), "--models", *args.models,
        "--seeds", *[str(s) for s in args.seeds], "--epochs", str(args.epochs),
        "--patience", str(args.patience), "--batch-size", str(args.batch_size),
        "--lr", str(args.lr), "--dense-lr", str(args.dense_lr), "--grad-clip", str(args.grad_clip),
        "--state-clip", str(args.state_clip), "--timesteps", str(args.timesteps),
        "--per-task-train", str(args.per_task_train), "--per-task-val", str(args.per_task_val),
        "--prune-max-hops", str(args.prune_max_hops),
        "--prune-max-internal-nodes", str(args.prune_max_internal_nodes),
        "--data-seed", str(args.data_seed), "--init-seed", str(args.init_seed),
        "--_worker-device", str(dev), "--_worker-out", str(out),
        "--_worker-stream-ids", *[str(i) for i in ids],
    ]


# --- outputs ----------------------------------------------------------------
def write_outputs(output_dir: Path, df: pd.DataFrame, total_seconds: float, args):
    df.to_csv(output_dir / "metrics_by_stream.csv", index=False)
    agg = df.groupby("model").agg(
        acc_final=("acc_final", "mean"), acc_final_se=("acc_final", "sem"),
        bwt=("bwt", "mean"), bwt_se=("bwt", "sem"),
        forgetting=("forgetting", "mean"), forgetting_se=("forgetting", "sem"),
        learning_acc=("learning_acc_mean", "mean"), rep_drift=("rep_drift_mean", "mean"),
        w_rec_drift=("w_rec_drift", "mean"),
        trainable_params=("trainable_params", "first"),
    ).reset_index().sort_values("forgetting")
    agg.to_csv(output_dir / "cl_summary.csv", index=False)

    models = list(agg["model"])
    colors = [MODEL_COLORS.get(m, "#333333") for m in models]
    fig, (axb, axs) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150)
    x = np.arange(len(models))
    axb.bar(x, agg["forgetting"], yerr=agg["forgetting_se"].fillna(0), color=colors, capsize=3)
    axb.set_xticks(x); axb.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
    axb.set_ylabel("Forgetting F (higher = worse)")
    axb.set_title("Catastrophic forgetting by model")
    axb.grid(True, axis="y", alpha=0.25)
    for xi, m in zip(x, models):
        r = agg[agg.model == m].iloc[0]
        axs.errorbar(r["acc_final"], r["forgetting"], xerr=r["acc_final_se"], yerr=r["forgetting_se"],
                     fmt="o", color=MODEL_COLORS.get(m, "#333"), capsize=2)
        axs.annotate(m, (r["acc_final"], r["forgetting"]), fontsize=7,
                     xytext=(4, 3), textcoords="offset points")
    axs.set_xlabel("Final average accuracy (higher = better)")
    axs.set_ylabel("Forgetting F (lower = better)")
    axs.set_title("Accuracy vs forgetting (best = bottom-right)")
    axs.grid(True, alpha=0.25)
    fig.suptitle("Split-CIFAR-10 domain-incremental CL (single shared head, no replay)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "continual_learning_split_cifar10.png")
    plt.close(fig)

    # R-matrix heatmap grid (mean over seeds, canonical position indexing)
    K = len(TASK_PAIRS)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=130)
    for ax, model in zip(axes.ravel(), df["model"].unique()):
        Rs = np.stack([np.array(json.loads(r)) for r in df[df.model == model]["R"]])
        Rm = np.nanmean(Rs, axis=0)
        im = ax.imshow(Rm, vmin=0.5, vmax=1.0, cmap="viridis")
        ax.set_title(model, fontsize=8)
        ax.set_xlabel("after position j"); ax.set_ylabel("task position i")
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{Rm[i,j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if Rm[i, j] < 0.8 else "black")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="test acc")
    fig.suptitle("R[i][j] = acc on task-position i after training position j (mean over seeds)", fontsize=12)
    fig.savefig(output_dir / "continual_learning_R_matrices.png", bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Split-CIFAR-10 Domain-Incremental Continual Learning",
        "",
        "Single shared 2-logit head, 5 binary pair-tasks, no replay/regularizer.",
        f"Substrate cap {args.max_neurons} neurons, {args.timesteps} timesteps, seeds {args.seeds},",
        f"sparse lr {args.lr} / dense lr {args.dense_lr}. Total wall-clock {total_seconds/60:.1f} min.",
        "",
        "## Summary (mean over seeds, sorted by least forgetting)",
        "",
        "```",
        agg.round(4).to_string(index=False),
        "```",
        "",
        "Sign conventions: BWT<0 and F>0 both mean forgetting. `w_rec_drift` must be 0 for",
        "frozen models (freeze check). `rep_drift` attributes a low-forgetting result to genuine",
        "representational stability vs a static (non-learning) representation.",
        "",
    ]
    (output_dir / "continual_learning_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--pool-assignments", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/continual_learning"))
    p.add_argument("--data-dir", type=Path, default=Path("data/torchvision"))
    p.add_argument("--max-neurons", type=int, default=5000)
    p.add_argument("--models", nargs="+", choices=list(CL_MODEL_SPECS), default=list(DEFAULT_MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device-ids", nargs="+", type=int, default=None)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dense-lr", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--state-clip", type=float, default=5.0)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--per-task-train", type=int, default=5000)
    p.add_argument("--per-task-val", type=int, default=1000)
    p.add_argument("--prune-max-hops", type=int, default=2)
    p.add_argument("--prune-max-internal-nodes", type=int, default=1024)
    p.add_argument("--data-seed", type=int, default=12345)
    p.add_argument("--init-seed", type=int, default=7000)
    p.add_argument("--_worker-device", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-stream-ids", nargs="*", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
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
        bpu.load_task("cifar10", args.data_dir)  # pre-download once
        print(f"dispatch multi-gpu device_ids={args.device_ids} streams={len(streams)}", flush=True)
        df = dispatch_multi_gpu(streams, args)
    else:
        if args.device == "cpu" or not torch.cuda.is_available():
            device = torch.device("cpu")
        elif args.device_ids:
            device = torch.device(f"cuda:{args.device_ids[0]}")
        else:
            device = torch.device("cuda")
        print(f"single-device run device={device} streams={len(streams)}", flush=True)
        df = pd.DataFrame(run_streams(streams, args, device))

    total = time.monotonic() - t0
    write_outputs(args.output_dir, df, total, args)
    (args.output_dir / "run_config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
         if not k.startswith("_worker")}, indent=2, sort_keys=True))
    print(f"complete streams={len(streams)} total_wall={total/60:.1f}min "
          f"figure={args.output_dir / 'continual_learning_split_cifar10.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
