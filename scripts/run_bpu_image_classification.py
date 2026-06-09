#!/usr/bin/env python3
"""BPU-style image classification on the tasks from the BPU paper (MNIST, CIFAR-10).

This is the "second part" experiment: replicate the Biological Processing Unit
(BPU) recipe -- a connectome adjacency matrix used as a fixed recurrent layer,
with a trainable input projection and linear readout -- on the general-purpose
image-classification benchmarks the BPU paper used, and compare it head-to-head
against dense and size-matched alternatives.

The intent is an explicit, fair negative-result test: on standard AI tasks the
connectome-derived recurrent matrix is *not* expected to beat a trainable dense
matrix or a size-matched MLP. We report this directly, while the connectome's
value for "fly-native" tasks (optic flow, path integration, associative recall)
and for sample efficiency is established elsewhere in this repo.

Biological wiring of the task:
  * pixels are projected (trainable W_in) into the connectome's *sensory* pool
    (lamina/photoreceptor-side input neurons) only;
  * BPU dynamics run for T timesteps with the (constant) input current;
  * classification logits are read (trainable readout) from the *output* pool's
    final hidden state.

Models compared (each at its tuned LR; dense uses --dense-lr):
  * connectome_frozen     -- optic-lobe matrix, recurrent FROZEN (BPU-faithful)
  * connectome_trainable  -- optic-lobe matrix, recurrent trainable
  * dense_trainable       -- fully-connected trainable recurrent matrix
  * mlp                   -- size-matched MLP (input -> hidden -> ReLU -> output)
  * random_sparse_frozen  -- random sparse control, recurrent frozen
  * weight_shuffle_frozen -- connectome support with shuffled weights, frozen

Sample-efficiency: every model is trained at a sweep of training-data fractions
(5,10,15,...%) as nested prefixes of a fixed shuffled training set; validation
and the official test set are shared across all conditions.

A dense N x N recurrent matrix forces an --max-neurons cap on the optic lobe
(N = 96,816); the cap keeps the top-activity sub-network and is applied to every
model so N is matched across families.

Inputs: a prepared optic-lobe adjacency ``--matrix`` and an index-aligned
``--pool-assignments`` CSV (from ``run_optic_flow_benchmark.py --mode prepare``).
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

import run_mb_associative_learning as mb  # noqa: E402  (control-matrix generators)
import run_optic_flow_data_efficiency as de  # noqa: E402  (pool/truncate helpers)


# model -> (matrix variant, runtime, recurrent_trainable)
#   variant: connectome | random | weight_shuffle | dense | mlp
#   runtime: sparse | dense | mlp
MODEL_SPECS = {
    "connectome_frozen": ("connectome", "sparse", False),
    "connectome_trainable": ("connectome", "sparse", True),
    "dense_trainable": ("connectome", "dense", True),
    "mlp": ("mlp", "mlp", True),
    "random_sparse_frozen": ("random", "sparse", False),
    "weight_shuffle_frozen": ("weight_shuffle", "sparse", False),
}
DEFAULT_MODELS = tuple(MODEL_SPECS)
DEFAULT_FRACTIONS = (5, 10, 15, 20, 30, 50, 75, 100)
TASKS = {"mnist": (1 * 28 * 28, 10), "cifar10": (3 * 32 * 32, 10)}


# --- models -----------------------------------------------------------------
class BPUClassifier(nn.Module):
    """Connectome-as-recurrent image classifier with pool-restricted I/O."""

    def __init__(
        self,
        recurrent: sparse.spmatrix,
        input_dim: int,
        num_classes: int,
        sensory_idx: np.ndarray,
        output_idx: np.ndarray,
        runtime: str,
        recurrent_trainable: bool,
        timesteps: int,
        state_clip: float,
        seed: int,
    ) -> None:
        super().__init__()
        recurrent = recurrent.astype(np.float32).tocoo()
        recurrent.sum_duplicates()
        self.N = int(recurrent.shape[0])
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.runtime = runtime
        self.timesteps = int(timesteps)
        self.state_clip = float(state_clip)

        sensory_idx = np.asarray(sensory_idx, dtype=np.int64)
        output_idx = np.asarray(output_idx, dtype=np.int64)
        if sensory_idx.size == 0:
            sensory_idx = np.arange(self.N, dtype=np.int64)
        if output_idx.size == 0:
            output_idx = np.arange(self.N, dtype=np.int64)
        self.register_buffer("sensory_idx", torch.from_numpy(sensory_idx))
        self.register_buffer("output_idx", torch.from_numpy(output_idx))

        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        scale_in = 1.0 / math.sqrt(max(self.input_dim, 1))
        # input projection: pixels -> sensory neurons only
        self.W_in = nn.Parameter(
            torch.empty(sensory_idx.size, self.input_dim).uniform_(-scale_in, scale_in, generator=g)
        )
        self.b_rec = nn.Parameter(torch.zeros(self.N))
        self.readout = nn.Linear(output_idx.size, self.num_classes)
        nn.init.uniform_(self.readout.weight, -1.0 / math.sqrt(max(output_idx.size, 1)),
                         1.0 / math.sqrt(max(output_idx.size, 1)))
        nn.init.zeros_(self.readout.bias)

        if runtime == "dense":
            self.W_rec = nn.Parameter(torch.from_numpy(recurrent.toarray().astype(np.float32)),
                                      requires_grad=recurrent_trainable)
            self.register_buffer("edge_indices", torch.empty(2, 0, dtype=torch.long))
        else:
            idx = np.vstack([recurrent.row, recurrent.col]).astype(np.int64)
            self.register_buffer("edge_indices", torch.from_numpy(idx))
            self.W_rec_values = nn.Parameter(torch.from_numpy(recurrent.data.astype(np.float32)),
                                             requires_grad=recurrent_trainable)

    def recurrent_parameter_count(self) -> int:
        return int(self.W_rec.numel() if self.runtime == "dense" else self.W_rec_values.numel())

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def _recurrent_step(self, h: torch.Tensor) -> torch.Tensor:
        if self.runtime == "dense":
            return h @ self.W_rec.t()
        W = torch.sparse_coo_tensor(self.edge_indices, self.W_rec_values,
                                    size=(self.N, self.N), device=h.device).coalesce()
        return torch.sparse.mm(W, h.t()).t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        inj = x @ self.W_in.t()  # [b, n_sensory]
        current = x.new_zeros((b, self.N))
        current = current.index_add(1, self.sensory_idx, inj) + self.b_rec
        h = x.new_zeros((b, self.N))
        for _ in range(self.timesteps):
            h = self._recurrent_step(h) + current
            h = torch.relu(h)
            if self.state_clip > 0:
                h = torch.clamp(h, max=self.state_clip)
        return self.readout(h.index_select(1, self.output_idx))


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_classes: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes)
        )

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def recurrent_parameter_count(self) -> int:
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def matched_hidden(bpu_trainable: int, input_dim: int, num_classes: int) -> int:
    """Hidden width so an MLP matches the BPU's trainable-parameter count."""
    # params(H) = input_dim*H + H + H*num_classes + num_classes
    denom = input_dim + num_classes + 1
    h = max(1, int(round((bpu_trainable - num_classes) / denom)))
    return h


# --- data -------------------------------------------------------------------
def load_task(task: str, data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened, normalised (train_x, train_y, test_x, test_y) as numpy."""
    import torchvision
    import torchvision.transforms as T

    tfm = T.Compose([T.ToTensor()])
    if task == "mnist":
        tr = torchvision.datasets.MNIST(str(data_dir), train=True, download=True, transform=tfm)
        te = torchvision.datasets.MNIST(str(data_dir), train=False, download=True, transform=tfm)
    elif task == "cifar10":
        tr = torchvision.datasets.CIFAR10(str(data_dir), train=True, download=True, transform=tfm)
        te = torchvision.datasets.CIFAR10(str(data_dir), train=False, download=True, transform=tfm)
    else:
        raise ValueError(f"unknown task {task!r}")

    def to_arrays(ds):
        loader = torch.utils.data.DataLoader(ds, batch_size=4096, shuffle=False, num_workers=2)
        xs, ys = [], []
        for xb, yb in loader:
            xs.append(xb.reshape(xb.shape[0], -1).numpy())
            ys.append(yb.numpy())
        return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int64)

    train_x, train_y = to_arrays(tr)
    test_x, test_y = to_arrays(te)
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    return (train_x - mean) / std, train_y, (test_x - mean) / std, test_y


# --- training ---------------------------------------------------------------
@dataclass
class Job:
    task: str
    model: str
    fraction: int
    seed: int


def build_model(model: str, base_cap: sparse.csr_matrix, pools_cap: pd.DataFrame,
                input_dim: int, num_classes: int, args: argparse.Namespace,
                seed: int) -> tuple[nn.Module, str, int]:
    variant, runtime, rec_trainable = MODEL_SPECS[model]
    sensory = de.prune_mod.pool_indices(pools_cap, base_cap.shape[0], "sensory")
    output = de.prune_mod.pool_indices(pools_cap, base_cap.shape[0], "output")

    if runtime == "mlp":
        # match to the frozen-connectome trainable param count for a fair MLP
        ref = BPUClassifier(
            base_cap.tocoo(), input_dim, num_classes, sensory, output,
            runtime="sparse", recurrent_trainable=False, timesteps=args.timesteps,
            state_clip=args.state_clip, seed=seed,
        )
        hidden = matched_hidden(ref.trainable_parameter_count(), input_dim, num_classes)
        m = MLPBaseline(input_dim, hidden, num_classes, seed)
        return m, "mlp", hidden

    if variant == "connectome":
        rec = base_cap.tocoo()
    elif variant == "random":
        rec = mb.matrix_for_model(base_cap.tocoo(), mb.MODEL_RANDOM, seed)
    elif variant == "weight_shuffle":
        rec = mb.matrix_for_model(base_cap.tocoo(), mb.MODEL_WEIGHT_SHUFFLE, seed)
    else:
        raise ValueError(variant)

    m = BPUClassifier(
        rec, input_dim, num_classes, sensory, output, runtime=runtime,
        recurrent_trainable=rec_trainable, timesteps=args.timesteps,
        state_clip=args.state_clip, seed=seed,
    )
    return m, runtime, 0


@torch.no_grad()
def accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device,
             batch_size: int) -> tuple[float, float]:
    model.eval()
    correct = 0
    loss_sum = 0.0
    n = x.shape[0]
    ce = nn.CrossEntropyLoss(reduction="sum")
    for s in range(0, n, batch_size):
        xb = torch.from_numpy(x[s:s + batch_size]).to(device)
        yb = torch.from_numpy(y[s:s + batch_size]).to(device)
        logits = model(xb)
        loss_sum += float(ce(logits, yb).item())
        correct += int((logits.argmax(1) == yb).sum().item())
    return correct / n, loss_sum / n


def train_job(job: Job, base_cap, pools_cap, data, args, device) -> tuple[dict, list[dict]]:
    train_x, train_y, test_x, test_y = data[job.task]
    input_dim, num_classes = TASKS[job.task]
    _variant, _runtime, rec_trainable = MODEL_SPECS[job.model]
    torch.manual_seed(job.seed)
    np.random.seed(job.seed)

    model, runtime, hidden = build_model(
        job.model, base_cap, pools_cap, input_dim, num_classes, args, seed=args.init_seed + job.seed
    )
    model = model.to(device)
    lr = args.dense_lr if runtime == "dense" else args.lr
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    # fixed shuffled train order -> nested fraction prefixes; last 10% held for val
    rng = np.random.default_rng(args.data_seed)
    order = rng.permutation(train_x.shape[0])
    n_val = max(args.batch_size, int(0.1 * order.size))
    val_idx, pool_idx = order[:n_val], order[n_val:]
    n_use = max(args.batch_size, int(round(pool_idx.size * job.fraction / 100.0)))
    use_idx = pool_idx[:n_use]
    val_x, val_y = train_x[val_idx], train_y[val_idx]

    t0 = time.monotonic()
    print(f"job-start task={job.task} model={job.model} runtime={runtime} "
          f"fraction={job.fraction}% n_train={n_use} seed={job.seed} N={getattr(model,'N','-')} "
          f"hidden={hidden} trainable_params={model.trainable_parameter_count()} lr={lr}", flush=True)

    ce = nn.CrossEntropyLoss()
    epoch_rng = np.random.default_rng(args.data_seed + job.seed + 1)
    best_val = float("inf")
    best_state = None
    wait = 0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep = epoch_rng.permutation(use_idx)
        for s in range(0, ep.size, args.batch_size):
            idx = ep[s:s + args.batch_size]
            xb = torch.from_numpy(train_x[idx]).to(device)
            yb = torch.from_numpy(train_y[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = ce(model(xb), yb)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        val_acc, val_loss = accuracy(model, val_x, val_y, device, args.batch_size)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        history.append({"task": job.task, "model": job.model, "fraction": job.fraction,
                        "seed": job.seed, "epoch": epoch, "val_acc": val_acc, "val_loss": val_loss,
                        "best_val_loss": best_val, "patience_wait": wait})
        if args.patience > 0 and wait >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, test_loss = accuracy(model, test_x, test_y, device, args.batch_size)
    elapsed = time.monotonic() - t0
    metrics = {
        "task": job.task, "model": job.model, "runtime": runtime, "fraction": job.fraction,
        "seed": job.seed, "n_train": int(n_use), "hidden": int(hidden),
        "recurrent_trainable": int(bool(rec_trainable)),
        "trainable_params": model.trainable_parameter_count(),
        "recurrent_params": model.recurrent_parameter_count(),
        "test_acc": test_acc, "test_loss": test_loss, "best_val_loss": best_val,
        "wall_seconds": round(elapsed, 2),
    }
    print(f"job-done task={job.task} model={job.model} fraction={job.fraction}% seed={job.seed} "
          f"test_acc={test_acc:.4f} wall={elapsed:.1f}s", flush=True)
    return metrics, history


# --- orchestration ----------------------------------------------------------
def enumerate_jobs(tasks, models, fractions, seeds) -> list[Job]:
    return [Job(t, m, f, s) for t in tasks for m in models for f in fractions for s in seeds]


def pool_aware_truncate(matrix, pools, max_neurons, force_pools=("sensory",)):
    """Cap to max_neurons, force-keeping every neuron in ``force_pools`` (e.g. the
    photoreceptor/sensory inputs), filling the remaining budget by top activity."""
    matrix = matrix.tocsr()
    n = int(matrix.shape[0])
    if max_neurons <= 0 or n <= max_neurons:
        return matrix, np.arange(n, dtype=np.int64)
    forced = np.unique(np.concatenate(
        [de.prune_mod.pool_indices(pools, n, p) for p in force_pools] or [np.empty(0, np.int64)]
    )).astype(np.int64)
    forced = forced[:max_neurons]
    activity = np.asarray(matrix.sum(0)).ravel() + np.asarray(matrix.sum(1)).ravel()
    remaining = int(max_neurons - forced.size)
    keep = forced
    if remaining > 0:
        mask = np.ones(n, dtype=bool)
        mask[forced] = False
        rest = np.where(mask)[0]
        top = rest[np.argsort(activity[rest])[-remaining:]]
        keep = np.concatenate([forced, top])
    return matrix[np.sort(keep)][:, np.sort(keep)].tocsr(), np.sort(keep).astype(np.int64)


def prepare_inputs(args):
    base = de.ofb.load_matrix(args.matrix)
    pools_full = de.load_pools_aligned(args.pool_assignments, base.shape[0])
    base_cap, keep = pool_aware_truncate(base, pools_full, args.max_neurons)
    pools_cap = de.remap_pools(pools_full, keep)
    data = {t: load_task(t, args.data_dir) for t in args.tasks}
    return base_cap, pools_cap, data


def run_jobs(jobs, args, device) -> tuple[list[dict], list[dict]]:
    base_cap, pools_cap, data = prepare_inputs(args)
    ns = de.prune_mod.pool_indices(pools_cap, base_cap.shape[0], "sensory").size
    no = de.prune_mod.pool_indices(pools_cap, base_cap.shape[0], "output").size
    print(f"prepared base_cap N={base_cap.shape[0]} edges={base_cap.nnz} sensory={ns} output={no} "
          f"tasks={args.tasks} jobs={len(jobs)}", flush=True)
    metrics_rows, history_rows = [], []
    for job in jobs:
        m, h = train_job(job, base_cap, pools_cap, data, args, device)
        metrics_rows.append(m)
        history_rows.extend(h)
    return metrics_rows, history_rows


def dispatch_multi_gpu(jobs, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    device_ids = args.device_ids
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partitions = {d: [] for d in device_ids}
    for ji in range(len(jobs)):
        partitions[device_ids[ji % len(device_ids)]].append(ji)
    procs, part_files = [], []
    for dev, job_ids in partitions.items():
        if not job_ids:
            continue
        out = args.output_dir / f"_worker_dev{dev}.json"
        part_files.append(out)
        cmd = [sys.executable, str(Path(__file__).resolve())] + _worker_argv(args, dev, job_ids, out)
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
    metrics_rows, history_rows = [], []
    for pf in part_files:
        payload = json.loads(pf.read_text())
        metrics_rows.extend(payload["metrics"])
        history_rows.extend(payload["history"])
    return pd.DataFrame(metrics_rows), pd.DataFrame(history_rows)


def _worker_argv(args, device_id, job_ids, out) -> list[str]:
    return [
        "--matrix", str(args.matrix), "--pool-assignments", str(args.pool_assignments),
        "--output-dir", str(args.output_dir), "--data-dir", str(args.data_dir),
        "--max-neurons", str(args.max_neurons), "--tasks", *args.tasks,
        "--models", *args.models, "--fractions", *[str(f) for f in args.fractions],
        "--seeds", *[str(s) for s in args.seeds], "--epochs", str(args.epochs),
        "--patience", str(args.patience), "--batch-size", str(args.batch_size),
        "--lr", str(args.lr), "--dense-lr", str(args.dense_lr), "--grad-clip", str(args.grad_clip),
        "--state-clip", str(args.state_clip), "--timesteps", str(args.timesteps),
        "--data-seed", str(args.data_seed), "--init-seed", str(args.init_seed),
        "--_worker-device", str(device_id), "--_worker-out", str(out),
        "--_worker-job-ids", *[str(j) for j in job_ids],
    ]


# --- outputs ----------------------------------------------------------------
MODEL_COLORS = {
    "connectome_frozen": "#1f77b4", "connectome_trainable": "#17becf",
    "dense_trainable": "#d62728", "mlp": "#7f7f7f",
    "random_sparse_frozen": "#ff7f0e", "weight_shuffle_frozen": "#2ca02c",
}


def write_plots(output_dir: Path, metrics: pd.DataFrame) -> None:
    tasks = sorted(metrics["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4.6), dpi=150, squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = metrics[metrics["task"] == task]
        for model, grp in sub.groupby("model"):
            agg = grp.groupby("fraction").agg(
                acc=("test_acc", "mean"), std=("test_acc", "std")).reset_index().sort_values("fraction")
            ax.errorbar(agg["fraction"], agg["acc"] * 100, yerr=(agg["std"].fillna(0.0) * 100),
                        marker="o", markersize=3, capsize=2, linewidth=1.6,
                        color=MODEL_COLORS.get(model), label=model)
        ax.set_title(f"{task.upper()} test accuracy vs training data")
        ax.set_xlabel("Training data (% of pool)")
        ax.set_ylabel("Test accuracy (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("BPU image classification: connectome vs dense vs MLP", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "bpu_image_classification_data_efficiency.png")
    plt.close(fig)


def write_report(output_dir: Path, metrics: pd.DataFrame, args, total_seconds: float) -> None:
    full = metrics[metrics["fraction"] == metrics["fraction"].max()]
    table = (full.groupby(["task", "model"]).agg(
        test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
        trainable_params=("trainable_params", "first"), N=("recurrent_params", "first"),
    ).reset_index())
    table.to_csv(output_dir / "full_data_accuracy.csv", index=False)
    metrics.groupby(["task", "model", "fraction"]).agg(
        test_acc_mean=("test_acc", "mean")).reset_index().to_csv(
        output_dir / "data_efficiency_summary.csv", index=False)
    lines = [
        "# BPU Image Classification (MNIST / CIFAR-10)",
        "",
        "Connectome-as-fixed-recurrent (BPU-faithful) vs trainable-recurrent vs dense",
        "vs size-matched MLP vs random/weight-shuffle controls, on the BPU paper's",
        "image-classification tasks, swept over training-data fraction.",
        "",
        f"- Optic-lobe matrix cap (`--max-neurons`): {args.max_neurons}",
        f"- Timesteps: {args.timesteps}; epochs: {args.epochs}; seeds: {args.seeds}",
        f"- Sparse lr: {args.lr}; dense lr: {args.dense_lr}",
        f"- Total wall-clock: {total_seconds/60.0:.1f} min",
        "",
        "## Full-data test accuracy (mean over seeds)",
        "",
        "```",
        table.to_string(index=False),
        "```",
        "",
        "Interpretation: if dense/MLP match or beat the frozen connectome on these",
        "general tasks, that is the intended explicit negative result vs the BPU paper.",
        "",
    ]
    (output_dir / "bpu_classification_report.md").write_text("\n".join(lines), encoding="utf-8")


# --- args / main ------------------------------------------------------------
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--pool-assignments", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/bpu_image_classification"))
    p.add_argument("--data-dir", type=Path, default=Path("data/torchvision"))
    p.add_argument("--max-neurons", type=int, default=3000,
                   help="Top-activity cap so the dense family is tractable and N is matched.")
    p.add_argument("--tasks", nargs="+", choices=list(TASKS), default=["mnist", "cifar10"])
    p.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=list(DEFAULT_MODELS))
    p.add_argument("--fractions", nargs="+", type=int, default=list(DEFAULT_FRACTIONS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--device-ids", nargs="+", type=int, default=None)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dense-lr", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--state-clip", type=float, default=5.0)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--data-seed", type=int, default=12345)
    p.add_argument("--init-seed", type=int, default=7000)
    p.add_argument("--_worker-device", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-job-ids", nargs="*", type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    bad = [f for f in args.fractions if not (0 < f <= 100)]
    if bad:
        p.error(f"--fractions must be in (0, 100]; got {bad}")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    all_jobs = enumerate_jobs(args.tasks, args.models, args.fractions, args.seeds)

    if args._worker_job_ids is not None:
        device = de.torch.device(f"cuda:{args._worker_device}") if args._worker_device is not None \
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        jobs = [all_jobs[i] for i in args._worker_job_ids]
        m, h = run_jobs(jobs, args, device)
        args._worker_out.write_text(json.dumps({"metrics": m, "history": h}))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if args.device_ids and len(args.device_ids) > 1:
        print(f"dispatch multi-gpu device_ids={args.device_ids} jobs={len(all_jobs)}", flush=True)
        # Pre-download datasets once so parallel workers don't race on the cache.
        for _t in args.tasks:
            load_task(_t, args.data_dir)
        metrics, history = dispatch_multi_gpu(all_jobs, args)
    else:
        if args.device == "cpu" or not torch.cuda.is_available():
            device = torch.device("cpu")
        elif args.device_ids:
            device = torch.device(f"cuda:{args.device_ids[0]}")
        else:
            device = torch.device("cuda")
        print(f"single-device run device={device} jobs={len(all_jobs)}", flush=True)
        m, h = run_jobs(all_jobs, args, device)
        metrics, history = pd.DataFrame(m), pd.DataFrame(h)

    total = time.monotonic() - t0
    metrics.to_csv(args.output_dir / "metrics_by_run.csv", index=False)
    history.to_csv(args.output_dir / "loss_history.csv", index=False)
    write_plots(args.output_dir, metrics)
    write_report(args.output_dir, metrics, args, total)
    (args.output_dir / "run_config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                    if not k.startswith("_worker")}, indent=2, sort_keys=True))
    print(f"complete jobs={len(all_jobs)} total_wall={total/60.0:.1f}min "
          f"figure={args.output_dir / 'bpu_image_classification_data_efficiency.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
