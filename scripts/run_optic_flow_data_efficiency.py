#!/usr/bin/env python3
"""Data-efficiency sweep on the optic-lobe optic-flow task.

This runner measures how three *connectome-derived* recurrent-matrix families
learn the already-defined hex-lattice optic-flow regression task when only a
limited fraction of the training data is available (5%, 10%, 15%, ... ).

Families compared (each with a matched random control):

  * sparse  -- the flywire optic-lobe sparse recurrent matrix (sparse runtime)
  * pruned  -- the sensory->output short-path-bridge pruned sparse matrix
               (same pruning strategy used for the mushroom-body comparison)
  * dense   -- a fully-connected trainable recurrent matrix initialised from the
               connectome support (dense runtime)

The task itself (stimulus rendering, targets, metrics) is reused verbatim from
``run_optic_flow_benchmark``; only the data-budget regime and the extra
dense/pruned families are new here.

Key differences from the streaming benchmark:

  * The training set is a FIXED, pre-generated pool of episodes. A data fraction
    selects a NESTED prefix of that pool (5% subset is contained in the 10%
    subset, etc.), so the curves isolate the effect of data quantity.
  * Validation and test pools are fixed and SHARED across every family, fraction
    and seed, so all conditions are scored on identical held-out data.

Because a dense N x N recurrent matrix is required for the dense family, the
optic lobe (N = 96,816) must be capped with ``--max-neurons`` to keep the dense
matrix tractable (e.g. 3000 neurons -> 9M dense params). The cap keeps the
top-activity sub-network and is applied identically to every family.

NOTE: this script intentionally does not download/prepare the connectome. Pass a
prepared adjacency ``--matrix`` and an index-aligned ``--pool-assignments`` CSV
(the same artifacts produced by ``run_optic_flow_benchmark.py --mode prepare``).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
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

import run_optic_flow_benchmark as ofb  # noqa: E402
import run_pruned_mb_associative_comparison as prune_mod  # noqa: E402


# --- families ---------------------------------------------------------------
# family -> (structural variant, runtime). "variant" picks the matrix builder.
FAMILY_RUNTIME = {
    "sparse_connectome": ("connectome", "sparse"),
    "sparse_random": ("random", "sparse"),
    "pruned_connectome": ("connectome", "pruned"),
    "pruned_random": ("random", "pruned"),
    "dense_connectome": ("connectome", "dense"),
    "dense_random": ("random", "dense"),
}
DEFAULT_FAMILIES = tuple(FAMILY_RUNTIME)
DEFAULT_FRACTIONS = (5, 10, 15, 20, 30, 50, 75, 100)


# --- model ------------------------------------------------------------------
class DataEffRNN(nn.Module):
    """Optic-flow recurrent regressor supporting sparse and dense recurrence.

    Mirrors ``run_optic_flow_benchmark.SparseOpticFlowRNN`` but the recurrent
    matrix can be a trainable dense ``W_rec`` (dense runtime) or a sparse COO
    parameter over a fixed support (sparse runtime).
    """

    def __init__(
        self,
        recurrent: sparse.spmatrix,
        input_dim: int,
        output_dim: int,
        runtime: str,
        state_clip: float,
        seed: int,
    ) -> None:
        super().__init__()
        if runtime not in ("sparse", "dense"):
            raise ValueError(f"runtime must be 'sparse' or 'dense', got {runtime!r}")
        recurrent = recurrent.astype(np.float32).tocoo()
        recurrent.sum_duplicates()
        if recurrent.shape[0] != recurrent.shape[1]:
            raise ValueError("recurrent matrix must be square.")
        self.N = int(recurrent.shape[0])
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.runtime = runtime
        self.state_clip = float(state_clip)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        scale_in = 1.0 / math.sqrt(max(self.input_dim, 1))
        scale_out = 1.0 / math.sqrt(max(self.N, 1))
        self.W_in = nn.Parameter(
            torch.empty(self.N, self.input_dim, dtype=torch.float32).uniform_(
                -scale_in, scale_in, generator=generator
            )
        )
        self.b_rec = nn.Parameter(torch.zeros(self.N, dtype=torch.float32))
        self.readout = nn.Linear(self.N, self.output_dim)
        nn.init.uniform_(self.readout.weight, -scale_out, scale_out)
        nn.init.zeros_(self.readout.bias)

        if runtime == "dense":
            dense = recurrent.toarray().astype(np.float32)
            self.W_rec = nn.Parameter(torch.from_numpy(dense))
            self.register_buffer("edge_indices", torch.empty(2, 0, dtype=torch.long))
        else:
            indices = np.vstack([recurrent.row, recurrent.col]).astype(np.int64)
            self.register_buffer("edge_indices", torch.from_numpy(indices))
            self.W_rec_values = nn.Parameter(torch.from_numpy(recurrent.data.astype(np.float32)))

    def recurrent_parameter_count(self) -> int:
        if self.runtime == "dense":
            return int(self.W_rec.numel())
        return int(self.W_rec_values.numel())

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def _recurrent_step(self, h: torch.Tensor) -> torch.Tensor:
        if self.runtime == "dense":
            return h @ self.W_rec.t()
        W = torch.sparse_coo_tensor(
            self.edge_indices, self.W_rec_values, size=(self.N, self.N), device=h.device
        ).coalesce()
        return torch.sparse.mm(W, h.t()).t()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"inputs must have shape [batch, T, {self.input_dim}], got {tuple(inputs.shape)}"
            )
        batch, T, _ = inputs.shape
        h = inputs.new_zeros((batch, self.N))
        outputs: list[torch.Tensor] = []
        for t in range(T):
            h = self._recurrent_step(h) + inputs[:, t, :] @ self.W_in.t() + self.b_rec
            h = torch.relu(h)
            if self.state_clip > 0:
                h = torch.clamp(h, max=self.state_clip)
            outputs.append(self.readout(h))
        return torch.stack(outputs, dim=1)


# --- matrix construction ----------------------------------------------------
def truncate_with_keep(matrix: sparse.csr_matrix, max_neurons: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Keep the top-activity ``max_neurons`` nodes; return matrix and kept indices."""
    matrix = matrix.tocsr()
    n = int(matrix.shape[0])
    if max_neurons <= 0 or n <= max_neurons:
        return matrix, np.arange(n, dtype=np.int64)
    activity = np.asarray(matrix.sum(axis=0)).ravel() + np.asarray(matrix.sum(axis=1)).ravel()
    keep = np.sort(np.argsort(activity)[-int(max_neurons):]).astype(np.int64)
    return matrix[keep][:, keep].tocsr(), keep


def load_pools_aligned(path: Path, n: int) -> pd.DataFrame:
    """Load pool assignments with an ``index`` column aligned to matrix rows.

    Accepts the mushroom-body style (already has ``index``) or a connectome
    export keyed by row order. Falls back to positional indexing only when no
    explicit index column is present, and validates against ``n``.
    """
    pools = pd.read_csv(path)
    if "index" not in pools.columns:
        for cand in ("matrix_index", "pruned_index", "row", "node_index"):
            if cand in pools.columns:
                pools = pools.rename(columns={cand: "index"})
                break
    if "index" not in pools.columns:
        if len(pools) != n:
            raise ValueError(
                f"pool assignments have no 'index' column and row count {len(pools)} "
                f"!= matrix size {n}; cannot align positionally."
            )
        pools = pools.copy()
        pools["index"] = np.arange(len(pools), dtype=np.int64)
    if "pool" not in pools.columns:
        raise ValueError(f"{path} must contain a 'pool' column")
    return _coerce_pools(pools, n)


def _coerce_pools(pools: pd.DataFrame, n: int) -> pd.DataFrame:
    pools = pools.copy()
    pools["index"] = pools["index"].astype(int)
    pools["pool"] = pools["pool"].astype(str)
    pools = pools[(pools["index"] >= 0) & (pools["index"] < int(n))]
    if pools["index"].duplicated().any():
        raise ValueError("pool assignments contain duplicate indices after alignment")
    return pools.sort_values("index").reset_index(drop=True)


def remap_pools(pools: pd.DataFrame, keep: np.ndarray) -> pd.DataFrame:
    """Reindex pool assignments after a top-activity truncation."""
    newpos = {int(old): i for i, old in enumerate(keep.tolist())}
    sub = pools[pools["index"].isin(newpos)].copy()
    sub["index"] = sub["index"].map(newpos).astype(int)
    return sub.sort_values("index").reset_index(drop=True)


def build_recurrent(
    base_cap: sparse.csr_matrix,
    pools_cap: pd.DataFrame,
    variant: str,
    runtime: str,
    seed: int,
    prune_max_hops: int,
    prune_max_internal_nodes: int,
) -> tuple[sparse.coo_matrix, str]:
    """Return the recurrent matrix and the torch runtime for a family member."""
    if runtime == "pruned":
        pruned = prune_mod.prune_recurrent_matrix(
            base_cap.tocoo(),
            pools_cap,
            max_hops=prune_max_hops,
            max_internal_nodes=prune_max_internal_nodes,
        ).matrix
        if variant == "connectome":
            return pruned.tocoo(), "sparse"
        return ofb.random_sparse_normal_like(pruned.tocsr(), seed).tocoo(), "sparse"

    torch_runtime = "dense" if runtime == "dense" else "sparse"
    if variant == "connectome":
        return base_cap.tocoo(), torch_runtime
    return ofb.random_sparse_normal_like(base_cap, seed).tocoo(), torch_runtime


# --- data -------------------------------------------------------------------
def generate_pool(spec: ofb.OpticFlowSpec, n_episodes: int, seed: int, chunk: int = 256
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Pre-generate a fixed pool of optic-flow episodes."""
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    remaining = int(n_episodes)
    while remaining > 0:
        b = min(chunk, remaining)
        batch = ofb.generate_optic_flow_batch(spec, b, rng)
        xs.append(batch.inputs)
        ys.append(batch.targets)
        remaining -= b
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def compute_metrics(pred: np.ndarray, target: np.ndarray, output_dim: int) -> dict[str, float]:
    err = pred - target
    component_rmse = np.sqrt(np.mean(err**2, axis=(0, 1)))
    target_var = np.var(target.reshape(-1, output_dim), axis=0) + 1e-8
    r2 = 1.0 - np.mean(err.reshape(-1, output_dim) ** 2, axis=0) / target_var
    return {
        "loss": float(np.mean(err**2)),
        "overall_rmse": float(np.sqrt(np.mean(err**2))),
        "yaw_rmse": float(component_rmse[0]),
        "forward_rmse": float(component_rmse[1]),
        "lateral_rmse": float(component_rmse[2]),
        "translation_rmse": float(np.sqrt(np.mean(err[..., 1:3] ** 2))),
        "yaw_r2": float(r2[0]),
        "forward_r2": float(r2[1]),
        "lateral_r2": float(r2[2]),
    }


@torch.no_grad()
def evaluate_on_pool(model: DataEffRNN, inputs: np.ndarray, targets: np.ndarray,
                     device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    for start in range(0, inputs.shape[0], batch_size):
        x = torch.from_numpy(inputs[start:start + batch_size]).to(device)
        preds.append(model(x).detach().cpu().numpy())
    return compute_metrics(np.concatenate(preds, axis=0), targets, model.output_dim)


# --- training ---------------------------------------------------------------
@dataclass
class Job:
    family: str
    fraction: int
    seed: int


def train_job(
    job: Job,
    base_cap: sparse.csr_matrix,
    pools_cap: pd.DataFrame,
    spec: ofb.OpticFlowSpec,
    pools: dict[str, tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    variant, runtime = FAMILY_RUNTIME[job.family]
    torch.manual_seed(job.seed)
    np.random.seed(job.seed)

    recurrent, torch_runtime = build_recurrent(
        base_cap, pools_cap, variant, runtime, seed=args.init_seed + job.seed,
        prune_max_hops=args.prune_max_hops, prune_max_internal_nodes=args.prune_max_internal_nodes,
    )
    model = DataEffRNN(
        recurrent=recurrent, input_dim=spec.input_dim, output_dim=spec.output_dim,
        runtime=torch_runtime, state_clip=args.state_clip, seed=args.init_seed + job.seed,
    ).to(device)
    lr = args.dense_lr if torch_runtime == "dense" else args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_x, train_y = pools["train"]
    n_full = train_x.shape[0]
    n_use = max(args.batch_size, int(round(n_full * job.fraction / 100.0)))
    n_use = min(n_use, n_full)
    sub_x, sub_y = train_x[:n_use], train_y[:n_use]  # nested prefix
    val_x, val_y = pools["val"]
    test_x, test_y = pools["test"]

    print(
        f"job-start family={job.family} variant={variant} runtime={torch_runtime} "
        f"fraction={job.fraction}% n_train={n_use}/{n_full} seed={job.seed} N={model.N} "
        f"edges={recurrent.nnz} trainable_params={model.trainable_parameter_count()} lr={lr}",
        flush=True,
    )

    rng = np.random.default_rng(args.data_seed + job.seed)
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(n_use)
        losses: list[float] = []
        for start in range(0, n_use, args.batch_size):
            idx = order[start:start + args.batch_size]
            x = torch.from_numpy(sub_x[idx]).to(device)
            y = torch.from_numpy(sub_y[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x) - y) ** 2)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate_on_pool(model, val_x, val_y, device, args.batch_size)
        improved = val["loss"] < best_val - 1e-9
        if improved:
            best_val = float(val["loss"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        history.append({
            "family": job.family, "variant": variant, "runtime": torch_runtime,
            "fraction": job.fraction, "seed": job.seed, "epoch": epoch,
            "n_train": n_use, "train_loss": float(np.mean(losses)),
            "val_loss": float(val["loss"]), "val_overall_rmse": float(val["overall_rmse"]),
            "val_yaw_r2": float(val["yaw_r2"]), "best_val_loss": best_val, "patience_wait": wait,
        })
        if args.patience > 0 and wait >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate_on_pool(model, test_x, test_y, device, args.batch_size)
    metrics = {
        "family": job.family, "variant": variant, "runtime": torch_runtime,
        "fraction": job.fraction, "seed": job.seed, "n_train": n_use, "N": model.N,
        "init_nonzero_edges": int(recurrent.nnz),
        "recurrent_params": model.recurrent_parameter_count(),
        "trainable_params": model.trainable_parameter_count(),
        "best_val_loss": best_val, "test_loss": test["loss"],
        "test_overall_rmse": test["overall_rmse"], "test_yaw_rmse": test["yaw_rmse"],
        "test_forward_rmse": test["forward_rmse"], "test_lateral_rmse": test["lateral_rmse"],
        "test_translation_rmse": test["translation_rmse"], "test_yaw_r2": test["yaw_r2"],
        "test_forward_r2": test["forward_r2"], "test_lateral_r2": test["lateral_r2"],
    }
    print(
        f"job-done family={job.family} fraction={job.fraction}% seed={job.seed} "
        f"test_overall_rmse={test['overall_rmse']:.6g} test_yaw_r2={test['yaw_r2']:.4f}",
        flush=True,
    )
    return metrics, history


# --- orchestration ----------------------------------------------------------
def enumerate_jobs(families: list[str], fractions: list[int], seeds: list[int]) -> list[Job]:
    return [Job(f, frac, s) for f in families for frac in fractions for s in seeds]


def prepare_inputs(args: argparse.Namespace) -> tuple[
    sparse.csr_matrix, pd.DataFrame, ofb.OpticFlowSpec, dict[str, tuple[np.ndarray, np.ndarray]]
]:
    base = ofb.load_matrix(args.matrix)
    pools_full = load_pools_aligned(args.pool_assignments, base.shape[0])
    base_cap, keep = truncate_with_keep(base, args.max_neurons)
    pools_cap = remap_pools(pools_full, keep)
    spec = ofb.OpticFlowSpec(
        hex_rings=args.hex_rings, timesteps=args.timesteps,
        sensor_noise_std=args.sensor_noise_std,
    )
    pools = {
        "train": generate_pool(spec, args.full_train_episodes, seed=args.data_seed),
        "val": generate_pool(spec, args.val_episodes, seed=args.val_seed),
        "test": generate_pool(spec, args.test_episodes, seed=args.test_seed),
    }
    return base_cap, pools_cap, spec, pools


def run_jobs(jobs: list[Job], args: argparse.Namespace, device: torch.device,
             ) -> tuple[list[dict], list[dict]]:
    base_cap, pools_cap, spec, pools = prepare_inputs(args)
    print(
        f"prepared base_cap N={base_cap.shape[0]} edges={base_cap.nnz} "
        f"train_pool={pools['train'][0].shape[0]} val_pool={pools['val'][0].shape[0]} "
        f"test_pool={pools['test'][0].shape[0]} jobs={len(jobs)}",
        flush=True,
    )
    metrics_rows: list[dict] = []
    history_rows: list[dict] = []
    for job in jobs:
        m, h = train_job(job, base_cap, pools_cap, spec, pools, args, device)
        metrics_rows.append(m)
        history_rows.extend(h)
    return metrics_rows, history_rows


def dispatch_multi_gpu(jobs: list[Job], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split jobs across device ids by re-invoking this script as workers."""
    device_ids = args.device_ids
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partitions: dict[int, list[int]] = {d: [] for d in device_ids}
    for ji, _job in enumerate(jobs):
        partitions[device_ids[ji % len(device_ids)]].append(ji)

    procs = []
    part_files = []
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


def _worker_argv(args: argparse.Namespace, device_id: int, job_ids: list[int], out: Path) -> list[str]:
    argv = [
        "--matrix", str(args.matrix), "--pool-assignments", str(args.pool_assignments),
        "--output-dir", str(args.output_dir), "--max-neurons", str(args.max_neurons),
        "--families", *args.families, "--fractions", *[str(f) for f in args.fractions],
        "--seeds", *[str(s) for s in args.seeds],
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--batch-size", str(args.batch_size), "--lr", str(args.lr), "--dense-lr", str(args.dense_lr),
        "--grad-clip", str(args.grad_clip), "--state-clip", str(args.state_clip),
        "--full-train-episodes", str(args.full_train_episodes),
        "--val-episodes", str(args.val_episodes), "--test-episodes", str(args.test_episodes),
        "--hex-rings", str(args.hex_rings), "--timesteps", str(args.timesteps),
        "--sensor-noise-std", str(args.sensor_noise_std),
        "--prune-max-hops", str(args.prune_max_hops),
        "--prune-max-internal-nodes", str(args.prune_max_internal_nodes),
        "--data-seed", str(args.data_seed), "--init-seed", str(args.init_seed),
        "--val-seed", str(args.val_seed), "--test-seed", str(args.test_seed),
        "--_worker-device", str(device_id), "--_worker-out", str(out),
        "--_worker-job-ids", *[str(j) for j in job_ids],
    ]
    return argv


# --- outputs ----------------------------------------------------------------
def write_plots(output_dir: Path, metrics: pd.DataFrame) -> None:
    variant_color = {"connectome": "#1f77b4", "random": "#ff7f0e"}
    fam_style = {"sparse": "-", "pruned": "--", "dense": ":"}
    fig, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    for family, grp in metrics.groupby("family"):
        runtime = family.split("_")[0]  # sparse|pruned|dense
        variant = "connectome" if family.endswith("connectome") else "random"
        agg = grp.groupby("fraction").agg(
            rmse_mean=("test_overall_rmse", "mean"), rmse_std=("test_overall_rmse", "std"),
            r2_mean=("test_yaw_r2", "mean"), r2_std=("test_yaw_r2", "std"),
        ).reset_index().sort_values("fraction")
        color, style = variant_color[variant], fam_style.get(runtime, "-")
        ax_rmse.errorbar(agg["fraction"], agg["rmse_mean"], yerr=agg["rmse_std"].fillna(0.0),
                         color=color, linestyle=style, marker="o", markersize=3,
                         capsize=2, linewidth=1.6, label=family)
        ax_r2.errorbar(agg["fraction"], agg["r2_mean"], yerr=agg["r2_std"].fillna(0.0),
                       color=color, linestyle=style, marker="o", markersize=3,
                       capsize=2, linewidth=1.6, label=family)
    ax_rmse.set_title("Test overall RMSE vs training data")
    ax_rmse.set_xlabel("Training data (% of full pool)")
    ax_rmse.set_ylabel("Overall RMSE")
    ax_r2.set_title("Test yaw R² vs training data")
    ax_r2.set_xlabel("Training data (% of full pool)")
    ax_r2.set_ylabel("Yaw R²")
    for ax in (ax_rmse, ax_r2):
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Optic-flow data efficiency: sparse vs pruned vs dense connectome models", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "optic_flow_data_efficiency_curves.png")
    plt.close(fig)


def write_report(output_dir: Path, metrics: pd.DataFrame, args: argparse.Namespace) -> None:
    summary = (
        metrics.groupby(["family", "fraction"])
        .agg(test_overall_rmse_mean=("test_overall_rmse", "mean"),
             test_yaw_r2_mean=("test_yaw_r2", "mean"),
             N=("N", "first"), trainable_params=("trainable_params", "first"))
        .reset_index()
    )
    summary.to_csv(output_dir / "data_efficiency_summary.csv", index=False)
    lines = [
        "# Optic-Flow Data-Efficiency Sweep",
        "",
        "Sparse vs pruned vs dense connectome-derived recurrent models on the",
        "optic-lobe optic-flow regression task, as a function of how much of a",
        "fixed training pool is used (nested prefixes). Validation/test pools are",
        "shared across all conditions.",
        "",
        f"- Matrix cap (`--max-neurons`): {args.max_neurons}",
        f"- Full training pool: {args.full_train_episodes} episodes",
        f"- Fractions (%): {', '.join(str(f) for f in args.fractions)}",
        f"- Sparse lr: {args.lr}; dense lr: {args.dense_lr}; epochs: {args.epochs}; seeds: {args.seeds}",
        "",
        "## Summary (mean over seeds)",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
    ]
    (output_dir / "data_efficiency_report.md").write_text("\n".join(lines), encoding="utf-8")


# --- args / main ------------------------------------------------------------
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True,
                   help="Prepared optic-lobe adjacency npz (run_optic_flow_benchmark --mode prepare).")
    p.add_argument("--pool-assignments", type=Path, required=True,
                   help="Pool assignment CSV aligned to matrix indices (sensory/internal/output).")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/optic_flow_data_efficiency"))
    p.add_argument("--max-neurons", type=int, default=3000,
                   help="Top-activity cap so the dense family is tractable. 0 keeps full N (dense may OOM).")
    p.add_argument("--families", nargs="+", choices=list(FAMILY_RUNTIME), default=list(DEFAULT_FAMILIES))
    p.add_argument("--fractions", nargs="+", type=int, default=list(DEFAULT_FRACTIONS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device-ids", nargs="+", type=int, default=None,
                   help="GPU ids to spread jobs across (e.g. 0 1). Default: single device.")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dense-lr", type=float, default=1e-4, help="Tuned lr for the dense family.")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--state-clip", type=float, default=5.0)
    p.add_argument("--full-train-episodes", type=int, default=4000)
    p.add_argument("--val-episodes", type=int, default=600)
    p.add_argument("--test-episodes", type=int, default=1200)
    p.add_argument("--hex-rings", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=16)
    p.add_argument("--sensor-noise-std", type=float, default=0.07)
    p.add_argument("--prune-max-hops", type=int, default=2)
    p.add_argument("--prune-max-internal-nodes", type=int, default=1024)
    p.add_argument("--data-seed", type=int, default=12345)
    p.add_argument("--init-seed", type=int, default=7000)
    p.add_argument("--val-seed", type=int, default=22000)
    p.add_argument("--test-seed", type=int, default=33000)
    # hidden worker hooks
    p.add_argument("--_worker-device", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-job-ids", nargs="*", type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    bad = [f for f in args.fractions if not (0 < f <= 100)]
    if bad:
        p.error(f"--fractions must be in (0, 100]; got {bad}")
    return args


def resolve_device(args: argparse.Namespace, explicit_id: int | None) -> torch.device:
    if explicit_id is not None:
        return torch.device(f"cuda:{explicit_id}")
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if args.device == "cuda":
        raise RuntimeError("--device cuda requested but CUDA unavailable")
    return torch.device("cpu")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    all_jobs = enumerate_jobs(args.families, args.fractions, args.seeds)

    # Worker mode: run only the assigned job indices, dump partial results.
    if args._worker_job_ids is not None:
        device = resolve_device(args, args._worker_device)
        jobs = [all_jobs[i] for i in args._worker_job_ids]
        metrics_rows, history_rows = run_jobs(jobs, args, device)
        args._worker_out.write_text(json.dumps({"metrics": metrics_rows, "history": history_rows}))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if args.device_ids and len(args.device_ids) > 1:
        print(f"dispatch multi-gpu device_ids={args.device_ids} jobs={len(all_jobs)}", flush=True)
        metrics, history = dispatch_multi_gpu(all_jobs, args)
    else:
        device = resolve_device(args, args.device_ids[0] if args.device_ids else None)
        print(f"single-device run device={device} jobs={len(all_jobs)}", flush=True)
        m, h = run_jobs(all_jobs, args, device)
        metrics, history = pd.DataFrame(m), pd.DataFrame(h)

    metrics.to_csv(args.output_dir / "metrics_by_run.csv", index=False)
    history.to_csv(args.output_dir / "loss_history.csv", index=False)
    write_plots(args.output_dir, metrics)
    write_report(args.output_dir, metrics, args)
    (args.output_dir / "run_config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                    if not k.startswith("_worker")}, indent=2, sort_keys=True)
    )
    print(
        f"complete jobs={len(all_jobs)} metrics={args.output_dir / 'metrics_by_run.csv'} "
        f"figure={args.output_dir / 'optic_flow_data_efficiency_curves.png'} "
        f"elapsed={ofb._format_seconds(time.monotonic() - t0)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
