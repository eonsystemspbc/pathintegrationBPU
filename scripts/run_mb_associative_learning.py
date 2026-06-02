#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_manifest import write_artifact_manifest  # noqa: E402


MODEL_HEMIBRAIN = "hemibrain_seeded"
MODEL_HEMIBRAIN_DENSE = "hemibrain_dense"
MODEL_RANDOM = "random_sparse"
MODEL_RANDOM_DENSE = "random_dense"
MODEL_WEIGHT_SHUFFLE = "weight_shuffle"
MODEL_CHOICES = (
    MODEL_HEMIBRAIN,
    MODEL_HEMIBRAIN_DENSE,
    MODEL_RANDOM,
    MODEL_WEIGHT_SHUFFLE,
    MODEL_RANDOM_DENSE,
)
DEFAULT_MODELS = (MODEL_HEMIBRAIN, MODEL_RANDOM, MODEL_WEIGHT_SHUFFLE)
RUNTIME_CHOICES = ("sparse", "dense")


@dataclass(frozen=True)
class EpisodeSpec:
    num_odors: int
    odor_dim: int
    odors_per_episode: int
    reversal_count: int
    reversal_repeats: int
    odor_sparsity: float
    odor_noise_std: float

    @property
    def input_dim(self) -> int:
        return self.odor_dim + 3

    @property
    def timesteps(self) -> int:
        return (
            self.odors_per_episode
            + self.odors_per_episode
            + self.reversal_count * self.reversal_repeats
            + self.odors_per_episode
        )


@dataclass(frozen=True)
class Batch:
    inputs: np.ndarray
    targets: np.ndarray
    query_mask: np.ndarray
    initial_query_mask: np.ndarray
    final_query_mask: np.ndarray


class AssociativeRNN(nn.Module):
    def __init__(
        self,
        recurrent: sparse.spmatrix,
        input_dim: int,
        runtime: str,
        state_clip: float,
        seed: int,
    ) -> None:
        super().__init__()
        if runtime not in RUNTIME_CHOICES:
            raise ValueError(f"runtime must be one of {RUNTIME_CHOICES}")
        recurrent = recurrent.astype(np.float32).tocoo()
        recurrent.sum_duplicates()
        if recurrent.shape[0] != recurrent.shape[1]:
            raise ValueError("recurrent matrix must be square.")
        self.N = int(recurrent.shape[0])
        self.input_dim = int(input_dim)
        self.runtime = runtime
        self.state_clip = float(state_clip)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        scale_in = 1.0 / math.sqrt(max(input_dim, 1))
        scale_out = 1.0 / math.sqrt(max(self.N, 1))
        self.W_in = nn.Parameter(
            torch.empty(self.N, input_dim, dtype=torch.float32).uniform_(
                -scale_in, scale_in, generator=generator
            )
        )
        self.b_rec = nn.Parameter(torch.zeros(self.N, dtype=torch.float32))
        self.readout = nn.Linear(self.N, 1)
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
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))

    def _recurrent_step(self, h: torch.Tensor) -> torch.Tensor:
        if self.runtime == "dense":
            next_h = h @ self.W_rec.t()
        else:
            W = torch.sparse_coo_tensor(
                self.edge_indices,
                self.W_rec_values,
                size=(self.N, self.N),
                device=h.device,
            ).coalesce()
            next_h = torch.sparse.mm(W, h.t()).t()
        return next_h

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"inputs must have shape [batch, T, {self.input_dim}], got {tuple(inputs.shape)}"
            )
        batch, T, _ = inputs.shape
        h = inputs.new_zeros((batch, self.N))
        outputs: list[torch.Tensor] = []
        for t in range(T):
            next_h = self._recurrent_step(h)
            next_h = next_h + inputs[:, t, :] @ self.W_in.t() + self.b_rec
            h = torch.relu(next_h)
            if self.state_clip > 0:
                h = torch.clamp(h, max=self.state_clip)
            outputs.append(self.readout(h).squeeze(-1))
        return torch.stack(outputs, dim=1)


def _format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, rem = divmod(minutes, 60)
    return f"{hours}h{rem:02d}m"


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_odor_bank(spec: EpisodeSpec, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bank = rng.normal(0.0, 1.0, size=(spec.num_odors, spec.odor_dim)).astype(np.float32)
    mask = rng.random(bank.shape) < spec.odor_sparsity
    bank *= mask.astype(np.float32)
    norms = np.linalg.norm(bank, axis=1, keepdims=True)
    empty = norms.squeeze(-1) == 0
    if np.any(empty):
        cols = rng.integers(0, spec.odor_dim, size=int(empty.sum()))
        bank[empty, cols] = 1.0
        norms = np.linalg.norm(bank, axis=1, keepdims=True)
    return (bank / np.maximum(norms, 1e-6)).astype(np.float32)


def generate_batch(
    odor_bank: np.ndarray,
    spec: EpisodeSpec,
    batch_size: int,
    rng: np.random.Generator,
) -> Batch:
    T = spec.timesteps
    inputs = np.zeros((batch_size, T, spec.input_dim), dtype=np.float32)
    targets = np.zeros((batch_size, T), dtype=np.float32)
    query_mask = np.zeros((batch_size, T), dtype=np.float32)
    initial_query_mask = np.zeros((batch_size, T), dtype=np.float32)
    final_query_mask = np.zeros((batch_size, T), dtype=np.float32)

    reward_col = spec.odor_dim
    punishment_col = spec.odor_dim + 1
    query_col = spec.odor_dim + 2

    for b in range(batch_size):
        odor_ids = rng.choice(spec.num_odors, size=spec.odors_per_episode, replace=False)
        initial_valence = rng.integers(0, 2, size=spec.odors_per_episode, dtype=np.int64)
        reversal_local = rng.choice(
            spec.odors_per_episode, size=spec.reversal_count, replace=False
        )
        final_valence = initial_valence.copy()
        final_valence[reversal_local] = 1 - final_valence[reversal_local]

        step = 0
        for local_idx in rng.permutation(spec.odors_per_episode):
            _write_stimulus(
                inputs[b, step],
                odor_bank[odor_ids[local_idx]],
                spec.odor_noise_std,
                rng,
            )
            if initial_valence[local_idx] == 1:
                inputs[b, step, punishment_col] = 1.0
            else:
                inputs[b, step, reward_col] = 1.0
            step += 1

        for local_idx in rng.permutation(spec.odors_per_episode):
            _write_stimulus(
                inputs[b, step],
                odor_bank[odor_ids[local_idx]],
                spec.odor_noise_std,
                rng,
            )
            inputs[b, step, query_col] = 1.0
            targets[b, step] = float(initial_valence[local_idx])
            query_mask[b, step] = 1.0
            initial_query_mask[b, step] = 1.0
            step += 1

        for _ in range(spec.reversal_repeats):
            for local_idx in rng.permutation(reversal_local):
                _write_stimulus(
                    inputs[b, step],
                    odor_bank[odor_ids[local_idx]],
                    spec.odor_noise_std,
                    rng,
                )
                if final_valence[local_idx] == 1:
                    inputs[b, step, punishment_col] = 1.0
                else:
                    inputs[b, step, reward_col] = 1.0
                step += 1

        for local_idx in rng.permutation(spec.odors_per_episode):
            _write_stimulus(
                inputs[b, step],
                odor_bank[odor_ids[local_idx]],
                spec.odor_noise_std,
                rng,
            )
            inputs[b, step, query_col] = 1.0
            targets[b, step] = float(final_valence[local_idx])
            query_mask[b, step] = 1.0
            final_query_mask[b, step] = 1.0
            step += 1

        if step != T:
            raise AssertionError(f"internal timestep mismatch: {step} != {T}")

    return Batch(inputs, targets, query_mask, initial_query_mask, final_query_mask)


def _write_stimulus(
    dest: np.ndarray,
    odor: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> None:
    dest[: odor.shape[0]] = odor
    if noise_std > 0:
        dest[: odor.shape[0]] += rng.normal(0.0, noise_std, size=odor.shape).astype(
            np.float32
        )


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(
    model: AssociativeRNN,
    odor_bank: np.ndarray,
    spec: EpisodeSpec,
    batch_size: int,
    batches: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    loss_sum = 0.0
    query_correct = 0.0
    query_total = 0.0
    initial_correct = 0.0
    initial_total = 0.0
    final_correct = 0.0
    final_total = 0.0
    for _ in range(batches):
        batch = generate_batch(odor_bank, spec, batch_size, rng)
        x, y, mask, initial_mask, final_mask = batch_to_torch(batch, device)
        logits = model(x)
        loss = masked_bce_loss(logits, y, mask)
        pred = (torch.sigmoid(logits) >= 0.5).float()
        correct = (pred == y).float()
        loss_sum += float(loss.item())
        query_correct += float((correct * mask).sum().item())
        query_total += float(mask.sum().item())
        initial_correct += float((correct * initial_mask).sum().item())
        initial_total += float(initial_mask.sum().item())
        final_correct += float((correct * final_mask).sum().item())
        final_total += float(final_mask.sum().item())
    return {
        "loss": loss_sum / max(batches, 1),
        "query_accuracy": query_correct / max(query_total, 1.0),
        "initial_probe_accuracy": initial_correct / max(initial_total, 1.0),
        "reversal_probe_accuracy": final_correct / max(final_total, 1.0),
    }


def batch_to_torch(
    batch: Batch, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(batch.inputs).to(device),
        torch.from_numpy(batch.targets).to(device),
        torch.from_numpy(batch.query_mask).to(device),
        torch.from_numpy(batch.initial_query_mask).to(device),
        torch.from_numpy(batch.final_query_mask).to(device),
    )


def load_base_matrix(matrix_path: Path, max_neurons: int) -> sparse.coo_matrix:
    matrix = sparse.load_npz(matrix_path).astype(np.float32).tocoo()
    matrix.sum_duplicates()
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got {matrix.shape}")
    if max_neurons > 0 and max_neurons < matrix.shape[0]:
        matrix = matrix.tocsr()[:max_neurons, :max_neurons].tocoo()
        matrix.sum_duplicates()
    if matrix.nnz == 0:
        raise ValueError("recurrent matrix has no nonzero entries after filtering.")
    return matrix


def matrix_for_model(base: sparse.coo_matrix, model_name: str, seed: int) -> sparse.coo_matrix:
    if model_name in {MODEL_HEMIBRAIN, MODEL_HEMIBRAIN_DENSE}:
        return base.copy().astype(np.float32).tocoo()
    if model_name == MODEL_WEIGHT_SHUFFLE:
        rng = np.random.default_rng(seed)
        shuffled = base.copy().astype(np.float32).tocoo()
        shuffled.data = rng.permutation(shuffled.data).astype(np.float32)
        return shuffled
    if model_name in {MODEL_RANDOM, MODEL_RANDOM_DENSE}:
        return random_sparse_like(base, seed)
    raise ValueError(f"unknown model name: {model_name}")


def runtime_for_model(model_name: str, requested_runtime: str) -> str:
    if requested_runtime not in RUNTIME_CHOICES:
        raise ValueError(f"requested_runtime must be one of {RUNTIME_CHOICES}")
    if model_name in {MODEL_HEMIBRAIN_DENSE, MODEL_RANDOM_DENSE}:
        return "dense"
    return requested_runtime


def random_sparse_like(base: sparse.coo_matrix, seed: int) -> sparse.coo_matrix:
    base = base.astype(np.float32).tocoo()
    base.sum_duplicates()
    rng = np.random.default_rng(seed)
    N = int(base.shape[0])
    self_loop_count = int(np.sum(base.row == base.col))
    off_count = int(base.nnz - self_loop_count)
    if self_loop_count > N:
        raise ValueError("cannot place more self-loops than neurons without duplicates.")

    self_nodes = (
        rng.choice(N, size=self_loop_count, replace=False).astype(np.int64)
        if self_loop_count
        else np.empty(0, dtype=np.int64)
    )
    off_linear = sample_unique_integers(N * (N - 1), off_count, rng)
    off_rows = off_linear // (N - 1)
    off_cols = off_linear % (N - 1)
    off_cols = off_cols + (off_cols >= off_rows)

    rows = np.concatenate([self_nodes, off_rows.astype(np.int64)])
    cols = np.concatenate([self_nodes, off_cols.astype(np.int64)])
    data = rng.permutation(base.data).astype(np.float32)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=base.shape, dtype=np.float32)
    matrix.sum_duplicates()
    if matrix.nnz != base.nnz:
        raise AssertionError("random support generation produced duplicate edges.")
    return matrix


def sample_unique_integers(
    population_size: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count > population_size:
        raise ValueError("cannot sample more unique integers than population size.")
    values = np.empty(0, dtype=np.int64)
    while values.size < count:
        needed = count - values.size
        draw = rng.integers(
            0,
            population_size,
            size=max(int(needed * 1.3), needed + 1024),
            dtype=np.int64,
        )
        values = np.unique(np.concatenate([values, draw]))
    return rng.choice(values, size=count, replace=False).astype(np.int64)


def train_model(
    model_name: str,
    seed: int,
    base_matrix: sparse.coo_matrix,
    odor_bank: np.ndarray,
    spec: EpisodeSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    init_matrix = matrix_for_model(base_matrix, model_name, seed=args.init_seed + seed)
    model_runtime = runtime_for_model(model_name, args.recurrent_runtime)
    model = AssociativeRNN(
        recurrent=init_matrix,
        input_dim=spec.input_dim,
        runtime=model_runtime,
        state_clip=args.state_clip,
        seed=args.init_seed + seed,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_rng = np.random.default_rng(args.data_seed + seed)

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_wait = 0
    history: list[dict[str, float | int | str]] = []
    print(
        "model-start "
        f"model={model_name} seed={seed} runtime={model_runtime} "
        f"N={model.N} recurrent_params={model.recurrent_parameter_count()} "
        f"trainable_params={model.trainable_parameter_count()} "
        f"init_nonzero_edges={init_matrix.nnz} "
        f"timesteps={spec.timesteps}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_started = time.time()
        last_log = epoch_started
        train_loss_sum = 0.0
        for batch_idx in range(1, args.train_batches + 1):
            batch = generate_batch(odor_bank, spec, args.batch_size, train_rng)
            x, y, mask, _, _ = batch_to_torch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = masked_bce_loss(logits, y, mask)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss_sum += float(loss.item())

            now = time.time()
            if args.log_every_seconds > 0 and now - last_log >= args.log_every_seconds:
                running = train_loss_sum / batch_idx
                print(
                    "progress "
                    f"model={model_name} seed={seed} epoch={epoch}/{args.epochs} "
                    f"batch={batch_idx}/{args.train_batches} "
                    f"batch_loss={loss.item():.6g} running_train_loss={running:.6g} "
                    f"elapsed={_format_seconds(now - epoch_started)}",
                    flush=True,
                )
                last_log = now

        train_loss = train_loss_sum / max(args.train_batches, 1)
        val_metrics = evaluate(
            model,
            odor_bank,
            spec,
            args.batch_size,
            args.val_batches,
            device,
            seed=args.val_seed + seed,
        )
        improved = val_metrics["loss"] < best_val_loss - 1e-7
        if improved:
            best_val_loss = float(val_metrics["loss"])
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            patience_wait = 0
        else:
            patience_wait += 1

        row = {
            "model": model_name,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_query_accuracy": val_metrics["query_accuracy"],
            "val_initial_probe_accuracy": val_metrics["initial_probe_accuracy"],
            "val_reversal_probe_accuracy": val_metrics["reversal_probe_accuracy"],
            "best_val_loss": best_val_loss,
            "patience_wait": patience_wait,
        }
        history.append(row)
        print(
            "loss "
            f"model={model_name} seed={seed} epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.6g} val_loss={val_metrics['loss']:.6g} "
            f"val_acc={val_metrics['query_accuracy']:.4f} "
            f"val_reversal_acc={val_metrics['reversal_probe_accuracy']:.4f} "
            f"best_val_loss={best_val_loss:.6g} patience_wait={patience_wait}",
            flush=True,
        )
        if patience_wait >= args.patience:
            print(
                f"early-stop model={model_name} seed={seed} epoch={epoch} "
                f"best_val_loss={best_val_loss:.6g}",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(
        model,
        odor_bank,
        spec,
        args.batch_size,
        args.test_batches,
        device,
        seed=args.test_seed + seed,
    )
    metrics: dict[str, float | int | str] = {
        "model": model_name,
        "seed": seed,
        "runtime": model_runtime,
        "N": model.N,
        "init_nonzero_edges": int(init_matrix.nnz),
        "recurrent_params": model.recurrent_parameter_count(),
        "trainable_params": model.trainable_parameter_count(),
        "timesteps": spec.timesteps,
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_query_accuracy": test_metrics["query_accuracy"],
        "test_initial_probe_accuracy": test_metrics["initial_probe_accuracy"],
        "test_reversal_probe_accuracy": test_metrics["reversal_probe_accuracy"],
    }
    print(
        "model-done "
        f"model={model_name} seed={seed} best_val_loss={best_val_loss:.6g} "
        f"test_acc={test_metrics['query_accuracy']:.4f} "
        f"test_reversal_acc={test_metrics['reversal_probe_accuracy']:.4f}",
        flush=True,
    )
    return metrics, history


def write_outputs(
    output_dir: Path,
    metrics_rows: list[dict[str, float | int | str]],
    history_rows: list[dict[str, float | int | str]],
    args: argparse.Namespace,
    spec: EpisodeSpec,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metrics_rows)
    history = pd.DataFrame(history_rows)
    metrics.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    history.to_csv(output_dir / "loss_history.csv", index=False)
    summary = (
        metrics.groupby("model")
        .agg(
            best_val_loss_mean=("best_val_loss", "mean"),
            best_val_loss_std=("best_val_loss", "std"),
            test_query_accuracy_mean=("test_query_accuracy", "mean"),
            test_query_accuracy_std=("test_query_accuracy", "std"),
            test_initial_probe_accuracy_mean=("test_initial_probe_accuracy", "mean"),
            test_reversal_probe_accuracy_mean=("test_reversal_probe_accuracy", "mean"),
            test_loss_mean=("test_loss", "mean"),
            runtime=("runtime", "first"),
            init_nonzero_edges=("init_nonzero_edges", "first"),
            trainable_params=("trainable_params", "first"),
            recurrent_params=("recurrent_params", "first"),
            N=("N", "first"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": serializable_args(args),
                "episode_spec": spec.__dict__,
                "task": "odor_valence_associative_reversal",
                "task_rationale": (
                    "Odor prototypes are paired with reward or punishment inside each "
                    "episode, then queried without reinforcement and partly reversed. "
                    "This tests rapid associative olfactory memory and updating."
                ),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    plot_metrics(output_dir, metrics, history)
    write_report(output_dir, metrics, summary, args, spec)
    write_artifact_manifest(
        output_dir,
        config={"args": serializable_args(args), "episode_spec": spec.__dict__},
        extra={"stage": "associative_learning"},
    )


def serializable_args(args: argparse.Namespace) -> dict:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def plot_metrics(output_dir: Path, metrics: pd.DataFrame, history: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for model, group in history.groupby("model"):
        curve = group.groupby("epoch", as_index=False).agg(
            train_loss=("train_loss", "mean"),
            val_loss=("val_loss", "mean"),
        )
        ax.plot(curve["epoch"], curve["train_loss"], alpha=0.6, linestyle="--", label=f"{model} train")
        ax.plot(curve["epoch"], curve["val_loss"], label=f"{model} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked BCE loss")
    ax.set_title("Associative learning loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "associative_loss.png")
    plt.close(fig)

    acc_cols = [
        ("test_initial_probe_accuracy", "Initial recall"),
        ("test_reversal_probe_accuracy", "After reversal"),
        ("test_query_accuracy", "All probes"),
    ]
    summary = metrics.groupby("model", as_index=False).mean(numeric_only=True)
    x = np.arange(len(summary))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for idx, (col, label) in enumerate(acc_cols):
        ax.bar(x + (idx - 1) * width, summary[col], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Odor-valence associative memory")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "associative_accuracy.png")
    plt.close(fig)


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    args: argparse.Namespace,
    spec: EpisodeSpec,
) -> None:
    lines = [
        "# Mushroom Body Associative Learning Benchmark",
        "",
        "Task: odor prototypes are paired with reward or punishment, queried from odor alone, then a subset of associations reverses and is queried again.",
        "",
        "Real-world analogue: adaptive chemical-sensor hazard learning, where a system must rapidly bind sparse odor signatures to safety labels and update those labels when feedback changes.",
        "",
        f"Requested recurrent runtime: `{args.recurrent_runtime}`",
        "Per-model runtime is recorded in the metrics table. `hemibrain_dense` and `random_dense` always use dense recurrence.",
        f"Episode timesteps: `{spec.timesteps}`",
        f"Odors per episode: `{spec.odors_per_episode}`",
        f"Reversed odors per episode: `{spec.reversal_count}`",
        "",
        "## Summary",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "## Per-Seed Metrics",
        "",
        "```",
        metrics.to_string(index=False),
        "```",
        "",
        "Interpretation note: a positive result should be framed as evidence that the hemibrain initialization/support helps this matched associative-memory benchmark, not as a broad claim that the connectome is universally better.",
        "",
    ]
    (output_dir / "associative_learning_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mushroom-body-style odor-valence associative learning benchmark "
            "with hemibrain-seeded and matched-control recurrent models."
        )
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz"),
        help="Prepared hemibrain mushroom-body adjacency npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mb_associative_learning"),
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(DEFAULT_MODELS))
    parser.add_argument("--recurrent-runtime", choices=RUNTIME_CHOICES, default="sparse")
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=0,
        help="Use the leading N-neuron submatrix for smoke tests. 0 keeps the full matrix.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-batches", type=int, default=200)
    parser.add_argument("--val-batches", type=int, default=40)
    parser.add_argument("--test-batches", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--state-clip", type=float, default=5.0)
    parser.add_argument("--log-every-seconds", type=float, default=30.0)
    parser.add_argument("--num-odors", type=int, default=64)
    parser.add_argument("--odor-dim", type=int, default=64)
    parser.add_argument("--odors-per-episode", type=int, default=6)
    parser.add_argument("--reversal-count", type=int, default=3)
    parser.add_argument("--reversal-repeats", type=int, default=1)
    parser.add_argument("--odor-sparsity", type=float, default=0.20)
    parser.add_argument("--odor-noise-std", type=float, default=0.03)
    parser.add_argument("--data-seed", type=int, default=12345)
    parser.add_argument("--init-seed", type=int, default=7000)
    parser.add_argument("--val-seed", type=int, default=22000)
    parser.add_argument("--test-seed", type=int, default=33000)
    args = parser.parse_args(argv)
    if args.odors_per_episode > args.num_odors:
        parser.error("--odors-per-episode cannot exceed --num-odors")
    if args.reversal_count > args.odors_per_episode:
        parser.error("--reversal-count cannot exceed --odors-per-episode")
    if not (0.0 < args.odor_sparsity <= 1.0):
        parser.error("--odor-sparsity must be in (0, 1]")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    print(
        "run-start "
        f"task=odor_valence_associative_reversal output_dir={args.output_dir} "
        f"matrix={args.matrix} device={device}",
        flush=True,
    )
    spec = EpisodeSpec(
        num_odors=args.num_odors,
        odor_dim=args.odor_dim,
        odors_per_episode=args.odors_per_episode,
        reversal_count=args.reversal_count,
        reversal_repeats=args.reversal_repeats,
        odor_sparsity=args.odor_sparsity,
        odor_noise_std=args.odor_noise_std,
    )
    base_matrix = load_base_matrix(args.matrix, args.max_neurons)
    odor_bank = make_odor_bank(spec, seed=args.data_seed)
    print(
        "data-ready "
        f"N={base_matrix.shape[0]} edges={base_matrix.nnz} input_dim={spec.input_dim} "
        f"timesteps={spec.timesteps} models={','.join(args.models)}",
        flush=True,
    )
    metrics_rows: list[dict[str, float | int | str]] = []
    history_rows: list[dict[str, float | int | str]] = []
    for model_name in args.models:
        for seed in args.seeds:
            metrics, history = train_model(
                model_name, seed, base_matrix, odor_bank, spec, args, device
            )
            metrics_rows.append(metrics)
            history_rows.extend(history)
    write_outputs(args.output_dir, metrics_rows, history_rows, args, spec)
    print(
        f"complete metrics={args.output_dir / 'metrics_by_seed.csv'} "
        f"figures={args.output_dir / 'associative_accuracy.png'},{args.output_dir / 'associative_loss.png'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
