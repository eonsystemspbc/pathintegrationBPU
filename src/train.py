from __future__ import annotations

import copy
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import OUTPUT_DIM, RHO_TARGET, OutputPaths, TaskSpec, TrainConfig, resolve_device
from .connectome import (
    PreparedGraph,
    degree_preserving_shuffle_matrix,
    load_prepared_graph,
    pool_indices,
    random_control_matrix,
    spectral_radius,
    weight_shuffled_control_matrix,
)
from .models import CXBPU, GRUBaseline, assert_bpu_trainable_surface, count_trainable_parameters
from .plots import write_plots
from .task import ensure_splits, load_split


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: Path) -> None:
        data = load_split(path)
        self.inputs = torch.as_tensor(data["inputs"], dtype=torch.float32)
        self.targets = torch.as_tensor(data["targets"], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def _loader(
    path: Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        SequenceDataset(path),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


def _to_device(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = batch
    return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def _scale_control(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    rho = spectral_radius(matrix)
    if rho <= 0:
        return matrix.astype(np.float32).tocsr()
    return (matrix * (RHO_TARGET / rho)).astype(np.float32).tocsr()


def _control_matrix(primary: sparse.csr_matrix, name: str, seed: int) -> sparse.csr_matrix:
    if name in {"cx_bpu", "no_recurrence"}:
        return primary
    if name == "random":
        return _scale_control(random_control_matrix(primary, seed=10_000 + seed))
    if name == "degree_shuffle":
        return _scale_control(degree_preserving_shuffle_matrix(primary, seed=20_000 + seed))
    if name == "weight_shuffle":
        return _scale_control(weight_shuffled_control_matrix(primary, seed=30_000 + seed))
    raise ValueError(f"Unknown control: {name}")


def _make_model(
    graph: PreparedGraph,
    model_name: str,
    seed: int,
    device: torch.device,
    include_gru_hidden: int | None = None,
) -> nn.Module:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if model_name == "gru":
        hidden = include_gru_hidden or min(256, int(graph.metadata["N"]))
        return GRUBaseline(hidden_size=hidden).to(device)
    indices = pool_indices(graph.pools)
    matrix = _control_matrix(graph.matrix.astype(np.float32).tocsr(), model_name, seed)
    K = int(graph.metadata["estimated_K"])
    model = CXBPU(
        matrix,
        sensory_indices=indices["sensory"],
        output_indices=indices["output"],
        K=K,
        reset_each_timestep=(model_name == "no_recurrence"),
    ).to(device)
    assert_bpu_trainable_surface(model)
    return model


def _loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-1] != OUTPUT_DIM:
        raise ValueError("model output dimension mismatch")
    return torch.mean((pred - target) ** 2)


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    epochs_without_improvement = 0
    history: dict[str, float] = {"epochs_ran": 0, "best_val_loss": best_val}
    for epoch in range(config.epochs):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            inputs, targets = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_fn(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = evaluate_loss(model, val_loader, device)
        history["epochs_ran"] = epoch + 1
        history["train_loss"] = float(np.mean(train_losses)) if train_losses else float("nan")
        history["best_val_loss"] = float(best_val)
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            history["best_val_loss"] = float(best_val)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break
    model.load_state_dict(best_state)
    history["best_val_loss"] = float(best_val)
    return history


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        inputs, targets = _to_device(batch, device)
        losses.append(float(_loss_fn(model(inputs), targets).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _angular_error(pred: np.ndarray, target: np.ndarray) -> float:
    pred_theta = np.arctan2(pred[..., 1], pred[..., 0])
    target_theta = np.arctan2(target[..., 1], target[..., 0])
    diff = (pred_theta - target_theta + np.pi) % (2 * np.pi) - np.pi
    return float(np.mean(np.abs(diff)))


def _position_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred[..., 2:4] - target[..., 2:4]) ** 2)))


def _final_home_vector_cosine(pred: np.ndarray, target: np.ndarray) -> float:
    pred_final = pred[:, -1, 2:4]
    target_final = target[:, -1, 2:4]
    pred_norm = np.linalg.norm(pred_final, axis=1)
    target_norm = np.linalg.norm(target_final, axis=1)
    denom = pred_norm * target_norm
    valid = denom > 1e-8
    if not np.any(valid):
        return float("nan")
    cos = np.sum(pred_final[valid] * target_final[valid], axis=1) / denom[valid]
    return float(np.mean(np.clip(cos, -1.0, 1.0)))


def _final_displacement_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred[:, -1, 2:4] - target[:, -1, 2:4], axis=1)))


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    losses: list[float] = []
    for batch in loader:
        inputs, targets = _to_device(batch, device)
        pred = model(inputs)
        losses.append(float(_loss_fn(pred, targets).detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())
    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets_all, axis=0)
    return {
        "mse": float(np.mean(losses)),
        "heading_angular_error": _angular_error(pred_np, target_np),
        "position_rmse": _position_rmse(pred_np, target_np),
        "final_home_vector_cosine": _final_home_vector_cosine(pred_np, target_np),
        "final_displacement_error": _final_displacement_error(pred_np, target_np),
    }


@torch.no_grad()
def measure_latency_ms_per_sequence(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    repeats: int = 5,
) -> float:
    model.eval()
    batch = next(iter(loader))
    inputs, _ = _to_device(batch, device)
    for _ in range(2):
        _ = model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return float((elapsed * 1000.0) / (repeats * inputs.shape[0]))


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "mse",
        "heading_angular_error",
        "position_rmse",
        "final_home_vector_cosine",
        "final_displacement_error",
        "drift_slope_vs_T",
        "latency_ms_per_sequence",
    ]
    grouped = metrics.groupby(["model", "split", "T", "noise_std"], dropna=False)
    summary = grouped[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns
    ]
    first_cols = grouped[["trainable_parameter_count", "frozen_edge_count", "K"]].first().reset_index()
    return summary.merge(first_cols, on=["model", "split", "T", "noise_std"], how="left")


def _add_drift_slopes(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics["drift_slope_vs_T"] = np.nan
    clean = metrics[(metrics["split"] == "test") & (metrics["noise_std"] == 0.0)]
    for (model, seed), group in clean.groupby(["model", "seed"]):
        if group["T"].nunique() < 2:
            continue
        coeff = np.polyfit(group["T"].astype(float), group["final_displacement_error"], deg=1)
        metrics.loc[(metrics["model"] == model) & (metrics["seed"] == seed), "drift_slope_vs_T"] = float(
            coeff[0]
        )
    return metrics


def run_training(
    paths: OutputPaths,
    train_config: TrainConfig,
    task_spec: TaskSpec,
) -> pd.DataFrame:
    graph = load_prepared_graph(paths)
    device = resolve_device(train_config.device)
    splits = ensure_splits(paths.sequence_dir, task_spec)
    train_split = next(split for split in splits if split.name == "train")
    val_split = next(split for split in splits if split.name == "val")
    train_loader = _loader(
        train_split.path,
        train_config.batch_size,
        train_config.num_workers,
        shuffle=True,
        device=device,
    )
    val_loader = _loader(
        val_split.path,
        train_config.batch_size,
        train_config.num_workers,
        shuffle=False,
        device=device,
    )
    eval_splits = [split for split in splits if split.name in {"test", "test_noise"}]
    model_names = ["cx_bpu", "no_recurrence", "random", "degree_shuffle", "weight_shuffle"]
    if train_config.include_gru:
        model_names.append("gru")

    rows: list[dict[str, object]] = []
    iterator = tqdm(
        [(seed, name) for seed in train_config.seeds for name in model_names],
        desc="training benchmark models",
    )
    for seed, model_name in iterator:
        iterator.set_postfix(seed=seed, model=model_name)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = _make_model(graph, model_name, seed, device)
        history = train_one_model(model, train_loader, val_loader, train_config, device)
        latency_loader = _loader(
            val_split.path,
            min(train_config.batch_size, 64),
            0,
            shuffle=False,
            device=device,
        )
        latency = measure_latency_ms_per_sequence(model, latency_loader, device)
        k_value = int(getattr(model, "K", 1))
        frozen_edges = int(getattr(model, "W_rec", torch.empty(0)).count_nonzero().item()) if hasattr(model, "W_rec") else 0
        trainable_params = count_trainable_parameters(model)
        for split in eval_splits:
            loader = _loader(
                split.path,
                train_config.batch_size,
                train_config.num_workers,
                shuffle=False,
                device=device,
            )
            metric = evaluate_metrics(model, loader, device)
            rows.append(
                {
                    "seed": int(seed),
                    "model": model_name,
                    "split": split.name,
                    "T": int(split.T),
                    "noise_std": float(split.noise_std),
                    "epochs_ran": int(history["epochs_ran"]),
                    "best_val_loss": float(history["best_val_loss"]),
                    "trainable_parameter_count": trainable_params,
                    "frozen_edge_count": frozen_edges,
                    "K": k_value,
                    "latency_ms_per_sequence": latency,
                    **metric,
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metrics = _add_drift_slopes(pd.DataFrame(rows))
    metrics.to_csv(paths.metrics_by_seed_csv, index=False)
    summary = _summarize_metrics(metrics)
    summary.to_csv(paths.metrics_summary_csv, index=False)
    write_plots(paths)
    return metrics


def smoke_train_config(seed: int = 0) -> TrainConfig:
    return TrainConfig(seeds=(seed,), epochs=1, batch_size=4, num_workers=0, patience=1, device="cpu")


def smoke_task_spec() -> TaskSpec:
    return replace(
        TaskSpec(),
        train_count=16,
        val_count=8,
        test_count=8,
        train_T=12,
        test_T=(12, 16),
        noise_stds=(0.0, 0.10),
    )
