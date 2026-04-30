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

from .config import (
    DEFAULT_BPU_MODELS,
    OUTPUT_DIM,
    RHO_TARGET,
    TASK_CARTESIAN,
    TASK_CX_POLAR_BUMP,
    OutputPaths,
    TaskSpec,
    TrainConfig,
    output_dim_for_task,
    resolve_device,
)
from .connectome import (
    PreparedGraph,
    degree_preserving_shuffle_matrix,
    load_prepared_graph,
    pool_indices,
    power_iteration_radius,
    random_control_matrix,
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


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes:d}m{sec:02d}s"
    return f"{sec:d}s"


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
    start = time.perf_counter()
    tqdm.write(f"control-scale-start edges={matrix.nnz}")
    rho = power_iteration_radius(matrix, iters=120)
    if rho <= 0:
        tqdm.write(
            f"control-scale-done rho={rho:.6g} scale=1 elapsed={_format_duration(time.perf_counter() - start)}"
        )
        return matrix.astype(np.float32).tocsr()
    scale = RHO_TARGET / rho
    scaled = (matrix * scale).astype(np.float32).tocsr()
    tqdm.write(
        f"control-scale-done rho={rho:.6g} scale={scale:.6g} elapsed={_format_duration(time.perf_counter() - start)}"
    )
    return scaled


def _control_matrix(primary: sparse.csr_matrix, name: str, seed: int) -> sparse.csr_matrix:
    if name in {"cx_bpu", "no_recurrence"}:
        tqdm.write(f"control-build-reuse model={name} edges={primary.nnz}")
        return primary
    if name == "random":
        start = time.perf_counter()
        tqdm.write(
            f"control-build-start model=random seed={seed} N={primary.shape[0]} edges={primary.nnz}"
        )
        matrix = random_control_matrix(primary, seed=10_000 + seed)
        tqdm.write(
            f"control-build-done model=random edges={matrix.nnz} elapsed={_format_duration(time.perf_counter() - start)}"
        )
        return _scale_control(matrix)
    if name == "degree_shuffle":
        start = time.perf_counter()
        tqdm.write(
            f"control-build-start model=degree_shuffle seed={seed} N={primary.shape[0]} edges={primary.nnz}"
        )
        matrix = degree_preserving_shuffle_matrix(primary, seed=20_000 + seed)
        tqdm.write(
            f"control-build-done model=degree_shuffle edges={matrix.nnz} elapsed={_format_duration(time.perf_counter() - start)}"
        )
        return _scale_control(matrix)
    if name == "weight_shuffle":
        start = time.perf_counter()
        tqdm.write(
            f"control-build-start model=weight_shuffle seed={seed} N={primary.shape[0]} edges={primary.nnz}"
        )
        matrix = weight_shuffled_control_matrix(primary, seed=30_000 + seed)
        tqdm.write(
            f"control-build-done model=weight_shuffle edges={matrix.nnz} elapsed={_format_duration(time.perf_counter() - start)}"
        )
        return _scale_control(matrix)
    raise ValueError(f"Unknown control: {name}")


def _make_model(
    graph: PreparedGraph,
    model_name: str,
    seed: int,
    device: torch.device,
    task_spec: TaskSpec,
    include_gru_hidden: int | None = None,
) -> nn.Module:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    output_dim = output_dim_for_task(task_spec)
    if model_name == "gru":
        hidden = include_gru_hidden or min(256, int(graph.metadata["N"]))
        return GRUBaseline(hidden_size=hidden, output_dim=output_dim).to(device)
    indices = pool_indices(graph.pools)
    matrix = _control_matrix(graph.matrix.astype(np.float32).tocsr(), model_name, seed)
    K = int(graph.metadata["estimated_K"])
    model = CXBPU(
        matrix,
        sensory_indices=indices["sensory"],
        output_indices=indices["output"],
        K=K,
        reset_each_timestep=(model_name == "no_recurrence"),
        output_dim=output_dim,
    ).to(device)
    assert_bpu_trainable_surface(model)
    return model


def _loss_fn(pred: torch.Tensor, target: torch.Tensor, task_spec: TaskSpec) -> torch.Tensor:
    expected_dim = output_dim_for_task(task_spec)
    if pred.shape[-1] != expected_dim or target.shape[-1] != expected_dim:
        raise ValueError(
            f"model/target output dimension mismatch for {task_spec.kind}: "
            f"pred={pred.shape[-1]}, target={target.shape[-1]}, expected={expected_dim}"
        )
    if task_spec.kind == TASK_CARTESIAN:
        return torch.mean((pred - target) ** 2)
    if task_spec.kind == TASK_CX_POLAR_BUMP:
        bins = task_spec.heading_bins
        pred_bump = torch.sigmoid(pred[..., :bins])
        target_bump = target[..., :bins]
        bump_loss = torch.mean((pred_bump - target_bump) ** 2)
        bearing_loss = torch.mean((pred[..., bins : bins + 2] - target[..., bins : bins + 2]) ** 2)
        distance_loss = torch.mean((pred[..., bins + 2] - target[..., bins + 2]) ** 2)
        return bump_loss + bearing_loss + 0.5 * distance_loss
    raise ValueError(f"Unknown task kind: {task_spec.kind}")


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    model_name: str,
    seed: int,
    task_spec: TaskSpec,
) -> dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, object]] = []
    history: dict[str, object] = {"epochs_ran": 0, "best_val_loss": best_val}
    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)
    tqdm.write(
        "model-start "
        f"model={model_name} seed={seed} epochs={config.epochs} "
        f"train_batches={total_train_batches} val_batches={total_val_batches} "
        f"batch_size={config.batch_size}"
    )
    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        last_log = epoch_start
        model.train()
        train_losses: list[float] = []
        tqdm.write(
            "epoch-start "
            f"model={model_name} seed={seed} epoch={epoch + 1}/{config.epochs}"
        )
        for batch_index, batch in enumerate(train_loader, start=1):
            inputs, targets = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_fn(model(inputs), targets, task_spec)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            batch_loss = float(loss.detach().cpu())
            train_losses.append(batch_loss)
            now = time.perf_counter()
            should_log = (
                config.log_every_seconds > 0
                and (now - last_log >= config.log_every_seconds)
            )
            if should_log:
                tqdm.write(
                    "progress "
                    f"phase=train model={model_name} seed={seed} "
                    f"epoch={epoch + 1}/{config.epochs} "
                    f"batch={batch_index}/{total_train_batches} "
                    f"batch_mse={batch_loss:.6g} "
                    f"running_train_mse={float(np.mean(train_losses)):.6g} "
                    f"elapsed={_format_duration(now - epoch_start)}"
                )
                last_log = now
        tqdm.write(
            "val-start "
            f"model={model_name} seed={seed} epoch={epoch + 1}/{config.epochs}"
        )
        val_loss = evaluate_loss(
            model,
            val_loader,
            device,
            task_spec,
            log_context=f"model={model_name} seed={seed} epoch={epoch + 1}/{config.epochs}",
            log_every_seconds=config.log_every_seconds,
        )
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        history["epochs_ran"] = epoch + 1
        history["train_loss"] = train_loss
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history["best_val_loss"] = float(best_val)
        row = {
            "seed": int(seed),
            "model": model_name,
            "epoch": int(epoch + 1),
            "train_mse": train_loss,
            "val_mse": float(val_loss),
            "best_val_mse": float(best_val),
            "patience_wait": int(epochs_without_improvement),
        }
        epoch_rows.append(row)
        tqdm.write(
            "loss "
            f"model={model_name} seed={seed} epoch={epoch + 1}/{config.epochs} "
            f"train_mse={train_loss:.6g} val_mse={val_loss:.6g} "
            f"best_val_mse={best_val:.6g} patience_wait={epochs_without_improvement}"
        )
        if epochs_without_improvement >= config.patience:
            break
    model.load_state_dict(best_state)
    history["best_val_loss"] = float(best_val)
    history["epoch_rows"] = epoch_rows
    tqdm.write(
        "model-done "
        f"model={model_name} seed={seed} epochs_ran={history['epochs_ran']} "
        f"best_val_mse={best_val:.6g}"
    )
    return history


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_spec: TaskSpec,
    log_context: str | None = None,
    log_every_seconds: float = 60.0,
) -> float:
    model.eval()
    losses: list[float] = []
    total_batches = len(loader)
    start = time.perf_counter()
    last_log = start
    for batch_index, batch in enumerate(loader, start=1):
        inputs, targets = _to_device(batch, device)
        losses.append(float(_loss_fn(model(inputs), targets, task_spec).detach().cpu()))
        now = time.perf_counter()
        if (
            log_context is not None
            and log_every_seconds > 0
            and now - last_log >= log_every_seconds
        ):
            tqdm.write(
                "progress "
                f"phase=val {log_context} batch={batch_index}/{total_batches} "
                f"running_val_mse={float(np.mean(losses)):.6g} "
                f"elapsed={_format_duration(now - start)}"
            )
            last_log = now
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


def _circular_error(pred_angle: np.ndarray, target_angle: np.ndarray) -> np.ndarray:
    return (pred_angle - target_angle + np.pi) % (2 * np.pi) - np.pi


def _decode_bump_angle(bump: np.ndarray) -> np.ndarray:
    bins = bump.shape[-1]
    angles = np.linspace(-np.pi, np.pi, bins, endpoint=False, dtype=np.float32)
    sin_sum = np.sum(bump * np.sin(angles), axis=-1)
    cos_sum = np.sum(bump * np.cos(angles), axis=-1)
    return np.arctan2(sin_sum, cos_sum)


def _evaluate_cartesian_metrics(
    pred_np: np.ndarray, target_np: np.ndarray, losses: list[float]
) -> dict[str, float]:
    return {
        "mse": float(np.mean(losses)),
        "heading_angular_error": _angular_error(pred_np, target_np),
        "position_rmse": _position_rmse(pred_np, target_np),
        "final_home_vector_cosine": _final_home_vector_cosine(pred_np, target_np),
        "final_displacement_error": _final_displacement_error(pred_np, target_np),
    }


def _evaluate_cx_polar_bump_metrics(
    pred_np: np.ndarray,
    target_np: np.ndarray,
    losses: list[float],
    task_spec: TaskSpec,
) -> dict[str, float]:
    bins = task_spec.heading_bins
    pred_bump = 1.0 / (1.0 + np.exp(-pred_np[..., :bins]))
    target_bump = target_np[..., :bins]
    pred_heading = _decode_bump_angle(pred_bump)
    target_heading = _decode_bump_angle(target_bump)
    heading_error = np.abs(_circular_error(pred_heading, target_heading))
    pred_bearing = np.arctan2(pred_np[..., bins + 1], pred_np[..., bins])
    target_bearing = np.arctan2(target_np[..., bins + 1], target_np[..., bins])
    bearing_error = np.abs(_circular_error(pred_bearing, target_bearing))
    pred_distance = pred_np[..., bins + 2] * task_spec.home_distance_scale
    target_distance = target_np[..., bins + 2] * task_spec.home_distance_scale
    distance_error = pred_distance - target_distance
    final_bearing_error = bearing_error[:, -1]
    final_distance_error = np.abs(distance_error[:, -1])
    return {
        "mse": float(np.mean(losses)),
        "heading_angular_error": float(np.mean(heading_error)),
        "position_rmse": float(np.sqrt(np.mean(distance_error**2))),
        "final_home_vector_cosine": float(np.mean(np.cos(final_bearing_error))),
        "final_displacement_error": float(np.mean(final_distance_error)),
        "bump_mse": float(np.mean((pred_bump - target_bump) ** 2)),
        "heading_bump_angular_error": float(np.mean(heading_error)),
        "home_bearing_angular_error": float(np.mean(bearing_error)),
        "home_distance_rmse": float(np.sqrt(np.mean(distance_error**2))),
        "final_home_bearing_angular_error": float(np.mean(final_bearing_error)),
        "final_home_distance_error": float(np.mean(final_distance_error)),
    }


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_spec: TaskSpec,
    log_context: str | None = None,
    log_every_seconds: float = 60.0,
) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    losses: list[float] = []
    total_batches = len(loader)
    start = time.perf_counter()
    last_log = start
    for batch_index, batch in enumerate(loader, start=1):
        inputs, targets = _to_device(batch, device)
        pred = model(inputs)
        losses.append(float(_loss_fn(pred, targets, task_spec).detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())
        now = time.perf_counter()
        if (
            log_context is not None
            and log_every_seconds > 0
            and now - last_log >= log_every_seconds
        ):
            tqdm.write(
                "progress "
                f"phase=eval {log_context} batch={batch_index}/{total_batches} "
                f"running_eval_mse={float(np.mean(losses)):.6g} "
                f"elapsed={_format_duration(now - start)}"
            )
            last_log = now
    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets_all, axis=0)
    if task_spec.kind == TASK_CARTESIAN:
        return _evaluate_cartesian_metrics(pred_np, target_np, losses)
    if task_spec.kind == TASK_CX_POLAR_BUMP:
        return _evaluate_cx_polar_bump_metrics(pred_np, target_np, losses, task_spec)
    raise ValueError(f"Unknown task kind: {task_spec.kind}")


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
        "bump_mse",
        "heading_bump_angular_error",
        "home_bearing_angular_error",
        "home_distance_rmse",
        "final_home_bearing_angular_error",
        "final_home_distance_error",
        "drift_slope_vs_T",
        "latency_ms_per_sequence",
        "final_train_loss",
        "best_val_loss",
    ]
    metric_cols = [col for col in metric_cols if col in metrics.columns]
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
    tqdm.write(f"run-start output_dir={paths.output_dir} cache_dir={paths.cache_dir}")
    graph = load_prepared_graph(paths)
    device = resolve_device(train_config.device)
    tqdm.write(f"device-selected device={device}")
    tqdm.write("data-start ensuring cached synthetic splits")
    splits = ensure_splits(paths.sequence_dir, task_spec)
    tqdm.write(f"data-ready splits={len(splits)} sequence_dir={paths.sequence_dir}")
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
    model_names = list(train_config.models or DEFAULT_BPU_MODELS)
    if train_config.include_gru and "gru" not in model_names:
        model_names.append("gru")
    tqdm.write(
        "run-config "
        f"models={','.join(model_names)} seeds={','.join(map(str, train_config.seeds))} "
        f"epochs={train_config.epochs} batch_size={train_config.batch_size} "
        f"task={task_spec.kind} output_dim={output_dim_for_task(task_spec)} "
        f"log_every_seconds={train_config.log_every_seconds:g}"
    )

    rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []
    iterator = tqdm(
        [(seed, name) for seed in train_config.seeds for name in model_names],
        desc="training benchmark models",
    )
    for seed, model_name in iterator:
        iterator.set_postfix(seed=seed, model=model_name)
        torch.manual_seed(seed)
        np.random.seed(seed)
        tqdm.write(f"build-model model={model_name} seed={seed}")
        model = _make_model(graph, model_name, seed, device, task_spec)
        history = train_one_model(
            model,
            train_loader,
            val_loader,
            train_config,
            device,
            model_name=model_name,
            seed=seed,
            task_spec=task_spec,
        )
        loss_rows.extend(history["epoch_rows"])
        pd.DataFrame(loss_rows).to_csv(paths.loss_history_csv, index=False)
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
            tqdm.write(
                "eval-start "
                f"model={model_name} seed={seed} split={split.name} "
                f"T={split.T} noise_std={split.noise_std}"
            )
            loader = _loader(
                split.path,
                train_config.batch_size,
                train_config.num_workers,
                shuffle=False,
                device=device,
            )
            metric = evaluate_metrics(
                model,
                loader,
                device,
                task_spec,
                log_context=(
                    f"model={model_name} seed={seed} split={split.name} "
                    f"T={split.T} noise_std={split.noise_std}"
                ),
                log_every_seconds=train_config.log_every_seconds,
            )
            rows.append(
                {
                    "seed": int(seed),
                    "model": model_name,
                    "task": task_spec.kind,
                    "split": split.name,
                    "T": int(split.T),
                    "noise_std": float(split.noise_std),
                    "epochs_ran": int(history["epochs_ran"]),
                    "final_train_loss": float(history["train_loss"]),
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
