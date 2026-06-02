#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_optic_flow_benchmark as optic  # noqa: E402
from src.config import OutputPaths  # noqa: E402
from src.run_manifest import write_artifact_manifest  # noqa: E402


POSE_CANDIDATES = (
    "pose_lcam_front.txt",
    "pose_lcam.txt",
    "pose_left.txt",
    "pose.txt",
)


@dataclass(frozen=True)
class TartanAirSpec:
    data_root: Path
    envs: tuple[str, ...]
    difficulties: tuple[str, ...]
    camera_name: str
    sequence_len: int
    hex_rings: int
    acceptance_pixel_sigma: float
    acceptance_samples: int
    flow_clip: float
    target_yaw_scale: float
    target_translation_scale: float
    pose_convention: str
    frame_stride: int
    sample_stride: int
    max_windows: int
    split_seed: int
    split_by_trajectory: bool
    train_fraction: float
    val_fraction: float

    @property
    def input_dim(self) -> int:
        return int(2 * (1 + 3 * self.hex_rings * (self.hex_rings + 1)))

    @property
    def output_dim(self) -> int:
        return 3


@dataclass(frozen=True)
class TrainSpec:
    seeds: tuple[int, ...]
    models: tuple[str, ...]
    epochs: int
    patience: int
    batch_size: int
    train_batches: int
    val_batches: int
    test_batches: int
    lr: float
    grad_clip: float
    state_clip: float
    device: str
    log_every_seconds: float
    flow_cache_size: int


@dataclass(frozen=True)
class FlowFrame:
    path: Path
    start_index: int
    end_index: int
    target: np.ndarray


@dataclass(frozen=True)
class FlowWindow:
    trajectory_key: str
    frames: tuple[FlowFrame, ...]


@dataclass(frozen=True)
class SplitData:
    train: tuple[FlowWindow, ...]
    val: tuple[FlowWindow, ...]
    test: tuple[FlowWindow, ...]
    target_mean: np.ndarray
    target_std: np.ndarray


class FlowCache:
    def __init__(self, max_items: int) -> None:
        self.max_items = max(0, int(max_items))
        self._cache: OrderedDict[Path, tuple[np.ndarray, np.ndarray | None]] = OrderedDict()

    def get(self, path: Path) -> tuple[np.ndarray, np.ndarray | None]:
        path = path.resolve()
        if self.max_items > 0 and path in self._cache:
            value = self._cache.pop(path)
            self._cache[path] = value
            return value
        value = read_flow(path)
        if self.max_items > 0:
            self._cache[path] = value
            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)
        return value


def _format_seconds(seconds: float) -> str:
    return optic._format_seconds(seconds)


def log_event(message: str) -> None:
    print(message, flush=True)


def numeric_sort_key(path: Path) -> tuple[int, ...]:
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        return (0,)
    return tuple(int(x) for x in nums)


def infer_flow_indices(path: Path, fallback_index: int, frame_stride: int) -> tuple[int, int]:
    nums = re.findall(r"\d+", path.stem)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    if len(nums) == 1:
        start = int(nums[0])
        return start, start + frame_stride
    return fallback_index, fallback_index + frame_stride


def find_pose_file(traj_dir: Path, camera_name: str) -> Path:
    dynamic = (
        f"pose_{camera_name}.txt",
        f"pose_{camera_name.replace('_front', '')}.txt",
    )
    for name in (*dynamic, *POSE_CANDIDATES):
        path = traj_dir / name
        if path.exists():
            return path
    candidates = sorted(traj_dir.glob("pose*.txt"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No TartanAir pose file found in {traj_dir}")


def flow_dir_for(traj_dir: Path, camera_name: str) -> Path:
    candidates = (
        traj_dir / f"flow_{camera_name}",
        traj_dir / "flow_lcam_front",
        traj_dir / "flow",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def read_pose_file(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.shape[1] != 7:
        raise ValueError(f"Pose file must have 7 columns [tx ty tz qx qy qz qw]: {path}")
    return poses


def relative_target(
    poses: np.ndarray,
    start_index: int,
    end_index: int,
    pose_convention: str,
    yaw_scale: float,
    translation_scale: float,
) -> np.ndarray:
    if start_index < 0 or end_index >= len(poses):
        raise IndexError(
            f"Pose indices out of range: start={start_index} end={end_index} len={len(poses)}"
        )
    p0 = poses[start_index, :3]
    p1 = poses[end_index, :3]
    r0 = Rotation.from_quat(poses[start_index, 3:7])
    r1 = Rotation.from_quat(poses[end_index, 3:7])
    if pose_convention == "camera_to_world":
        delta_local = r0.inv().apply(p1 - p0)
        relative_rotation = r0.inv() * r1
    elif pose_convention == "world_to_camera":
        delta_local = r0.apply(p1 - p0)
        relative_rotation = r1 * r0.inv()
    else:
        raise ValueError(f"Unknown pose convention: {pose_convention}")
    yaw = float(relative_rotation.as_euler("zyx", degrees=False)[0])
    forward = float(delta_local[0])
    lateral = float(delta_local[1])
    return np.array(
        [
            yaw / max(yaw_scale, 1e-8),
            forward / max(translation_scale, 1e-8),
            lateral / max(translation_scale, 1e-8),
        ],
        dtype=np.float32,
    )


def read_flow(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if path.suffix == ".npz":
        data = np.load(path)
        flow_key = next(
            (key for key in ("flow_fwd", "flow", "flow_fw", "flow_forward") if key in data),
            None,
        )
        if flow_key is None:
            raise KeyError(f"No forward flow field found in {path}; keys={list(data.keys())}")
        flow = np.asarray(data[flow_key], dtype=np.float32)
        mask = None
        for key in ("covisible_mask_fwd", "fov_mask_fwd", "mask", "valid_mask"):
            if key in data:
                mask = np.asarray(data[key]).astype(bool)
                break
    elif path.suffix.lower() == ".png":
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Reading old TartanAir flow PNGs requires opencv-python or opencv-contrib-python."
            ) from exc
        flow16 = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if flow16 is None:
            raise FileNotFoundError(path)
        flow = flow16[:, :, :2].astype(np.float32)
        flow = (flow - 32768.0) / 64.0
        mask = flow16[:, :, 2].astype(np.uint8).astype(bool) if flow16.shape[2] > 2 else None
    else:
        raise ValueError(f"Unsupported TartanAir flow format: {path}")
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Flow must have shape [H, W, 2], got {flow.shape} from {path}")
    return flow.astype(np.float32), mask


def _hex_pixel_centers(height: int, width: int, rings: int) -> np.ndarray:
    lattice = optic.hex_lattice(rings)
    x = (lattice[:, 0] + 1.0) * 0.5 * (width - 1)
    y = (lattice[:, 1] + 1.0) * 0.5 * (height - 1)
    return np.stack([x, y], axis=1).astype(np.float32)


def _acceptance_offsets(sample_count: int, sigma_px: float) -> np.ndarray:
    if sample_count <= 1 or sigma_px <= 0:
        return np.zeros((1, 2), dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * math.pi, sample_count - 1, endpoint=False)
    radius = float(sigma_px)
    offsets = [(0.0, 0.0)]
    offsets.extend((radius * math.cos(a), radius * math.sin(a)) for a in angles)
    return np.asarray(offsets, dtype=np.float32)


def _bilinear_flow_sample(
    flow: np.ndarray,
    mask: np.ndarray | None,
    xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width, _ = flow.shape
    x = np.clip(xy[:, 0], 0.0, width - 1.0)
    y = np.clip(xy[:, 1], 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = (x - x0).astype(np.float32)[:, None]
    wy = (y - y0).astype(np.float32)[:, None]
    top = (1.0 - wx) * flow[y0, x0] + wx * flow[y0, x1]
    bottom = (1.0 - wx) * flow[y1, x0] + wx * flow[y1, x1]
    values = ((1.0 - wy) * top + wy * bottom).astype(np.float32)
    if mask is None:
        valid = np.ones((xy.shape[0],), dtype=np.float32)
    else:
        valid = (
            mask[y0, x0].astype(np.float32)
            + mask[y0, x1].astype(np.float32)
            + mask[y1, x0].astype(np.float32)
            + mask[y1, x1].astype(np.float32)
        )
        valid = (valid > 0).astype(np.float32)
    return values, valid


def hex_pool_flow(
    flow: np.ndarray,
    mask: np.ndarray | None,
    rings: int,
    sigma_px: float,
    samples: int,
    flow_clip: float,
) -> np.ndarray:
    centers = _hex_pixel_centers(flow.shape[0], flow.shape[1], rings)
    offsets = _acceptance_offsets(samples, sigma_px)
    acc = np.zeros((centers.shape[0], 2), dtype=np.float32)
    valid_sum = np.zeros((centers.shape[0], 1), dtype=np.float32)
    for offset in offsets:
        values, valid = _bilinear_flow_sample(flow, mask, centers + offset[None, :])
        valid = valid[:, None]
        acc += values * valid
        valid_sum += valid
    pooled = acc / np.maximum(valid_sum, 1.0)
    pooled = np.clip(pooled / max(flow_clip, 1e-8), -1.0, 1.0)
    return pooled.reshape(-1).astype(np.float32)


def enumerate_tartanair_windows(spec: TartanAirSpec) -> tuple[FlowWindow, ...]:
    start_time = time.monotonic()
    windows: list[FlowWindow] = []
    log_event(
        "tartanair-index-start "
        f"root={spec.data_root} envs={','.join(spec.envs)} difficulties={','.join(spec.difficulties)} "
        f"camera={spec.camera_name} sequence_len={spec.sequence_len}"
    )
    for env in spec.envs:
        for difficulty in spec.difficulties:
            difficulty_dir = spec.data_root / env / f"Data_{difficulty}"
            if not difficulty_dir.exists():
                log_event(f"tartanair-index-skip missing={difficulty_dir}")
                continue
            for traj_dir in sorted(difficulty_dir.glob("P*")):
                if not traj_dir.is_dir():
                    continue
                flow_dir = flow_dir_for(traj_dir, spec.camera_name)
                flow_paths = sorted(
                    [*flow_dir.glob("*.npz"), *flow_dir.glob("*.png")],
                    key=numeric_sort_key,
                )
                if not flow_paths:
                    log_event(f"tartanair-index-skip no_flow_dir={flow_dir}")
                    continue
                pose_path = find_pose_file(traj_dir, spec.camera_name)
                poses = read_pose_file(pose_path)
                frames: list[FlowFrame] = []
                for fallback_idx, flow_path in enumerate(flow_paths):
                    start_idx, end_idx = infer_flow_indices(
                        flow_path,
                        fallback_index=fallback_idx,
                        frame_stride=spec.frame_stride,
                    )
                    if end_idx >= len(poses):
                        continue
                    target = relative_target(
                        poses,
                        start_idx,
                        end_idx,
                        pose_convention=spec.pose_convention,
                        yaw_scale=spec.target_yaw_scale,
                        translation_scale=spec.target_translation_scale,
                    )
                    frames.append(
                        FlowFrame(
                            path=flow_path,
                            start_index=start_idx,
                            end_index=end_idx,
                            target=target,
                        )
                    )
                for start in range(0, len(frames) - spec.sequence_len + 1, spec.sample_stride):
                    seq = tuple(frames[start : start + spec.sequence_len])
                    windows.append(
                        FlowWindow(
                            trajectory_key=f"{env}/{difficulty}/{traj_dir.name}",
                            frames=seq,
                        )
                    )
    if spec.max_windows > 0 and len(windows) > spec.max_windows:
        rng = np.random.default_rng(spec.split_seed)
        keep = np.sort(rng.choice(len(windows), size=spec.max_windows, replace=False))
        windows = [windows[int(i)] for i in keep]
    log_event(
        f"tartanair-index-done windows={len(windows)} elapsed={_format_seconds(time.monotonic() - start_time)}"
    )
    if not windows:
        raise RuntimeError(
            "No TartanAir flow windows found. Download/generate flow_lcam_front first, "
            "or point --tartanair-root to an existing TartanAirV2 tree."
        )
    return tuple(windows)


def split_windows(windows: tuple[FlowWindow, ...], spec: TartanAirSpec) -> SplitData:
    rng = np.random.default_rng(spec.split_seed)
    if spec.split_by_trajectory:
        by_traj: dict[str, list[FlowWindow]] = {}
        for window in windows:
            by_traj.setdefault(window.trajectory_key, []).append(window)
        keys = sorted(by_traj)
        rng.shuffle(keys)
        if len(keys) >= 3:
            n_train = max(1, int(round(len(keys) * spec.train_fraction)))
            n_val = max(1, int(round(len(keys) * spec.val_fraction)))
            n_train = min(n_train, len(keys) - 2)
            n_val = min(n_val, len(keys) - n_train - 1)
            train_keys = set(keys[:n_train])
            val_keys = set(keys[n_train : n_train + n_val])
            test_keys = set(keys[n_train + n_val :])
            train = tuple(w for key in train_keys for w in by_traj[key])
            val = tuple(w for key in val_keys for w in by_traj[key])
            test = tuple(w for key in test_keys for w in by_traj[key])
        else:
            log_event("split-fallback reason=too_few_trajectories split=window")
            train, val, test = split_windows_by_index(windows, spec, rng)
    else:
        train, val, test = split_windows_by_index(windows, spec, rng)
    train_targets = np.concatenate(
        [np.stack([frame.target for frame in window.frames], axis=0) for window in train],
        axis=0,
    )
    target_mean = train_targets.mean(axis=0).astype(np.float32)
    target_std = np.maximum(train_targets.std(axis=0), 1e-4).astype(np.float32)
    log_event(
        "split-done "
        f"train={len(train)} val={len(val)} test={len(test)} "
        f"target_mean={target_mean.tolist()} target_std={target_std.tolist()}"
    )
    return SplitData(
        train=tuple(train),
        val=tuple(val),
        test=tuple(test),
        target_mean=target_mean,
        target_std=target_std,
    )


def split_windows_by_index(
    windows: tuple[FlowWindow, ...],
    spec: TartanAirSpec,
    rng: np.random.Generator,
) -> tuple[tuple[FlowWindow, ...], tuple[FlowWindow, ...], tuple[FlowWindow, ...]]:
    order = np.arange(len(windows))
    rng.shuffle(order)
    n_train = max(1, int(round(len(order) * spec.train_fraction)))
    n_val = max(1, int(round(len(order) * spec.val_fraction)))
    if n_train + n_val >= len(order):
        n_train = max(1, len(order) - 2)
        n_val = 1
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    if len(test_idx) == 0:
        test_idx = val_idx
    return (
        tuple(windows[int(i)] for i in train_idx),
        tuple(windows[int(i)] for i in val_idx),
        tuple(windows[int(i)] for i in test_idx),
    )


def build_batch(
    windows: tuple[FlowWindow, ...],
    batch_size: int,
    rng: np.random.Generator,
    spec: TartanAirSpec,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    cache: FlowCache,
) -> optic.OpticBatch:
    if not windows:
        raise ValueError("Cannot build a batch from an empty split.")
    chosen = rng.integers(0, len(windows), size=batch_size)
    inputs = np.zeros((batch_size, spec.sequence_len, spec.input_dim), dtype=np.float32)
    targets = np.zeros((batch_size, spec.sequence_len, spec.output_dim), dtype=np.float32)
    for b, window_idx in enumerate(chosen):
        window = windows[int(window_idx)]
        for t, frame in enumerate(window.frames):
            flow, mask = cache.get(frame.path)
            inputs[b, t] = hex_pool_flow(
                flow,
                mask,
                rings=spec.hex_rings,
                sigma_px=spec.acceptance_pixel_sigma,
                samples=spec.acceptance_samples,
                flow_clip=spec.flow_clip,
            )
            targets[b, t] = (frame.target - target_mean) / target_std
    return optic.OpticBatch(inputs=inputs, targets=targets)


def _batch_to_torch(batch: optic.OpticBatch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(batch.inputs).to(device=device, dtype=torch.float32),
        torch.from_numpy(batch.targets).to(device=device, dtype=torch.float32),
    )


def _denormalize(values: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray) -> np.ndarray:
    return values * target_std[None, None, :] + target_mean[None, None, :]


def regression_metrics(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> dict[str, float]:
    pred = _denormalize(pred_norm, target_mean, target_std)
    target = _denormalize(target_norm, target_mean, target_std)
    err = pred - target
    component_rmse = np.sqrt(np.mean(err**2, axis=(0, 1)))
    target_var = np.var(target.reshape(-1, target.shape[-1]), axis=0) + 1e-8
    r2 = 1.0 - np.mean(err.reshape(-1, target.shape[-1]) ** 2, axis=0) / target_var
    return {
        "norm_loss": float(np.mean((pred_norm - target_norm) ** 2)),
        "overall_rmse": float(np.sqrt(np.mean(err**2))),
        "yaw_rmse": float(component_rmse[0]),
        "forward_rmse": float(component_rmse[1]),
        "lateral_rmse": float(component_rmse[2]),
        "translation_rmse": float(np.sqrt(np.mean(err[..., 1:3] ** 2))),
        "yaw_r2": float(r2[0]),
        "forward_r2": float(r2[1]),
        "lateral_r2": float(r2[2]),
    }


def evaluate_model(
    model: optic.SparseOpticFlowRNN,
    windows: tuple[FlowWindow, ...],
    spec: TartanAirSpec,
    train_spec: TrainSpec,
    device: torch.device,
    seed: int,
    batches: int,
    phase: str,
    model_name: str,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    cache = FlowCache(train_spec.flow_cache_size)
    model.eval()
    losses: list[float] = []
    pred_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    start = time.monotonic()
    last_log = start
    io_seconds = 0.0
    eval_seconds = 0.0
    with torch.no_grad():
        for batch_idx in range(1, batches + 1):
            io_start = time.monotonic()
            batch = build_batch(
                windows,
                train_spec.batch_size,
                rng,
                spec,
                target_mean,
                target_std,
                cache,
            )
            io_seconds += time.monotonic() - io_start
            x, y = _batch_to_torch(batch, device)
            eval_start = time.monotonic()
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            eval_seconds += time.monotonic() - eval_start
            losses.append(float(loss.detach().cpu()))
            pred_chunks.append(pred.detach().cpu().numpy())
            target_chunks.append(y.detach().cpu().numpy())
            now = time.monotonic()
            if train_spec.log_every_seconds > 0 and now - last_log >= train_spec.log_every_seconds:
                log_event(
                    "progress "
                    f"phase={phase} model={model_name} batch={batch_idx}/{batches} "
                    f"running_norm_loss={np.mean(losses):.6g} io_elapsed={_format_seconds(io_seconds)} "
                    f"eval_elapsed={_format_seconds(eval_seconds)} elapsed={_format_seconds(now - start)}"
                )
                last_log = now
    pred_np = np.concatenate(pred_chunks, axis=0)
    target_np = np.concatenate(target_chunks, axis=0)
    metrics = regression_metrics(pred_np, target_np, target_mean, target_std)
    metrics["norm_loss"] = float(np.mean(losses))
    metrics["io_seconds"] = float(io_seconds)
    metrics["eval_seconds"] = float(eval_seconds)
    log_event(
        "eval-done "
        f"phase={phase} model={model_name} norm_loss={metrics['norm_loss']:.6g} "
        f"overall_rmse={metrics['overall_rmse']:.6g} yaw_rmse={metrics['yaw_rmse']:.6g} "
        f"translation_rmse={metrics['translation_rmse']:.6g} elapsed={_format_seconds(time.monotonic() - start)}"
    )
    return metrics


def train_one_model(
    base_matrix: sparse.csr_matrix,
    split: SplitData,
    model_name: str,
    seed: int,
    spec: TartanAirSpec,
    train_spec: TrainSpec,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    build_start = time.monotonic()
    log_event(
        f"model-build-start model={model_name} seed={seed} base_N={base_matrix.shape[0]} base_edges={base_matrix.nnz}"
    )
    recurrent = optic.model_matrix(base_matrix, model_name, seed + 10_000)
    log_event(
        f"model-build-matrix-done model={model_name} seed={seed} edges={recurrent.nnz} elapsed={_format_seconds(time.monotonic() - build_start)}"
    )
    model = optic.SparseOpticFlowRNN(
        recurrent=recurrent,
        input_dim=spec.input_dim,
        output_dim=spec.output_dim,
        state_clip=train_spec.state_clip,
        seed=seed + 1_000,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_spec.lr)
    rng = np.random.default_rng(seed + 12345)
    cache = FlowCache(train_spec.flow_cache_size)
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    history: list[dict[str, object]] = []
    log_event(
        "model-start "
        f"model={model_name} seed={seed} N={model.N} edges={recurrent.nnz} "
        f"trainable_params={model.trainable_parameter_count()} recurrent_params={model.recurrent_parameter_count()} "
        f"input_dim={spec.input_dim} sequence_len={spec.sequence_len}"
    )
    for epoch in range(1, train_spec.epochs + 1):
        model.train()
        losses: list[float] = []
        start = time.monotonic()
        last_log = start
        io_seconds = 0.0
        train_seconds = 0.0
        for batch_idx in range(1, train_spec.train_batches + 1):
            io_start = time.monotonic()
            batch = build_batch(
                split.train,
                train_spec.batch_size,
                rng,
                spec,
                split.target_mean,
                split.target_std,
                cache,
            )
            io_seconds += time.monotonic() - io_start
            x, y = _batch_to_torch(batch, device)
            step_start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            if train_spec.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_spec.grad_clip)
            optimizer.step()
            train_seconds += time.monotonic() - step_start
            losses.append(float(loss.detach().cpu()))
            now = time.monotonic()
            if train_spec.log_every_seconds > 0 and now - last_log >= train_spec.log_every_seconds:
                log_event(
                    "progress "
                    f"phase=train model={model_name} seed={seed} epoch={epoch}/{train_spec.epochs} "
                    f"batch={batch_idx}/{train_spec.train_batches} batch_norm_loss={losses[-1]:.6g} "
                    f"running_norm_loss={np.mean(losses):.6g} io_elapsed={_format_seconds(io_seconds)} "
                    f"train_elapsed={_format_seconds(train_seconds)} elapsed={_format_seconds(now - start)}"
                )
                last_log = now
        val = evaluate_model(
            model,
            split.val,
            spec,
            train_spec,
            device,
            seed=seed + 50_000 + epoch,
            batches=train_spec.val_batches,
            phase=f"val_epoch_{epoch}",
            model_name=model_name,
            target_mean=split.target_mean,
            target_std=split.target_std,
        )
        train_loss = float(np.mean(losses))
        if val["norm_loss"] < best_val:
            best_val = float(val["norm_loss"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        row = {
            "model": model_name,
            "seed": int(seed),
            "epoch": int(epoch),
            "train_norm_loss": train_loss,
            "val_norm_loss": float(val["norm_loss"]),
            "val_overall_rmse": float(val["overall_rmse"]),
            "val_yaw_rmse": float(val["yaw_rmse"]),
            "val_translation_rmse": float(val["translation_rmse"]),
            "val_yaw_r2": float(val["yaw_r2"]),
            "best_val_norm_loss": float(best_val),
            "patience_wait": int(wait),
            "io_seconds": float(io_seconds),
            "train_seconds": float(train_seconds),
        }
        history.append(row)
        log_event(
            "loss "
            f"model={model_name} seed={seed} epoch={epoch}/{train_spec.epochs} "
            f"train_norm_loss={train_loss:.6g} val_norm_loss={val['norm_loss']:.6g} "
            f"val_overall_rmse={val['overall_rmse']:.6g} val_yaw_rmse={val['yaw_rmse']:.6g} "
            f"val_translation_rmse={val['translation_rmse']:.6g} best_val_norm_loss={best_val:.6g} "
            f"patience_wait={wait}"
        )
        if train_spec.patience > 0 and wait >= train_spec.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate_model(
        model,
        split.test,
        spec,
        train_spec,
        device,
        seed=seed + 90_000,
        batches=train_spec.test_batches,
        phase="test",
        model_name=model_name,
        target_mean=split.target_mean,
        target_std=split.target_std,
    )
    metrics = {
        "model": model_name,
        "seed": int(seed),
        "N": int(model.N),
        "init_nonzero_edges": int(recurrent.nnz),
        "recurrent_params": int(model.recurrent_parameter_count()),
        "trainable_params": int(model.trainable_parameter_count()),
        "best_val_norm_loss": float(best_val),
        "test_norm_loss": float(test["norm_loss"]),
        "test_overall_rmse": float(test["overall_rmse"]),
        "test_yaw_rmse": float(test["yaw_rmse"]),
        "test_forward_rmse": float(test["forward_rmse"]),
        "test_lateral_rmse": float(test["lateral_rmse"]),
        "test_translation_rmse": float(test["translation_rmse"]),
        "test_yaw_r2": float(test["yaw_r2"]),
        "test_forward_r2": float(test["forward_r2"]),
        "test_lateral_r2": float(test["lateral_r2"]),
    }
    log_event(
        "model-done "
        f"model={model_name} seed={seed} best_val_norm_loss={best_val:.6g} "
        f"test_overall_rmse={test['overall_rmse']:.6g} test_yaw_rmse={test['yaw_rmse']:.6g} "
        f"test_translation_rmse={test['translation_rmse']:.6g}"
    )
    return metrics, history


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("model", as_index=False)
        .agg(
            best_val_norm_loss_mean=("best_val_norm_loss", "mean"),
            best_val_norm_loss_std=("best_val_norm_loss", "std"),
            test_overall_rmse_mean=("test_overall_rmse", "mean"),
            test_overall_rmse_std=("test_overall_rmse", "std"),
            test_yaw_rmse_mean=("test_yaw_rmse", "mean"),
            test_translation_rmse_mean=("test_translation_rmse", "mean"),
            test_yaw_r2_mean=("test_yaw_r2", "mean"),
            trainable_params=("trainable_params", "first"),
            recurrent_params=("recurrent_params", "first"),
            N=("N", "first"),
        )
        .sort_values("test_overall_rmse_mean")
    )


def write_plots(output_dir: Path, metrics: pd.DataFrame, history: pd.DataFrame) -> None:
    if not history.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
        for model, group in history.groupby("model"):
            by_epoch = group.groupby("epoch", as_index=False).agg(
                val_loss=("val_norm_loss", "mean"),
                val_sem=(
                    "val_norm_loss",
                    lambda x: x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0,
                ),
            )
            ax.plot(by_epoch["epoch"], by_epoch["val_loss"], label=model)
            ax.fill_between(
                by_epoch["epoch"],
                by_epoch["val_loss"] - by_epoch["val_sem"],
                by_epoch["val_loss"] + by_epoch["val_sem"],
                alpha=0.15,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation normalized MSE")
        ax.set_title("TartanAirV2 optic-flow learning curves")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "tartanair_optic_flow_loss.png")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    order = metrics.groupby("model")["test_overall_rmse"].mean().sort_values().index.tolist()
    values = [metrics.loc[metrics["model"] == model, "test_overall_rmse"].to_numpy() for model in order]
    means = [float(v.mean()) for v in values]
    sems = [float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0 for v in values]
    ax.bar(range(len(order)), means, yerr=sems, capsize=4, alpha=0.85)
    for idx, vals in enumerate(values):
        ax.scatter([idx] * len(vals), vals, color="black", s=18, zorder=3)
    ax.set_xticks(range(len(order)), order, rotation=15, ha="right")
    ax.set_ylabel("Test ego-motion RMSE")
    ax.set_title("TartanAirV2 optic-flow test performance")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "tartanair_optic_flow_rmse.png")
    plt.close(fig)


def write_report(
    output_dir: Path,
    config: dict[str, object],
    summary: pd.DataFrame,
    split: SplitData,
) -> None:
    lines = [
        "# TartanAirV2 Optic-Flow Connectome Benchmark",
        "",
        "This run trains size-matched sparse recurrent models to decode local ego-motion from",
        "TartanAirV2 dense optical flow sampled onto a fly-like hexagonal retinal lattice.",
        "",
        "## Inputs And Targets",
        "",
        "- Inputs: TartanAir flow fields from `flow_lcam_front`, pooled over a hex lattice with acceptance blur.",
        "- Targets: pose-derived local yaw, forward translation, and lateral translation between the flow frame pair.",
        "- Training loss: normalized MSE over those three target components.",
        "",
        "## Models",
        "",
        "- `optic_lobe_seeded`: observed optic-lobe support and scaled connectome weights.",
        "- `random_weight_topology`: observed optic-lobe support with random Gaussian edge weights.",
        "- `shuffled_topology`: same neuron and edge count, randomized support, same weight multiset.",
        "- `random_sparse`: same neuron and edge count, randomized support, Gaussian random weights.",
        "",
        "All recurrent edge slots, input weights, recurrent biases, and readout weights are trainable.",
        "",
        "## Split",
        "",
        f"- train windows: {len(split.train)}",
        f"- val windows: {len(split.val)}",
        f"- test windows: {len(split.test)}",
        f"- target mean: {split.target_mean.tolist()}",
        f"- target std: {split.target_std.tolist()}",
        "",
        "## Summary",
        "",
        "```",
        summary.to_string(index=False) if not summary.empty else "No metrics available.",
        "```",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, indent=2, sort_keys=True),
        "```",
    ]
    (output_dir / "tartanair_optic_flow_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def download_tartanair(args: argparse.Namespace) -> None:
    try:
        import tartanair as ta  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Download/generate mode requires the optional TartanAirV2 toolkit. "
            "Install it with: python -m pip install tartanair"
        ) from exc
    args.tartanair_root.mkdir(parents=True, exist_ok=True)
    ta.init(str(args.tartanair_root))
    log_event(
        "tartanair-download-start "
        f"root={args.tartanair_root} envs={','.join(args.envs)} difficulties={','.join(args.tartanair_difficulties)} "
        f"modalities={','.join(args.download_modalities)} camera={args.camera_name}"
    )
    ta.download(
        env=list(args.envs),
        difficulty=list(args.tartanair_difficulties),
        modality=list(args.download_modalities),
        camera_name=[args.camera_name],
        unzip=True,
        delete_zip=args.delete_zip,
        num_workers=args.num_workers,
        data_source=args.data_source,
    )
    if args.generate_flow:
        log_event("tartanair-flow-generate-start")
        for env in args.envs:
            for difficulty in args.tartanair_difficulties:
                ta.customize_flow(
                    env=env,
                    difficulty=difficulty,
                    trajectory_id=list(args.trajectory_ids),
                    camera_name=[args.camera_name],
                    num_workers=args.num_workers,
                    frame_sep=args.frame_stride,
                    device=args.flow_device,
                )
        log_event("tartanair-flow-generate-done")
    log_event("tartanair-download-done")


def train_benchmark(
    matrix: sparse.csr_matrix,
    output_dir: Path,
    spec: TartanAirSpec,
    train_spec: TrainSpec,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = optic.select_device(train_spec.device)
    windows = enumerate_tartanair_windows(spec)
    split = split_windows(windows, spec)
    log_event(
        "run-start "
        f"output_dir={output_dir} device={device} N={matrix.shape[0]} edges={matrix.nnz} "
        f"input_dim={spec.input_dim} sequence_len={spec.sequence_len} models={','.join(train_spec.models)}"
    )
    metrics_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    for model_name in train_spec.models:
        for seed in train_spec.seeds:
            metrics, history = train_one_model(
                matrix,
                split,
                model_name,
                seed,
                spec,
                train_spec,
                device,
            )
            metrics_rows.append(metrics)
            history_rows.extend(history)
    metrics = pd.DataFrame(metrics_rows)
    history = pd.DataFrame(history_rows)
    summary = summarize_metrics(metrics)
    metrics.to_csv(output_dir / "tartanair_metrics_by_seed.csv", index=False)
    summary.to_csv(output_dir / "tartanair_metrics_summary.csv", index=False)
    history.to_csv(output_dir / "tartanair_loss_history.csv", index=False)
    config = {
        "tartanair_spec": {
            **asdict(spec),
            "data_root": str(spec.data_root),
            "envs": list(spec.envs),
            "difficulties": list(spec.difficulties),
        },
        "train_spec": {
            **asdict(train_spec),
            "seeds": list(train_spec.seeds),
            "models": list(train_spec.models),
        },
    }
    (output_dir / "tartanair_run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_plots(output_dir, metrics, history)
    write_report(output_dir, config, summary, split)
    write_artifact_manifest(
        output_dir,
        config=config,
        extra={"stage": "tartanair_optic_flow_training"},
    )
    log_event(
        f"complete metrics={output_dir / 'tartanair_metrics_by_seed.csv'} "
        f"summary={output_dir / 'tartanair_metrics_summary.csv'}"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate optic-lobe connectomic priors on TartanAirV2 optical flow."
    )
    parser.add_argument("--mode", choices=("download", "train", "all"), default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tartanair_optic_lobe_flow"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--matrix", type=Path, default=None, help="Prepared optic-lobe adjacency npz.")
    parser.add_argument("--tartanair-root", type=Path, required=True)
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["ArchVizTinyHouseDay", "AbandonedFactory"],
        help="TartanAirV2 environments to use.",
    )
    parser.add_argument(
        "--tartanair-difficulties",
        nargs="+",
        choices=("easy", "hard"),
        default=["easy"],
    )
    parser.add_argument("--camera-name", default="lcam_front")
    parser.add_argument("--download-modalities", nargs="+", default=["image", "depth"])
    parser.add_argument("--data-source", choices=("huggingface", "airlab"), default="huggingface")
    parser.add_argument("--delete-zip", action="store_true")
    parser.add_argument("--generate-flow", action="store_true")
    parser.add_argument("--flow-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--trajectory-ids", nargs="+", default=[])
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--sequence-len", type=int, default=8)
    parser.add_argument("--hex-rings", type=int, default=4)
    parser.add_argument("--acceptance-pixel-sigma", type=float, default=2.0)
    parser.add_argument("--acceptance-samples", type=int, default=7)
    parser.add_argument("--flow-clip", type=float, default=128.0)
    parser.add_argument("--target-yaw-scale", type=float, default=0.10)
    parser.add_argument("--target-translation-scale", type=float, default=0.25)
    parser.add_argument(
        "--pose-convention",
        choices=("camera_to_world", "world_to_camera"),
        default="camera_to_world",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=1729)
    parser.add_argument("--split-by-trajectory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)

    parser.add_argument("--models", nargs="+", choices=optic.MODEL_CHOICES, default=list(optic.DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-batches", type=int, default=120)
    parser.add_argument("--val-batches", type=int, default=30)
    parser.add_argument("--test-batches", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--state-clip", type=float, default=5.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--log-every-seconds", type=float, default=30.0)
    parser.add_argument("--flow-cache-size", type=int, default=256)
    parser.add_argument("--max-neurons", type=int, default=0, help="Optional top-activity cap for smoke tests.")
    args = parser.parse_args(argv)
    if args.cache_dir is None:
        args.cache_dir = args.output_dir
    if args.sequence_len < 1:
        parser.error("--sequence-len must be positive")
    if args.hex_rings < 0:
        parser.error("--hex-rings must be non-negative")
    if args.acceptance_samples < 1:
        parser.error("--acceptance-samples must be positive")
    if args.flow_clip <= 0:
        parser.error("--flow-clip must be positive")
    if args.train_fraction <= 0 or args.val_fraction <= 0 or args.train_fraction + args.val_fraction >= 1:
        parser.error("--train-fraction and --val-fraction must be positive and leave room for test")
    return args


def spec_from_args(args: argparse.Namespace) -> TartanAirSpec:
    return TartanAirSpec(
        data_root=args.tartanair_root.resolve(),
        envs=tuple(args.envs),
        difficulties=tuple(args.tartanair_difficulties),
        camera_name=args.camera_name,
        sequence_len=args.sequence_len,
        hex_rings=args.hex_rings,
        acceptance_pixel_sigma=args.acceptance_pixel_sigma,
        acceptance_samples=args.acceptance_samples,
        flow_clip=args.flow_clip,
        target_yaw_scale=args.target_yaw_scale,
        target_translation_scale=args.target_translation_scale,
        pose_convention=args.pose_convention,
        frame_stride=args.frame_stride,
        sample_stride=args.sample_stride,
        max_windows=args.max_windows,
        split_seed=args.split_seed,
        split_by_trajectory=args.split_by_trajectory,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )


def train_spec_from_args(args: argparse.Namespace) -> TrainSpec:
    return TrainSpec(
        seeds=tuple(args.seeds),
        models=tuple(args.models),
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        test_batches=args.test_batches,
        lr=args.lr,
        grad_clip=args.grad_clip,
        state_clip=args.state_clip,
        device=args.device,
        log_every_seconds=args.log_every_seconds,
        flow_cache_size=args.flow_cache_size,
    )


def matrix_for_training(args: argparse.Namespace) -> sparse.csr_matrix:
    if args.matrix is None:
        paths = OutputPaths(args.output_dir, args.cache_dir)
        path = paths.adjacency_unsigned_npz
    else:
        path = args.matrix
    return optic.maybe_truncate_matrix(optic.load_matrix(path), args.max_neurons)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.tartanair_root = args.tartanair_root.resolve()
    if args.mode in {"download", "all"}:
        download_tartanair(args)
    if args.mode in {"train", "all"}:
        matrix = matrix_for_training(args)
        train_benchmark(
            matrix=matrix,
            output_dir=args.output_dir,
            spec=spec_from_args(args),
            train_spec=train_spec_from_args(args),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
