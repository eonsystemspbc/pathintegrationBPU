#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
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
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.acquire import (  # noqa: E402
    _ensure_flywire_file,
    _flywire_filename,
    _read_flywire_connections,
    _write_flywire_connections,
    _write_flywire_neurons,
    _write_flywire_roi_counts,
)
from src.config import DEFAULT_FLYWIRE_RELEASE, OutputPaths, RHO_TARGET  # noqa: E402
from src.connectome import (  # noqa: E402
    assign_presynaptic_signs,
    build_raw_adjacency,
    build_signed_adjacency,
    choose_primary_matrix,
    random_control_matrix,
    scale_to_spectral_radius,
    sign_coverage,
)
from src.pools import assign_pools  # noqa: E402
from src.run_manifest import write_artifact_manifest  # noqa: E402


MODEL_OPTIC_LOBE = "optic_lobe_seeded"
MODEL_RANDOM_WEIGHT_TOPOLOGY = "random_weight_topology"
MODEL_SHUFFLED = "shuffled_topology"
MODEL_RANDOM = "random_sparse"
MODEL_CHOICES = (
    MODEL_OPTIC_LOBE,
    MODEL_RANDOM_WEIGHT_TOPOLOGY,
    MODEL_SHUFFLED,
    MODEL_RANDOM,
)
DEFAULT_MODELS = MODEL_CHOICES

DEFAULT_OPTIC_ROIS = (
    "LA_L",
    "LA_R",
    "ME_L",
    "ME_R",
    "LO_L",
    "LO_R",
    "LOP_L",
    "LOP_R",
    "AME_L",
    "AME_R",
)

DIFFICULTY_PRESETS = {
    "easy": {
        "contrast": 1.0,
        "sensor_noise_std": 0.02,
        "acceptance_angle_deg": 2.0,
        "blur_samples": 8,
        "train_batches": 80,
        "val_batches": 20,
        "test_batches": 40,
    },
    "medium": {
        "contrast": 0.65,
        "sensor_noise_std": 0.07,
        "acceptance_angle_deg": 4.0,
        "blur_samples": 6,
        "train_batches": 140,
        "val_batches": 30,
        "test_batches": 60,
    },
    "hard": {
        "contrast": 0.38,
        "sensor_noise_std": 0.14,
        "acceptance_angle_deg": 6.0,
        "blur_samples": 4,
        "train_batches": 220,
        "val_batches": 45,
        "test_batches": 80,
    },
}


@dataclass(frozen=True)
class OpticFlowSpec:
    hex_rings: int = 4
    timesteps: int = 16
    panorama_width: int = 256
    panorama_height: int = 96
    fov_azimuth_deg: float = 150.0
    fov_elevation_deg: float = 95.0
    acceptance_angle_deg: float = 4.0
    blur_samples: int = 6
    contrast: float = 0.65
    sensor_noise_std: float = 0.07
    texture_mode: str = "mixed"
    max_yaw_rate: float = 0.25
    max_forward: float = 0.55
    max_lateral: float = 0.35
    motion_scale: float = 0.55
    temporal_contrast_jitter: float = 0.08

    @property
    def input_dim(self) -> int:
        return int(1 + 3 * self.hex_rings * (self.hex_rings + 1))

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
    freeze_recurrent: bool
    recurrent_prior_l2: float
    device: str
    log_every_seconds: float


@dataclass(frozen=True)
class OpticBatch:
    inputs: np.ndarray
    targets: np.ndarray


def _format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, rem = divmod(minutes, 60)
    return f"{hours}h{rem:02d}m"


def log_event(message: str) -> None:
    print(message, flush=True)


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def hex_lattice(rings: int) -> np.ndarray:
    if rings < 0:
        raise ValueError("hex_rings must be non-negative.")
    points: list[tuple[float, float]] = []
    for q in range(-rings, rings + 1):
        r_min = max(-rings, -q - rings)
        r_max = min(rings, -q + rings)
        for r in range(r_min, r_max + 1):
            x = math.sqrt(3.0) * (q + r / 2.0)
            y = 1.5 * r
            points.append((x, y))
    arr = np.array(points, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    max_abs = float(np.max(np.abs(arr)))
    if max_abs > 0:
        arr = arr / max_abs
    return arr.astype(np.float32)


def lattice_angles(spec: OpticFlowSpec) -> np.ndarray:
    lattice = hex_lattice(spec.hex_rings)
    az = lattice[:, 0] * math.radians(spec.fov_azimuth_deg) / 2.0
    el = lattice[:, 1] * math.radians(spec.fov_elevation_deg) / 2.0
    return np.stack([az, el], axis=1).astype(np.float32)


def _smooth_noise(
    rng: np.random.Generator,
    height: int,
    width: int,
    harmonics: int = 8,
) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.linspace(-math.pi, math.pi, height, endpoint=False, dtype=np.float32),
        np.linspace(-math.pi, math.pi, width, endpoint=False, dtype=np.float32),
        indexing="ij",
    )
    image = np.zeros((height, width), dtype=np.float32)
    for _ in range(harmonics):
        fx = int(rng.integers(1, 8))
        fy = int(rng.integers(0, 5))
        phase = float(rng.uniform(-math.pi, math.pi))
        amp = float(rng.uniform(0.15, 1.0)) / math.sqrt(fx * fx + fy * fy + 1.0)
        image += amp * np.sin(fx * xx + fy * yy + phase).astype(np.float32)
    image += 0.25 * rng.normal(size=(height, width)).astype(np.float32)
    image -= float(image.mean())
    image /= float(image.std() + 1e-6)
    return image.astype(np.float32)


def make_panorama(rng: np.random.Generator, spec: OpticFlowSpec) -> np.ndarray:
    pano = _smooth_noise(rng, spec.panorama_height, spec.panorama_width)
    if spec.texture_mode in {"naturalistic", "mixed"}:
        yy = np.linspace(-1.0, 1.0, spec.panorama_height, dtype=np.float32)[:, None]
        horizon = np.exp(-(yy / 0.28) ** 2).astype(np.float32)
        vertical_grad = 0.35 * yy
        pano = 0.75 * pano + horizon * rng.normal(0.0, 0.45, size=pano.shape).astype(np.float32)
        pano = pano + vertical_grad
    if spec.texture_mode in {"checker", "mixed"} and rng.random() < 0.35:
        tile = int(rng.integers(6, 18))
        yy, xx = np.indices((spec.panorama_height, spec.panorama_width))
        checker = (((xx // tile) + (yy // tile)) % 2).astype(np.float32) * 2.0 - 1.0
        pano = 0.7 * pano + 0.3 * checker
    pano -= float(pano.mean())
    pano /= float(pano.std() + 1e-6)
    pano = 0.5 + 0.5 * spec.contrast * pano
    return np.clip(pano, 0.0, 1.0).astype(np.float32)


def _sample_periodic_bilinear(pano: np.ndarray, az: np.ndarray, el: np.ndarray) -> np.ndarray:
    height, width = pano.shape
    x = ((az + math.pi) / (2.0 * math.pi) * width) % width
    y = np.clip((el + math.pi / 2.0) / math.pi * (height - 1), 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64) % width
    x1 = (x0 + 1) % width
    y0 = np.floor(y).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    top = (1.0 - wx) * pano[y0, x0] + wx * pano[y0, x1]
    bottom = (1.0 - wx) * pano[y1, x0] + wx * pano[y1, x1]
    return ((1.0 - wy) * top + wy * bottom).astype(np.float32)


def _blur_offsets(
    rng: np.random.Generator,
    sample_count: int,
    acceptance_angle_rad: float,
) -> np.ndarray:
    if sample_count <= 1 or acceptance_angle_rad <= 0:
        return np.zeros((1, 2), dtype=np.float32)
    sigma = acceptance_angle_rad / 2.0
    offsets = rng.normal(0.0, sigma, size=(sample_count, 2)).astype(np.float32)
    offsets[0] = 0.0
    return offsets


def render_hex_frame(
    pano: np.ndarray,
    base_angles: np.ndarray,
    yaw: float,
    forward: float,
    lateral: float,
    spec: OpticFlowSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    az0 = base_angles[:, 0]
    el0 = base_angles[:, 1]
    acceptance = math.radians(spec.acceptance_angle_deg)
    offsets = _blur_offsets(rng, spec.blur_samples, acceptance)
    samples: list[np.ndarray] = []
    for off_az, off_el in offsets:
        az = az0 + float(off_az)
        el = el0 + float(off_el)
        depth_gain = 0.65 + 0.35 * np.cos(el)
        trans_az = spec.motion_scale * (
            forward * np.sin(az) - lateral * np.cos(az)
        ) * depth_gain
        trans_el = spec.motion_scale * forward * np.sin(el) * np.cos(az)
        samples.append(_sample_periodic_bilinear(pano, az + yaw + trans_az, el + trans_el))
    frame = np.mean(samples, axis=0).astype(np.float32)
    return frame


def generate_optic_flow_batch(
    spec: OpticFlowSpec,
    batch_size: int,
    rng: np.random.Generator,
) -> OpticBatch:
    base_angles = lattice_angles(spec)
    inputs = np.zeros((batch_size, spec.timesteps, spec.input_dim), dtype=np.float32)
    targets = np.zeros((batch_size, spec.timesteps, spec.output_dim), dtype=np.float32)
    for b in range(batch_size):
        pano = make_panorama(rng, spec)
        yaw_rate = float(rng.uniform(-spec.max_yaw_rate, spec.max_yaw_rate))
        forward = float(rng.uniform(-spec.max_forward, spec.max_forward))
        lateral = float(rng.uniform(-spec.max_lateral, spec.max_lateral))
        if abs(forward) + abs(lateral) < 0.08 and abs(yaw_rate) < 0.04:
            forward += 0.12 * np.sign(rng.normal() or 1.0)
        phase = float(rng.uniform(-math.pi, math.pi))
        for t in range(spec.timesteps):
            frac = t / max(spec.timesteps - 1, 1)
            jitter = 1.0 + spec.temporal_contrast_jitter * math.sin(2.0 * math.pi * frac + phase)
            yaw = yaw_rate * t
            frame = render_hex_frame(
                pano,
                base_angles,
                yaw=yaw,
                forward=forward * t,
                lateral=lateral * t,
                spec=spec,
                rng=rng,
            )
            frame = 0.5 + jitter * (frame - 0.5)
            if spec.sensor_noise_std > 0:
                frame = frame + rng.normal(0.0, spec.sensor_noise_std, size=frame.shape).astype(
                    np.float32
                )
            inputs[b, t] = np.clip(frame, 0.0, 1.0)
            targets[b, t] = np.array([yaw_rate, forward, lateral], dtype=np.float32)
    return OpticBatch(inputs=inputs, targets=targets)


class SparseOpticFlowRNN(nn.Module):
    def __init__(
        self,
        recurrent: sparse.spmatrix,
        input_dim: int,
        output_dim: int,
        state_clip: float,
        seed: int,
        freeze_recurrent: bool = False,
    ) -> None:
        super().__init__()
        recurrent = recurrent.astype(np.float32).tocoo()
        recurrent.sum_duplicates()
        if recurrent.shape[0] != recurrent.shape[1]:
            raise ValueError("recurrent matrix must be square.")
        self.N = int(recurrent.shape[0])
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
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
        indices = np.vstack([recurrent.row, recurrent.col]).astype(np.int64)
        self.register_buffer("edge_indices", torch.from_numpy(indices))
        values = recurrent.data.astype(np.float32)
        self.W_rec_values = nn.Parameter(torch.from_numpy(values))
        self.register_buffer("W_rec_initial_values", torch.from_numpy(values.copy()))
        if freeze_recurrent:
            self.W_rec_values.requires_grad_(False)

    def recurrent_parameter_count(self) -> int:
        return int(self.W_rec_values.numel())

    def trainable_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))

    def recurrent_prior_loss(self) -> torch.Tensor:
        return nn.functional.mse_loss(self.W_rec_values, self.W_rec_initial_values)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"inputs must have shape [batch, T, {self.input_dim}], got {tuple(inputs.shape)}"
            )
        batch, T, _ = inputs.shape
        h = inputs.new_zeros((batch, self.N))
        W = torch.sparse_coo_tensor(
            self.edge_indices,
            self.W_rec_values,
            size=(self.N, self.N),
            device=inputs.device,
        ).coalesce()
        outputs: list[torch.Tensor] = []
        for t in range(T):
            h = torch.sparse.mm(W, h.t()).t() + inputs[:, t, :] @ self.W_in.t() + self.b_rec
            h = torch.relu(h)
            if self.state_clip > 0:
                h = torch.clamp(h, max=self.state_clip)
            outputs.append(self.readout(h))
        return torch.stack(outputs, dim=1)


def random_sparse_normal_like(matrix: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    rows, cols, weights = _matrix_triplets(matrix)
    n = matrix.shape[0]
    rng = np.random.default_rng(seed)
    self_count = int(np.sum(rows == cols))
    self_nodes = rng.choice(n, size=self_count, replace=False) if self_count else np.array([], dtype=int)
    target_nonself = len(weights) - self_count
    total_nonself = n * (n - 1)
    if target_nonself > total_nonself:
        raise ValueError("Cannot sample requested non-self edge count.")
    selected: set[int] = set()
    chunks: list[np.ndarray] = []
    while len(selected) < target_nonself:
        remaining = target_nonself - len(selected)
        draw_count = max(4096, remaining * 2)
        codes = rng.integers(0, total_nonself, size=draw_count, dtype=np.int64)
        keep: list[int] = []
        for code in codes:
            value = int(code)
            if value not in selected:
                selected.add(value)
                keep.append(value)
                if len(selected) == target_nonself:
                    break
        if keep:
            chunks.append(np.asarray(keep, dtype=np.int64))
    codes = np.concatenate(chunks) if chunks else np.array([], dtype=np.int64)
    nonself_posts = (codes // (n - 1)).astype(np.int64)
    nonself_pres = (codes % (n - 1)).astype(np.int64)
    nonself_pres = nonself_pres + (nonself_pres >= nonself_posts)
    mean_abs = float(np.mean(np.abs(weights))) if len(weights) else 1.0
    random_weights = rng.normal(0.0, mean_abs, size=len(weights)).astype(np.float32)
    out_rows = np.concatenate([self_nodes.astype(np.int64), nonself_posts])
    out_cols = np.concatenate([self_nodes.astype(np.int64), nonself_pres])
    return sparse.coo_matrix((random_weights, (out_rows, out_cols)), shape=matrix.shape).tocsr()


def random_weight_same_topology(matrix: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    rows, cols, weights = _matrix_triplets(matrix)
    rng = np.random.default_rng(seed)
    mean_abs = float(np.mean(np.abs(weights))) if len(weights) else 1.0
    random_weights = rng.normal(0.0, mean_abs, size=len(weights)).astype(np.float32)
    return sparse.coo_matrix((random_weights, (rows, cols)), shape=matrix.shape).tocsr()


def _matrix_triplets(matrix: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coo = matrix.tocoo()
    return coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data.astype(np.float32)


def model_matrix(base: sparse.csr_matrix, model_name: str, seed: int) -> sparse.csr_matrix:
    if model_name == MODEL_OPTIC_LOBE:
        return base.copy().astype(np.float32).tocsr()
    if model_name == MODEL_RANDOM_WEIGHT_TOPOLOGY:
        return random_weight_same_topology(base, seed).astype(np.float32).tocsr()
    if model_name == MODEL_SHUFFLED:
        return random_control_matrix(base, seed).astype(np.float32).tocsr()
    if model_name == MODEL_RANDOM:
        return random_sparse_normal_like(base, seed).astype(np.float32).tocsr()
    raise ValueError(f"Unknown model: {model_name}")


def load_matrix(path: Path) -> sparse.csr_matrix:
    start = time.monotonic()
    log_event(f"matrix-load-start path={path}")
    matrix = sparse.load_npz(path).astype(np.float32).tocsr()
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square: {matrix.shape}")
    if matrix.nnz == 0:
        raise ValueError("matrix has no edges.")
    log_event(
        f"matrix-load-done path={path} N={matrix.shape[0]} edges={matrix.nnz} elapsed={_format_seconds(time.monotonic() - start)}"
    )
    return matrix


def maybe_truncate_matrix(matrix: sparse.csr_matrix, max_neurons: int) -> sparse.csr_matrix:
    if max_neurons <= 0 or matrix.shape[0] <= max_neurons:
        log_event(f"matrix-ready N={matrix.shape[0]} edges={matrix.nnz} max_neurons={max_neurons}")
        return matrix
    log_event(
        f"matrix-truncate-start N={matrix.shape[0]} edges={matrix.nnz} max_neurons={max_neurons}"
    )
    activity = np.asarray(matrix.sum(axis=0)).ravel() + np.asarray(matrix.sum(axis=1)).ravel()
    keep = np.sort(np.argsort(activity)[-max_neurons:])
    truncated = matrix[keep][:, keep].tocsr()
    log_event(f"matrix-truncate-done N={truncated.shape[0]} edges={truncated.nnz}")
    return truncated


def download_flywire_optic_lobe_exports(
    paths: OutputPaths,
    release: str,
    download_dir: Path | None,
    optic_rois: tuple[str, ...],
    max_neurons: int = 0,
) -> dict[str, object]:
    download_dir = (
        Path(download_dir)
        if download_dir is not None
        else paths.cache_dir / f"flywire_release_{release}"
    )
    download_dir.mkdir(parents=True, exist_ok=True)
    log_event(
        "download-start "
        f"connectome=flywire_optic_lobe release={release} download_dir={download_dir} "
        f"optic_rois={','.join(optic_rois)} max_neurons={max_neurons}"
    )
    root_ids_path = _ensure_flywire_file(
        download_dir, _flywire_filename("proofread_root_ids", release)
    )
    log_event(f"download-file-ready kind=proofread_root_ids path={root_ids_path}")
    connections_path = _ensure_flywire_file(
        download_dir, _flywire_filename("proofread_connections", release)
    )
    log_event(f"download-file-ready kind=proofread_connections path={connections_path}")
    start = time.monotonic()
    log_event("flywire-read-start file=proofread_connections")
    connections = _read_flywire_connections(connections_path)
    log_event(
        f"flywire-read-done rows={len(connections)} elapsed={_format_seconds(time.monotonic() - start)}"
    )
    if "neuropil" not in connections.columns:
        raise RuntimeError("FlyWire proofread connections do not include neuropil labels.")
    optic_connections = connections[connections["neuropil"].isin(optic_rois)].copy()
    if optic_connections.empty:
        raise RuntimeError(f"No FlyWire connections found in optic-lobe ROIs: {optic_rois}")
    rois = sorted(map(str, optic_connections["neuropil"].dropna().unique()))
    log_event(f"optic-filter-done optic_rows={len(optic_connections)} rois={','.join(rois)}")
    body_ids = pd.Index(
        pd.concat(
            [optic_connections["pre_pt_root_id"], optic_connections["post_pt_root_id"]],
            ignore_index=True,
        )
        .dropna()
        .astype("int64")
        .unique()
    )
    log_event(f"optic-bodyids-ready count={len(body_ids)}")
    if max_neurons > 0 and len(body_ids) > max_neurons:
        log_event(f"optic-bodyids-truncate-start count={len(body_ids)} max_neurons={max_neurons}")
        pre = optic_connections.groupby("pre_pt_root_id")["syn_count"].sum()
        post = optic_connections.groupby("post_pt_root_id")["syn_count"].sum()
        activity = (
            pd.DataFrame({"bodyId": body_ids})
            .merge(pre.rename("pre"), how="left", left_on="bodyId", right_index=True)
            .merge(post.rename("post"), how="left", left_on="bodyId", right_index=True)
            .fillna(0.0)
        )
        activity["total"] = activity["pre"] + activity["post"]
        body_ids = pd.Index(
            activity.sort_values("total", ascending=False)["bodyId"].head(max_neurons).astype("int64")
        )
        optic_connections = optic_connections[
            optic_connections["pre_pt_root_id"].isin(body_ids)
            & optic_connections["post_pt_root_id"].isin(body_ids)
        ].copy()
        log_event(
            f"optic-bodyids-truncate-done count={len(body_ids)} optic_rows={len(optic_connections)}"
        )
    selected_global = connections[
        connections["pre_pt_root_id"].isin(body_ids)
        | connections["post_pt_root_id"].isin(body_ids)
    ].copy()
    log_event(
        f"optic-export-write-start selected_global_rows={len(selected_global)} optic_rows={len(optic_connections)}"
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    neurons = _write_flywire_neurons(
        paths, root_ids_path, selected_global, body_ids_subset=body_ids
    )
    _write_flywire_roi_counts(paths, selected_global)
    aggregated = _write_flywire_connections(paths, optic_connections)
    log_event(
        f"optic-export-write-done neurons={len(neurons)} aggregated_edges={len(aggregated)} output_dir={paths.output_dir}"
    )
    source_metadata = {
        "connectome": "flywire_optic_lobe",
        "release": release,
        "optic_rois": list(optic_rois),
        "root_ids_path": str(root_ids_path),
        "proofread_connections_path": str(connections_path),
        "source_rows": int(len(connections)),
        "optic_source_rows": int(len(optic_connections)),
        "aggregated_edge_count": int(len(aggregated)),
        "max_neurons": int(max_neurons),
    }
    with (paths.output_dir / "flywire_sources.json").open("w", encoding="utf-8") as f:
        json.dump(source_metadata, f, indent=2, sort_keys=True)
    return {
        "primary_rois": optic_rois,
        "neuron_count": int(len(neurons)),
        "edge_count": int(len(aggregated)),
        "flywire_sources_path": str(paths.output_dir / "flywire_sources.json"),
        "download_dir": str(download_dir),
    }


def prepare_optic_lobe_connectome(
    paths: OutputPaths,
    optic_rois: tuple[str, ...],
    signed_policy: str = "auto",
) -> sparse.csr_matrix:
    start = time.monotonic()
    log_event(
        f"prepare-start connectome=flywire_optic_lobe output_dir={paths.output_dir} signed_policy={signed_policy}"
    )
    neurons = pd.read_csv(paths.neurons_csv)
    roi_counts = pd.read_csv(paths.roi_counts_csv)
    connections = pd.read_csv(paths.connections_csv)
    log_event(
        f"prepare-read-done neurons={len(neurons)} roi_rows={len(roi_counts)} connection_rows={len(connections)}"
    )
    adjacency_start = time.monotonic()
    unsigned_raw, body_to_index, aggregated_edges = build_raw_adjacency(neurons, connections)
    log_event(
        "prepare-adjacency-built "
        f"N={unsigned_raw.shape[0]} raw_edges={len(aggregated_edges)} nnz={unsigned_raw.nnz} "
        f"elapsed={_format_seconds(time.monotonic() - adjacency_start)}"
    )
    signs = assign_presynaptic_signs(neurons, body_to_index)
    signed_raw = build_signed_adjacency(unsigned_raw, signs)
    coverage = sign_coverage(unsigned_raw, signs)
    primary_name = choose_primary_matrix(signed_policy, coverage, signed_raw)
    log_event(
        f"prepare-signs-done signed_neurons={len(signs)} sign_coverage={coverage:.4f} primary_matrix={primary_name}"
    )
    scale_start = time.monotonic()
    unsigned, signed, raw_rho, scale = scale_to_spectral_radius(
        unsigned_raw,
        signed_raw,
        primary_name=primary_name,
        rho_target=RHO_TARGET,
    )
    log_event(
        f"prepare-scale-done raw_rho={raw_rho:.6g} scale={scale:.6g} elapsed={_format_seconds(time.monotonic() - scale_start)}"
    )
    assignments = assign_pools(neurons, roi_counts, primary_rois=optic_rois)
    assignments.to_csv(paths.pool_assignments_csv, index=False)
    log_event(f"prepare-pools-done pool_counts={assignments['pool'].value_counts().to_dict()}")
    primary = signed if primary_name == "signed" else unsigned
    if primary is None:
        raise RuntimeError("Primary optic-lobe adjacency could not be constructed.")
    body_ids = [int(x) for x in neurons["bodyId"].astype("int64").drop_duplicates().sort_values()]
    metadata = {
        "connectome": "flywire_optic_lobe",
        "primary_rois": list(optic_rois),
        "N": int(primary.shape[0]),
        "body_ids": body_ids,
        "orientation": "W_rec[post_index, pre_index]",
        "unsigned_edge_count": int(unsigned.nnz),
        "raw_edge_count": int(len(aggregated_edges)),
        "self_loop_count": int(np.sum(unsigned.tocoo().row == unsigned.tocoo().col)),
        "signed_edge_count": int(signed.nnz) if signed is not None else 0,
        "signed_presynaptic_neuron_count": int(len(signs)),
        "sign_coverage": float(coverage),
        "signed_policy": signed_policy,
        "primary_matrix": primary_name,
        "raw_primary_spectral_radius": float(raw_rho),
        "spectral_scale": float(scale),
        "rho_target": float(RHO_TARGET),
        "pool_counts": assignments["pool"].value_counts().to_dict(),
    }
    sparse.save_npz(paths.adjacency_unsigned_npz, unsigned)
    if signed is not None:
        sparse.save_npz(paths.adjacency_signed_npz, signed)
    with paths.graph_metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    log_event(
        "prepare-done "
        f"N={primary.shape[0]} edges={primary.nnz} primary_matrix={primary_name} "
        f"adjacency={paths.adjacency_unsigned_npz} elapsed={_format_seconds(time.monotonic() - start)}"
    )
    return primary.astype(np.float32).tocsr()


def _batch_to_torch(batch: OpticBatch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(batch.inputs).to(device=device, dtype=torch.float32),
        torch.from_numpy(batch.targets).to(device=device, dtype=torch.float32),
    )


def evaluate_model(
    model: SparseOpticFlowRNN,
    spec: OpticFlowSpec,
    device: torch.device,
    seed: int,
    batches: int,
    batch_size: int,
    phase: str = "eval",
    model_name: str = "",
    log_every_seconds: float = 0.0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    model.eval()
    losses: list[float] = []
    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    start = time.monotonic()
    last_log = start
    data_gen_seconds = 0.0
    eval_seconds = 0.0
    log_event(
        "data-gen-start "
        f"phase={phase} model={model_name or 'unknown'} batches={batches} batch_size={batch_size} "
        f"timesteps={spec.timesteps} input_dim={spec.input_dim} seed={seed}"
    )
    with torch.no_grad():
        for batch_idx in range(1, batches + 1):
            gen_start = time.monotonic()
            batch = generate_optic_flow_batch(spec, batch_size, rng)
            data_gen_seconds += time.monotonic() - gen_start
            x, y = _batch_to_torch(batch, device)
            eval_start = time.monotonic()
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            eval_seconds += time.monotonic() - eval_start
            losses.append(float(loss.detach().cpu()))
            preds_all.append(pred.detach().cpu().numpy())
            targets_all.append(y.detach().cpu().numpy())
            now = time.monotonic()
            if log_every_seconds > 0 and now - last_log >= log_every_seconds:
                log_event(
                    "data-gen-progress "
                    f"phase={phase} model={model_name or 'unknown'} batch={batch_idx}/{batches} "
                    f"running_loss={np.mean(losses):.6g} data_gen_elapsed={_format_seconds(data_gen_seconds)} "
                    f"eval_elapsed={_format_seconds(eval_seconds)} elapsed={_format_seconds(now - start)}"
                )
                last_log = now
    pred_np = np.concatenate(preds_all, axis=0)
    target_np = np.concatenate(targets_all, axis=0)
    err = pred_np - target_np
    component_rmse = np.sqrt(np.mean(err**2, axis=(0, 1)))
    target_var = np.var(target_np.reshape(-1, spec.output_dim), axis=0) + 1e-8
    r2 = 1.0 - np.mean(err.reshape(-1, spec.output_dim) ** 2, axis=0) / target_var
    translation_err = err[..., 1:3]
    translation_rmse = float(np.sqrt(np.mean(translation_err**2)))
    metrics = {
        "loss": float(np.mean(losses)),
        "overall_rmse": float(np.sqrt(np.mean(err**2))),
        "yaw_rmse": float(component_rmse[0]),
        "forward_rmse": float(component_rmse[1]),
        "lateral_rmse": float(component_rmse[2]),
        "translation_rmse": translation_rmse,
        "yaw_r2": float(r2[0]),
        "forward_r2": float(r2[1]),
        "lateral_r2": float(r2[2]),
    }
    log_event(
        "data-gen-done "
        f"phase={phase} model={model_name or 'unknown'} loss={metrics['loss']:.6g} "
        f"data_gen_elapsed={_format_seconds(data_gen_seconds)} eval_elapsed={_format_seconds(eval_seconds)} "
        f"elapsed={_format_seconds(time.monotonic() - start)}"
    )
    return metrics


def train_one_model(
    base_matrix: sparse.csr_matrix,
    model_name: str,
    seed: int,
    spec: OpticFlowSpec,
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
    recurrent = model_matrix(base_matrix, model_name, seed + 10_000)
    log_event(
        f"model-build-matrix-done model={model_name} seed={seed} edges={recurrent.nnz} elapsed={_format_seconds(time.monotonic() - build_start)}"
    )
    model = SparseOpticFlowRNN(
        recurrent=recurrent,
        input_dim=spec.input_dim,
        output_dim=spec.output_dim,
        state_clip=train_spec.state_clip,
        seed=seed + 1_000,
        freeze_recurrent=train_spec.freeze_recurrent,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_spec.lr)
    rng = np.random.default_rng(seed + 12345)
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    history: list[dict[str, object]] = []
    print(
        "model-start "
        f"model={model_name} seed={seed} N={model.N} edges={recurrent.nnz} "
        f"trainable_params={model.trainable_parameter_count()} recurrent_params={model.recurrent_parameter_count()}",
        flush=True,
    )
    for epoch in range(1, train_spec.epochs + 1):
        model.train()
        losses: list[float] = []
        prior_losses: list[float] = []
        start = time.monotonic()
        last_log = start
        data_gen_seconds = 0.0
        train_step_seconds = 0.0
        log_event(
            "epoch-start "
            f"model={model_name} seed={seed} epoch={epoch}/{train_spec.epochs} "
            f"synthetic_train_batches={train_spec.train_batches} batch_size={train_spec.batch_size} "
            f"timesteps={spec.timesteps} input_dim={spec.input_dim}"
        )
        for batch_idx in range(1, train_spec.train_batches + 1):
            gen_start = time.monotonic()
            batch = generate_optic_flow_batch(spec, train_spec.batch_size, rng)
            data_gen_seconds += time.monotonic() - gen_start
            x, y = _batch_to_torch(batch, device)
            step_start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            task_loss = torch.mean((pred - y) ** 2)
            prior_loss = task_loss.new_tensor(0.0)
            if train_spec.recurrent_prior_l2 > 0:
                prior_loss = model.recurrent_prior_loss()
            loss = task_loss + train_spec.recurrent_prior_l2 * prior_loss
            loss.backward()
            if train_spec.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_spec.grad_clip)
            optimizer.step()
            train_step_seconds += time.monotonic() - step_start
            losses.append(float(task_loss.detach().cpu()))
            prior_losses.append(float(prior_loss.detach().cpu()))
            now = time.monotonic()
            if now - last_log >= train_spec.log_every_seconds:
                print(
                    "progress "
                    f"model={model_name} seed={seed} epoch={epoch}/{train_spec.epochs} "
                    f"batch={batch_idx}/{train_spec.train_batches} batch_loss={losses[-1]:.6g} "
                    f"prior_loss={prior_losses[-1]:.6g} "
                    f"running_train_loss={np.mean(losses):.6g} "
                    f"data_gen_elapsed={_format_seconds(data_gen_seconds)} "
                    f"train_step_elapsed={_format_seconds(train_step_seconds)} "
                    f"elapsed={_format_seconds(now - start)}",
                    flush=True,
                )
                last_log = now
        val = evaluate_model(
            model,
            spec,
            device,
            seed=seed + 50_000 + epoch,
            batches=train_spec.val_batches,
            batch_size=train_spec.batch_size,
            phase=f"val_epoch_{epoch}",
            model_name=model_name,
            log_every_seconds=train_spec.log_every_seconds,
        )
        train_loss = float(np.mean(losses))
        train_prior_loss = float(np.mean(prior_losses)) if prior_losses else 0.0
        if val["loss"] < best_val:
            best_val = float(val["loss"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        row = {
            "model": model_name,
            "seed": int(seed),
            "epoch": int(epoch),
            "train_loss": train_loss,
            "val_loss": float(val["loss"]),
            "val_overall_rmse": float(val["overall_rmse"]),
            "val_yaw_rmse": float(val["yaw_rmse"]),
            "val_translation_rmse": float(val["translation_rmse"]),
            "val_yaw_r2": float(val["yaw_r2"]),
            "best_val_loss": float(best_val),
            "recurrent_prior_loss": train_prior_loss,
            "patience_wait": int(wait),
            "train_data_gen_seconds": float(data_gen_seconds),
            "train_step_seconds": float(train_step_seconds),
        }
        history.append(row)
        print(
            "loss "
            f"model={model_name} seed={seed} epoch={epoch}/{train_spec.epochs} "
            f"train_loss={train_loss:.6g} val_loss={val['loss']:.6g} "
            f"prior_loss={train_prior_loss:.6g} "
            f"val_yaw_rmse={val['yaw_rmse']:.6g} val_translation_rmse={val['translation_rmse']:.6g} "
            f"data_gen_elapsed={_format_seconds(data_gen_seconds)} "
            f"train_step_elapsed={_format_seconds(train_step_seconds)} "
            f"best_val_loss={best_val:.6g} patience_wait={wait}",
            flush=True,
        )
        if train_spec.patience > 0 and wait >= train_spec.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate_model(
        model,
        spec,
        device,
        seed=seed + 90_000,
        batches=train_spec.test_batches,
        batch_size=train_spec.batch_size,
        phase="test",
        model_name=model_name,
        log_every_seconds=train_spec.log_every_seconds,
    )
    metrics = {
        "model": model_name,
        "seed": int(seed),
        "N": int(model.N),
        "init_nonzero_edges": int(recurrent.nnz),
        "recurrent_params": int(model.recurrent_parameter_count()),
        "trainable_params": int(model.trainable_parameter_count()),
        "freeze_recurrent": int(bool(train_spec.freeze_recurrent)),
        "recurrent_prior_l2": float(train_spec.recurrent_prior_l2),
        "best_val_loss": float(best_val),
        "test_loss": float(test["loss"]),
        "test_overall_rmse": float(test["overall_rmse"]),
        "test_yaw_rmse": float(test["yaw_rmse"]),
        "test_forward_rmse": float(test["forward_rmse"]),
        "test_lateral_rmse": float(test["lateral_rmse"]),
        "test_translation_rmse": float(test["translation_rmse"]),
        "test_yaw_r2": float(test["yaw_r2"]),
        "test_forward_r2": float(test["forward_r2"]),
        "test_lateral_r2": float(test["lateral_r2"]),
    }
    print(
        "model-done "
        f"model={model_name} seed={seed} best_val_loss={best_val:.6g} "
        f"test_overall_rmse={test['overall_rmse']:.6g} test_yaw_rmse={test['yaw_rmse']:.6g}",
        flush=True,
    )
    return metrics, history


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("model", as_index=False)
        .agg(
            best_val_loss_mean=("best_val_loss", "mean"),
            best_val_loss_std=("best_val_loss", "std"),
            test_overall_rmse_mean=("test_overall_rmse", "mean"),
            test_overall_rmse_std=("test_overall_rmse", "std"),
            test_yaw_rmse_mean=("test_yaw_rmse", "mean"),
            test_translation_rmse_mean=("test_translation_rmse", "mean"),
            test_yaw_r2_mean=("test_yaw_r2", "mean"),
            trainable_params=("trainable_params", "first"),
            recurrent_params=("recurrent_params", "first"),
            N=("N", "first"),
            freeze_recurrent=("freeze_recurrent", "first"),
            recurrent_prior_l2=("recurrent_prior_l2", "first"),
        )
        .sort_values("test_overall_rmse_mean")
    )


def write_plots(output_dir: Path, metrics: pd.DataFrame, history: pd.DataFrame) -> None:
    if not history.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
        for model, group in history.groupby("model"):
            by_epoch = group.groupby("epoch", as_index=False).agg(
                val_loss=("val_loss", "mean"),
                val_sem=("val_loss", lambda x: x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0),
            )
            ax.plot(by_epoch["epoch"], by_epoch["val_loss"], label=model)
            ax.fill_between(
                by_epoch["epoch"],
                by_epoch["val_loss"] - by_epoch["val_sem"],
                by_epoch["val_loss"] + by_epoch["val_sem"],
                alpha=0.15,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE")
        ax.set_title("Optic-flow learning curves")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "optic_flow_loss.png")
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
    ax.set_ylabel("Test optic-flow RMSE")
    ax.set_title("Optic-flow test performance")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "optic_flow_rmse.png")
    plt.close(fig)


def write_report(output_dir: Path, config: dict[str, object], summary: pd.DataFrame) -> None:
    lines = [
        "# Optic-Flow Connectome Benchmark",
        "",
        "This run trains size-matched sparse recurrent models on synthetic optic-flow decoding.",
        "Inputs are hex-lattice samples from procedural panoramas with acceptance-angle blur.",
        "Targets are known ego-motion components: yaw rate, forward translation, and lateral translation.",
        "",
        "## Models",
        "",
        "- `optic_lobe_seeded`: observed optic-lobe support and scaled connectome weights.",
        "- `random_weight_topology`: observed optic-lobe support with random Gaussian edge weights.",
        "- `shuffled_topology`: same neuron and edge count, randomized support, same weight multiset.",
        "- `random_sparse`: same neuron and edge count, randomized support, Gaussian random weights.",
        "",
        (
            "Recurrent edge slots are trainable unless `freeze_recurrent` is true; "
            "`recurrent_prior_l2` penalizes drift from the initialized connectome/control weights."
        ),
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
    (output_dir / "optic_flow_report.md").write_text("\n".join(lines), encoding="utf-8")


def train_benchmark(
    matrix: sparse.csr_matrix,
    output_dir: Path,
    optic_spec: OpticFlowSpec,
    train_spec: TrainSpec,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(train_spec.device)
    print(
        "run-start "
        f"output_dir={output_dir} device={device} N={matrix.shape[0]} edges={matrix.nnz} "
        f"input_dim={optic_spec.input_dim} timesteps={optic_spec.timesteps} models={','.join(train_spec.models)}",
        flush=True,
    )
    log_event(
        "data-config "
        f"hex_rings={optic_spec.hex_rings} samples={optic_spec.input_dim} "
        f"panorama={optic_spec.panorama_width}x{optic_spec.panorama_height} "
        f"acceptance_angle_deg={optic_spec.acceptance_angle_deg} blur_samples={optic_spec.blur_samples} "
        f"contrast={optic_spec.contrast} sensor_noise_std={optic_spec.sensor_noise_std} "
        f"train_batches={train_spec.train_batches} val_batches={train_spec.val_batches} "
        f"test_batches={train_spec.test_batches} batch_size={train_spec.batch_size}"
    )
    metrics_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    for model_name in train_spec.models:
        for seed in train_spec.seeds:
            metrics, history = train_one_model(
                matrix,
                model_name,
                seed,
                optic_spec,
                train_spec,
                device,
            )
            metrics_rows.append(metrics)
            history_rows.extend(history)
    metrics = pd.DataFrame(metrics_rows)
    history = pd.DataFrame(history_rows)
    summary = summarize_metrics(metrics)
    metrics.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    history.to_csv(output_dir / "loss_history.csv", index=False)
    config = {
        "optic_spec": asdict(optic_spec),
        "train_spec": {
            **asdict(train_spec),
            "seeds": list(train_spec.seeds),
            "models": list(train_spec.models),
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_plots(output_dir, metrics, history)
    write_report(output_dir, config, summary)
    write_artifact_manifest(
        output_dir,
        config=config,
        extra={"stage": "optic_flow_training"},
    )
    print(
        f"complete metrics={output_dir / 'metrics_by_seed.csv'} "
        f"summary={output_dir / 'metrics_summary.csv'}",
        flush=True,
    )


def apply_difficulty(args: argparse.Namespace) -> argparse.Namespace:
    preset = DIFFICULTY_PRESETS[args.difficulty]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fly optic-lobe optic-flow benchmark.")
    parser.add_argument("--mode", choices=("download", "prepare", "train", "all"), default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/flywire_optic_lobe_flow"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--matrix", type=Path, default=None, help="Prepared adjacency npz for train mode.")
    parser.add_argument("--flywire-release", default=DEFAULT_FLYWIRE_RELEASE)
    parser.add_argument("--flywire-download-dir", type=Path, default=None)
    parser.add_argument("--optic-rois", nargs="+", default=list(DEFAULT_OPTIC_ROIS))
    parser.add_argument("--signed-policy", choices=("auto", "force_unsigned", "force_signed"), default="auto")
    parser.add_argument("--max-neurons", type=int, default=0, help="Optional top-activity cap for AWS smoke runs.")

    parser.add_argument("--difficulty", choices=tuple(DIFFICULTY_PRESETS), default="medium")
    parser.add_argument("--hex-rings", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--panorama-width", type=int, default=256)
    parser.add_argument("--panorama-height", type=int, default=96)
    parser.add_argument("--fov-azimuth-deg", type=float, default=150.0)
    parser.add_argument("--fov-elevation-deg", type=float, default=95.0)
    parser.add_argument("--acceptance-angle-deg", type=float, default=None)
    parser.add_argument("--blur-samples", type=int, default=None)
    parser.add_argument("--contrast", type=float, default=None)
    parser.add_argument("--sensor-noise-std", type=float, default=None)
    parser.add_argument("--texture-mode", choices=("naturalistic", "checker", "mixed"), default="mixed")
    parser.add_argument("--max-yaw-rate", type=float, default=0.25)
    parser.add_argument("--max-forward", type=float, default=0.55)
    parser.add_argument("--max-lateral", type=float, default=0.35)
    parser.add_argument("--motion-scale", type=float, default=0.55)

    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-batches", type=int, default=None)
    parser.add_argument("--val-batches", type=int, default=None)
    parser.add_argument("--test-batches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--state-clip", type=float, default=5.0)
    parser.add_argument(
        "--freeze-recurrent",
        action="store_true",
        help="Freeze recurrent connectome/control weights and train only input/readout parameters.",
    )
    parser.add_argument(
        "--recurrent-prior-l2",
        type=float,
        default=0.0,
        help="L2 penalty weight that keeps trainable recurrent weights near their initialization.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--log-every-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    args = apply_difficulty(args)
    if args.cache_dir is None:
        args.cache_dir = args.output_dir
    if args.hex_rings < 0:
        parser.error("--hex-rings must be non-negative")
    if args.timesteps < 2:
        parser.error("--timesteps must be at least 2")
    if args.blur_samples < 1:
        parser.error("--blur-samples must be at least 1")
    if args.contrast <= 0:
        parser.error("--contrast must be positive")
    if args.sensor_noise_std < 0:
        parser.error("--sensor-noise-std must be non-negative")
    if args.train_batches < 1 or args.val_batches < 1 or args.test_batches < 1:
        parser.error("batch counts must be positive")
    if args.max_neurons < 0:
        parser.error("--max-neurons must be non-negative")
    if args.recurrent_prior_l2 < 0:
        parser.error("--recurrent-prior-l2 must be nonnegative")
    return args


def optic_spec_from_args(args: argparse.Namespace) -> OpticFlowSpec:
    return OpticFlowSpec(
        hex_rings=args.hex_rings,
        timesteps=args.timesteps,
        panorama_width=args.panorama_width,
        panorama_height=args.panorama_height,
        fov_azimuth_deg=args.fov_azimuth_deg,
        fov_elevation_deg=args.fov_elevation_deg,
        acceptance_angle_deg=args.acceptance_angle_deg,
        blur_samples=args.blur_samples,
        contrast=args.contrast,
        sensor_noise_std=args.sensor_noise_std,
        texture_mode=args.texture_mode,
        max_yaw_rate=args.max_yaw_rate,
        max_forward=args.max_forward,
        max_lateral=args.max_lateral,
        motion_scale=args.motion_scale,
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
        freeze_recurrent=bool(args.freeze_recurrent),
        recurrent_prior_l2=float(args.recurrent_prior_l2),
        device=args.device,
        log_every_seconds=args.log_every_seconds,
    )


def matrix_for_training(args: argparse.Namespace, paths: OutputPaths) -> sparse.csr_matrix:
    if args.matrix is not None:
        matrix = load_matrix(args.matrix)
    else:
        matrix = load_matrix(paths.adjacency_unsigned_npz)
    return maybe_truncate_matrix(matrix, args.max_neurons)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    paths = OutputPaths(args.output_dir, args.cache_dir)
    optic_rois = tuple(args.optic_rois)
    if args.mode in {"download", "all"}:
        info = download_flywire_optic_lobe_exports(
            paths,
            release=args.flywire_release,
            download_dir=args.flywire_download_dir,
            optic_rois=optic_rois,
            max_neurons=args.max_neurons,
        )
        print(f"download-complete {info}", flush=True)
        write_artifact_manifest(
            args.output_dir,
            config=vars(args),
            extra={"stage": "optic_flow_download"},
        )
    if args.mode in {"prepare", "all"}:
        matrix = prepare_optic_lobe_connectome(
            paths,
            optic_rois=optic_rois,
            signed_policy=args.signed_policy,
        )
        print(
            f"prepare-complete matrix={paths.adjacency_unsigned_npz} N={matrix.shape[0]} edges={matrix.nnz}",
            flush=True,
        )
        write_artifact_manifest(
            args.output_dir,
            config=vars(args),
            extra={"stage": "optic_flow_prepare"},
        )
    if args.mode in {"train", "all"}:
        matrix = matrix_for_training(args, paths)
        train_benchmark(
            matrix=matrix,
            output_dir=args.output_dir,
            optic_spec=optic_spec_from_args(args),
            train_spec=train_spec_from_args(args),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
