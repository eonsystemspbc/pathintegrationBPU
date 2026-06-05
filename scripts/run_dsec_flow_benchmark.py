#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None

import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODEL_CONNECTOME_SEEDED = "connectome_seeded"
MODEL_CONNECTOME_WEIGHT_SHUFFLE = "connectome_weight_shuffle"
MODEL_RANDOM_INIT = "random_init"
MODEL_ALIASES = {
    "optic_lobe_seeded": MODEL_CONNECTOME_SEEDED,
    "weight_shuffle": MODEL_CONNECTOME_WEIGHT_SHUFFLE,
    "random_sparse": MODEL_RANDOM_INIT,
}
MODEL_CHOICES = (
    MODEL_CONNECTOME_SEEDED,
    MODEL_CONNECTOME_WEIGHT_SHUFFLE,
    MODEL_RANDOM_INIT,
    *MODEL_ALIASES.keys(),
)
DEFAULT_MODELS = (
    MODEL_CONNECTOME_SEEDED,
    MODEL_CONNECTOME_WEIGHT_SHUFFLE,
    MODEL_RANDOM_INIT,
)

DSEC_HEIGHT = 480
DSEC_WIDTH = 640
FLOW_SCALE = 128.0
FLOW_OFFSET = 2**15


@dataclass(frozen=True)
class DSECDataSpec:
    data_root: Path
    event_bins: int = 12
    temporal_groups: int = 5
    crop_height: int = 256
    crop_width: int = 320
    sensor_height: int = DSEC_HEIGHT
    sensor_width: int = DSEC_WIDTH
    rectify_events: bool = True
    require_rectify: bool = False
    augment: bool = True
    val_fraction: float = 0.1
    split_seed: int = 1337
    train_sequences: tuple[str, ...] = ()
    val_sequences: tuple[str, ...] = ()
    max_train_samples: int = 0
    max_val_samples: int = 0
    event_count_clip: float = 5.0

    @property
    def input_channels(self) -> int:
        return int(self.event_bins * 2)


@dataclass(frozen=True)
class TrainSpec:
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    train_steps: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    grad_clip: float
    gamma: float
    max_flow: float
    flow_iters: int
    hidden_dim: int
    ssm_blocks: int
    corr_radius: int
    connectome_neurons: int
    connectome_steps: int
    connectome_prior_l2: float
    validate_every_steps: int
    val_batches: int
    log_every_steps: int
    save_every_steps: int
    mixed_precision: bool
    compile_model: bool
    device: str


@dataclass(frozen=True)
class FlowSample:
    sequence: str
    event_path: Path
    rectify_path: Path | None
    flow_path: Path | None
    t0_us: int
    t1_us: int
    file_index: str


class RunLogger:
    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / "dsec_flow.log"
        self.handle = self.path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def canonical_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _first_existing(base: Path, rels: Sequence[str]) -> Path | None:
    for rel in rels:
        path = base / rel
        if path.exists():
            return path
    return None


def find_event_file(seq_dir: Path) -> Path | None:
    return _first_existing(
        seq_dir,
        (
            "events/left/events.h5",
            "events_left/events.h5",
            "events/events_left/events.h5",
            "events.h5",
        ),
    )


def find_rectify_file(seq_dir: Path) -> Path | None:
    return _first_existing(
        seq_dir,
        (
            "events/left/rectify_map.h5",
            "events_left/rectify_map.h5",
            "calibration/left/rectify_map.h5",
            "calibration/rectify_map.h5",
            "rectify_map.h5",
        ),
    )


def find_flow_dir(seq_dir: Path) -> Path | None:
    candidates = (
        "optical_flow/forward/event",
        "optical_flow/forward",
        "optical_flow_forward_event",
        "flow/forward",
        "flow_forward",
    )
    for rel in candidates:
        path = seq_dir / rel
        if path.exists() and any(path.glob("*.png")):
            return path
    return None


def find_flow_timestamps(seq_dir: Path, sequence: str) -> Path | None:
    return _first_existing(
        seq_dir,
        (
            "optical_flow/forward_timestamps.txt",
            "optical_flow_forward_timestamps.txt",
            "flow/forward_timestamps.txt",
            "flow_forward_timestamps.txt",
            f"{sequence}_optical_flow_forward_timestamps.txt",
        ),
    )


def sequence_dirs(root: Path) -> list[Path]:
    roots = [root]
    if (root / "train").exists():
        roots.insert(0, root / "train")
    out: list[Path] = []
    seen: set[Path] = set()
    for base in roots:
        if not base.exists():
            continue
        for path in sorted(base.iterdir()):
            if not path.is_dir():
                continue
            if find_event_file(path) is None:
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(path)
    return out


def _read_numeric_rows(path: Path) -> list[list[int]]:
    rows: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace(",", " ")
            parts = [part for part in line.split() if part]
            values: list[int] = []
            for part in parts:
                try:
                    values.append(int(float(part)))
                except ValueError:
                    values = []
                    break
            if values:
                rows.append(values)
    return rows


def discover_labeled_samples(root: Path) -> list[FlowSample]:
    samples: list[FlowSample] = []
    for seq_dir in sequence_dirs(root):
        sequence = seq_dir.name
        event_path = find_event_file(seq_dir)
        flow_dir = find_flow_dir(seq_dir)
        ts_path = find_flow_timestamps(seq_dir, sequence)
        if event_path is None or flow_dir is None or ts_path is None:
            continue
        flow_files = sorted(flow_dir.glob("*.png"))
        rows = _read_numeric_rows(ts_path)
        count = min(len(flow_files), len(rows))
        rectify_path = find_rectify_file(seq_dir)
        for idx in range(count):
            row = rows[idx]
            if len(row) < 2:
                t0_us = int(row[0])
                t1_us = t0_us + 100_000
            else:
                t0_us = int(row[0])
                t1_us = int(row[1])
            samples.append(
                FlowSample(
                    sequence=sequence,
                    event_path=event_path,
                    rectify_path=rectify_path,
                    flow_path=flow_files[idx],
                    t0_us=t0_us,
                    t1_us=t1_us,
                    file_index=flow_files[idx].stem,
                )
            )
    return samples


def _sequence_from_timestamp_file(path: Path) -> str:
    stem = path.stem
    suffixes = (
        "_forward_optical_flow_timestamps",
        "_optical_flow_forward_timestamps",
        "_flow_forward_timestamps",
        "_timestamps",
    )
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def discover_prediction_samples(root: Path, timestamps_dir: Path) -> list[FlowSample]:
    seq_by_name = {path.name: path for path in sequence_dirs(root)}
    samples: list[FlowSample] = []
    for ts_path in sorted([*timestamps_dir.glob("*.csv"), *timestamps_dir.glob("*.txt")]):
        sequence = _sequence_from_timestamp_file(ts_path)
        seq_dir = seq_by_name.get(sequence)
        if seq_dir is None:
            continue
        event_path = find_event_file(seq_dir)
        if event_path is None:
            continue
        rows = _read_numeric_rows(ts_path)
        rectify_path = find_rectify_file(seq_dir)
        for row_index, row in enumerate(rows):
            if len(row) >= 3:
                t0_us, t1_us, file_index = int(row[0]), int(row[1]), f"{int(row[2]):06d}"
            elif len(row) >= 2:
                t0_us, t1_us, file_index = int(row[0]), int(row[1]), f"{row_index:06d}"
            else:
                t0_us, t1_us, file_index = int(row[0]), int(row[0]) + 100_000, f"{row_index:06d}"
            samples.append(
                FlowSample(
                    sequence=sequence,
                    event_path=event_path,
                    rectify_path=rectify_path,
                    flow_path=None,
                    t0_us=t0_us,
                    t1_us=t1_us,
                    file_index=file_index,
                )
            )
    return samples


def split_samples(
    samples: Sequence[FlowSample],
    spec: DSECDataSpec,
) -> tuple[list[FlowSample], list[FlowSample]]:
    if spec.train_sequences or spec.val_sequences:
        train_set = set(spec.train_sequences)
        val_set = set(spec.val_sequences)
        train = [sample for sample in samples if sample.sequence in train_set] if train_set else []
        val = [sample for sample in samples if sample.sequence in val_set]
        if not train and val_set:
            train = [sample for sample in samples if sample.sequence not in val_set]
    else:
        rng = np.random.default_rng(spec.split_seed)
        indices = np.arange(len(samples))
        rng.shuffle(indices)
        val_count = max(1, int(round(len(samples) * spec.val_fraction))) if samples else 0
        val_idx = set(indices[:val_count].tolist())
        val = [sample for i, sample in enumerate(samples) if i in val_idx]
        train = [sample for i, sample in enumerate(samples) if i not in val_idx]
    if spec.max_train_samples > 0:
        train = train[: spec.max_train_samples]
    if spec.max_val_samples > 0:
        val = val[: spec.max_val_samples]
    return train, val


class DSECEventReader:
    def __init__(
        self,
        event_path: Path,
        rectify_path: Path | None,
        rectify_events: bool,
        require_rectify: bool,
    ) -> None:
        self.event_path = event_path
        self.rectify_path = rectify_path
        self.rectify_events = rectify_events
        self.require_rectify = require_rectify
        self._events: h5py.File | None = None
        self._rectify_map: np.ndarray | None = None

    def _open(self) -> h5py.File:
        if self._events is None:
            self._events = h5py.File(self.event_path, "r")
        return self._events

    def _load_rectify_map(self) -> np.ndarray | None:
        if not self.rectify_events:
            return None
        if self._rectify_map is not None:
            return self._rectify_map
        if self.rectify_path is None:
            if self.require_rectify:
                raise FileNotFoundError(
                    f"Rectification requested but no rectify_map.h5 was found for {self.event_path}"
                )
            return None
        with h5py.File(self.rectify_path, "r") as handle:
            key = "rectify_map" if "rectify_map" in handle else next(iter(handle.keys()))
            self._rectify_map = np.asarray(handle[key], dtype=np.float32)
        return self._rectify_map

    def close(self) -> None:
        if self._events is not None:
            self._events.close()
            self._events = None

    def __del__(self) -> None:
        self.close()

    def events_between(
        self,
        t0_global_us: int,
        t1_global_us: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        handle = self._open()
        events = handle["events"]
        t_offset = int(np.asarray(handle.get("t_offset", 0)))
        t0 = max(0, int(t0_global_us) - t_offset)
        t1 = max(t0 + 1, int(t1_global_us) - t_offset)
        t_dataset = events["t"]
        i0, i1 = self._event_index_bounds(handle, t_dataset, t0, t1)
        ts = np.asarray(t_dataset[i0:i1], dtype=np.int64)
        if ts.size:
            keep = (ts >= t0) & (ts <= t1)
        else:
            keep = np.zeros((0,), dtype=bool)
        xs = np.asarray(events["x"][i0:i1], dtype=np.int64)[keep]
        ys = np.asarray(events["y"][i0:i1], dtype=np.int64)[keep]
        ps = np.asarray(events["p"][i0:i1], dtype=np.int64)[keep]
        ts = ts[keep]
        rectify_map = self._load_rectify_map()
        if rectify_map is not None and xs.size:
            valid_raw = (
                (ys >= 0)
                & (ys < rectify_map.shape[0])
                & (xs >= 0)
                & (xs < rectify_map.shape[1])
            )
            xs = xs[valid_raw]
            ys = ys[valid_raw]
            ps = ps[valid_raw]
            ts = ts[valid_raw]
            rectified = rectify_map[ys, xs]
            xs = np.rint(rectified[:, 0]).astype(np.int64)
            ys = np.rint(rectified[:, 1]).astype(np.int64)
        ts = ts + t_offset
        return xs, ys, ps, ts

    def _event_index_bounds(
        self,
        handle: h5py.File,
        t_dataset: h5py.Dataset,
        t0: int,
        t1: int,
    ) -> tuple[int, int]:
        if "ms_to_idx" in handle:
            ms_to_idx = handle["ms_to_idx"]
            ms0 = max(0, min(int(t0 // 1000), len(ms_to_idx) - 1))
            ms1 = max(0, min(int(math.ceil(t1 / 1000)), len(ms_to_idx) - 1))
            lo = max(0, int(ms_to_idx[ms0]) - 8)
            hi = min(len(t_dataset), int(ms_to_idx[ms1]) + 8)
            if hi > lo:
                local = np.asarray(t_dataset[lo:hi], dtype=np.int64)
                i0 = lo + int(np.searchsorted(local, t0, side="left"))
                i1 = lo + int(np.searchsorted(local, t1, side="right"))
                return i0, min(max(i1, i0), len(t_dataset))
        ts = np.asarray(t_dataset, dtype=np.int64)
        return (
            int(np.searchsorted(ts, t0, side="left")),
            int(np.searchsorted(ts, t1, side="right")),
        )


def read_dsec_flow_png(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2  # type: ignore

        raw_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw_bgr is None:
            raise FileNotFoundError(path)
        raw = raw_bgr[..., ::-1]
    except Exception:
        try:
            raw = imageio.imread(path, format="PNG-FI")
        except Exception:
            raw = imageio.imread(path)
    if raw.ndim != 3 or raw.shape[-1] < 3:
        raise ValueError(f"DSEC flow PNG must have three channels: {path}")
    raw = raw.astype(np.float32)
    flow = (raw[..., :2] - FLOW_OFFSET) / FLOW_SCALE
    valid = raw[..., 2] > 0
    return flow.astype(np.float32), valid.astype(bool)


def write_dsec_flow_png(path: Path, flow: np.ndarray, valid: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError("flow must have shape [H, W, 2]")
    if valid is None:
        valid = np.ones(flow.shape[:2], dtype=bool)
    out = np.zeros((*flow.shape[:2], 3), dtype=np.uint16)
    encoded = np.rint(flow * FLOW_SCALE + FLOW_OFFSET)
    out[..., :2] = np.clip(encoded, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    out[..., 2] = valid.astype(np.uint16)
    try:
        import cv2  # type: ignore

        ok = cv2.imwrite(str(path), out[..., ::-1])
        if not ok:
            raise RuntimeError(f"Failed to write DSEC flow PNG: {path}")
    except Exception:
        imageio.imwrite(path, out, format="PNG-FI")


def events_to_voxel(
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
    ts: np.ndarray,
    t0_us: int,
    t1_us: int,
    top: int,
    left: int,
    height: int,
    width: int,
    bins: int,
    count_clip: float,
) -> np.ndarray:
    volume = np.zeros((bins, 2, height, width), dtype=np.float32)
    if xs.size == 0:
        return volume.reshape(bins * 2, height, width)
    x = xs.astype(np.int64) - int(left)
    y = ys.astype(np.int64) - int(top)
    keep = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(keep):
        return volume.reshape(bins * 2, height, width)
    x = x[keep]
    y = y[keep]
    p = (ps[keep] > 0).astype(np.int64)
    t = ts[keep].astype(np.float32)
    denom = max(float(t1_us - t0_us), 1.0)
    t_norm = np.clip((t - float(t0_us)) / denom, 0.0, 1.0) * float(bins - 1)
    b0 = np.floor(t_norm).astype(np.int64)
    b1 = np.minimum(b0 + 1, bins - 1)
    w1 = (t_norm - b0.astype(np.float32)).astype(np.float32)
    w0 = 1.0 - w1
    np.add.at(volume, (b0, p, y, x), w0)
    np.add.at(volume, (b1, p, y, x), w1)
    if count_clip > 0:
        volume = np.clip(volume, 0.0, count_clip) / count_clip
    return volume.reshape(bins * 2, height, width).astype(np.float32)


class DSECFlowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        samples: Sequence[FlowSample],
        spec: DSECDataSpec,
        train: bool,
    ) -> None:
        self.samples = list(samples)
        self.spec = spec
        self.train = bool(train)
        self._readers: dict[Path, DSECEventReader] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _reader_for(self, sample: FlowSample) -> DSECEventReader:
        reader = self._readers.get(sample.event_path)
        if reader is None:
            reader = DSECEventReader(
                sample.event_path,
                sample.rectify_path,
                rectify_events=self.spec.rectify_events,
                require_rectify=self.spec.require_rectify,
            )
            self._readers[sample.event_path] = reader
        return reader

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        if sample.flow_path is None:
            raise ValueError("DSECFlowDataset requires labeled samples with flow_path.")
        flow_hw2, valid_hw = read_dsec_flow_png(sample.flow_path)
        height, width = flow_hw2.shape[:2]
        crop_h = min(self.spec.crop_height, height)
        crop_w = min(self.spec.crop_width, width)
        if self.train and self.spec.augment:
            top = int(np.random.randint(0, max(height - crop_h + 1, 1)))
            left = int(np.random.randint(0, max(width - crop_w + 1, 1)))
        else:
            top = max((height - crop_h) // 2, 0)
            left = max((width - crop_w) // 2, 0)
        xs, ys, ps, ts = self._reader_for(sample).events_between(sample.t0_us, sample.t1_us)
        volume = events_to_voxel(
            xs,
            ys,
            ps,
            ts,
            sample.t0_us,
            sample.t1_us,
            top,
            left,
            crop_h,
            crop_w,
            self.spec.event_bins,
            self.spec.event_count_clip,
        )
        flow = flow_hw2[top : top + crop_h, left : left + crop_w].copy()
        valid = valid_hw[top : top + crop_h, left : left + crop_w].copy()
        if self.train and self.spec.augment:
            if np.random.random() < 0.5:
                volume = volume[..., ::-1].copy()
                flow = flow[:, ::-1].copy()
                valid = valid[:, ::-1].copy()
                flow[..., 0] *= -1.0
            if np.random.random() < 0.1:
                volume = volume[:, ::-1, :].copy()
                flow = flow[::-1, :].copy()
                valid = valid[::-1, :].copy()
                flow[..., 1] *= -1.0
        return (
            torch.from_numpy(volume),
            torch.from_numpy(flow.transpose(2, 0, 1).astype(np.float32)),
            torch.from_numpy(valid[None].astype(np.float32)),
        )


class DSECPredictionDataset(Dataset[tuple[torch.Tensor, str, str]]):
    def __init__(self, samples: Sequence[FlowSample], spec: DSECDataSpec) -> None:
        self.samples = list(samples)
        self.spec = spec
        self._readers: dict[Path, DSECEventReader] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _reader_for(self, sample: FlowSample) -> DSECEventReader:
        reader = self._readers.get(sample.event_path)
        if reader is None:
            reader = DSECEventReader(
                sample.event_path,
                sample.rectify_path,
                rectify_events=self.spec.rectify_events,
                require_rectify=self.spec.require_rectify,
            )
            self._readers[sample.event_path] = reader
        return reader

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        sample = self.samples[index]
        xs, ys, ps, ts = self._reader_for(sample).events_between(sample.t0_us, sample.t1_us)
        volume = events_to_voxel(
            xs,
            ys,
            ps,
            ts,
            sample.t0_us,
            sample.t1_us,
            top=0,
            left=0,
            height=self.spec.sensor_height,
            width=self.spec.sensor_width,
            bins=self.spec.event_bins,
            count_clip=self.spec.event_count_clip,
        )
        return torch.from_numpy(volume), sample.sequence, sample.file_index


def matrix_triplets(matrix: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coo = matrix.tocoo()
    return coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data.astype(np.float32)


def load_sparse_matrix(path: Path) -> sparse.csr_matrix:
    if path.suffix == ".npz":
        return sparse.load_npz(path).astype(np.float32).tocsr()
    if path.suffix == ".npy":
        return sparse.csr_matrix(np.load(path).astype(np.float32))
    if path.suffix in {".csv", ".tsv"}:
        sep = "\t" if path.suffix == ".tsv" else ","
        frame = pd.read_csv(path, sep=sep)
        cols = set(frame.columns)
        if {"row", "col", "weight"}.issubset(cols):
            rows = frame["row"].to_numpy(np.int64)
            cols_np = frame["col"].to_numpy(np.int64)
            data = frame["weight"].to_numpy(np.float32)
        elif {"bodyId_pre", "bodyId_post", "weight"}.issubset(cols):
            body_ids = pd.Index(
                pd.concat([frame["bodyId_pre"], frame["bodyId_post"]], ignore_index=True)
                .drop_duplicates()
                .sort_values()
            )
            body_to_idx = {int(body): i for i, body in enumerate(body_ids)}
            rows = frame["bodyId_post"].map(body_to_idx).to_numpy(np.int64)
            cols_np = frame["bodyId_pre"].map(body_to_idx).to_numpy(np.int64)
            data = frame["weight"].to_numpy(np.float32)
            return sparse.coo_matrix((data, (rows, cols_np)), shape=(len(body_ids), len(body_ids))).tocsr()
        else:
            raise ValueError(f"CSV matrix must contain row/col/weight or bodyId_pre/bodyId_post/weight columns: {path}")
        n = int(max(rows.max(initial=0), cols_np.max(initial=0)) + 1)
        return sparse.coo_matrix((data, (rows, cols_np)), shape=(n, n)).tocsr()
    raise ValueError(f"Unsupported matrix file extension: {path}")


def top_activity_submatrix(matrix: sparse.csr_matrix, max_neurons: int) -> sparse.csr_matrix:
    matrix = matrix.astype(np.float32).tocsr()
    if max_neurons <= 0 or matrix.shape[0] <= max_neurons:
        return matrix
    activity = np.asarray(np.abs(matrix).sum(axis=0)).ravel() + np.asarray(np.abs(matrix).sum(axis=1)).ravel()
    keep = np.sort(np.argsort(activity)[-max_neurons:])
    sub = matrix[keep][:, keep].tocsr()
    if sub.nnz == 0:
        raise ValueError("The selected connectome submatrix has no edges; increase --connectome-neurons.")
    return sub


def scale_matrix_spectral(matrix: sparse.csr_matrix, rho_target: float = 0.95) -> sparse.csr_matrix:
    matrix = matrix.astype(np.float32).tocsr()
    if matrix.nnz == 0:
        return matrix
    n = matrix.shape[0]
    if n <= 256:
        rho = float(np.max(np.abs(np.linalg.eigvals(matrix.toarray().astype(np.float64)))))
    else:
        rng = np.random.default_rng(0)
        x = rng.normal(size=n)
        x /= np.linalg.norm(x) + 1e-12
        rho = 0.0
        for _ in range(120):
            y = matrix @ x
            rho = float(np.linalg.norm(y))
            if rho <= 0:
                break
            x = y / rho
    if rho <= 1e-8:
        return matrix
    return (matrix * (rho_target / rho)).astype(np.float32).tocsr()


def random_sparse_like(matrix: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    rows, cols, weights = matrix_triplets(matrix)
    n = matrix.shape[0]
    rng = np.random.default_rng(seed)
    edge_count = len(weights)
    total = n * n
    if edge_count > total:
        raise ValueError("edge count exceeds dense matrix size")
    codes = rng.choice(total, size=edge_count, replace=False)
    out_rows = (codes // n).astype(np.int64)
    out_cols = (codes % n).astype(np.int64)
    std = float(np.std(weights)) if len(weights) > 1 else float(np.mean(np.abs(weights)) or 1.0)
    random_weights = rng.normal(0.0, max(std, 1e-3), size=edge_count).astype(np.float32)
    return sparse.coo_matrix((random_weights, (out_rows, out_cols)), shape=matrix.shape).tocsr()


def weight_shuffle_matrix(matrix: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    rows, cols, weights = matrix_triplets(matrix)
    rng = np.random.default_rng(seed)
    return sparse.coo_matrix((rng.permutation(weights), (rows, cols)), shape=matrix.shape).tocsr()


def connectome_variant(matrix_path: Path, model_name: str, max_neurons: int, seed: int) -> sparse.csr_matrix:
    base = scale_matrix_spectral(top_activity_submatrix(load_sparse_matrix(matrix_path), max_neurons))
    model_name = canonical_model_name(model_name)
    if model_name == MODEL_CONNECTOME_SEEDED:
        return base
    if model_name == MODEL_CONNECTOME_WEIGHT_SHUFFLE:
        return scale_matrix_spectral(weight_shuffle_matrix(base, seed + 10_000))
    if model_name == MODEL_RANDOM_INIT:
        return scale_matrix_spectral(random_sparse_like(base, seed + 20_000))
    raise ValueError(f"Unknown model: {model_name}")


def _norm_groups(channels: int) -> int:
    groups = min(8, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(_norm_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        mid = max(hidden_dim // 2, 16)
        self.net = nn.Sequential(
            ConvBlock(in_channels, mid, stride=2),
            ConvBlock(mid, hidden_dim, stride=2),
            ConvBlock(hidden_dim, hidden_dim, stride=2),
            ConvBlock(hidden_dim, hidden_dim, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerturbedStateSpace2d(nn.Module):
    def __init__(self, channels: int, seed: int = 0) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(channels, channels * 2, 1)
        self.dw = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.norm = nn.GroupNorm(_norm_groups(channels), channels)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        rng = np.random.default_rng(seed)
        base = np.linspace(-2.2, -0.25, channels, dtype=np.float32)
        ptd_noise = rng.normal(0.0, 0.1, size=(4, channels)).astype(np.float32)
        self.A_logit = nn.Parameter(torch.from_numpy(base[None, :] + ptd_noise))
        self.res_scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    @staticmethod
    def _scan_width(x: torch.Tensor, decay: torch.Tensor, reverse: bool) -> torch.Tensor:
        if reverse:
            x = torch.flip(x, dims=(-1,))
        acc = torch.zeros_like(x[..., 0])
        outs: list[torch.Tensor] = []
        d = decay[None, :, None]
        for idx in range(x.shape[-1]):
            acc = d * acc + x[..., idx]
            outs.append(acc)
        y = torch.stack(outs, dim=-1)
        if reverse:
            y = torch.flip(y, dims=(-1,))
        return y

    @staticmethod
    def _scan_height(x: torch.Tensor, decay: torch.Tensor, reverse: bool) -> torch.Tensor:
        xt = x.transpose(-1, -2)
        yt = PerturbedStateSpace2d._scan_width(xt, decay, reverse)
        return yt.transpose(-1, -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, gate = self.in_proj(x).chunk(2, dim=1)
        u = self.dw(u)
        decays = torch.sigmoid(self.A_logit) * 0.995
        y = (
            self._scan_width(u, decays[0], reverse=False)
            + self._scan_width(u, decays[1], reverse=True)
            + self._scan_height(u, decays[2], reverse=False)
            + self._scan_height(u, decays[3], reverse=True)
        ) * 0.25
        y = self.out_proj(self.norm(y) * torch.sigmoid(gate))
        return x + self.res_scale * y


class ConnectomeMixer2d(nn.Module):
    def __init__(self, matrix: sparse.csr_matrix, hidden_dim: int, steps: int) -> None:
        super().__init__()
        matrix = matrix.astype(np.float32).tocoo()
        matrix.sum_duplicates()
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("connectome matrix must be square")
        if matrix.nnz == 0:
            raise ValueError("connectome matrix has no nonzero edges")
        self.N = int(matrix.shape[0])
        self.steps = int(steps)
        self.in_proj = nn.Conv2d(hidden_dim, self.N, 1)
        self.out_proj = nn.Conv2d(self.N, hidden_dim, 1)
        self.norm = nn.GroupNorm(_norm_groups(self.N), self.N)
        self.bias = nn.Parameter(torch.zeros(self.N, dtype=torch.float32))
        indices = np.vstack([matrix.row, matrix.col]).astype(np.int64)
        values = matrix.data.astype(np.float32)
        self.register_buffer("edge_indices", torch.from_numpy(indices))
        self.edge_values = nn.Parameter(torch.from_numpy(values))
        self.register_buffer("edge_initial_values", torch.from_numpy(values.copy()))
        self.res_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    def recurrent_prior_loss(self) -> torch.Tensor:
        return F.mse_loss(self.edge_values, self.edge_initial_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        z = self.in_proj(x)
        tokens = z.permute(0, 2, 3, 1).reshape(-1, self.N)
        rec = torch.sparse_coo_tensor(
            self.edge_indices,
            self.edge_values,
            size=(self.N, self.N),
            device=x.device,
        ).coalesce()
        state = F.silu(tokens)
        for _ in range(max(self.steps, 1)):
            mixed = torch.sparse.mm(rec, state.t()).t()
            state = F.silu(mixed + state + self.bias)
            state = F.layer_norm(state, (self.N,))
        y = state.reshape(b, h, w, self.N).permute(0, 3, 1, 2).contiguous()
        y = self.out_proj(self.norm(y))
        return x + self.res_scale * y


def split_event_groups(x: torch.Tensor, bins: int, groups: int) -> torch.Tensor:
    b, channels, h, w = x.shape
    if channels != bins * 2:
        raise ValueError(f"expected {bins * 2} event channels, got {channels}")
    if groups < 2 or groups > bins:
        raise ValueError("--temporal-groups must be between 2 and --event-bins")
    xb = x.reshape(b, bins, 2, h, w)
    boundaries = torch.linspace(0, bins, groups + 1, device=x.device).round().long()
    out: list[torch.Tensor] = []
    for idx in range(groups):
        lo = int(boundaries[idx].item())
        hi = int(boundaries[idx + 1].item())
        hi = max(hi, lo + 1)
        out.append(xb[:, lo:hi].mean(dim=1))
    return torch.stack(out, dim=1)


def local_correlation(f1: torch.Tensor, f2: torch.Tensor, radius: int) -> torch.Tensor:
    f1 = F.normalize(f1, dim=1)
    f2 = F.normalize(f2, dim=1)
    padded = F.pad(f2, (radius, radius, radius, radius))
    corrs: list[torch.Tensor] = []
    h, w = f1.shape[-2:]
    scale = math.sqrt(max(f1.shape[1], 1))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0 = dy + radius
            x0 = dx + radius
            shifted = padded[:, :, y0 : y0 + h, x0 : x0 + w]
            corrs.append((f1 * shifted).sum(dim=1, keepdim=True) / scale)
    return torch.cat(corrs, dim=1)


class BidirectionalTemporalCorrelation(nn.Module):
    def __init__(self, event_bins: int, groups: int, hidden_dim: int, radius: int) -> None:
        super().__init__()
        self.event_bins = int(event_bins)
        self.groups = int(groups)
        self.radius = int(radius)
        group_dim = max(hidden_dim // 2, 32)
        self.group_encoder = nn.Sequential(
            ConvBlock(2, group_dim, stride=2),
            ConvBlock(group_dim, group_dim, stride=2),
            ConvBlock(group_dim, hidden_dim, stride=2),
        )
        pair_count = 2 * (groups - 1)
        corr_channels = pair_count * (2 * radius + 1) ** 2
        self.project = nn.Sequential(
            nn.Conv2d(corr_channels, hidden_dim, 1),
            nn.GroupNorm(_norm_groups(hidden_dim), hidden_dim),
            nn.SiLU(inplace=True),
            ConvBlock(hidden_dim, hidden_dim, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grouped = split_event_groups(x, self.event_bins, self.groups)
        b, g, c, h, w = grouped.shape
        flat = grouped.reshape(b * g, c, h, w)
        encoded = self.group_encoder(flat)
        feats = encoded.reshape(b, g, encoded.shape[1], encoded.shape[2], encoded.shape[3])
        first = feats[:, 0]
        last = feats[:, -1]
        corrs: list[torch.Tensor] = []
        for idx in range(1, g):
            corrs.append(local_correlation(first, feats[:, idx], self.radius))
        for idx in range(g - 1):
            corrs.append(local_correlation(last, feats[:, idx], self.radius))
        return self.project(torch.cat(corrs, dim=1))


class ConvGRUCell(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int) -> None:
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1.0 - z) * h + z * q


class FlowHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DSECConnectomeFlowNet(nn.Module):
    def __init__(
        self,
        matrix: sparse.csr_matrix,
        event_bins: int,
        temporal_groups: int,
        hidden_dim: int,
        ssm_blocks: int,
        corr_radius: int,
        connectome_steps: int,
    ) -> None:
        super().__init__()
        in_channels = event_bins * 2
        self.event_bins = int(event_bins)
        self.encoder = BasicEncoder(in_channels, hidden_dim)
        self.ssm = nn.Sequential(
            *[PerturbedStateSpace2d(hidden_dim, seed=17 + idx) for idx in range(ssm_blocks)]
        )
        self.connectome = ConnectomeMixer2d(matrix, hidden_dim=hidden_dim, steps=connectome_steps)
        self.temporal_corr = BidirectionalTemporalCorrelation(
            event_bins=event_bins,
            groups=temporal_groups,
            hidden_dim=hidden_dim,
            radius=corr_radius,
        )
        self.fuse = ConvBlock(hidden_dim * 2, hidden_dim, stride=1)
        self.context_hidden = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.context_input = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(hidden_dim + 2, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gru = ConvGRUCell(hidden_dim=hidden_dim, input_dim=hidden_dim * 2)
        self.flow_head = FlowHead(hidden_dim)

    def recurrent_prior_loss(self) -> torch.Tensor:
        return self.connectome.recurrent_prior_loss()

    def _upsample_flow(self, flow: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        up = F.interpolate(flow, size=target_hw, mode="bilinear", align_corners=True)
        scale_y = target_hw[0] / max(flow.shape[-2], 1)
        scale_x = target_hw[1] / max(flow.shape[-1], 1)
        scale = flow.new_tensor([scale_x, scale_y]).view(1, 2, 1, 1)
        return up * scale

    def forward(self, events: torch.Tensor, iters: int = 12) -> list[torch.Tensor]:
        target_hw = (events.shape[-2], events.shape[-1])
        features = self.encoder(events)
        features = self.ssm(features)
        features = self.connectome(features)
        temporal = self.temporal_corr(events)
        fused = self.fuse(torch.cat([features, temporal], dim=1))
        hidden = torch.tanh(self.context_hidden(fused))
        context = self.context_input(fused)
        flow = events.new_zeros((events.shape[0], 2, fused.shape[-2], fused.shape[-1]))
        predictions: list[torch.Tensor] = []
        for _ in range(iters):
            motion = self.motion_encoder(torch.cat([temporal, flow], dim=1))
            hidden = self.gru(hidden, torch.cat([context, motion], dim=1))
            flow = flow + self.flow_head(hidden)
            predictions.append(self._upsample_flow(flow, target_hw))
        return predictions


def sequence_flow_loss(
    predictions: Sequence[torch.Tensor],
    flow_gt: torch.Tensor,
    valid: torch.Tensor,
    gamma: float,
    max_flow: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mag = torch.linalg.norm(flow_gt, dim=1, keepdim=True)
    valid_mask = (valid > 0.5) & torch.isfinite(flow_gt).all(dim=1, keepdim=True) & (mag < max_flow)
    if not torch.any(valid_mask):
        zero = flow_gt.new_tensor(0.0)
        return zero, {"epe": float("nan"), "1pe": float("nan"), "2pe": float("nan"), "3pe": float("nan"), "ae": float("nan")}
    loss = flow_gt.new_tensor(0.0)
    n = len(predictions)
    valid_float = valid_mask.float()
    denom = valid_float.sum().clamp_min(1.0)
    for idx, pred in enumerate(predictions):
        weight = gamma ** (n - idx - 1)
        error = torch.abs(pred - flow_gt).sum(dim=1, keepdim=True)
        loss = loss + weight * (valid_float * error).sum() / denom
    with torch.no_grad():
        final = predictions[-1]
        epe_map = torch.linalg.norm(final - flow_gt, dim=1, keepdim=True)
        epe_vals = epe_map[valid_mask]
        pred3 = torch.cat([final, torch.ones_like(final[:, :1])], dim=1)
        gt3 = torch.cat([flow_gt, torch.ones_like(flow_gt[:, :1])], dim=1)
        cos = (pred3 * gt3).sum(dim=1, keepdim=True) / (
            torch.linalg.norm(pred3, dim=1, keepdim=True) * torch.linalg.norm(gt3, dim=1, keepdim=True)
        ).clamp_min(1e-6)
        ae_vals = torch.rad2deg(torch.acos(torch.clamp(cos[valid_mask], -1.0, 1.0)))
        metrics = {
            "epe": float(epe_vals.mean().detach().cpu()),
            "1pe": float((epe_vals > 1.0).float().mean().detach().cpu() * 100.0),
            "2pe": float((epe_vals > 2.0).float().mean().detach().cpu() * 100.0),
            "3pe": float((epe_vals > 3.0).float().mean().detach().cpu() * 100.0),
            "ae": float(ae_vals.mean().detach().cpu()),
        }
    return loss, metrics


def infinite_loader(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def move_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    events, flow, valid = batch
    return (
        events.to(device, non_blocking=True, dtype=torch.float32),
        flow.to(device, non_blocking=True, dtype=torch.float32),
        valid.to(device, non_blocking=True, dtype=torch.float32),
    )


@torch.no_grad()
def evaluate(
    model: DSECConnectomeFlowNet,
    loader: DataLoader,
    device: torch.device,
    train_spec: TrainSpec,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    for batch_idx, batch in enumerate(loader, start=1):
        events, flow, valid = move_batch(batch, device)
        predictions = model(events, iters=train_spec.flow_iters)
        _, metrics = sequence_flow_loss(
            predictions,
            flow,
            valid,
            gamma=train_spec.gamma,
            max_flow=train_spec.max_flow,
        )
        rows.append(metrics)
        if max_batches > 0 and batch_idx >= max_batches:
            break
    if not rows:
        return {"epe": float("nan"), "1pe": float("nan"), "2pe": float("nan"), "3pe": float("nan"), "ae": float("nan")}
    return {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0]}


def parameter_count(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    best_epe: float,
    config: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": int(step),
            "best_epe": float(best_epe),
            "config": config,
        },
        path,
    )


def train_one(
    model_name: str,
    seed: int,
    matrix_path: Path,
    data_spec: DSECDataSpec,
    train_spec: TrainSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    logger: RunLogger,
) -> dict[str, object]:
    device = select_device(train_spec.device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    canonical = canonical_model_name(model_name)
    matrix = connectome_variant(matrix_path, canonical, train_spec.connectome_neurons, seed)
    model = DSECConnectomeFlowNet(
        matrix=matrix,
        event_bins=data_spec.event_bins,
        temporal_groups=data_spec.temporal_groups,
        hidden_dim=train_spec.hidden_dim,
        ssm_blocks=train_spec.ssm_blocks,
        corr_radius=train_spec.corr_radius,
        connectome_steps=train_spec.connectome_steps,
    ).to(device)
    if train_spec.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_spec.lr,
        weight_decay=train_spec.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_spec.lr,
        total_steps=train_spec.train_steps,
        pct_start=0.05,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=train_spec.mixed_precision and device.type == "cuda")
    run_dir = output_dir / f"{canonical}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": canonical,
        "seed": int(seed),
        "data_spec": {**asdict(data_spec), "data_root": str(data_spec.data_root)},
        "train_spec": asdict(train_spec),
        "matrix_path": str(matrix_path),
        "connectome_N": int(matrix.shape[0]),
        "connectome_edges": int(matrix.nnz),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    logger.log(
        "model-start "
        f"model={canonical} seed={seed} params={parameter_count(model)} "
        f"connectome_N={matrix.shape[0]} connectome_edges={matrix.nnz} train_steps={train_spec.train_steps}"
    )
    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "train_loss",
                "train_epe",
                "train_1pe",
                "train_2pe",
                "train_3pe",
                "train_ae",
                "val_epe",
                "val_1pe",
                "val_2pe",
                "val_3pe",
                "val_ae",
                "lr",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        loader_iter = infinite_loader(train_loader)
        best_epe = float("inf")
        start = time.monotonic()
        last_metrics = {"epe": float("nan"), "1pe": float("nan"), "2pe": float("nan"), "3pe": float("nan"), "ae": float("nan")}
        for step in range(1, train_spec.train_steps + 1):
            model.train()
            events, flow, valid = move_batch(next(loader_iter), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=train_spec.mixed_precision and device.type == "cuda"):
                predictions = model(events, iters=train_spec.flow_iters)
                loss, last_metrics = sequence_flow_loss(
                    predictions,
                    flow,
                    valid,
                    gamma=train_spec.gamma,
                    max_flow=train_spec.max_flow,
                )
                if train_spec.connectome_prior_l2 > 0:
                    core = model._orig_mod if hasattr(model, "_orig_mod") else model
                    loss = loss + train_spec.connectome_prior_l2 * core.recurrent_prior_loss()
            scaler.scale(loss).backward()
            if train_spec.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_spec.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            val_metrics = {"epe": float("nan"), "1pe": float("nan"), "2pe": float("nan"), "3pe": float("nan"), "ae": float("nan")}
            should_validate = step == 1 or step % train_spec.validate_every_steps == 0 or step == train_spec.train_steps
            if should_validate:
                val_metrics = evaluate(model, val_loader, device, train_spec, train_spec.val_batches)
                if val_metrics["epe"] < best_epe:
                    best_epe = val_metrics["epe"]
                    save_checkpoint(run_dir / "checkpoint_best.pt", model, optimizer, scheduler, step, best_epe, config)
            if step % train_spec.save_every_steps == 0 or step == train_spec.train_steps:
                save_checkpoint(run_dir / "checkpoint_latest.pt", model, optimizer, scheduler, step, best_epe, config)
            if step % train_spec.log_every_steps == 0 or should_validate:
                elapsed = time.monotonic() - start
                row = {
                    "step": step,
                    "train_loss": float(loss.detach().cpu()),
                    "train_epe": last_metrics["epe"],
                    "train_1pe": last_metrics["1pe"],
                    "train_2pe": last_metrics["2pe"],
                    "train_3pe": last_metrics["3pe"],
                    "train_ae": last_metrics["ae"],
                    "val_epe": val_metrics["epe"],
                    "val_1pe": val_metrics["1pe"],
                    "val_2pe": val_metrics["2pe"],
                    "val_3pe": val_metrics["3pe"],
                    "val_ae": val_metrics["ae"],
                    "lr": float(scheduler.get_last_lr()[0]),
                    "elapsed_seconds": float(elapsed),
                }
                writer.writerow(row)
                handle.flush()
                logger.log(
                    "progress "
                    f"model={canonical} seed={seed} step={step}/{train_spec.train_steps} "
                    f"train_loss={row['train_loss']:.5g} train_epe={row['train_epe']:.4g} "
                    f"val_epe={row['val_epe']:.4g} best_epe={best_epe:.4g} "
                    f"lr={row['lr']:.3g} elapsed={_format_duration(elapsed)}"
                )
    final_metrics = evaluate(model, val_loader, device, train_spec, train_spec.val_batches)
    metrics = {
        "model": canonical,
        "seed": int(seed),
        "best_val_epe": float(best_epe),
        "final_val_epe": float(final_metrics["epe"]),
        "final_val_1pe": float(final_metrics["1pe"]),
        "final_val_2pe": float(final_metrics["2pe"]),
        "final_val_3pe": float(final_metrics["3pe"]),
        "final_val_ae": float(final_metrics["ae"]),
        "trainable_params": parameter_count(model),
        "connectome_N": int(matrix.shape[0]),
        "connectome_edges": int(matrix.nnz),
        "run_dir": str(run_dir),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    logger.log(
        "model-done "
        f"model={canonical} seed={seed} best_val_epe={best_epe:.4g} final_val_epe={final_metrics['epe']:.4g}"
    )
    return metrics


def train(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    logger = RunLogger(output_dir)
    try:
        data_spec = data_spec_from_args(args)
        train_spec = train_spec_from_args(args)
        samples = discover_labeled_samples(data_spec.data_root)
        train_samples, val_samples = split_samples(samples, data_spec)
        if not train_samples or not val_samples:
            raise RuntimeError(
                f"Need non-empty train and val samples. Found train={len(train_samples)} val={len(val_samples)} from {data_spec.data_root}"
            )
        logger.log(
            "data-ready "
            f"root={data_spec.data_root} total={len(samples)} train={len(train_samples)} val={len(val_samples)} "
            f"event_bins={data_spec.event_bins} crop={data_spec.crop_height}x{data_spec.crop_width}"
        )
        train_loader = DataLoader(
            DSECFlowDataset(train_samples, data_spec, train=True),
            batch_size=train_spec.batch_size,
            shuffle=True,
            num_workers=train_spec.num_workers,
            pin_memory=(select_device(train_spec.device).type == "cuda"),
            drop_last=True,
        )
        val_loader = DataLoader(
            DSECFlowDataset(val_samples, data_spec, train=False),
            batch_size=train_spec.batch_size,
            shuffle=False,
            num_workers=train_spec.num_workers,
            pin_memory=(select_device(train_spec.device).type == "cuda"),
            drop_last=False,
        )
        if len(train_loader) == 0:
            raise RuntimeError(
                "Training loader has zero batches; lower --batch-size or increase --max-train-samples."
            )
        metrics_rows = []
        for model_name in train_spec.models:
            for seed in train_spec.seeds:
                metrics_rows.append(
                    train_one(
                        model_name,
                        seed,
                        args.matrix.resolve(),
                        data_spec,
                        train_spec,
                        train_loader,
                        val_loader,
                        output_dir,
                        logger,
                    )
                )
        metrics = pd.DataFrame(metrics_rows)
        metrics.to_csv(output_dir / "dsec_metrics_by_seed.csv", index=False)
        summary = (
            metrics.groupby("model", as_index=False)
            .agg(
                best_val_epe_mean=("best_val_epe", "mean"),
                best_val_epe_std=("best_val_epe", "std"),
                final_val_epe_mean=("final_val_epe", "mean"),
                final_val_1pe_mean=("final_val_1pe", "mean"),
                final_val_2pe_mean=("final_val_2pe", "mean"),
                final_val_3pe_mean=("final_val_3pe", "mean"),
                final_val_ae_mean=("final_val_ae", "mean"),
                trainable_params=("trainable_params", "first"),
                connectome_N=("connectome_N", "first"),
                connectome_edges=("connectome_edges", "first"),
            )
            .sort_values("best_val_epe_mean")
        )
        summary.to_csv(output_dir / "dsec_metrics_summary.csv", index=False)
        write_report(output_dir, data_spec, train_spec, summary)
        logger.log(f"complete metrics={output_dir / 'dsec_metrics_by_seed.csv'}")
    finally:
        logger.close()


def _load_state_dict_for_model(model: nn.Module, checkpoint: dict[str, object]) -> None:
    state = checkpoint["model"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint['model'] must be a state_dict")
    try:
        model.load_state_dict(state)
    except RuntimeError:
        stripped = {
            (key.removeprefix("_orig_mod.") if isinstance(key, str) else key): value
            for key, value in state.items()
        }
        model.load_state_dict(stripped)


@torch.no_grad()
def predict(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    logger = RunLogger(output_dir)
    try:
        if args.checkpoint is None:
            raise RuntimeError("--checkpoint is required for --mode predict")
        if args.eval_timestamps_dir is None:
            raise RuntimeError("--eval-timestamps-dir is required for --mode predict")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        model_name = canonical_model_name(args.predict_model or checkpoint_config.get("model", MODEL_CONNECTOME_SEEDED))
        seed = int(args.predict_seed if args.predict_seed is not None else checkpoint_config.get("seed", 0))
        data_spec = data_spec_from_args(args)
        samples = discover_prediction_samples(data_spec.data_root, args.eval_timestamps_dir.resolve())
        if not samples:
            raise RuntimeError(
                f"No prediction samples found under {data_spec.data_root} using {args.eval_timestamps_dir}"
            )
        matrix = connectome_variant(args.matrix.resolve(), model_name, args.connectome_neurons, seed)
        model = DSECConnectomeFlowNet(
            matrix=matrix,
            event_bins=data_spec.event_bins,
            temporal_groups=data_spec.temporal_groups,
            hidden_dim=args.hidden_dim,
            ssm_blocks=args.ssm_blocks,
            corr_radius=args.corr_radius,
            connectome_steps=args.connectome_steps,
        )
        _load_state_dict_for_model(model, checkpoint)
        device = select_device(args.device)
        model.to(device)
        model.eval()
        dataset = DSECPredictionDataset(samples, data_spec)
        loader = DataLoader(
            dataset,
            batch_size=args.predict_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        submission_dir = output_dir / "submission"
        logger.log(
            "predict-start "
            f"samples={len(samples)} model={model_name} seed={seed} output={submission_dir}"
        )
        written = 0
        start = time.monotonic()
        for events, sequences, file_indices in loader:
            events = events.to(device, dtype=torch.float32, non_blocking=True)
            pred = model(events, iters=args.flow_iters)[-1].detach().cpu().numpy()
            for item_idx, sequence in enumerate(sequences):
                flow = pred[item_idx].transpose(1, 2, 0)
                file_index = str(file_indices[item_idx])
                if not file_index.endswith(".png"):
                    file_index = f"{int(file_index):06d}.png" if file_index.isdigit() else f"{file_index}.png"
                write_dsec_flow_png(submission_dir / str(sequence) / file_index, flow)
                written += 1
                if written % max(args.log_every_steps, 1) == 0:
                    logger.log(
                        f"predict-progress written={written}/{len(samples)} elapsed={_format_duration(time.monotonic() - start)}"
                    )
        if args.zip_submission:
            archive = shutil.make_archive(str(output_dir / "dsec_flow_submission"), "zip", submission_dir)
            logger.log(f"predict-zip archive={archive}")
        logger.log(
            f"predict-done written={written} submission_dir={submission_dir} elapsed={_format_duration(time.monotonic() - start)}"
        )
    finally:
        logger.close()


def write_report(output_dir: Path, data_spec: DSECDataSpec, train_spec: TrainSpec, summary: pd.DataFrame) -> None:
    lines = [
        "# DSEC-Flow Optic-Lobe Connectome Run",
        "",
        "This run trains a P-SSE/BAT-inspired event optical-flow model with a matched optic-lobe connectome mixer.",
        "",
        "## Model Arms",
        "",
        "- `connectome_seeded`: observed optic-lobe topology and synaptic weights.",
        "- `connectome_weight_shuffle`: identical topology with shuffled observed weights.",
        "- `random_init`: same connectome-block size and edge count with random sparse support and Gaussian weights.",
        "",
        "## Summary",
        "",
        "```",
        summary.to_string(index=False) if not summary.empty else "No metrics.",
        "```",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(
            {
                "data_spec": {**asdict(data_spec), "data_root": str(data_spec.data_root)},
                "train_spec": asdict(train_spec),
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
    ]
    (output_dir / "dsec_flow_report.md").write_text("\n".join(lines), encoding="utf-8")


def data_spec_from_args(args: argparse.Namespace) -> DSECDataSpec:
    return DSECDataSpec(
        data_root=args.dsec_root.resolve(),
        event_bins=args.event_bins,
        temporal_groups=args.temporal_groups,
        crop_height=args.crop_height,
        crop_width=args.crop_width,
        sensor_height=args.sensor_height,
        sensor_width=args.sensor_width,
        rectify_events=not args.no_rectify,
        require_rectify=args.require_rectify,
        augment=not args.no_augment,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        train_sequences=tuple(args.train_sequences or ()),
        val_sequences=tuple(args.val_sequences or ()),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        event_count_clip=args.event_count_clip,
    )


def train_spec_from_args(args: argparse.Namespace) -> TrainSpec:
    return TrainSpec(
        models=tuple(canonical_model_name(name) for name in args.models),
        seeds=tuple(args.seeds),
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        gamma=args.gamma,
        max_flow=args.max_flow,
        flow_iters=args.flow_iters,
        hidden_dim=args.hidden_dim,
        ssm_blocks=args.ssm_blocks,
        corr_radius=args.corr_radius,
        connectome_neurons=args.connectome_neurons,
        connectome_steps=args.connectome_steps,
        connectome_prior_l2=args.connectome_prior_l2,
        validate_every_steps=args.validate_every_steps,
        val_batches=args.val_batches,
        log_every_steps=args.log_every_steps,
        save_every_steps=args.save_every_steps,
        mixed_precision=args.mixed_precision,
        compile_model=args.compile_model,
        device=args.device,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DSEC-Flow optic-lobe connectome benchmark.")
    parser.add_argument("--mode", choices=("train", "predict"), default="train")
    parser.add_argument("--dsec-root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True, help="Prepared optic-lobe adjacency .npz/.npy/.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dsec_flow_connectome"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-timestamps-dir", type=Path, default=None)
    parser.add_argument("--predict-model", choices=MODEL_CHOICES, default=None)
    parser.add_argument("--predict-seed", type=int, default=None)
    parser.add_argument("--predict-batch-size", type=int, default=1)
    parser.add_argument("--zip-submission", action="store_true")

    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--train-steps", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--max-flow", type=float, default=400.0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")

    parser.add_argument("--event-bins", type=int, default=12)
    parser.add_argument("--temporal-groups", type=int, default=5)
    parser.add_argument("--crop-height", type=int, default=256)
    parser.add_argument("--crop-width", type=int, default=320)
    parser.add_argument("--sensor-height", type=int, default=DSEC_HEIGHT)
    parser.add_argument("--sensor-width", type=int, default=DSEC_WIDTH)
    parser.add_argument("--event-count-clip", type=float, default=5.0)
    parser.add_argument("--no-rectify", action="store_true")
    parser.add_argument("--require-rectify", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--train-sequences", nargs="*", default=[])
    parser.add_argument("--val-sequences", nargs="*", default=[])
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ssm-blocks", type=int, default=2)
    parser.add_argument("--corr-radius", type=int, default=3)
    parser.add_argument("--flow-iters", type=int, default=12)
    parser.add_argument("--connectome-neurons", type=int, default=256)
    parser.add_argument("--connectome-steps", type=int, default=2)
    parser.add_argument("--connectome-prior-l2", type=float, default=0.0)

    parser.add_argument("--validate-every-steps", type=int, default=1000)
    parser.add_argument("--val-batches", type=int, default=64)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--save-every-steps", type=int, default=5000)
    args = parser.parse_args(argv)
    if args.event_bins < 2:
        parser.error("--event-bins must be at least 2")
    if args.temporal_groups < 2 or args.temporal_groups > args.event_bins:
        parser.error("--temporal-groups must be between 2 and --event-bins")
    if args.crop_height < 8 or args.crop_width < 8:
        parser.error("--crop-height/--crop-width must be at least 8")
    if args.train_steps < 1:
        parser.error("--train-steps must be positive")
    if args.connectome_neurons < 2:
        parser.error("--connectome-neurons must be at least 2")
    if args.validate_every_steps < 1 or args.log_every_steps < 1 or args.save_every_steps < 1:
        parser.error("logging, validation, and save step intervals must be positive")
    if args.predict_batch_size < 1:
        parser.error("--predict-batch-size must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
