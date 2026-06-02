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
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.connectome_expansion import expand_connectome_dcsbm  # noqa: E402
from src.run_manifest import write_artifact_manifest  # noqa: E402

import run_mb_associative_learning as mb  # noqa: E402


MODEL_GRU = "gru"
MODEL_NEAREST = "nearest_support"
MODEL_FAST_HEMIBRAIN = "hemibrain_fast_memory"
MODEL_FAST_RANDOM = "random_sparse_fast_memory"
MODEL_FAST_WEIGHT_SHUFFLE = "weight_shuffle_fast_memory"
FAST_MEMORY_MODEL_TO_BASE = {
    MODEL_FAST_HEMIBRAIN: mb.MODEL_HEMIBRAIN,
    MODEL_FAST_RANDOM: mb.MODEL_RANDOM,
    MODEL_FAST_WEIGHT_SHUFFLE: mb.MODEL_WEIGHT_SHUFFLE,
}
FAST_MEMORY_MODELS = tuple(FAST_MEMORY_MODEL_TO_BASE)
MODEL_CHOICES = mb.MODEL_CHOICES + FAST_MEMORY_MODELS + (MODEL_GRU, MODEL_NEAREST)
DEFAULT_MODELS = (
    mb.MODEL_HEMIBRAIN,
    mb.MODEL_RANDOM,
    mb.MODEL_WEIGHT_SHUFFLE,
    MODEL_GRU,
    MODEL_NEAREST,
)
DATASET_CHOICES = ("omniglot", "synthetic")
EMBEDDING_CHOICES = ("random_projection", "raw")


@dataclass(frozen=True)
class FeatureBank:
    name: str
    classes: tuple[np.ndarray, ...]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def feature_dim(self) -> int:
        if not self.classes:
            raise ValueError("feature bank has no classes.")
        return int(self.classes[0].shape[1])


@dataclass(frozen=True)
class EpisodeSpec:
    way: int
    shot: int
    queries_per_class: int
    reversal_count: int
    feature_dim: int
    feature_noise_std: float

    @property
    def input_dim(self) -> int:
        return self.feature_dim + self.way + 2

    @property
    def initial_support_steps(self) -> int:
        return self.way * self.shot

    @property
    def initial_query_steps(self) -> int:
        return self.way * self.queries_per_class

    @property
    def reversal_support_steps(self) -> int:
        return self.reversal_count * self.shot

    @property
    def reversal_query_steps(self) -> int:
        if self.reversal_count <= 0:
            return 0
        return self.way * self.queries_per_class

    @property
    def timesteps(self) -> int:
        return (
            self.initial_support_steps
            + self.initial_query_steps
            + self.reversal_support_steps
            + self.reversal_query_steps
        )


@dataclass(frozen=True)
class EpisodicBatch:
    inputs: np.ndarray
    targets: np.ndarray
    query_mask: np.ndarray
    initial_query_mask: np.ndarray
    reversal_query_mask: np.ndarray
    support_mask: np.ndarray


class MatrixEpisodicRNN(nn.Module):
    def __init__(
        self,
        recurrent: sparse.spmatrix,
        input_dim: int,
        output_dim: int,
        runtime: str,
        state_clip: float,
        seed: int,
        freeze_recurrent: bool = False,
    ) -> None:
        super().__init__()
        if runtime not in mb.RUNTIME_CHOICES:
            raise ValueError(f"runtime must be one of {mb.RUNTIME_CHOICES}")
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
        scale_in = 1.0 / math.sqrt(max(input_dim, 1))
        scale_out = 1.0 / math.sqrt(max(self.N, 1))
        self.W_in = nn.Parameter(
            torch.empty(self.N, input_dim, dtype=torch.float32).uniform_(
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
            self.register_buffer("W_rec_initial", torch.from_numpy(dense.copy()))
            self.register_buffer("edge_indices", torch.empty(2, 0, dtype=torch.long))
        else:
            indices = np.vstack([recurrent.row, recurrent.col]).astype(np.int64)
            self.register_buffer("edge_indices", torch.from_numpy(indices))
            values = recurrent.data.astype(np.float32)
            self.W_rec_values = nn.Parameter(torch.from_numpy(values))
            self.register_buffer("W_rec_initial_values", torch.from_numpy(values.copy()))
        if freeze_recurrent:
            if runtime == "dense":
                self.W_rec.requires_grad_(False)
            else:
                self.W_rec_values.requires_grad_(False)

    def recurrent_parameter_count(self) -> int:
        if self.runtime == "dense":
            return int(self.W_rec.numel())
        return int(self.W_rec_values.numel())

    def trainable_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))

    def recurrent_prior_loss(self) -> torch.Tensor:
        if self.runtime == "dense":
            return nn.functional.mse_loss(self.W_rec, self.W_rec_initial)
        return nn.functional.mse_loss(self.W_rec_values, self.W_rec_initial_values)

    def _recurrent_step(self, h: torch.Tensor) -> torch.Tensor:
        if self.runtime == "dense":
            return h @ self.W_rec.t()
        W = torch.sparse_coo_tensor(
            self.edge_indices,
            self.W_rec_values,
            size=(self.N, self.N),
            device=h.device,
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
            h = torch.relu(self._recurrent_step(h) + inputs[:, t, :] @ self.W_in.t() + self.b_rec)
            if self.state_clip > 0:
                h = torch.clamp(h, max=self.state_clip)
            outputs.append(self.readout(h))
        return torch.stack(outputs, dim=1)


class MatrixFastMemoryRNN(nn.Module):
    def __init__(
        self,
        recurrent: sparse.spmatrix,
        input_dim: int,
        output_dim: int,
        feature_dim: int,
        runtime: str,
        state_clip: float,
        memory_decay: float,
        memory_temperature: float,
        encoder_steps: int,
        seed: int,
        freeze_recurrent: bool = False,
    ) -> None:
        super().__init__()
        if runtime not in mb.RUNTIME_CHOICES:
            raise ValueError(f"runtime must be one of {mb.RUNTIME_CHOICES}")
        recurrent = recurrent.astype(np.float32).tocoo()
        recurrent.sum_duplicates()
        if recurrent.shape[0] != recurrent.shape[1]:
            raise ValueError("recurrent matrix must be square.")
        self.N = int(recurrent.shape[0])
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.feature_dim = int(feature_dim)
        self.runtime = runtime
        self.state_clip = float(state_clip)
        self.memory_decay = float(memory_decay)
        self.memory_temperature = float(memory_temperature)
        self.encoder_steps = int(encoder_steps)
        if self.encoder_steps < 1:
            raise ValueError("encoder_steps must be at least 1.")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        scale_feature = 1.0 / math.sqrt(max(feature_dim, 1))
        self.W_feature = nn.Parameter(
            torch.empty(self.N, feature_dim, dtype=torch.float32).uniform_(
                -scale_feature, scale_feature, generator=generator
            )
        )
        self.b_rec = nn.Parameter(torch.zeros(self.N, dtype=torch.float32))

        if runtime == "dense":
            dense = recurrent.toarray().astype(np.float32)
            self.W_rec = nn.Parameter(torch.from_numpy(dense))
            self.register_buffer("W_rec_initial", torch.from_numpy(dense.copy()))
            self.register_buffer("edge_indices", torch.empty(2, 0, dtype=torch.long))
        else:
            indices = np.vstack([recurrent.row, recurrent.col]).astype(np.int64)
            self.register_buffer("edge_indices", torch.from_numpy(indices))
            values = recurrent.data.astype(np.float32)
            self.W_rec_values = nn.Parameter(torch.from_numpy(values))
            self.register_buffer("W_rec_initial_values", torch.from_numpy(values.copy()))
        if freeze_recurrent:
            if runtime == "dense":
                self.W_rec.requires_grad_(False)
            else:
                self.W_rec_values.requires_grad_(False)

    def recurrent_parameter_count(self) -> int:
        if self.runtime == "dense":
            return int(self.W_rec.numel())
        return int(self.W_rec_values.numel())

    def trainable_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))

    def recurrent_prior_loss(self) -> torch.Tensor:
        if self.runtime == "dense":
            return nn.functional.mse_loss(self.W_rec, self.W_rec_initial)
        return nn.functional.mse_loss(self.W_rec_values, self.W_rec_initial_values)

    def _recurrent_step(self, h: torch.Tensor) -> torch.Tensor:
        if self.runtime == "dense":
            return h @ self.W_rec.t()
        W = torch.sparse_coo_tensor(
            self.edge_indices,
            self.W_rec_values,
            size=(self.N, self.N),
            device=h.device,
        ).coalesce()
        return torch.sparse.mm(W, h.t()).t()

    def encode_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must have shape [batch, {self.feature_dim}], got {tuple(features.shape)}"
            )
        h = features.new_zeros((features.shape[0], self.N))
        drive = features @ self.W_feature.t() + self.b_rec
        for _ in range(self.encoder_steps):
            h = torch.relu(self._recurrent_step(h) + drive)
            if self.state_clip > 0:
                h = torch.clamp(h, max=self.state_clip)
        return nn.functional.normalize(h, p=2, dim=1, eps=1e-6)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"inputs must have shape [batch, T, {self.input_dim}], got {tuple(inputs.shape)}"
            )
        batch, T, _ = inputs.shape
        memory = inputs.new_zeros((batch, self.N, self.output_dim))
        outputs: list[torch.Tensor] = []
        label_slice = slice(self.feature_dim, self.feature_dim + self.output_dim)
        support_col = self.feature_dim + self.output_dim
        temp = max(self.memory_temperature, 1e-6)
        for t in range(T):
            z = self.encode_features(inputs[:, t, : self.feature_dim])
            logits = torch.bmm(z.unsqueeze(1), memory).squeeze(1) / temp
            outputs.append(logits)

            support_gate = inputs[:, t, support_col].view(batch, 1, 1)
            label = inputs[:, t, label_slice].view(batch, 1, self.output_dim)
            memory = self.memory_decay * memory + support_gate * torch.bmm(
                z.unsqueeze(2),
                label,
            )
        return torch.stack(outputs, dim=1)


class GRUEpisodicClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_size = int(hidden_size)
        self.gru = nn.GRU(self.input_dim, self.hidden_size, batch_first=True)
        self.readout = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(inputs)
        return self.readout(h)

    def recurrent_parameter_count(self) -> int:
        return int(sum(param.numel() for name, param in self.named_parameters() if "gru" in name))

    def trainable_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))


def _format_seconds(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, rem = divmod(minutes, 60)
    return f"{hours}h{rem:02d}m"


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32, copy=False)
    features = features - features.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return (features / np.maximum(norm, 1e-6)).astype(np.float32)


def _sparsify_features(features: np.ndarray, keep_fraction: float) -> np.ndarray:
    if keep_fraction >= 1.0:
        return features.astype(np.float32, copy=False)
    if keep_fraction <= 0.0:
        raise ValueError("embedding sparsity must be in (0, 1].")
    k = max(1, int(round(features.shape[1] * keep_fraction)))
    threshold = np.partition(np.abs(features), -k, axis=1)[:, -k][:, None]
    return np.where(np.abs(features) >= threshold, features, 0.0).astype(np.float32)


def synthetic_feature_banks(
    feature_dim: int,
    samples_per_class: int,
    train_classes: int,
    val_classes: int,
    test_classes: int,
    seed: int,
    class_noise_std: float,
) -> tuple[FeatureBank, FeatureBank, FeatureBank]:
    rng = np.random.default_rng(seed)

    def make_bank(name: str, count: int) -> FeatureBank:
        classes: list[np.ndarray] = []
        prototypes = rng.normal(0.0, 1.0, size=(count, feature_dim)).astype(np.float32)
        prototypes = _l2_normalize(prototypes)
        for idx in range(count):
            samples = prototypes[idx][None, :] + rng.normal(
                0.0, class_noise_std, size=(samples_per_class, feature_dim)
            ).astype(np.float32)
            classes.append(_l2_normalize(samples))
        return FeatureBank(name=name, classes=tuple(classes))

    return (
        make_bank("synthetic_train", train_classes),
        make_bank("synthetic_val", val_classes),
        make_bank("synthetic_test", test_classes),
    )


def _pil_to_flat_array(image: object, image_size: int) -> np.ndarray:
    image = image.convert("L").resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = 1.0 - arr
    return arr.reshape(-1).astype(np.float32)


def _class_arrays_from_omniglot(dataset: object, image_size: int) -> tuple[np.ndarray, ...]:
    grouped: dict[int, list[np.ndarray]] = {}
    for image, target in dataset:
        grouped.setdefault(int(target), []).append(_pil_to_flat_array(image, image_size))
    return tuple(np.stack(grouped[key], axis=0).astype(np.float32) for key in sorted(grouped))


def _limit_classes(
    classes: tuple[np.ndarray, ...],
    max_classes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    if max_classes <= 0 or max_classes >= len(classes):
        return classes
    indices = np.sort(rng.choice(len(classes), size=max_classes, replace=False))
    return tuple(classes[int(idx)] for idx in indices)


def _embed_class_arrays(
    classes: tuple[np.ndarray, ...],
    embedding: str,
    embedding_dim: int,
    embedding_sparsity: float,
    projection: np.ndarray | None,
) -> tuple[np.ndarray, ...]:
    embedded: list[np.ndarray] = []
    for samples in classes:
        if embedding == "raw":
            features = samples.astype(np.float32)
        elif embedding == "random_projection":
            if projection is None:
                raise ValueError("projection is required for random_projection embedding.")
            features = samples @ projection
        else:
            raise ValueError(f"unknown embedding mode: {embedding}")
        features = _sparsify_features(_l2_normalize(features), embedding_sparsity)
        features = _l2_normalize(features)
        embedded.append(features.astype(np.float32))
    return tuple(embedded)


def load_omniglot_feature_banks(args: argparse.Namespace) -> tuple[FeatureBank, FeatureBank, FeatureBank]:
    try:
        from torchvision.datasets import Omniglot
    except Exception as exc:  # pragma: no cover - exercised only when torchvision is absent
        raise RuntimeError(
            "The real Omniglot benchmark requires torchvision. Install it in the "
            "experiment environment, or run with --dataset synthetic for a smoke test."
        ) from exc

    rng = np.random.default_rng(args.data_seed)
    trainval_dataset = Omniglot(
        root=str(args.data_root),
        background=True,
        download=args.download,
    )
    test_dataset = Omniglot(
        root=str(args.data_root),
        background=False,
        download=args.download,
    )
    trainval_classes = _class_arrays_from_omniglot(trainval_dataset, args.image_size)
    test_classes = _class_arrays_from_omniglot(test_dataset, args.image_size)
    trainval_classes = _limit_classes(trainval_classes, args.max_classes, rng)
    test_classes = _limit_classes(test_classes, args.max_classes, rng)
    if len(trainval_classes) < args.way * 2:
        raise ValueError(
            "Omniglot background split needs at least 2 * --way classes after filtering "
            "so train and validation class pools are disjoint."
        )
    if len(test_classes) < args.way:
        raise ValueError("Omniglot evaluation split has too few classes after filtering.")
    val_count = max(args.way, int(round(len(trainval_classes) * args.val_class_fraction)))
    val_count = min(val_count, len(trainval_classes) - args.way)
    shuffled = rng.permutation(len(trainval_classes))
    val_indices = set(int(idx) for idx in shuffled[:val_count])
    train_classes = tuple(
        trainval_classes[idx] for idx in range(len(trainval_classes)) if idx not in val_indices
    )
    val_classes = tuple(
        trainval_classes[idx] for idx in range(len(trainval_classes)) if idx in val_indices
    )

    pixel_dim = trainval_classes[0].shape[1]
    projection = None
    if args.embedding == "random_projection":
        projection = rng.normal(
            0.0,
            1.0 / math.sqrt(max(args.embedding_dim, 1)),
            size=(pixel_dim, args.embedding_dim),
        ).astype(np.float32)
    elif args.embedding_dim > 0 and args.embedding != "raw":
        raise ValueError("--embedding-dim is only used by random_projection.")

    return (
        FeatureBank(
            "omniglot_background_train",
            _embed_class_arrays(
                train_classes,
                args.embedding,
                args.embedding_dim,
                args.embedding_sparsity,
                projection,
            ),
        ),
        FeatureBank(
            "omniglot_background_val",
            _embed_class_arrays(
                val_classes,
                args.embedding,
                args.embedding_dim,
                args.embedding_sparsity,
                projection,
            ),
        ),
        FeatureBank(
            "omniglot_evaluation_test",
            _embed_class_arrays(
                test_classes,
                args.embedding,
                args.embedding_dim,
                args.embedding_sparsity,
                projection,
            ),
        ),
    )


def load_feature_banks(args: argparse.Namespace) -> tuple[FeatureBank, FeatureBank, FeatureBank]:
    if args.dataset == "synthetic":
        return synthetic_feature_banks(
            feature_dim=args.synthetic_feature_dim,
            samples_per_class=args.synthetic_samples_per_class,
            train_classes=args.synthetic_train_classes,
            val_classes=args.synthetic_val_classes,
            test_classes=args.synthetic_test_classes,
            seed=args.data_seed,
            class_noise_std=args.synthetic_class_noise_std,
        )
    if args.dataset == "omniglot":
        return load_omniglot_feature_banks(args)
    raise ValueError(f"unknown dataset: {args.dataset}")


def _sample_class_features(
    class_features: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    replace = count > class_features.shape[0]
    indices = rng.choice(class_features.shape[0], size=count, replace=replace)
    return class_features[indices].astype(np.float32)


def _write_input_step(
    dest: np.ndarray,
    feature: np.ndarray,
    label: int,
    spec: EpisodeSpec,
    is_support: bool,
    rng: np.random.Generator,
) -> None:
    feat = feature.astype(np.float32, copy=True)
    if spec.feature_noise_std > 0:
        feat += rng.normal(0.0, spec.feature_noise_std, size=feat.shape).astype(np.float32)
        feat = _l2_normalize(feat[None, :])[0]
    dest[: spec.feature_dim] = feat
    if is_support:
        dest[spec.feature_dim + label] = 1.0
        dest[spec.feature_dim + spec.way] = 1.0
    else:
        dest[spec.feature_dim + spec.way + 1] = 1.0


def generate_episode_batch(
    bank: FeatureBank,
    spec: EpisodeSpec,
    batch_size: int,
    rng: np.random.Generator,
) -> EpisodicBatch:
    if bank.num_classes < spec.way:
        raise ValueError(
            f"feature bank {bank.name} has {bank.num_classes} classes, but way={spec.way}"
        )
    inputs = np.zeros((batch_size, spec.timesteps, spec.input_dim), dtype=np.float32)
    targets = np.zeros((batch_size, spec.timesteps), dtype=np.int64)
    query_mask = np.zeros((batch_size, spec.timesteps), dtype=np.float32)
    initial_query_mask = np.zeros((batch_size, spec.timesteps), dtype=np.float32)
    reversal_query_mask = np.zeros((batch_size, spec.timesteps), dtype=np.float32)
    support_mask = np.zeros((batch_size, spec.timesteps), dtype=np.float32)

    for batch_idx in range(batch_size):
        local_classes = rng.choice(bank.num_classes, size=spec.way, replace=False)
        initial_labels = rng.permutation(spec.way)
        final_labels = initial_labels.copy()
        reversed_local: np.ndarray
        if spec.reversal_count > 0:
            reversed_local = rng.choice(spec.way, size=spec.reversal_count, replace=False)
            final_labels[reversed_local] = np.roll(final_labels[reversed_local], 1)
        else:
            reversed_local = np.empty(0, dtype=np.int64)
        reversed_set = set(reversed_local.tolist())

        sampled_by_local: dict[int, np.ndarray] = {}
        for local_idx, class_idx in enumerate(local_classes):
            needed = spec.shot + spec.queries_per_class
            if spec.reversal_count > 0:
                needed += spec.queries_per_class
            if local_idx in reversed_set:
                needed += spec.shot
            sampled_by_local[local_idx] = _sample_class_features(
                bank.classes[int(class_idx)], needed, rng
            )

        phases: list[tuple[str, int, np.ndarray, int]] = []
        offsets = {local_idx: 0 for local_idx in range(spec.way)}
        for local_idx in rng.permutation(spec.way):
            for _ in range(spec.shot):
                feature = sampled_by_local[int(local_idx)][offsets[int(local_idx)]]
                offsets[int(local_idx)] += 1
                phases.append(("support", int(local_idx), feature, int(initial_labels[local_idx])))
        for local_idx in rng.permutation(np.repeat(np.arange(spec.way), spec.queries_per_class)):
            feature = sampled_by_local[int(local_idx)][offsets[int(local_idx)]]
            offsets[int(local_idx)] += 1
            phases.append(("initial_query", int(local_idx), feature, int(initial_labels[local_idx])))
        if spec.reversal_count > 0:
            for local_idx in rng.permutation(np.repeat(reversed_local, spec.shot)):
                feature = sampled_by_local[int(local_idx)][offsets[int(local_idx)]]
                offsets[int(local_idx)] += 1
                phases.append(("support", int(local_idx), feature, int(final_labels[local_idx])))
            for local_idx in rng.permutation(np.repeat(np.arange(spec.way), spec.queries_per_class)):
                feature = sampled_by_local[int(local_idx)][offsets[int(local_idx)]]
                offsets[int(local_idx)] += 1
                phases.append(("reversal_query", int(local_idx), feature, int(final_labels[local_idx])))

        if len(phases) != spec.timesteps:
            raise AssertionError(f"internal timestep mismatch: {len(phases)} != {spec.timesteps}")
        for step, (phase, _local_idx, feature, label) in enumerate(phases):
            is_support = phase == "support"
            _write_input_step(
                inputs[batch_idx, step],
                feature,
                label,
                spec,
                is_support=is_support,
                rng=rng,
            )
            targets[batch_idx, step] = label
            if is_support:
                support_mask[batch_idx, step] = 1.0
            else:
                query_mask[batch_idx, step] = 1.0
                if phase == "initial_query":
                    initial_query_mask[batch_idx, step] = 1.0
                elif phase == "reversal_query":
                    reversal_query_mask[batch_idx, step] = 1.0

    return EpisodicBatch(
        inputs=inputs,
        targets=targets,
        query_mask=query_mask,
        initial_query_mask=initial_query_mask,
        reversal_query_mask=reversal_query_mask,
        support_mask=support_mask,
    )


def batch_to_torch(
    batch: EpisodicBatch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(batch.inputs).to(device),
        torch.from_numpy(batch.targets).to(device),
        torch.from_numpy(batch.query_mask).to(device),
        torch.from_numpy(batch.initial_query_mask).to(device),
        torch.from_numpy(batch.reversal_query_mask).to(device),
        torch.from_numpy(batch.support_mask).to(device),
    )


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    selected = mask.bool()
    if not bool(selected.any()):
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[selected], targets[selected].long())


def _accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    selected = mask.bool()
    total = float(selected.sum().item())
    if total <= 0:
        return float("nan"), 0.0
    pred = torch.argmax(logits, dim=-1)
    correct = ((pred == targets.long()).float() * mask).sum().item()
    return float(correct / total), total


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    bank: FeatureBank,
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
    reversal_correct = 0.0
    reversal_total = 0.0
    for _ in range(batches):
        batch = generate_episode_batch(bank, spec, batch_size, rng)
        x, y, mask, initial_mask, reversal_mask, _ = batch_to_torch(batch, device)
        logits = model(x)
        loss_sum += float(masked_cross_entropy(logits, y, mask).item())
        acc, total = _accuracy_from_logits(logits, y, mask)
        query_correct += acc * total
        query_total += total
        acc, total = _accuracy_from_logits(logits, y, initial_mask)
        if total:
            initial_correct += acc * total
            initial_total += total
        acc, total = _accuracy_from_logits(logits, y, reversal_mask)
        if total:
            reversal_correct += acc * total
            reversal_total += total
    return {
        "loss": loss_sum / max(batches, 1),
        "query_accuracy": query_correct / max(query_total, 1.0),
        "initial_query_accuracy": initial_correct / max(initial_total, 1.0),
        "reversal_query_accuracy": (
            reversal_correct / reversal_total if reversal_total > 0 else float("nan")
        ),
    }


def nearest_support_logits(batch: EpisodicBatch, spec: EpisodeSpec, temperature: float) -> np.ndarray:
    logits = np.zeros((batch.inputs.shape[0], batch.inputs.shape[1], spec.way), dtype=np.float32)
    large = 1e6
    temp = max(float(temperature), 1e-6)
    for batch_idx in range(batch.inputs.shape[0]):
        memory_features: list[np.ndarray] = []
        memory_labels: list[int] = []
        for step in range(batch.inputs.shape[1]):
            x = batch.inputs[batch_idx, step]
            feature = x[: spec.feature_dim]
            if batch.support_mask[batch_idx, step] > 0.5:
                label_vec = x[spec.feature_dim : spec.feature_dim + spec.way]
                memory_features.append(feature)
                memory_labels.append(int(np.argmax(label_vec)))
            if batch.query_mask[batch_idx, step] > 0.5:
                if not memory_features:
                    continue
                mem = np.stack(memory_features, axis=0)
                labels = np.asarray(memory_labels, dtype=np.int64)
                dist = np.sum((mem - feature[None, :]) ** 2, axis=1)
                class_dist = np.full((spec.way,), large, dtype=np.float32)
                for label in range(spec.way):
                    values = dist[labels == label]
                    if values.size:
                        class_dist[label] = float(np.min(values))
                logits[batch_idx, step] = -class_dist / temp
    return logits


def evaluate_nearest_support(
    bank: FeatureBank,
    spec: EpisodeSpec,
    batch_size: int,
    batches: int,
    seed: int,
    temperature: float,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    loss_sum = 0.0
    query_correct = 0.0
    query_total = 0.0
    initial_correct = 0.0
    initial_total = 0.0
    reversal_correct = 0.0
    reversal_total = 0.0
    for _ in range(batches):
        batch = generate_episode_batch(bank, spec, batch_size, rng)
        logits_np = nearest_support_logits(batch, spec, temperature)
        logits = torch.from_numpy(logits_np)
        targets = torch.from_numpy(batch.targets)
        mask = torch.from_numpy(batch.query_mask)
        initial_mask = torch.from_numpy(batch.initial_query_mask)
        reversal_mask = torch.from_numpy(batch.reversal_query_mask)
        loss_sum += float(masked_cross_entropy(logits, targets, mask).item())
        acc, total = _accuracy_from_logits(logits, targets, mask)
        query_correct += acc * total
        query_total += total
        acc, total = _accuracy_from_logits(logits, targets, initial_mask)
        if total:
            initial_correct += acc * total
            initial_total += total
        acc, total = _accuracy_from_logits(logits, targets, reversal_mask)
        if total:
            reversal_correct += acc * total
            reversal_total += total
    return {
        "loss": loss_sum / max(batches, 1),
        "query_accuracy": query_correct / max(query_total, 1.0),
        "initial_query_accuracy": initial_correct / max(initial_total, 1.0),
        "reversal_query_accuracy": (
            reversal_correct / reversal_total if reversal_total > 0 else float("nan")
        ),
    }


def build_model(
    model_name: str,
    base_matrix: sparse.coo_matrix,
    spec: EpisodeSpec,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, str, int, int]:
    if model_name == MODEL_GRU:
        model = GRUEpisodicClassifier(
            input_dim=spec.input_dim,
            output_dim=spec.way,
            hidden_size=args.gru_hidden,
            seed=args.init_seed + seed,
        ).to(device)
        return (
            model,
            "dense",
            model.recurrent_parameter_count(),
            model.trainable_parameter_count(),
        )

    if model_name in FAST_MEMORY_MODEL_TO_BASE:
        base_model_name = FAST_MEMORY_MODEL_TO_BASE[model_name]
        init_matrix = mb.matrix_for_model(base_matrix, base_model_name, seed=args.init_seed + seed)
        runtime = mb.runtime_for_model(base_model_name, args.recurrent_runtime)
        model = MatrixFastMemoryRNN(
            recurrent=init_matrix,
            input_dim=spec.input_dim,
            output_dim=spec.way,
            feature_dim=spec.feature_dim,
            runtime=runtime,
            state_clip=args.state_clip,
            memory_decay=args.fast_memory_decay,
            memory_temperature=args.fast_memory_temperature,
            encoder_steps=args.fast_memory_encoder_steps,
            seed=args.init_seed + seed,
            freeze_recurrent=args.freeze_recurrent,
        ).to(device)
        runtime_label = f"{runtime}_fast_memory"
        if args.freeze_recurrent:
            runtime_label += "_frozen"
        elif args.recurrent_prior_l2 > 0:
            runtime_label += "_prior_l2"
        return (
            model,
            runtime_label,
            model.recurrent_parameter_count(),
            model.trainable_parameter_count(),
        )

    init_matrix = mb.matrix_for_model(base_matrix, model_name, seed=args.init_seed + seed)
    runtime = mb.runtime_for_model(model_name, args.recurrent_runtime)
    model = MatrixEpisodicRNN(
        recurrent=init_matrix,
        input_dim=spec.input_dim,
        output_dim=spec.way,
        runtime=runtime,
        state_clip=args.state_clip,
        seed=args.init_seed + seed,
        freeze_recurrent=args.freeze_recurrent,
    ).to(device)
    runtime_label = runtime
    if args.freeze_recurrent:
        runtime_label += "_frozen"
    elif args.recurrent_prior_l2 > 0:
        runtime_label += "_prior_l2"
    return model, runtime_label, model.recurrent_parameter_count(), model.trainable_parameter_count()


def train_one_model(
    model_name: str,
    seed: int,
    base_matrix: sparse.coo_matrix,
    train_bank: FeatureBank,
    val_bank: FeatureBank,
    test_bank: FeatureBank,
    spec: EpisodeSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    if model_name == MODEL_NEAREST:
        val_metrics = evaluate_nearest_support(
            val_bank,
            spec,
            args.batch_size,
            args.val_batches,
            seed=args.val_seed + seed,
            temperature=args.nearest_temperature,
        )
        test_metrics = evaluate_nearest_support(
            test_bank,
            spec,
            args.batch_size,
            args.test_batches,
            seed=args.test_seed + seed,
            temperature=args.nearest_temperature,
        )
        return {
            "model": model_name,
            "seed": seed,
            "runtime": "none",
            "N": 0,
            "init_nonzero_edges": 0,
            "recurrent_params": 0,
            "trainable_params": 0,
            "timesteps": spec.timesteps,
            "freeze_recurrent": int(bool(args.freeze_recurrent)),
            "recurrent_prior_l2": float(args.recurrent_prior_l2),
            "best_val_loss": val_metrics["loss"],
            "test_loss": test_metrics["loss"],
            "test_query_accuracy": test_metrics["query_accuracy"],
            "test_initial_query_accuracy": test_metrics["initial_query_accuracy"],
            "test_reversal_query_accuracy": test_metrics["reversal_query_accuracy"],
        }, []

    torch.manual_seed(seed)
    np.random.seed(seed)
    model, runtime, recurrent_params, trainable_params = build_model(
        model_name, base_matrix, spec, args, seed, device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_rng = np.random.default_rng(args.data_seed + seed)
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_wait = 0
    history: list[dict[str, float | int | str]] = []
    print(
        "model-start "
        f"model={model_name} seed={seed} runtime={runtime} "
        f"N={getattr(model, 'N', getattr(model, 'hidden_size', 0))} "
        f"recurrent_params={recurrent_params} trainable_params={trainable_params} "
        f"timesteps={spec.timesteps} input_dim={spec.input_dim} output_dim={spec.way}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        last_log = started
        train_loss_sum = 0.0
        train_prior_sum = 0.0
        for batch_idx in range(1, args.train_batches + 1):
            batch = generate_episode_batch(train_bank, spec, args.batch_size, train_rng)
            x, targets, query_mask, _, _, _ = batch_to_torch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            task_loss = masked_cross_entropy(logits, targets, query_mask)
            prior_loss = task_loss.new_tensor(0.0)
            if args.recurrent_prior_l2 > 0 and hasattr(model, "recurrent_prior_loss"):
                prior_loss = model.recurrent_prior_loss()
            loss = task_loss + args.recurrent_prior_l2 * prior_loss
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss_sum += float(task_loss.item())
            train_prior_sum += float(prior_loss.item())
            now = time.time()
            if args.log_every_seconds > 0 and now - last_log >= args.log_every_seconds:
                print(
                    "progress "
                    f"model={model_name} seed={seed} epoch={epoch}/{args.epochs} "
                    f"batch={batch_idx}/{args.train_batches} "
                    f"batch_loss={task_loss.item():.6g} "
                    f"prior_loss={prior_loss.item():.6g} "
                    f"running_train_loss={train_loss_sum / batch_idx:.6g} "
                    f"elapsed={_format_seconds(now - started)}",
                    flush=True,
                )
                last_log = now

        train_loss = train_loss_sum / max(args.train_batches, 1)
        train_prior_loss = train_prior_sum / max(args.train_batches, 1)
        val_metrics = evaluate_model(
            model,
            val_bank,
            spec,
            args.batch_size,
            args.val_batches,
            device,
            seed=args.val_seed + seed,
        )
        if val_metrics["loss"] < best_val_loss - 1e-7:
            best_val_loss = float(val_metrics["loss"])
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            patience_wait = 0
        else:
            patience_wait += 1
        history.append(
            {
                "model": model_name,
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_query_accuracy": val_metrics["query_accuracy"],
                "val_initial_query_accuracy": val_metrics["initial_query_accuracy"],
                "val_reversal_query_accuracy": val_metrics["reversal_query_accuracy"],
                "best_val_loss": best_val_loss,
                "recurrent_prior_loss": train_prior_loss,
                "patience_wait": patience_wait,
            }
        )
        print(
            "loss "
            f"model={model_name} seed={seed} epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.6g} val_loss={val_metrics['loss']:.6g} "
            f"prior_loss={train_prior_loss:.6g} "
            f"val_acc={val_metrics['query_accuracy']:.4f} "
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
    test_metrics = evaluate_model(
        model,
        test_bank,
        spec,
        args.batch_size,
        args.test_batches,
        device,
        seed=args.test_seed + seed,
    )
    metrics: dict[str, float | int | str] = {
        "model": model_name,
        "seed": seed,
        "runtime": runtime,
        "N": int(getattr(model, "N", getattr(model, "hidden_size", 0))),
        "init_nonzero_edges": int(base_matrix.nnz if model_name != MODEL_GRU else 0),
        "recurrent_params": recurrent_params,
        "trainable_params": trainable_params,
        "timesteps": spec.timesteps,
        "freeze_recurrent": int(bool(args.freeze_recurrent)),
        "recurrent_prior_l2": float(args.recurrent_prior_l2),
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_query_accuracy": test_metrics["query_accuracy"],
        "test_initial_query_accuracy": test_metrics["initial_query_accuracy"],
        "test_reversal_query_accuracy": test_metrics["reversal_query_accuracy"],
    }
    print(
        "model-done "
        f"model={model_name} seed={seed} best_val_loss={best_val_loss:.6g} "
        f"test_acc={test_metrics['query_accuracy']:.4f}",
        flush=True,
    )
    return metrics, history


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def plot_outputs(output_dir: Path, metrics: pd.DataFrame, history: pd.DataFrame) -> None:
    summary = metrics.groupby("model", as_index=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    x = np.arange(len(summary))
    width = 0.25
    bars = [
        ("test_initial_query_accuracy", "Initial queries"),
        ("test_reversal_query_accuracy", "After reversal"),
        ("test_query_accuracy", "All queries"),
    ]
    present = [(col, label) for col, label in bars if col in summary and not summary[col].isna().all()]
    for idx, (col, label) in enumerate(present):
        ax.bar(x + (idx - (len(present) - 1) / 2.0) * width, summary[col], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Episodic associative classification")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "omniglot_associative_accuracy.png")
    plt.close(fig)

    if not history.empty:
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        for model, group in history.groupby("model"):
            curve = group.groupby("epoch", as_index=False).agg(
                train_loss=("train_loss", "mean"),
                val_loss=("val_loss", "mean"),
            )
            ax.plot(curve["epoch"], curve["train_loss"], linestyle="--", alpha=0.6, label=f"{model} train")
            ax.plot(curve["epoch"], curve["val_loss"], label=f"{model} val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Query cross entropy")
        ax.set_title("Episodic associative learning loss")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "omniglot_associative_loss.png")
        plt.close(fig)


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    args: argparse.Namespace,
    spec: EpisodeSpec,
    train_bank: FeatureBank,
    val_bank: FeatureBank,
    test_bank: FeatureBank,
) -> None:
    benchmark_name = (
        "Omniglot episodic few-shot classification"
        if args.dataset == "omniglot"
        else "Synthetic episodic few-shot smoke benchmark"
    )
    reversal_text = (
        f"with `{spec.reversal_count}` relabeled classes per episode"
        if spec.reversal_count
        else "without within-episode label reversal"
    )
    lines = [
        "# Omniglot Associative Benchmark",
        "",
        f"Benchmark: {benchmark_name}, {spec.way}-way {spec.shot}-shot, {reversal_text}.",
        "",
        "Task mapping: support examples are presented as sensory features plus a label channel; query examples omit the label and require the recurrent system to recall the episode-local association. Optional reversal support steps relabel a subset of classes and test online updating.",
        "",
        f"Train bank: `{train_bank.name}` with `{train_bank.num_classes}` classes.",
        f"Validation bank: `{val_bank.name}` with `{val_bank.num_classes}` classes.",
        f"Test bank: `{test_bank.name}` with `{test_bank.num_classes}` classes.",
        f"Feature dimension: `{spec.feature_dim}`; input dimension: `{spec.input_dim}`; timesteps: `{spec.timesteps}`.",
        f"Connectome expansion factor: `{getattr(args, 'expand_factor', 1.0)}`; target neurons: `{getattr(args, 'expand_target_neurons', 0)}`.",
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
        "Interpretation note: compare connectome-seeded models primarily against same-size random and weight-shuffled controls, then against the GRU and nearest-support baselines to determine whether the biological prior is helping the episodic associative-learning problem rather than merely solving an easy embedding task.",
        "",
    ]
    (output_dir / "omniglot_associative_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_outputs(
    output_dir: Path,
    metrics_rows: list[dict[str, float | int | str]],
    history_rows: list[dict[str, float | int | str]],
    args: argparse.Namespace,
    spec: EpisodeSpec,
    train_bank: FeatureBank,
    val_bank: FeatureBank,
    test_bank: FeatureBank,
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
            test_initial_query_accuracy_mean=("test_initial_query_accuracy", "mean"),
            test_reversal_query_accuracy_mean=("test_reversal_query_accuracy", "mean"),
            test_loss_mean=("test_loss", "mean"),
            runtime=("runtime", "first"),
            init_nonzero_edges=("init_nonzero_edges", "first"),
            trainable_params=("trainable_params", "first"),
            recurrent_params=("recurrent_params", "first"),
            N=("N", "first"),
            timesteps=("timesteps", "first"),
            freeze_recurrent=("freeze_recurrent", "first"),
            recurrent_prior_l2=("recurrent_prior_l2", "first"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": serializable_args(args),
                "episode_spec": spec.__dict__,
                "train_bank": {
                    "name": train_bank.name,
                    "num_classes": train_bank.num_classes,
                },
                "val_bank": {
                    "name": val_bank.name,
                    "num_classes": val_bank.num_classes,
                },
                "test_bank": {
                    "name": test_bank.name,
                    "num_classes": test_bank.num_classes,
                },
                "task": "episodic_few_shot_associative_classification",
            },
            f,
            indent=2,
            sort_keys=True,
        )
    plot_outputs(output_dir, metrics, history)
    write_report(output_dir, metrics, summary, args, spec, train_bank, val_bank, test_bank)
    write_artifact_manifest(
        output_dir,
        config={
            "args": serializable_args(args),
            "episode_spec": spec.__dict__,
        },
        extra={"stage": "omniglot_associative_learning"},
    )


def add_connectome_expansion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--expand-factor",
        type=float,
        default=1.0,
        help=(
            "Expand the recurrent connectome with a directed signed degree-corrected "
            "SBM before training. 1.0 keeps the original matrix."
        ),
    )
    parser.add_argument(
        "--expand-target-neurons",
        type=int,
        default=0,
        help="Optional explicit expanded neuron count. 0 uses --expand-factor.",
    )
    parser.add_argument(
        "--expand-seed",
        type=int,
        default=9100,
        help="Random seed for DCSBM connectome expansion.",
    )


def validate_connectome_expansion_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.expand_factor < 1.0:
        parser.error("--expand-factor must be at least 1.0")
    if args.expand_target_neurons < 0:
        parser.error("--expand-target-neurons must be nonnegative")


def load_benchmark_matrix(args: argparse.Namespace) -> sparse.coo_matrix:
    base_matrix = mb.load_base_matrix(args.matrix, args.max_neurons)
    target_neurons = args.expand_target_neurons if args.expand_target_neurons > 0 else None
    if args.expand_factor <= 1.0 and target_neurons is None:
        return base_matrix
    result = expand_connectome_dcsbm(
        base_matrix,
        factor=args.expand_factor,
        target_neurons=target_neurons,
        seed=args.expand_seed,
    )
    if not result.preserved_original_submatrix:
        raise RuntimeError("connectome expansion failed to preserve the original submatrix")
    metadata = result.metadata()
    (args.output_dir / "connectome_expansion.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "connectome-expanded "
        f"method={metadata['method']} original_N={result.original_n} "
        f"target_N={result.target_n} original_edges={result.original_edges} "
        f"expanded_edges={result.expanded_edges} sampled_events={result.sampled_edge_events} "
        f"seed={result.seed}",
        flush=True,
    )
    return result.matrix


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Established episodic few-shot associative benchmark for testing "
            "mushroom-body connectome priors on Omniglot-style label binding."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="omniglot",
        help="Use real Omniglot or a deterministic synthetic class bank for smoke tests.",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "outputs" / "hemibrain_mushroom_body_plume" / "adjacency_unsigned.npz",
        help="Prepared mushroom-body adjacency npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "omniglot_associative",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "omniglot")
    parser.add_argument("--download", action="store_true", help="Download Omniglot if needed.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(DEFAULT_MODELS))
    parser.add_argument("--recurrent-runtime", choices=mb.RUNTIME_CHOICES, default="sparse")
    parser.add_argument("--max-neurons", type=int, default=0)
    add_connectome_expansion_args(parser)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-batches", type=int, default=200)
    parser.add_argument("--val-batches", type=int, default=40)
    parser.add_argument("--test-batches", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--state-clip", type=float, default=5.0)
    parser.add_argument(
        "--freeze-recurrent",
        action="store_true",
        help="Freeze recurrent connectome/control weights and train only input/readout or fast-memory encoder parameters.",
    )
    parser.add_argument(
        "--recurrent-prior-l2",
        type=float,
        default=0.0,
        help="L2 penalty weight that keeps trainable recurrent weights near their initialization.",
    )
    parser.add_argument(
        "--fast-memory-decay",
        type=float,
        default=0.92,
        help="Per-timestep decay for fast associative-memory models.",
    )
    parser.add_argument(
        "--fast-memory-temperature",
        type=float,
        default=0.2,
        help="Similarity temperature for fast associative-memory logits.",
    )
    parser.add_argument(
        "--fast-memory-encoder-steps",
        type=int,
        default=2,
        help="Recurrent sensory-encoder refinement steps for fast associative-memory models.",
    )
    parser.add_argument("--log-every-seconds", type=float, default=30.0)
    parser.add_argument("--way", type=int, default=20)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--queries-per-class", type=int, default=1)
    parser.add_argument(
        "--reversal-count",
        type=int,
        default=0,
        help="Number of episode-local classes to relabel after initial queries. 0 matches standard few-shot.",
    )
    parser.add_argument("--feature-noise-std", type=float, default=0.0)
    parser.add_argument("--embedding", choices=EMBEDDING_CHOICES, default="random_projection")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--embedding-sparsity", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--val-class-fraction", type=float, default=0.15)
    parser.add_argument("--max-classes", type=int, default=0)
    parser.add_argument("--gru-hidden", type=int, default=256)
    parser.add_argument("--nearest-temperature", type=float, default=0.1)
    parser.add_argument("--synthetic-feature-dim", type=int, default=64)
    parser.add_argument("--synthetic-samples-per-class", type=int, default=20)
    parser.add_argument("--synthetic-train-classes", type=int, default=120)
    parser.add_argument("--synthetic-val-classes", type=int, default=40)
    parser.add_argument("--synthetic-test-classes", type=int, default=40)
    parser.add_argument("--synthetic-class-noise-std", type=float, default=0.08)
    parser.add_argument("--data-seed", type=int, default=12345)
    parser.add_argument("--init-seed", type=int, default=7100)
    parser.add_argument("--val-seed", type=int, default=22000)
    parser.add_argument("--test-seed", type=int, default=33000)
    args = parser.parse_args(argv)
    if args.way < 2:
        parser.error("--way must be at least 2")
    if args.shot < 1:
        parser.error("--shot must be at least 1")
    if args.queries_per_class < 1:
        parser.error("--queries-per-class must be at least 1")
    if args.reversal_count < 0 or args.reversal_count > args.way:
        parser.error("--reversal-count must be between 0 and --way")
    if args.reversal_count == 1:
        parser.error("--reversal-count must be 0 or at least 2 so labels can be exchanged")
    if not (0.0 < args.embedding_sparsity <= 1.0):
        parser.error("--embedding-sparsity must be in (0, 1]")
    if args.embedding == "random_projection" and args.embedding_dim < 1:
        parser.error("--embedding-dim must be positive for random_projection")
    if not (0.0 <= args.fast_memory_decay <= 1.0):
        parser.error("--fast-memory-decay must be in [0, 1]")
    if args.fast_memory_temperature <= 0:
        parser.error("--fast-memory-temperature must be positive")
    if args.fast_memory_encoder_steps < 1:
        parser.error("--fast-memory-encoder-steps must be at least 1")
    if args.recurrent_prior_l2 < 0:
        parser.error("--recurrent-prior-l2 must be nonnegative")
    if args.dataset == "synthetic":
        for name in ("synthetic_train_classes", "synthetic_val_classes", "synthetic_test_classes"):
            if getattr(args, name) < args.way:
                parser.error(f"--{name.replace('_', '-')} must be at least --way")
    validate_connectome_expansion_args(parser, args)
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = mb.select_device(args.device)
    print(
        "run-start "
        f"task=episodic_few_shot_associative_classification dataset={args.dataset} "
        f"output_dir={args.output_dir} matrix={args.matrix} device={device}",
        flush=True,
    )
    base_matrix = load_benchmark_matrix(args)
    train_bank, val_bank, test_bank = load_feature_banks(args)
    spec = EpisodeSpec(
        way=args.way,
        shot=args.shot,
        queries_per_class=args.queries_per_class,
        reversal_count=args.reversal_count,
        feature_dim=train_bank.feature_dim,
        feature_noise_std=args.feature_noise_std,
    )
    print(
        "data-ready "
        f"train_classes={train_bank.num_classes} val_classes={val_bank.num_classes} "
        f"test_classes={test_bank.num_classes} feature_dim={spec.feature_dim} "
        f"input_dim={spec.input_dim} timesteps={spec.timesteps} "
        f"N={base_matrix.shape[0]} edges={base_matrix.nnz} "
        f"models={','.join(args.models)}",
        flush=True,
    )
    metrics_rows: list[dict[str, float | int | str]] = []
    history_rows: list[dict[str, float | int | str]] = []
    for model_name in args.models:
        for seed in args.seeds:
            metrics, history = train_one_model(
                model_name,
                seed,
                base_matrix,
                train_bank,
                val_bank,
                test_bank,
                spec,
                args,
                device,
            )
            metrics_rows.append(metrics)
            history_rows.extend(history)
    write_outputs(
        args.output_dir,
        metrics_rows,
        history_rows,
        args,
        spec,
        train_bank,
        val_bank,
        test_bank,
    )
    print(
        f"complete metrics={args.output_dir / 'metrics_by_seed.csv'} "
        f"report={args.output_dir / 'omniglot_associative_report.md'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
