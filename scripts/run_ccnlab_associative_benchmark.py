#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "hemibrain_cx_bpu_matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "hemibrain_cx_bpu_xdg_cache"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_mb_associative_learning import (  # noqa: E402
    MODEL_DEGREE_PRESERVING,
    MODEL_HEMIBRAIN,
    MODEL_RANDOM,
    MODEL_WEIGHT_SHUFFLE,
    load_base_matrix,
    matrix_for_model,
)
from src.run_manifest import write_artifact_manifest  # noqa: E402


import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MODEL_RESCORLA_WAGNER = "rescorla_wagner"
MODEL_KALMAN_FILTER = "kalman_filter"
MODEL_TEMPORAL_DIFFERENCE = "temporal_difference"
MODEL_CONNECTOME_RW = "connectome_rescorla_wagner"
MODEL_RANDOM_RW = "random_sparse_rescorla_wagner"
MODEL_DEGREE_RW = "degree_preserving_rescorla_wagner"
MODEL_WEIGHT_RW = "weight_shuffle_rescorla_wagner"
MODEL_CONNECTOME_KALMAN = "connectome_kalman_filter"
MODEL_RANDOM_KALMAN = "random_sparse_kalman_filter"
MODEL_DEGREE_KALMAN = "degree_preserving_kalman_filter"
MODEL_WEIGHT_KALMAN = "weight_shuffle_kalman_filter"
MODEL_CONNECTOME_TD = "connectome_temporal_difference"
MODEL_RANDOM_TD = "random_sparse_temporal_difference"
MODEL_DEGREE_TD = "degree_preserving_temporal_difference"
MODEL_WEIGHT_TD = "weight_shuffle_temporal_difference"
MODEL_CONNECTOME_ALIAS = "connectome_seeded"
MODEL_HEMIBRAIN_CONV_ALIAS = "hemibrain_conv_fast_memory"
MODEL_RANDOM_CONV_ALIAS = "random_sparse_conv_fast_memory"
MODEL_WEIGHT_CONV_ALIAS = "weight_shuffle_conv_fast_memory"

CONNECTOME_MODEL_ALIASES = {
    MODEL_HEMIBRAIN: MODEL_HEMIBRAIN,
    MODEL_CONNECTOME_ALIAS: MODEL_HEMIBRAIN,
    MODEL_HEMIBRAIN_CONV_ALIAS: MODEL_HEMIBRAIN,
    MODEL_RANDOM: MODEL_RANDOM,
    MODEL_RANDOM_CONV_ALIAS: MODEL_RANDOM,
    MODEL_DEGREE_PRESERVING: MODEL_DEGREE_PRESERVING,
    MODEL_WEIGHT_SHUFFLE: MODEL_WEIGHT_SHUFFLE,
    MODEL_WEIGHT_CONV_ALIAS: MODEL_WEIGHT_SHUFFLE,
}
CONNECTOME_MODELS = tuple(CONNECTOME_MODEL_ALIASES)
BASELINE_MODELS = (
    MODEL_RESCORLA_WAGNER,
    MODEL_KALMAN_FILTER,
    MODEL_TEMPORAL_DIFFERENCE,
)
GRAPH_FEATURE_MODEL_SPECS = {
    MODEL_CONNECTOME_RW: (MODEL_HEMIBRAIN, MODEL_RESCORLA_WAGNER),
    MODEL_RANDOM_RW: (MODEL_RANDOM, MODEL_RESCORLA_WAGNER),
    MODEL_DEGREE_RW: (MODEL_DEGREE_PRESERVING, MODEL_RESCORLA_WAGNER),
    MODEL_WEIGHT_RW: (MODEL_WEIGHT_SHUFFLE, MODEL_RESCORLA_WAGNER),
    MODEL_CONNECTOME_KALMAN: (MODEL_HEMIBRAIN, MODEL_KALMAN_FILTER),
    MODEL_RANDOM_KALMAN: (MODEL_RANDOM, MODEL_KALMAN_FILTER),
    MODEL_DEGREE_KALMAN: (MODEL_DEGREE_PRESERVING, MODEL_KALMAN_FILTER),
    MODEL_WEIGHT_KALMAN: (MODEL_WEIGHT_SHUFFLE, MODEL_KALMAN_FILTER),
    MODEL_CONNECTOME_TD: (MODEL_HEMIBRAIN, MODEL_TEMPORAL_DIFFERENCE),
    MODEL_RANDOM_TD: (MODEL_RANDOM, MODEL_TEMPORAL_DIFFERENCE),
    MODEL_DEGREE_TD: (MODEL_DEGREE_PRESERVING, MODEL_TEMPORAL_DIFFERENCE),
    MODEL_WEIGHT_TD: (MODEL_WEIGHT_SHUFFLE, MODEL_TEMPORAL_DIFFERENCE),
}
GRAPH_FEATURE_MODELS = tuple(GRAPH_FEATURE_MODEL_SPECS)
MODEL_CHOICES = CONNECTOME_MODELS + GRAPH_FEATURE_MODELS + BASELINE_MODELS
DEFAULT_MODELS = (
    MODEL_HEMIBRAIN,
    MODEL_RANDOM,
    MODEL_WEIGHT_SHUFFLE,
    MODEL_RESCORLA_WAGNER,
    MODEL_KALMAN_FILTER,
    MODEL_TEMPORAL_DIFFERENCE,
)
DEFAULT_EXPERIMENTS = (
    "Acquisition_ContinuousVsPartial",
    "Extinction_ContinuousVsPartial",
    "Generalization_NovelVsInhibitor",
    "Generalization_AddVsRemove",
    "Competition_OvershadowingAndForwardBlocking",
    "Recovery_Overshadowing",
    "HigherOrder_SensoryPreconditioning",
)


@dataclass(frozen=True)
class ModelMetadata:
    runtime: str
    N: int
    input_dim: int
    feature_dim: int
    encoded_dim: int
    init_nonzero_edges: int
    recurrent_params: int
    trainable_params: int


@dataclass
class ExperimentScore:
    model: str
    seed: int
    experiment: str
    score_type: str
    score: float
    finite: bool
    empirical_rows: int
    simulated_rows: int


def _format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, rem = divmod(minutes, 60)
    return f"{hours}h{rem:02d}m"


def configure_ccnlab_root(ccnlab_root: Path | None) -> None:
    if ccnlab_root is None:
        return
    root = ccnlab_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"--ccnlab-root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def import_ccnlab_modules() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import ccnlab.benchmarks.classical as classical
        import ccnlab.evaluation as evaluation
        from ccnlab.baselines.basic import KalmanFilter
        from ccnlab.baselines.basic import RescorlaWagner
        from ccnlab.baselines.basic import TemporalDifference
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        if missing != "ccnlab":
            raise ModuleNotFoundError(
                f"Could not import CCNLab because dependency {missing!r} is missing. "
                'Install CCNLab helper dependencies with: python -m pip install "seaborn>=0.13.2" "IPython>=8.0.0"'
            ) from exc
        raise ModuleNotFoundError(
            "Could not import CCNLab. Clone https://github.com/nikhilxb/ccnlab "
            "and pass --ccnlab-root /path/to/ccnlab."
        ) from exc
    return classical, evaluation, RescorlaWagner, KalmanFilter, TemporalDifference


def load_experiments(
    classical: Any,
    names: Iterable[str],
    registry_seed: int,
) -> list[Any]:
    random.seed(int(registry_seed))
    np.random.seed(int(registry_seed) % (2**32 - 1))
    requested = list(names)
    if requested == ["all"]:
        requested = ["*"]
    return list(classical.registry(*requested))


def normalize_recurrent(matrix: sparse.spmatrix, gain: float) -> sparse.csr_matrix:
    csr = matrix.astype(np.float32).tocsr()
    csr.sum_duplicates()
    row_abs = np.asarray(np.abs(csr).sum(axis=1)).ravel().astype(np.float32)
    inv = np.zeros_like(row_abs, dtype=np.float32)
    nonzero = row_abs > 0
    inv[nonzero] = float(gain) / np.maximum(row_abs[nonzero], 1e-6)
    normalized = sparse.diags(inv, dtype=np.float32).dot(csr).tocsr()
    normalized.sum_duplicates()
    return normalized


def vector_key(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 6) for value in values)


class ConnectomeCueEncoder:
    """Fixed graph-diffusion feature map for CCNLab cue/context/time inputs."""

    def __init__(
        self,
        recurrent: sparse.spmatrix,
        cue_dim: int,
        time_basis_dim: int,
        feature_dim: int,
        encoder_steps: int,
        recurrent_gain: float,
        input_scale: float,
        hidden_scale: float,
        raw_input_scale: float,
        state_clip: float,
        seed: int,
    ) -> None:
        if recurrent.shape[0] != recurrent.shape[1]:
            raise ValueError("recurrent matrix must be square.")
        if cue_dim <= 0:
            raise ValueError("cue_dim must be positive.")
        if time_basis_dim < 0:
            raise ValueError("time_basis_dim must be nonnegative.")
        self.N = int(recurrent.shape[0])
        self.recurrent = normalize_recurrent(recurrent, recurrent_gain)
        self.cue_dim = int(cue_dim)
        self.time_basis_dim = int(time_basis_dim)
        self.input_dim = self.cue_dim + self.time_basis_dim
        self.encoder_steps = int(encoder_steps)
        self.hidden_scale = float(hidden_scale)
        self.raw_input_scale = float(raw_input_scale)
        self.state_clip = float(state_clip)
        self.cache: dict[tuple[tuple[float, ...], tuple[float, ...], int], np.ndarray] = {}

        rng = np.random.default_rng(seed)
        scale = float(input_scale) / math.sqrt(max(self.input_dim, 1))
        self.W_in = rng.normal(0.0, scale, size=(self.N, self.input_dim)).astype(np.float32)
        self.feature_dim = min(int(feature_dim), self.N)
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        self.feature_indices = np.sort(
            rng.choice(self.N, size=self.feature_dim, replace=False)
        ).astype(np.int64)
        self.encoded_dim = self.input_dim + self.feature_dim

    def _input_vector(self, cs: Iterable[float], ctx: Iterable[float], t: int) -> np.ndarray:
        cue = np.asarray(list(cs) + list(ctx), dtype=np.float32)
        if cue.shape[0] != self.cue_dim:
            raise ValueError(f"expected cue dimension {self.cue_dim}, got {cue.shape[0]}")
        if self.time_basis_dim:
            time_basis = np.zeros(self.time_basis_dim, dtype=np.float32)
            time_basis[min(max(int(t), 0), self.time_basis_dim - 1)] = 1.0
            return np.concatenate([cue, time_basis]).astype(np.float32)
        return cue

    def encode(self, cs: Iterable[float], ctx: Iterable[float], t: int) -> np.ndarray:
        cs_key = vector_key(cs)
        ctx_key = vector_key(ctx)
        t_key = min(max(int(t), 0), max(self.time_basis_dim - 1, 0))
        key = (cs_key, ctx_key, t_key)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        x = self._input_vector(cs_key, ctx_key, t_key)
        drive = self.W_in.dot(x)
        h = np.maximum(drive, 0.0).astype(np.float32)
        for _ in range(self.encoder_steps):
            h = self.recurrent.dot(h).astype(np.float32, copy=False) + drive
            h = np.maximum(h, 0.0).astype(np.float32, copy=False)
            if self.state_clip > 0:
                np.clip(h, 0.0, self.state_clip, out=h)
        hidden = h[self.feature_indices] * self.hidden_scale
        z = np.concatenate([x * self.raw_input_scale, hidden]).astype(np.float32)
        norm = float(np.linalg.norm(z))
        if norm > 1e-6:
            z /= norm
        self.cache[key] = z
        return z


class ConnectomeRPEConditioningModel:
    """Online prediction-error learner over fixed connectome/control features."""

    def __init__(
        self,
        encoder: ConnectomeCueEncoder,
        alpha: float,
        alpha_bias: float,
        trace_decay: float,
        weight_decay: float,
        response_clip: float,
        nonnegative_response: bool,
    ) -> None:
        self.encoder = encoder
        self.alpha = float(alpha)
        self.alpha_bias = float(alpha_bias)
        self.trace_decay = float(trace_decay)
        self.weight_decay = float(weight_decay)
        self.response_clip = float(response_clip)
        self.nonnegative_response = bool(nonnegative_response)
        self.reset()

    def reset(self) -> None:
        self.w = np.zeros(self.encoder.encoded_dim, dtype=np.float32)
        self.eligibility = np.zeros_like(self.w)
        self.bias = 0.0

    def act(self, cs: list[float], ctx: list[float], us: float, t: int) -> float:
        if int(t) == 0:
            self.eligibility.fill(0.0)
        z = self.encoder.encode(cs, ctx, t)
        value = float(np.dot(self.w, z) + self.bias)
        response = value
        if self.nonnegative_response:
            response = max(response, 0.0)
        if self.response_clip > 0:
            response = float(np.clip(response, 0.0, self.response_clip))

        self.eligibility = self.trace_decay * self.eligibility + z
        update_features = self.eligibility if self.trace_decay > 0 else z
        prediction = response if self.nonnegative_response else value
        rpe = float(us) - prediction
        if self.weight_decay > 0:
            self.w *= max(0.0, 1.0 - self.weight_decay)
        self.w += self.alpha * rpe * update_features
        self.bias += self.alpha_bias * rpe
        return response


class FeatureRescorlaWagner:
    """CCNLab Rescorla-Wagner update over a graph feature basis."""

    def __init__(self, encoder: ConnectomeCueEncoder, alpha: float) -> None:
        self.encoder = encoder
        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        self.w = np.zeros(self.encoder.encoded_dim, dtype=np.float64)

    def act(self, cs: list[float], ctx: list[float], us: float, t: int) -> float:
        x = self.encoder.encode(cs, ctx, t).astype(np.float64, copy=False)
        value = float(self.w.dot(x))
        rpe = float(us) - float(self.w.dot(x))
        self.w = self.w + self.alpha * rpe * x
        return value


class FeatureKalmanFilter:
    """CCNLab Kalman-filter update over a graph feature basis."""

    def __init__(
        self,
        encoder: ConnectomeCueEncoder,
        tau2: float,
        sigma_r2: float,
        sigma_w2: float,
    ) -> None:
        self.encoder = encoder
        self.D = int(encoder.encoded_dim)
        self.tau2 = float(tau2)
        self.sigma_r2 = float(sigma_r2)
        self.sigma_w2 = float(sigma_w2)
        self.Q = self.tau2 * np.identity(self.D, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.w = np.zeros(self.D, dtype=np.float64)
        self.S = self.sigma_w2 * np.identity(self.D, dtype=np.float64)

    def act(self, cs: list[float], ctx: list[float], us: float, t: int) -> float:
        x = self.encoder.encode(cs, ctx, t).astype(np.float64, copy=False)
        value = float(self.w.dot(x))
        rpe = float(us) - float(self.w.dot(x))
        S = self.S + self.Q
        residual_covariance = float(x.dot(S).dot(x) + self.sigma_r2)
        k = S.dot(x) / residual_covariance
        self.w = self.w + k * rpe
        self.S = S - float(k.dot(x)) * S
        return value


class FeatureTemporalDifference:
    """CCNLab temporal-difference update over graph features."""

    def __init__(
        self,
        encoder: ConnectomeCueEncoder,
        num_timesteps: int,
        alpha: float,
        gamma: float,
    ) -> None:
        self.encoder = encoder
        self.D = int(encoder.encoded_dim)
        self.T = int(num_timesteps)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reset()

    def reset(self) -> None:
        self.w = np.zeros(self.D * self.T, dtype=np.float64)
        self.last_x = np.zeros(self.D * self.T, dtype=np.float64)
        self.last_r = 0.0

    def act(self, cs: list[float], ctx: list[float], us: float, t: int) -> float:
        if int(t) == 0:
            self.last_x = np.zeros(self.D * self.T, dtype=np.float64)
        x = np.zeros(self.D * self.T, dtype=np.float64)
        timestep = min(max(int(t), 0), self.T - 1)
        feature = self.encoder.encode(cs, ctx, timestep).astype(np.float64, copy=False)
        x[timestep * self.D : (timestep + 1) * self.D] = feature
        value = float(self.w.dot(x))
        self._update(x=x, reward=float(us))
        if timestep + 1 == self.T:
            self._update(x=np.zeros(self.D * self.T, dtype=np.float64), reward=0.0)
        return value

    def _update(self, x: np.ndarray, reward: float) -> None:
        last_rpe = self.last_r + self.gamma * self.w.dot(x) - self.w.dot(self.last_x)
        self.w = self.w + self.alpha * last_rpe * self.last_x
        self.last_x = x
        self.last_r = reward


def connectome_metadata(
    model_name: str,
    encoder: ConnectomeCueEncoder,
    recurrent: sparse.spmatrix,
) -> ModelMetadata:
    return ModelMetadata(
        runtime="connectome_feature_rpe",
        N=encoder.N,
        input_dim=encoder.input_dim,
        feature_dim=encoder.feature_dim,
        encoded_dim=encoder.encoded_dim,
        init_nonzero_edges=int(recurrent.nnz),
        recurrent_params=int(recurrent.nnz),
        trainable_params=int(encoder.encoded_dim + 1),
    )


def graph_feature_metadata(
    model_name: str,
    learner_name: str,
    encoder: ConnectomeCueEncoder,
    recurrent: sparse.spmatrix,
    max_timesteps: int,
) -> ModelMetadata:
    if learner_name == MODEL_TEMPORAL_DIFFERENCE:
        trainable_params = encoder.encoded_dim * int(max_timesteps)
        encoded_dim = trainable_params
    elif learner_name == MODEL_KALMAN_FILTER:
        trainable_params = encoder.encoded_dim + encoder.encoded_dim * encoder.encoded_dim
        encoded_dim = encoder.encoded_dim
    elif learner_name == MODEL_RESCORLA_WAGNER:
        trainable_params = encoder.encoded_dim
        encoded_dim = encoder.encoded_dim
    else:
        raise ValueError(f"unknown feature learner: {learner_name}")
    return ModelMetadata(
        runtime=f"graph_feature_{learner_name}",
        N=encoder.N,
        input_dim=encoder.input_dim,
        feature_dim=encoder.feature_dim,
        encoded_dim=int(encoded_dim),
        init_nonzero_edges=int(recurrent.nnz),
        recurrent_params=int(recurrent.nnz),
        trainable_params=int(trainable_params),
    )


def baseline_metadata(model_name: str, exp: Any) -> ModelMetadata:
    cue_dim = len(exp.cs_space) + len(exp.ctx_space)
    if model_name == MODEL_TEMPORAL_DIFFERENCE:
        max_timesteps = max(
            len(trial) for group in exp.stimuli.values() for trial in group
        )
        params = cue_dim * max_timesteps
    elif model_name == MODEL_KALMAN_FILTER:
        params = cue_dim + cue_dim * cue_dim
    else:
        params = cue_dim
    return ModelMetadata(
        runtime="ccnlab_baseline",
        N=0,
        input_dim=cue_dim,
        feature_dim=cue_dim,
        encoded_dim=cue_dim,
        init_nonzero_edges=0,
        recurrent_params=0,
        trainable_params=int(params),
    )


def aggregate_metadata(rows: list[ModelMetadata]) -> ModelMetadata:
    if not rows:
        raise ValueError("metadata rows cannot be empty.")
    first = rows[0]
    return ModelMetadata(
        runtime=first.runtime,
        N=max(row.N for row in rows),
        input_dim=max(row.input_dim for row in rows),
        feature_dim=max(row.feature_dim for row in rows),
        encoded_dim=max(row.encoded_dim for row in rows),
        init_nonzero_edges=max(row.init_nonzero_edges for row in rows),
        recurrent_params=max(row.recurrent_params for row in rows),
        trainable_params=max(row.trainable_params for row in rows),
    )


def make_connectome_factory(
    model_name: str,
    exp: Any,
    base_matrix: sparse.coo_matrix,
    seed: int,
    args: argparse.Namespace,
) -> tuple[Callable[[str, int], ConnectomeRPEConditioningModel], ModelMetadata]:
    resolved = CONNECTOME_MODEL_ALIASES[model_name]
    recurrent = matrix_for_model(base_matrix, resolved, seed=args.init_seed + seed)
    cue_dim = len(exp.cs_space) + len(exp.ctx_space)
    encoder = ConnectomeCueEncoder(
        recurrent=recurrent,
        cue_dim=cue_dim,
        time_basis_dim=args.time_basis_dim,
        feature_dim=args.feature_dim,
        encoder_steps=args.encoder_steps,
        recurrent_gain=args.recurrent_gain,
        input_scale=args.input_scale,
        hidden_scale=args.hidden_scale,
        raw_input_scale=args.raw_input_scale,
        state_clip=args.state_clip,
        seed=args.init_seed + seed,
    )
    metadata = connectome_metadata(model_name, encoder, recurrent)

    def factory(group_name: str, subject: int) -> ConnectomeRPEConditioningModel:
        return ConnectomeRPEConditioningModel(
            encoder=encoder,
            alpha=args.alpha,
            alpha_bias=args.alpha_bias,
            trace_decay=args.trace_decay,
            weight_decay=args.weight_decay,
            response_clip=args.response_clip,
            nonnegative_response=not args.allow_negative_response,
        )

    return factory, metadata


def make_graph_feature_factory(
    model_name: str,
    exp: Any,
    base_matrix: sparse.coo_matrix,
    seed: int,
    args: argparse.Namespace,
) -> tuple[Callable[[str, int], Any], ModelMetadata]:
    topology_model, learner_name = GRAPH_FEATURE_MODEL_SPECS[model_name]
    recurrent = matrix_for_model(base_matrix, topology_model, seed=args.init_seed + seed)
    cue_dim = len(exp.cs_space) + len(exp.ctx_space)
    feature_dim = int(args.feature_learner_dim or args.feature_dim)
    max_timesteps = max(
        len(trial) for group in exp.stimuli.values() for trial in group
    )
    encoder = ConnectomeCueEncoder(
        recurrent=recurrent,
        cue_dim=cue_dim,
        time_basis_dim=args.feature_time_basis_dim,
        feature_dim=feature_dim,
        encoder_steps=args.encoder_steps,
        recurrent_gain=args.recurrent_gain,
        input_scale=args.input_scale,
        hidden_scale=args.hidden_scale,
        raw_input_scale=args.raw_input_scale,
        state_clip=args.state_clip,
        seed=args.init_seed + seed,
    )
    metadata = graph_feature_metadata(
        model_name=model_name,
        learner_name=learner_name,
        encoder=encoder,
        recurrent=recurrent,
        max_timesteps=max_timesteps,
    )

    if learner_name == MODEL_RESCORLA_WAGNER:

        def factory(group_name: str, subject: int) -> FeatureRescorlaWagner:
            return FeatureRescorlaWagner(encoder=encoder, alpha=args.rw_alpha)

    elif learner_name == MODEL_KALMAN_FILTER:

        def factory(group_name: str, subject: int) -> FeatureKalmanFilter:
            return FeatureKalmanFilter(
                encoder=encoder,
                tau2=args.kalman_tau2,
                sigma_r2=args.kalman_sigma_r2,
                sigma_w2=args.kalman_sigma_w2,
            )

    elif learner_name == MODEL_TEMPORAL_DIFFERENCE:

        def factory(group_name: str, subject: int) -> FeatureTemporalDifference:
            return FeatureTemporalDifference(
                encoder=encoder,
                num_timesteps=max_timesteps,
                alpha=args.td_alpha,
                gamma=args.td_gamma,
            )

    else:
        raise ValueError(f"unknown feature learner: {learner_name}")
    return factory, metadata


def make_baseline_factory(
    model_name: str,
    exp: Any,
    args: argparse.Namespace,
    RescorlaWagner: Any,
    KalmanFilter: Any,
    TemporalDifference: Any,
) -> tuple[Callable[[str, int], Any], ModelMetadata]:
    cs_dim = len(exp.cs_space)
    ctx_dim = len(exp.ctx_space)
    metadata = baseline_metadata(model_name, exp)

    if model_name == MODEL_RESCORLA_WAGNER:

        def factory(group_name: str, subject: int) -> Any:
            return RescorlaWagner(cs_dim=cs_dim, ctx_dim=ctx_dim, alpha=args.rw_alpha)

    elif model_name == MODEL_KALMAN_FILTER:

        def factory(group_name: str, subject: int) -> Any:
            return KalmanFilter(
                cs_dim=cs_dim,
                ctx_dim=ctx_dim,
                tau2=args.kalman_tau2,
                sigma_r2=args.kalman_sigma_r2,
                sigma_w2=args.kalman_sigma_w2,
            )

    elif model_name == MODEL_TEMPORAL_DIFFERENCE:
        max_timesteps = max(
            len(trial) for group in exp.stimuli.values() for trial in group
        )

        def factory(group_name: str, subject: int) -> Any:
            return TemporalDifference(
                cs_dim=cs_dim,
                ctx_dim=ctx_dim,
                num_timesteps=max_timesteps,
                alpha=args.td_alpha,
                gamma=args.td_gamma,
            )

    else:
        raise ValueError(f"unknown baseline model: {model_name}")
    return factory, metadata


def make_model_factory(
    model_name: str,
    exp: Any,
    base_matrix: sparse.coo_matrix | None,
    seed: int,
    args: argparse.Namespace,
    RescorlaWagner: Any,
    KalmanFilter: Any,
    TemporalDifference: Any,
) -> tuple[Callable[[str, int], Any], ModelMetadata]:
    if model_name in CONNECTOME_MODEL_ALIASES:
        if base_matrix is None:
            raise ValueError(f"{model_name} requires --matrix")
        return make_connectome_factory(model_name, exp, base_matrix, seed, args)
    if model_name in GRAPH_FEATURE_MODEL_SPECS:
        if base_matrix is None:
            raise ValueError(f"{model_name} requires --matrix")
        return make_graph_feature_factory(model_name, exp, base_matrix, seed, args)
    return make_baseline_factory(
        model_name,
        exp,
        args,
        RescorlaWagner,
        KalmanFilter,
        TemporalDifference,
    )


def simulate_experiment(
    exp: Any,
    model_factory: Callable[[str, int], Any],
    subjects: int,
) -> pd.DataFrame:
    exp.reset()
    for g, group in exp.stimuli.items():
        for subject in range(subjects):
            model = model_factory(str(g), subject)
            for i, trial in enumerate(group):
                for t, timestep in enumerate(trial):
                    cs, ctx, us = exp.stimulus(g, i, t, vector=True)
                    response = model.act(cs, ctx, us, t)
                    exp.data[g][i][t]["response"].append(response)
    return exp.simulated_results()


def score_experiment(evaluation: Any, empirical: pd.DataFrame, simulated: pd.DataFrame) -> tuple[str, float]:
    if len(list(empirical.value)) == 2:
        return "ratio_of_ratios", float(evaluation.ratio_of_ratios(empirical, simulated))
    return "correlation", float(evaluation.correlation(empirical, simulated))


def run_model_seed(
    model_name: str,
    seed: int,
    args: argparse.Namespace,
    base_matrix: sparse.coo_matrix | None,
    classical: Any,
    evaluation: Any,
    RescorlaWagner: Any,
    KalmanFilter: Any,
    TemporalDifference: Any,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    started = time.time()
    experiments = load_experiments(
        classical,
        args.experiments,
        registry_seed=args.registry_seed + seed,
    )
    if not experiments:
        raise RuntimeError(f"No CCNLab experiments matched {args.experiments}")

    print(
        "model-start "
        f"model={model_name} seed={seed} experiments={len(experiments)} "
        f"subjects={args.subjects}",
        flush=True,
    )

    score_rows: list[ExperimentScore] = []
    metadata_rows: list[ModelMetadata] = []
    for exp_index, exp in enumerate(experiments, start=1):
        factory, metadata = make_model_factory(
            model_name,
            exp,
            base_matrix,
            seed,
            args,
            RescorlaWagner,
            KalmanFilter,
            TemporalDifference,
        )
        simulated = simulate_experiment(exp, factory, args.subjects)
        score_type, score = score_experiment(evaluation, exp.empirical_results, simulated)
        finite = bool(np.isfinite(score))
        score_rows.append(
            ExperimentScore(
                model=model_name,
                seed=seed,
                experiment=str(exp.name),
                score_type=score_type,
                score=score,
                finite=finite,
                empirical_rows=int(len(exp.empirical_results)),
                simulated_rows=int(len(simulated)),
            )
        )
        metadata_rows.append(metadata)
        print(
            "experiment-score "
            f"model={model_name} seed={seed} index={exp_index}/{len(experiments)} "
            f"experiment={exp.name} score_type={score_type} score={score:.6g} "
            f"finite={int(finite)} elapsed={_format_seconds(time.time() - started)}",
            flush=True,
        )

    frame = pd.DataFrame([row.__dict__ for row in score_rows])
    finite_scores = frame.loc[frame["finite"], "score"].astype(float)
    finite_corr = frame.loc[
        frame["finite"] & (frame["score_type"] == "correlation"), "score"
    ].astype(float)
    finite_ratio = frame.loc[
        frame["finite"] & (frame["score_type"] == "ratio_of_ratios"), "score"
    ].astype(float)
    metadata = aggregate_metadata(metadata_rows)
    metrics: dict[str, float | int | str] = {
        "model": model_name,
        "seed": seed,
        "runtime": metadata.runtime,
        "N": metadata.N,
        "input_dim": metadata.input_dim,
        "feature_dim": metadata.feature_dim,
        "encoded_dim": metadata.encoded_dim,
        "init_nonzero_edges": metadata.init_nonzero_edges,
        "recurrent_params": metadata.recurrent_params,
        "trainable_params": metadata.trainable_params,
        "test_ccnlab_score": float(finite_scores.mean()) if len(finite_scores) else float("nan"),
        "test_ccnlab_score_std": float(finite_scores.std(ddof=1))
        if len(finite_scores) > 1
        else float("nan"),
        "test_ccnlab_correlation": float(finite_corr.mean())
        if len(finite_corr)
        else float("nan"),
        "test_ccnlab_ratio": float(finite_ratio.mean()) if len(finite_ratio) else float("nan"),
        "test_ccnlab_finite_score_count": int(len(finite_scores)),
        "test_ccnlab_experiment_count": int(len(frame)),
        "subjects": int(args.subjects),
    }
    print(
        "model-done "
        f"model={model_name} seed={seed} test_ccnlab_score={metrics['test_ccnlab_score']:.6g} "
        f"finite_scores={metrics['test_ccnlab_finite_score_count']}/"
        f"{metrics['test_ccnlab_experiment_count']} elapsed={_format_seconds(time.time() - started)}",
        flush=True,
    )
    return metrics, [row.__dict__ for row in score_rows]


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def write_outputs(
    output_dir: Path,
    metrics_rows: list[dict[str, float | int | str]],
    experiment_rows: list[dict[str, float | int | str]],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metrics_rows)
    experiment_scores = pd.DataFrame(experiment_rows)
    metrics.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    experiment_scores.to_csv(output_dir / "experiment_scores.csv", index=False)

    summary = (
        metrics.groupby("model")
        .agg(
            test_ccnlab_score_mean=("test_ccnlab_score", "mean"),
            test_ccnlab_score_std=("test_ccnlab_score", "std"),
            test_ccnlab_correlation_mean=("test_ccnlab_correlation", "mean"),
            test_ccnlab_ratio_mean=("test_ccnlab_ratio", "mean"),
            test_ccnlab_finite_score_count=("test_ccnlab_finite_score_count", "mean"),
            test_ccnlab_experiment_count=("test_ccnlab_experiment_count", "first"),
            runtime=("runtime", "first"),
            init_nonzero_edges=("init_nonzero_edges", "first"),
            trainable_params=("trainable_params", "first"),
            recurrent_params=("recurrent_params", "first"),
            N=("N", "first"),
            feature_dim=("feature_dim", "first"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    leaderboard = summary.sort_values(
        "test_ccnlab_score_mean", ascending=False, na_position="last"
    ).reset_index(drop=True)
    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": serializable_args(args),
                "task": "ccnlab_classical_conditioning",
                "task_rationale": (
                    "CCNLab evaluates classical-conditioning models on empirical "
                    "phenomena. Connectome/control models use the same matrix "
                    "topology-control family as the mushroom-body associative benchmarks, "
                    "but with an online cue-value prediction-error architecture."
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    plot_metrics(output_dir, metrics, experiment_scores)
    write_report(output_dir, metrics, experiment_scores, summary, leaderboard, args)
    write_artifact_manifest(
        output_dir,
        config={"args": serializable_args(args)},
        extra={"stage": "ccnlab_associative_benchmark"},
    )


def plot_metrics(output_dir: Path, metrics: pd.DataFrame, experiment_scores: pd.DataFrame) -> None:
    summary = metrics.groupby("model", as_index=False).agg(
        test_ccnlab_score=("test_ccnlab_score", "mean")
    )
    summary = summary.sort_values("test_ccnlab_score", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.bar(summary["model"], summary["test_ccnlab_score"], color="#3267a8")
    ax.set_ylabel("Mean CCNLab score")
    ax.set_title("CCNLab associative-conditioning benchmark")
    ax.set_ylim(
        min(-0.1, float(summary["test_ccnlab_score"].min()) - 0.05),
        max(1.0, float(summary["test_ccnlab_score"].max()) + 0.05),
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=22)
    fig.tight_layout()
    fig.savefig(output_dir / "ccnlab_scores.png")
    plt.close(fig)

    finite = experiment_scores[experiment_scores["finite"]].copy()
    if finite.empty:
        return
    pivot = finite.pivot_table(
        index="experiment",
        columns="model",
        values="score",
        aggfunc="mean",
    )
    fig_height = max(4.0, 0.35 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(9, fig_height), dpi=150)
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_title("Per-experiment scores")
    fig.colorbar(image, ax=ax, label="score")
    fig.tight_layout()
    fig.savefig(output_dir / "ccnlab_experiment_scores.png")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in frame]
    if frame.empty or not available:
        return "_No rows._"
    lines = [
        "| " + " | ".join(available) + " |",
        "| " + " | ".join("---" for _ in available) + " |",
    ]
    for _, row in frame[available].iterrows():
        values = []
        for column in available:
            value = row[column]
            if isinstance(value, float):
                values.append("nan" if math.isnan(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    experiment_scores: pd.DataFrame,
    summary: pd.DataFrame,
    leaderboard: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    finite = experiment_scores[experiment_scores["finite"]].copy()
    per_exp = (
        finite.pivot_table(index="experiment", columns="model", values="score", aggfunc="mean")
        .reset_index()
        if not finite.empty
        else pd.DataFrame()
    )
    lines = [
        "# CCNLab Associative Benchmark",
        "",
        "Task: simulate CCNLab classical-conditioning experiments and score model outputs against empirical summaries using CCNLab's evaluation metrics.",
        "",
        "Connectome models use a fixed graph-diffusion cue/context/time encoder and an online reward-prediction-error readout. The random-sparse, degree-preserving, and weight-shuffle controls reuse the same topology-control semantics as the existing mushroom-body associative benchmarks.",
        "",
        f"Subjects per experiment group: `{args.subjects}`",
        f"Experiments: `{', '.join(args.experiments)}`",
        f"Feature neurons: `{args.feature_dim}`",
        f"Feature-learner neurons: `{args.feature_learner_dim or args.feature_dim}`",
        f"Encoder graph steps: `{args.encoder_steps}`",
        "",
        "## Leaderboard",
        "",
        _markdown_table(
            leaderboard,
            [
                "rank",
                "model",
                "test_ccnlab_score_mean",
                "test_ccnlab_score_std",
                "test_ccnlab_correlation_mean",
                "test_ccnlab_ratio_mean",
                "N",
                "feature_dim",
                "trainable_params",
            ],
        ),
        "",
        "## Per-Experiment Scores",
        "",
        _markdown_table(per_exp, list(per_exp.columns) if not per_exp.empty else []),
        "",
        "Interpretation note: CCNLab is a behavioral-fit benchmark, not an image-recognition benchmark. A useful connectome signal is the seeded model beating same-family random-sparse, degree-preserving, and weight-shuffled controls across seeds and across multiple conditioning phenomena.",
        "",
        "## Per-Seed Metrics",
        "",
        "```",
        metrics.to_string(index=False),
        "```",
        "",
    ]
    (output_dir / "ccnlab_associative_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run connectome-seeded and matched-control associative models on "
            "the CCNLab classical-conditioning benchmark."
        )
    )
    parser.add_argument(
        "--ccnlab-root",
        type=Path,
        default=None,
        help="Path to a clone of https://github.com/nikhilxb/ccnlab. If omitted, ccnlab must be importable.",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz"),
        help="Prepared connectome adjacency npz for seeded and control models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ccnlab_associative_benchmark"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Accepted for multi-GPU sweep compatibility. This runner uses CPU scipy/numpy.",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(DEFAULT_EXPERIMENTS),
        help="CCNLab experiment names/globs. Use '*' or 'all' for the full registry.",
    )
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=0,
        help="Use the leading N-neuron submatrix for smoke tests. 0 keeps the full matrix.",
    )
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument(
        "--feature-learner-dim",
        type=int,
        default=128,
        help=(
            "Graph feature count for connectome/random/degree-preserving/"
            "weight-shuffle RW, Kalman, and TD variants. Use 0 to reuse "
            "--feature-dim."
        ),
    )
    parser.add_argument(
        "--feature-time-basis-dim",
        type=int,
        default=0,
        help=(
            "Optional time basis appended before graph encoding for feature-learner "
            "variants. TD already adds complete-serial-compound time blocks, so 0 "
            "is the default."
        ),
    )
    parser.add_argument("--time-basis-dim", type=int, default=8)
    parser.add_argument("--encoder-steps", type=int, default=2)
    parser.add_argument("--recurrent-gain", type=float, default=0.7)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--hidden-scale", type=float, default=1.0)
    parser.add_argument("--raw-input-scale", type=float, default=1.0)
    parser.add_argument("--state-clip", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--alpha-bias", type=float, default=0.0)
    parser.add_argument("--trace-decay", type=float, default=0.90)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--response-clip", type=float, default=5.0)
    parser.add_argument("--allow-negative-response", action="store_true")
    parser.add_argument("--rw-alpha", type=float, default=0.3)
    parser.add_argument("--td-alpha", type=float, default=0.3)
    parser.add_argument("--td-gamma", type=float, default=0.98)
    parser.add_argument("--kalman-tau2", type=float, default=0.01)
    parser.add_argument("--kalman-sigma-r2", type=float, default=1.0)
    parser.add_argument("--kalman-sigma-w2", type=float, default=1.0)
    parser.add_argument("--init-seed", type=int, default=7000)
    parser.add_argument("--registry-seed", type=int, default=0)
    parser.add_argument("--log-every-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.subjects < 1:
        parser.error("--subjects must be positive")
    if args.max_neurons < 0:
        parser.error("--max-neurons must be nonnegative")
    if args.feature_dim < 1:
        parser.error("--feature-dim must be positive")
    if args.feature_learner_dim < 0:
        parser.error("--feature-learner-dim must be nonnegative")
    if args.feature_time_basis_dim < 0:
        parser.error("--feature-time-basis-dim must be nonnegative")
    if args.time_basis_dim < 0:
        parser.error("--time-basis-dim must be nonnegative")
    if args.encoder_steps < 0:
        parser.error("--encoder-steps must be nonnegative")
    if not (0.0 <= args.trace_decay < 1.0):
        parser.error("--trace-decay must be in [0, 1)")
    if args.response_clip < 0:
        parser.error("--response-clip must be nonnegative")
    if args.log_every_seconds < 0:
        parser.error("--log-every-seconds must be nonnegative")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_ccnlab_root(args.ccnlab_root)
    classical, evaluation, RescorlaWagner, KalmanFilter, TemporalDifference = (
        import_ccnlab_modules()
    )
    needs_matrix = any(
        model in CONNECTOME_MODEL_ALIASES or model in GRAPH_FEATURE_MODEL_SPECS
        for model in args.models
    )
    base_matrix = load_base_matrix(args.matrix, args.max_neurons) if needs_matrix else None
    print(
        "run-start "
        f"task=ccnlab_classical_conditioning output_dir={args.output_dir} "
        f"ccnlab_root={args.ccnlab_root or 'importable'} device={args.device} "
        f"models={','.join(args.models)} seeds={','.join(str(seed) for seed in args.seeds)}",
        flush=True,
    )
    if base_matrix is not None:
        print(
            "matrix-ready "
            f"N={base_matrix.shape[0]} edges={base_matrix.nnz} feature_dim={args.feature_dim} "
            f"feature_learner_dim={args.feature_learner_dim or args.feature_dim} "
            f"encoder_steps={args.encoder_steps}",
            flush=True,
        )

    metrics_rows: list[dict[str, float | int | str]] = []
    experiment_rows: list[dict[str, float | int | str]] = []
    for model_name in args.models:
        for seed in args.seeds:
            metrics, scores = run_model_seed(
                model_name,
                seed,
                args,
                base_matrix,
                classical,
                evaluation,
                RescorlaWagner,
                KalmanFilter,
                TemporalDifference,
            )
            metrics_rows.append(metrics)
            experiment_rows.extend(scores)
    write_outputs(args.output_dir, metrics_rows, experiment_rows, args)
    print(
        f"complete metrics={args.output_dir / 'metrics_by_seed.csv'} "
        f"report={args.output_dir / 'ccnlab_associative_report.md'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
