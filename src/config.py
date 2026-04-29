from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


HEMIBRAIN_DATASET = "hemibrain:v1.2.1"
NEUPRINT_SERVER = "neuprint.janelia.org"
CX_ROI_LABELS = ("EB", "PB", "FB", "NO")
RHO_TARGET = 0.95
SIGN_COVERAGE_THRESHOLD = 0.95
DATA_SEED = 12345
DT = 1.0
INPUT_DIM = 2
OUTPUT_DIM = 4
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_TRAIN_T = 50
DEFAULT_TEST_T = (50, 100, 200)
DEFAULT_NOISE_STDS = (0.0, 0.05, 0.10, 0.20)
DEFAULT_BPU_MODELS = (
    "cx_bpu",
    "no_recurrence",
    "random",
    "degree_shuffle",
    "weight_shuffle",
)
ALL_MODEL_NAMES = DEFAULT_BPU_MODELS + ("gru",)


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    cache_dir: Path

    @property
    def neurons_csv(self) -> Path:
        return self.output_dir / "neurons.csv"

    @property
    def roi_counts_csv(self) -> Path:
        return self.output_dir / "roi_counts.csv"

    @property
    def connections_csv(self) -> Path:
        return self.output_dir / "connections.csv"

    @property
    def pool_assignments_csv(self) -> Path:
        return self.output_dir / "pool_assignments.csv"

    @property
    def graph_metadata_json(self) -> Path:
        return self.output_dir / "graph_metadata.json"

    @property
    def adjacency_unsigned_npz(self) -> Path:
        return self.output_dir / "adjacency_unsigned.npz"

    @property
    def adjacency_signed_npz(self) -> Path:
        return self.output_dir / "adjacency_signed.npz"

    @property
    def data_validation_md(self) -> Path:
        return self.output_dir / "data_validation.md"

    @property
    def bpu_validation_md(self) -> Path:
        return self.output_dir / "bpu_validation.md"

    @property
    def control_validation_md(self) -> Path:
        return self.output_dir / "control_validation.md"

    @property
    def summary_md(self) -> Path:
        return self.output_dir / "summary.md"

    @property
    def metrics_by_seed_csv(self) -> Path:
        return self.output_dir / "metrics_by_seed.csv"

    @property
    def metrics_summary_csv(self) -> Path:
        return self.output_dir / "metrics_summary.csv"

    @property
    def loss_history_csv(self) -> Path:
        return self.output_dir / "loss_history.csv"

    @property
    def error_vs_sequence_length_png(self) -> Path:
        return self.output_dir / "error_vs_sequence_length.png"

    @property
    def error_vs_noise_png(self) -> Path:
        return self.output_dir / "error_vs_noise.png"

    @property
    def loss_curve_png(self) -> Path:
        return self.output_dir / "loss_curve.png"

    @property
    def sample_efficiency_png(self) -> Path:
        return self.output_dir / "sample_efficiency.png"

    @property
    def sequence_dir(self) -> Path:
        return self.cache_dir / "sequences"


@dataclass(frozen=True)
class TaskSpec:
    train_count: int = 10_000
    val_count: int = 2_000
    test_count: int = 2_000
    train_T: int = DEFAULT_TRAIN_T
    test_T: tuple[int, ...] = DEFAULT_TEST_T
    noise_stds: tuple[float, ...] = DEFAULT_NOISE_STDS
    data_seed: int = DATA_SEED


@dataclass(frozen=True)
class TrainConfig:
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    epochs: int = 20
    batch_size: int = 128
    num_workers: int = 2
    lr: float = 1e-3
    patience: int = 4
    grad_clip: float = 1.0
    include_gru: bool = False
    device: str = "auto"
    models: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CliConfig:
    mode: str
    device: str
    output_dir: Path
    cache_dir: Path
    signed_policy: str
    train: TrainConfig
    task: TaskSpec


def default_output_dir() -> Path:
    return Path("experiments/hemibrain_cx_bpu/outputs")


def parse_args(argv: Sequence[str] | None = None) -> CliConfig:
    parser = argparse.ArgumentParser(
        description="Hemibrain central-complex BPU benchmark."
    )
    parser.add_argument(
        "--mode",
        choices=("download", "prepare", "train", "validate", "all"),
        default="all",
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--signed-policy",
        choices=("auto", "force_unsigned", "force_signed"),
        default="auto",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODEL_NAMES,
        default=None,
        help="Optional subset of models to train for quick sanity checks.",
    )
    parser.add_argument("--include-gru", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    cache_dir = (args.cache_dir.resolve() if args.cache_dir else output_dir)
    train = TrainConfig(
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_gru=args.include_gru,
        device=args.device,
        models=tuple(args.models) if args.models is not None else None,
    )
    return CliConfig(
        mode=args.mode,
        device=args.device,
        output_dir=output_dir,
        cache_dir=cache_dir,
        signed_policy=args.signed_policy,
        train=train,
        task=TaskSpec(),
    )


def build_paths(output_dir: Path, cache_dir: Path | None = None) -> OutputPaths:
    output_dir = Path(output_dir).resolve()
    cache_dir = Path(cache_dir).resolve() if cache_dir else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return OutputPaths(output_dir=output_dir, cache_dir=cache_dir)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
