from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


HEMIBRAIN_DATASET = "hemibrain:v1.2.1"
NEUPRINT_SERVER = "neuprint.janelia.org"
CX_ROI_LABELS = ("EB", "PB", "FB", "NO")
HEMIBRAIN_MB_ROI_LABELS = ("MB(R)", "MB(L)")
MB_ROI_LABELS = (
    "MB_CA_L",
    "MB_CA_R",
    "MB_ML_L",
    "MB_ML_R",
    "MB_PED_L",
    "MB_PED_R",
    "MB_VL_L",
    "MB_VL_R",
)
CONNECTOME_HEMIBRAIN_CX = "hemibrain_cx"
CONNECTOME_HEMIBRAIN_MUSHROOM_BODY = "hemibrain_mushroom_body"
CONNECTOME_FLYWIRE_WHOLE = "flywire_whole"
CONNECTOME_FLYWIRE_MUSHROOM_BODY = "flywire_mushroom_body"
CONNECTOME_CHOICES = (
    CONNECTOME_HEMIBRAIN_CX,
    CONNECTOME_HEMIBRAIN_MUSHROOM_BODY,
    CONNECTOME_FLYWIRE_WHOLE,
    CONNECTOME_FLYWIRE_MUSHROOM_BODY,
)
DEFAULT_FLYWIRE_RELEASE = "783"
DEFAULT_WHOLE_BRAIN_POOL_FRACTION = 0.05
RHO_TARGET = 0.95
SIGN_COVERAGE_THRESHOLD = 0.95
DATA_SEED = 12345
TASK_CACHE_VERSION = 2
DT = 1.0
INPUT_DIM = 2
OUTPUT_DIM = 4
TASK_CARTESIAN = "cartesian"
TASK_CX_POLAR_BUMP = "cx_polar_bump"
TASK_CX_LANDMARK_BUMP = "cx_landmark_bump"
TASK_CHOICES = (TASK_CARTESIAN, TASK_CX_POLAR_BUMP, TASK_CX_LANDMARK_BUMP)
DEFAULT_HEADING_BINS = 32
DEFAULT_HOME_DISTANCE_SCALE = 25.0
DEFAULT_BUMP_KAPPA = 8.0
CX_LANDMARK_INPUT_DIM = 6
DEFAULT_LANDMARK_VISIBLE_PROB = 0.15
DEFAULT_LANDMARK_NOISE_STD = 0.05
DEFAULT_PASSIVE_DISPLACEMENT_PROB = 0.08
DEFAULT_PASSIVE_DISPLACEMENT_SCALE = 0.75
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
STRUCTURE_COMPARISON_MODELS = (
    "cx_bpu",
    "random",
    "degree_shuffle",
    "weight_shuffle",
    "no_recurrence",
)
WHOLE_BRAIN_COMPARISON_MODELS = (
    "connectome_bpu",
    "random",
    "weight_shuffle",
)
ALL_MODEL_NAMES = DEFAULT_BPU_MODELS + ("connectome_bpu", "gru")
RECURRENT_RUNTIME_CHOICES = ("auto", "dense", "sparse")
RECURRENT_TRAIN_CHOICES = ("frozen", "observed", "dense")


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
    cache_version: int = TASK_CACHE_VERSION
    kind: str = TASK_CARTESIAN
    heading_bins: int = DEFAULT_HEADING_BINS
    home_distance_scale: float = DEFAULT_HOME_DISTANCE_SCALE
    bump_kappa: float = DEFAULT_BUMP_KAPPA
    landmark_visible_prob: float = DEFAULT_LANDMARK_VISIBLE_PROB
    landmark_noise_std: float = DEFAULT_LANDMARK_NOISE_STD
    passive_displacement_prob: float = DEFAULT_PASSIVE_DISPLACEMENT_PROB
    passive_displacement_scale: float = DEFAULT_PASSIVE_DISPLACEMENT_SCALE


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
    log_every_seconds: float = 60.0
    recurrent_runtime: str = "auto"
    train_recurrent: str = "frozen"


@dataclass(frozen=True)
class CliConfig:
    mode: str
    device: str
    output_dir: Path
    cache_dir: Path
    signed_policy: str
    connectome: str
    flywire_release: str
    flywire_download_dir: Path | None
    whole_brain_pool_fraction: float
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
        "--connectome",
        choices=CONNECTOME_CHOICES,
        default=CONNECTOME_HEMIBRAIN_CX,
        help=(
            "Connectome substrate. 'hemibrain_cx' uses the original neuPrint CX "
            "query; 'hemibrain_mushroom_body' uses neuPrint MB(R)/MB(L); "
            "'flywire_whole' uses the FlyWire whole-brain release dump; "
            "'flywire_mushroom_body' filters the FlyWire release to MB neuropils."
        ),
    )
    parser.add_argument(
        "--flywire-release",
        default=DEFAULT_FLYWIRE_RELEASE,
        help="FlyWire release label for file names; currently tested with 783.",
    )
    parser.add_argument(
        "--flywire-download-dir",
        type=Path,
        default=None,
        help="Directory for raw FlyWire release files. Defaults to --cache-dir/flywire_release_<release>.",
    )
    parser.add_argument(
        "--whole-brain-pool-fraction",
        type=float,
        default=DEFAULT_WHOLE_BRAIN_POOL_FRACTION,
        help="Fraction of whole-brain neurons assigned to sensory and output pools each.",
    )
    parser.add_argument(
        "--signed-policy",
        choices=("auto", "force_unsigned", "force_signed"),
        default="auto",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--task",
        choices=TASK_CHOICES,
        default=TASK_CARTESIAN,
        help=(
            "Training target. 'cartesian' predicts [cos(theta), sin(theta), x, y]. "
            "'cx_polar_bump' predicts a heading bump plus home-vector polar readout. "
            "'cx_landmark_bump' adds intermittent home-landmark cues and passive "
            "displacements to stress cue correction."
        ),
    )
    parser.add_argument(
        "--heading-bins",
        type=int,
        default=DEFAULT_HEADING_BINS,
        help="Heading bump bins for --task cx_polar_bump.",
    )
    parser.add_argument(
        "--home-distance-scale",
        type=float,
        default=DEFAULT_HOME_DISTANCE_SCALE,
        help="Distance divisor for the cx_polar_bump home-distance target.",
    )
    parser.add_argument(
        "--bump-kappa",
        type=float,
        default=DEFAULT_BUMP_KAPPA,
        help="Concentration of the circular heading bump for --task cx_polar_bump.",
    )
    parser.add_argument(
        "--landmark-visible-prob",
        type=float,
        default=DEFAULT_LANDMARK_VISIBLE_PROB,
        help="Per-timestep probability of a home-vector landmark cue for --task cx_landmark_bump.",
    )
    parser.add_argument(
        "--landmark-noise-std",
        type=float,
        default=DEFAULT_LANDMARK_NOISE_STD,
        help="Noise added to visible landmark bearing/distance cue channels.",
    )
    parser.add_argument(
        "--passive-displacement-prob",
        type=float,
        default=DEFAULT_PASSIVE_DISPLACEMENT_PROB,
        help="Per-timestep probability of an unobserved world-frame displacement for --task cx_landmark_bump.",
    )
    parser.add_argument(
        "--passive-displacement-scale",
        type=float,
        default=DEFAULT_PASSIVE_DISPLACEMENT_SCALE,
        help="Scale of passive displacement jumps for --task cx_landmark_bump.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--log-every-seconds",
        type=float,
        default=60.0,
        help="Emit batch-level training/evaluation progress at this time interval. Use 0 to disable.",
    )
    parser.add_argument(
        "--comparison",
        choices=("default", "structure", "whole_brain"),
        default="default",
        help=(
            "Named model preset. 'structure' tests CX-BPU against same-size "
            "random, degree-preserving, weight-shuffled, and no-recurrence controls. "
            "'whole_brain' uses the scalable whole-brain preset cx_bpu/random/weight_shuffle."
        ),
    )
    parser.add_argument(
        "--recurrent-runtime",
        choices=RECURRENT_RUNTIME_CHOICES,
        default="auto",
        help="Use dense or sparse recurrent multiplication; auto selects sparse for large graphs.",
    )
    parser.add_argument(
        "--train-recurrent",
        choices=RECURRENT_TRAIN_CHOICES,
        default="frozen",
        help=(
            "Recurrent training mode. 'frozen' keeps the BPU connectome fixed; "
            "'observed' trains one recurrent parameter per observed edge while "
            "preserving support; 'dense' trains a full N x N recurrent matrix."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODEL_NAMES,
        default=None,
        help="Optional subset of models to train for quick sanity checks.",
    )
    parser.add_argument("--include-gru", action="store_true")
    args = parser.parse_args(argv)
    if args.heading_bins < 4:
        parser.error("--heading-bins must be at least 4.")
    if args.home_distance_scale <= 0:
        parser.error("--home-distance-scale must be positive.")
    if args.bump_kappa <= 0:
        parser.error("--bump-kappa must be positive.")
    if not (0.0 <= args.landmark_visible_prob <= 1.0):
        parser.error("--landmark-visible-prob must be in [0, 1].")
    if args.landmark_noise_std < 0:
        parser.error("--landmark-noise-std must be nonnegative.")
    if not (0.0 <= args.passive_displacement_prob <= 1.0):
        parser.error("--passive-displacement-prob must be in [0, 1].")
    if args.passive_displacement_scale < 0:
        parser.error("--passive-displacement-scale must be nonnegative.")
    if not (0.0 < args.whole_brain_pool_fraction < 0.5):
        parser.error("--whole-brain-pool-fraction must be in (0, 0.5).")

    output_dir = args.output_dir.resolve()
    cache_dir = (args.cache_dir.resolve() if args.cache_dir else output_dir)
    models = tuple(args.models) if args.models is not None else None
    if models is None and args.comparison == "structure":
        models = STRUCTURE_COMPARISON_MODELS
    if models is None and args.comparison == "whole_brain":
        models = WHOLE_BRAIN_COMPARISON_MODELS
    if models is None and args.connectome == CONNECTOME_FLYWIRE_WHOLE:
        models = WHOLE_BRAIN_COMPARISON_MODELS
    train = TrainConfig(
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_gru=args.include_gru,
        device=args.device,
        models=models,
        log_every_seconds=args.log_every_seconds,
        recurrent_runtime=args.recurrent_runtime,
        train_recurrent=args.train_recurrent,
    )
    return CliConfig(
        mode=args.mode,
        device=args.device,
        output_dir=output_dir,
        cache_dir=cache_dir,
        signed_policy=args.signed_policy,
        connectome=args.connectome,
        flywire_release=str(args.flywire_release),
        flywire_download_dir=(
            args.flywire_download_dir.resolve()
            if args.flywire_download_dir is not None
            else None
        ),
        whole_brain_pool_fraction=float(args.whole_brain_pool_fraction),
        train=train,
        task=TaskSpec(
            kind=args.task,
            heading_bins=args.heading_bins,
            home_distance_scale=args.home_distance_scale,
            bump_kappa=args.bump_kappa,
            landmark_visible_prob=args.landmark_visible_prob,
            landmark_noise_std=args.landmark_noise_std,
            passive_displacement_prob=args.passive_displacement_prob,
            passive_displacement_scale=args.passive_displacement_scale,
        ),
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


def output_dim_for_task(task: TaskSpec) -> int:
    if task.kind == TASK_CARTESIAN:
        return OUTPUT_DIM
    if task.kind in {TASK_CX_POLAR_BUMP, TASK_CX_LANDMARK_BUMP}:
        return int(task.heading_bins) + 3
    raise ValueError(f"Unknown task kind: {task.kind}")


def input_dim_for_task(task: TaskSpec) -> int:
    if task.kind == TASK_CX_LANDMARK_BUMP:
        return CX_LANDMARK_INPUT_DIM
    if task.kind in {TASK_CARTESIAN, TASK_CX_POLAR_BUMP}:
        return INPUT_DIM
    raise ValueError(f"Unknown task kind: {task.kind}")
