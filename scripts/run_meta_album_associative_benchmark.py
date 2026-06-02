#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_mb_associative_learning as mb  # noqa: E402
import run_omniglot_associative_benchmark as episodic  # noqa: E402
from src.run_manifest import write_artifact_manifest  # noqa: E402


DATASET_CHOICES = ("meta_album", "synthetic")
SPLIT_MODE_CHOICES = ("dataset", "class")
DEFAULT_MODELS = episodic.DEFAULT_MODELS
MODEL_CHOICES = episodic.MODEL_CHOICES


@dataclass(frozen=True)
class DatasetClasses:
    name: str
    root: Path
    classes: tuple[np.ndarray, ...]

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def _normalize_col(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _column_for(columns: Sequence[str], candidates: set[str]) -> str | None:
    by_normalized = {_normalize_col(col): col for col in columns}
    for candidate in candidates:
        if candidate in by_normalized:
            return by_normalized[candidate]
    return None


def _image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        for path in root.rglob(suffix):
            index.setdefault(path.name, path)
            index.setdefault(str(path.relative_to(root)), path)
    return index


def _resolve_image_path(
    dataset_root: Path,
    labels_dir: Path,
    filename: str,
    index: dict[str, Path],
) -> Path | None:
    raw = str(filename).strip()
    if not raw:
        return None
    path = Path(raw)
    candidates = [
        path if path.is_absolute() else labels_dir / path,
        dataset_root / path,
        dataset_root / "images" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return index.get(raw) or index.get(path.name)


def _load_image_vector(path: Path, image_size: int) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on optional runtime package
        raise RuntimeError(
            "Meta-Album image loading requires Pillow. Install it with `python -m pip install pillow`."
        ) from exc
    with Image.open(path) as image:
        return episodic._pil_to_flat_array(image, image_size)


def _labels_csv_paths(data_root: Path) -> list[Path]:
    return sorted(data_root.rglob("labels.csv"))


def discover_meta_album_roots(data_root: Path) -> list[Path]:
    roots = []
    for labels_csv in _labels_csv_paths(data_root):
        roots.append(labels_csv.parent)
    deduped: dict[Path, Path] = {}
    for root in roots:
        deduped[root.resolve()] = root
    return sorted(deduped.values())


def load_meta_album_dataset_dir(
    dataset_root: Path,
    image_size: int,
    min_samples_per_class: int,
    max_classes_per_dataset: int,
    seed: int,
) -> DatasetClasses:
    labels_candidates = sorted(dataset_root.rglob("labels.csv"))
    if not labels_candidates:
        raise FileNotFoundError(f"No labels.csv found under {dataset_root}")
    labels_csv = labels_candidates[0]
    labels = pd.read_csv(labels_csv)
    file_col = _column_for(
        labels.columns,
        {"filename", "file", "filepath", "image", "imagepath", "filenamepath", "name", "samplepath"},
    )
    class_col = _column_for(labels.columns, {"category", "label", "class", "classname", "target"})
    if file_col is None or class_col is None:
        raise ValueError(
            f"{labels_csv} must contain image filename and class/category columns; "
            f"columns={list(labels.columns)}"
        )
    index = _image_index(dataset_root)
    grouped: dict[str, list[np.ndarray]] = {}
    missing = 0
    for _, row in labels.iterrows():
        path = _resolve_image_path(dataset_root, labels_csv.parent, str(row[file_col]), index)
        if path is None:
            missing += 1
            continue
        label = str(row[class_col])
        grouped.setdefault(label, []).append(_load_image_vector(path, image_size))
    if missing:
        print(
            f"warning missing_images dataset={dataset_root.name} count={missing}",
            flush=True,
        )
    classes = [
        np.stack(samples, axis=0).astype(np.float32)
        for _, samples in sorted(grouped.items())
        if len(samples) >= min_samples_per_class
    ]
    if max_classes_per_dataset > 0 and len(classes) > max_classes_per_dataset:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(classes), size=max_classes_per_dataset, replace=False))
        classes = [classes[int(idx)] for idx in indices]
    if not classes:
        raise ValueError(
            f"{dataset_root} has no classes with at least {min_samples_per_class} usable samples."
        )
    return DatasetClasses(name=dataset_root.name, root=dataset_root, classes=tuple(classes))


def download_openml_ids(openml_ids: Sequence[int], data_root: Path) -> None:
    if not openml_ids:
        return
    try:
        import openml
    except Exception as exc:  # pragma: no cover - optional package
        raise RuntimeError(
            "OpenML download requires openml. Install it with `python -m pip install openml`."
        ) from exc
    openml.config.cache_directory = str(data_root / "openml_cache")
    for dataset_id in openml_ids:
        print(f"openml-download-start dataset_id={dataset_id}", flush=True)
        try:
            openml.datasets.get_dataset(
                int(dataset_id),
                download_data=True,
                download_qualities=False,
                download_features_meta_data=False,
                download_all_files=True,
            )
        except TypeError:
            openml.datasets.get_dataset(
                int(dataset_id),
                download_data=True,
                download_qualities=False,
                download_features_meta_data=False,
            )
        print(f"openml-download-done dataset_id={dataset_id}", flush=True)


def _required_samples_per_class(args: argparse.Namespace) -> int:
    needed = args.shot + args.queries_per_class
    if args.reversal_count > 0:
        needed += args.queries_per_class
        needed += args.shot
    return needed


def _embed_classes(
    class_groups: Sequence[np.ndarray],
    args: argparse.Namespace,
    projection: np.ndarray | None,
) -> tuple[np.ndarray, ...]:
    return episodic._embed_class_arrays(
        tuple(class_groups),
        args.embedding,
        args.embedding_dim,
        args.embedding_sparsity,
        projection,
    )


def _combined_bank(
    name: str,
    datasets: Sequence[DatasetClasses],
    args: argparse.Namespace,
    projection: np.ndarray | None,
) -> episodic.FeatureBank:
    class_groups = [cls for dataset in datasets for cls in dataset.classes]
    return episodic.FeatureBank(name=name, classes=_embed_classes(class_groups, args, projection))


def _select_by_name(
    datasets: Sequence[DatasetClasses],
    requested: Sequence[str],
    field_name: str,
) -> list[DatasetClasses]:
    if not requested:
        return []
    by_name = {dataset.name: dataset for dataset in datasets}
    selected = []
    missing = []
    for name in requested:
        if name in by_name:
            selected.append(by_name[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(
            f"Unknown {field_name} dataset(s): {missing}. Available: {sorted(by_name)}"
        )
    return selected


def _dataset_level_split(
    datasets: Sequence[DatasetClasses],
    args: argparse.Namespace,
) -> tuple[list[DatasetClasses], list[DatasetClasses], list[DatasetClasses]]:
    train = _select_by_name(datasets, args.train_datasets, "train")
    val = _select_by_name(datasets, args.val_datasets, "val")
    test = _select_by_name(datasets, args.test_datasets, "test")
    if train or val or test:
        used = {dataset.name for dataset in train + val + test}
        if len(used) != len(train) + len(val) + len(test):
            raise ValueError("Explicit train/val/test dataset lists must be disjoint.")
        remaining = [dataset for dataset in datasets if dataset.name not in used]
        if not test and remaining:
            test = [remaining.pop()]
        if not val and remaining:
            val = [remaining.pop()]
        if not train:
            train = remaining
        if not train or not val or not test:
            raise ValueError(
                "Explicit dataset split did not leave nonempty train, val, and test pools."
            )
        return train, val, test

    if len(datasets) < 3:
        raise ValueError(
            "--split-mode dataset requires at least 3 Meta-Album dataset directories "
            "unless explicit train/val/test lists are provided."
        )
    rng = np.random.default_rng(args.data_seed)
    order = list(rng.permutation(len(datasets)))
    shuffled = [datasets[int(idx)] for idx in order]
    test_count = max(1, int(round(len(shuffled) * args.test_dataset_fraction)))
    val_count = max(1, int(round(len(shuffled) * args.val_dataset_fraction)))
    if test_count + val_count >= len(shuffled):
        test_count = 1
        val_count = 1
    test = shuffled[:test_count]
    val = shuffled[test_count : test_count + val_count]
    train = shuffled[test_count + val_count :]
    return train, val, test


def _class_level_split(
    datasets: Sequence[DatasetClasses],
    args: argparse.Namespace,
    projection: np.ndarray | None,
) -> tuple[episodic.FeatureBank, episodic.FeatureBank, episodic.FeatureBank]:
    class_groups = [cls for dataset in datasets for cls in dataset.classes]
    if len(class_groups) < args.way * 3:
        raise ValueError("--split-mode class needs at least 3 * --way usable classes.")
    rng = np.random.default_rng(args.data_seed)
    order = list(rng.permutation(len(class_groups)))
    test_count = max(args.way, int(round(len(class_groups) * args.test_class_fraction)))
    val_count = max(args.way, int(round(len(class_groups) * args.val_class_fraction)))
    if test_count + val_count >= len(class_groups):
        test_count = args.way
        val_count = args.way
    test = [class_groups[int(idx)] for idx in order[:test_count]]
    val = [class_groups[int(idx)] for idx in order[test_count : test_count + val_count]]
    train = [class_groups[int(idx)] for idx in order[test_count + val_count :]]
    if len(train) < args.way:
        raise ValueError("Class-level split left too few train classes.")
    return (
        episodic.FeatureBank("meta_album_class_train", _embed_classes(train, args, projection)),
        episodic.FeatureBank("meta_album_class_val", _embed_classes(val, args, projection)),
        episodic.FeatureBank("meta_album_class_test", _embed_classes(test, args, projection)),
    )


def _projection_for(args: argparse.Namespace, pixel_dim: int) -> np.ndarray | None:
    if args.embedding == "raw":
        return None
    rng = np.random.default_rng(args.data_seed)
    return rng.normal(
        0.0,
        1.0 / math.sqrt(max(args.embedding_dim, 1)),
        size=(pixel_dim, args.embedding_dim),
    ).astype(np.float32)


def load_meta_album_feature_banks(
    args: argparse.Namespace,
) -> tuple[episodic.FeatureBank, episodic.FeatureBank, episodic.FeatureBank, dict[str, object]]:
    download_openml_ids(args.openml_ids, args.data_root)
    dataset_dirs = [Path(path) for path in args.dataset_dirs]
    if not dataset_dirs:
        dataset_dirs = discover_meta_album_roots(args.data_root)
    if not dataset_dirs:
        raise FileNotFoundError(
            f"No Meta-Album labels.csv files found under {args.data_root}. "
            "Pass --dataset-dirs or --openml-ids first."
        )
    min_samples = _required_samples_per_class(args)
    datasets = [
        load_meta_album_dataset_dir(
            path,
            image_size=args.image_size,
            min_samples_per_class=min_samples,
            max_classes_per_dataset=args.max_classes_per_dataset,
            seed=args.data_seed + idx,
        )
        for idx, path in enumerate(dataset_dirs)
    ]
    datasets = [dataset for dataset in datasets if dataset.num_classes >= args.way]
    if not datasets:
        raise ValueError(f"No dataset has at least --way={args.way} usable classes.")
    pixel_dim = datasets[0].classes[0].shape[1]
    projection = _projection_for(args, pixel_dim)
    if args.split_mode == "class":
        train_bank, val_bank, test_bank = _class_level_split(datasets, args, projection)
        split_info = {
            "split_mode": "class",
            "source_datasets": [dataset.name for dataset in datasets],
        }
    else:
        train_datasets, val_datasets, test_datasets = _dataset_level_split(datasets, args)
        train_bank = _combined_bank("meta_album_dataset_train", train_datasets, args, projection)
        val_bank = _combined_bank("meta_album_dataset_val", val_datasets, args, projection)
        test_bank = _combined_bank("meta_album_dataset_test", test_datasets, args, projection)
        split_info = {
            "split_mode": "dataset",
            "train_datasets": [dataset.name for dataset in train_datasets],
            "val_datasets": [dataset.name for dataset in val_datasets],
            "test_datasets": [dataset.name for dataset in test_datasets],
        }
    return train_bank, val_bank, test_bank, split_info


def load_feature_banks(
    args: argparse.Namespace,
) -> tuple[episodic.FeatureBank, episodic.FeatureBank, episodic.FeatureBank, dict[str, object]]:
    if args.dataset == "synthetic":
        train_bank, val_bank, test_bank = episodic.synthetic_feature_banks(
            feature_dim=args.synthetic_feature_dim,
            samples_per_class=args.synthetic_samples_per_class,
            train_classes=args.synthetic_train_classes,
            val_classes=args.synthetic_val_classes,
            test_classes=args.synthetic_test_classes,
            seed=args.data_seed,
            class_noise_std=args.synthetic_class_noise_std,
        )
        return train_bank, val_bank, test_bank, {"split_mode": "synthetic"}
    if args.dataset == "meta_album":
        return load_meta_album_feature_banks(args)
    raise ValueError(f"Unknown dataset: {args.dataset}")


def write_meta_album_outputs(
    output_dir: Path,
    metrics_rows: list[dict[str, float | int | str]],
    history_rows: list[dict[str, float | int | str]],
    args: argparse.Namespace,
    spec: episodic.EpisodeSpec,
    train_bank: episodic.FeatureBank,
    val_bank: episodic.FeatureBank,
    test_bank: episodic.FeatureBank,
    split_info: dict[str, object],
) -> None:
    episodic.write_outputs(
        output_dir,
        metrics_rows,
        history_rows,
        args,
        spec,
        train_bank,
        val_bank,
        test_bank,
    )
    report = output_dir / "omniglot_associative_report.md"
    if report.exists():
        report.rename(output_dir / "meta_album_associative_report.md")
    for old_name, new_name in (
        ("omniglot_associative_accuracy.png", "meta_album_associative_accuracy.png"),
        ("omniglot_associative_loss.png", "meta_album_associative_loss.png"),
    ):
        old_path = output_dir / old_name
        if old_path.exists():
            old_path.rename(output_dir / new_name)
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {}
    config["task"] = "meta_album_episodic_associative_classification"
    config["split_info"] = split_info
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Meta-Album Associative Benchmark",
        "",
        f"Benchmark: Meta-Album-style multi-domain few-shot classification, {spec.way}-way {spec.shot}-shot.",
        "",
        "Support examples provide a sensory embedding plus an episode-local label channel. Query examples omit the label. A nonzero reversal count adds within-episode relabeling as a continual-association variant.",
        "",
        f"Split info: `{json.dumps(split_info, sort_keys=True)}`",
        f"Train classes: `{train_bank.num_classes}`; validation classes: `{val_bank.num_classes}`; test classes: `{test_bank.num_classes}`.",
        f"Feature dimension: `{spec.feature_dim}`; timesteps: `{spec.timesteps}`.",
        f"Connectome expansion factor: `{getattr(args, 'expand_factor', 1.0)}`; target neurons: `{getattr(args, 'expand_target_neurons', 0)}`.",
        "",
        "## Summary",
        "",
        "```",
        pd.read_csv(output_dir / "metrics_summary.csv").to_string(index=False),
        "```",
        "",
        "Interpretation note: use dataset-level splits for the serious result. Class-level splits are useful for debugging but are less compelling because classes from the same source dataset may appear across train and test.",
        "",
    ]
    (output_dir / "meta_album_associative_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    write_artifact_manifest(
        output_dir,
        config={
            "args": episodic.serializable_args(args),
            "episode_spec": spec.__dict__,
            "split_info": split_info,
        },
        extra={"stage": "meta_album_associative_learning"},
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Meta-Album episodic associative benchmark for evaluating mushroom-body "
            "connectome priors on a serious multi-domain few-shot task."
        )
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="meta_album")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "outputs" / "hemibrain_mushroom_body_plume" / "adjacency_unsigned.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "meta_album_associative",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "meta_album")
    parser.add_argument("--dataset-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--openml-ids", nargs="*", type=int, default=[])
    parser.add_argument("--split-mode", choices=SPLIT_MODE_CHOICES, default="dataset")
    parser.add_argument("--train-datasets", nargs="*", default=[])
    parser.add_argument("--val-datasets", nargs="*", default=[])
    parser.add_argument("--test-datasets", nargs="*", default=[])
    parser.add_argument("--val-dataset-fraction", type=float, default=0.20)
    parser.add_argument("--test-dataset-fraction", type=float, default=0.20)
    parser.add_argument("--val-class-fraction", type=float, default=0.15)
    parser.add_argument("--test-class-fraction", type=float, default=0.15)
    parser.add_argument("--max-classes-per-dataset", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(DEFAULT_MODELS))
    parser.add_argument("--recurrent-runtime", choices=mb.RUNTIME_CHOICES, default="sparse")
    parser.add_argument("--max-neurons", type=int, default=0)
    episodic.add_connectome_expansion_args(parser)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-batches", type=int, default=240)
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--test-batches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--state-clip", type=float, default=5.0)
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
    parser.add_argument("--way", type=int, default=10)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--queries-per-class", type=int, default=1)
    parser.add_argument("--reversal-count", type=int, default=0)
    parser.add_argument("--feature-noise-std", type=float, default=0.0)
    parser.add_argument("--embedding", choices=episodic.EMBEDDING_CHOICES, default="random_projection")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--embedding-sparsity", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--gru-hidden", type=int, default=256)
    parser.add_argument("--nearest-temperature", type=float, default=0.1)
    parser.add_argument("--synthetic-feature-dim", type=int, default=64)
    parser.add_argument("--synthetic-samples-per-class", type=int, default=20)
    parser.add_argument("--synthetic-train-classes", type=int, default=120)
    parser.add_argument("--synthetic-val-classes", type=int, default=40)
    parser.add_argument("--synthetic-test-classes", type=int, default=40)
    parser.add_argument("--synthetic-class-noise-std", type=float, default=0.08)
    parser.add_argument("--data-seed", type=int, default=12345)
    parser.add_argument("--init-seed", type=int, default=7200)
    parser.add_argument("--val-seed", type=int, default=23000)
    parser.add_argument("--test-seed", type=int, default=34000)
    args = parser.parse_args(argv)
    episodic_args = argparse.Namespace(
        way=args.way,
        shot=args.shot,
        queries_per_class=args.queries_per_class,
        reversal_count=args.reversal_count,
        embedding_sparsity=args.embedding_sparsity,
        embedding=args.embedding,
        embedding_dim=args.embedding_dim,
        dataset="synthetic" if args.dataset == "synthetic" else "omniglot",
        synthetic_train_classes=args.synthetic_train_classes,
        synthetic_val_classes=args.synthetic_val_classes,
        synthetic_test_classes=args.synthetic_test_classes,
    )
    # Reuse the strict episodic argument checks without parsing again.
    if episodic_args.way < 2:
        parser.error("--way must be at least 2")
    if episodic_args.shot < 1:
        parser.error("--shot must be at least 1")
    if episodic_args.queries_per_class < 1:
        parser.error("--queries-per-class must be at least 1")
    if episodic_args.reversal_count < 0 or episodic_args.reversal_count > episodic_args.way:
        parser.error("--reversal-count must be between 0 and --way")
    if episodic_args.reversal_count == 1:
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
    if args.dataset == "synthetic":
        for name in ("synthetic_train_classes", "synthetic_val_classes", "synthetic_test_classes"):
            if getattr(args, name) < args.way:
                parser.error(f"--{name.replace('_', '-')} must be at least --way")
    episodic.validate_connectome_expansion_args(parser, args)
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = mb.select_device(args.device)
    print(
        "run-start "
        f"task=meta_album_episodic_associative_classification dataset={args.dataset} "
        f"output_dir={args.output_dir} matrix={args.matrix} device={device}",
        flush=True,
    )
    base_matrix = episodic.load_benchmark_matrix(args)
    train_bank, val_bank, test_bank, split_info = load_feature_banks(args)
    spec = episodic.EpisodeSpec(
        way=args.way,
        shot=args.shot,
        queries_per_class=args.queries_per_class,
        reversal_count=args.reversal_count,
        feature_dim=train_bank.feature_dim,
        feature_noise_std=args.feature_noise_std,
    )
    print(
        "data-ready "
        f"split={split_info} train_classes={train_bank.num_classes} "
        f"val_classes={val_bank.num_classes} test_classes={test_bank.num_classes} "
        f"feature_dim={spec.feature_dim} input_dim={spec.input_dim} "
        f"timesteps={spec.timesteps} N={base_matrix.shape[0]} edges={base_matrix.nnz} "
        f"models={','.join(args.models)}",
        flush=True,
    )
    metrics_rows: list[dict[str, float | int | str]] = []
    history_rows: list[dict[str, float | int | str]] = []
    for model_name in args.models:
        for seed in args.seeds:
            metrics, history = episodic.train_one_model(
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
    write_meta_album_outputs(
        args.output_dir,
        metrics_rows,
        history_rows,
        args,
        spec,
        train_bank,
        val_bank,
        test_bank,
        split_info,
    )
    print(
        f"complete metrics={args.output_dir / 'metrics_by_seed.csv'} "
        f"report={args.output_dir / 'meta_album_associative_report.md'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
