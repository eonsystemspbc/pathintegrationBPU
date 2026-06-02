from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_meta_album_associative_benchmark.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("meta_album_assoc", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_matrix(n: int = 10) -> sparse.coo_matrix:
    rows = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 7], dtype=np.int64)
    cols = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2], dtype=np.int64)
    data = np.linspace(0.05, 0.6, rows.size, dtype=np.float32)
    return sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)


def _write_fake_meta_album_dataset(root: Path, name: str, classes: int = 4, samples: int = 3) -> Path:
    image_mod = pytest.importorskip("PIL.Image")
    dataset_dir = root / name
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True)
    rows = []
    for class_idx in range(classes):
        for sample_idx in range(samples):
            arr = np.full((12, 12), class_idx * 30 + sample_idx, dtype=np.uint8)
            filename = f"{name}_{class_idx}_{sample_idx}.png"
            image_mod.fromarray(arr).save(images_dir / filename)
            rows.append({"FILE_NAME": f"images/{filename}", "CATEGORY": f"class_{class_idx}"})
    pd.DataFrame(rows).to_csv(dataset_dir / "labels.csv", index=False)
    return dataset_dir


def test_meta_album_labels_csv_loader_groups_classes(tmp_path: Path) -> None:
    meta = _load_module()
    dataset_dir = _write_fake_meta_album_dataset(tmp_path, "toy_meta", classes=5, samples=4)

    loaded = meta.load_meta_album_dataset_dir(
        dataset_dir,
        image_size=8,
        min_samples_per_class=3,
        max_classes_per_dataset=0,
        seed=1,
    )

    assert loaded.name == "toy_meta"
    assert loaded.num_classes == 5
    assert loaded.classes[0].shape == (4, 64)


def test_dataset_level_split_uses_disjoint_dataset_names(tmp_path: Path) -> None:
    meta = _load_module()
    datasets = [
        meta.DatasetClasses(
            name=f"dataset_{idx}",
            root=tmp_path / f"dataset_{idx}",
            classes=tuple(np.ones((3, 4), dtype=np.float32) for _ in range(4)),
        )
        for idx in range(5)
    ]
    args = meta.parse_args(
        [
            "--dataset",
            "synthetic",
            "--train-datasets",
            "dataset_0",
            "dataset_1",
            "dataset_2",
            "--val-datasets",
            "dataset_3",
            "--test-datasets",
            "dataset_4",
            "--way",
            "3",
        ]
    )

    train, val, test = meta._dataset_level_split(datasets, args)

    assert [dataset.name for dataset in train] == ["dataset_0", "dataset_1", "dataset_2"]
    assert [dataset.name for dataset in val] == ["dataset_3"]
    assert [dataset.name for dataset in test] == ["dataset_4"]


def test_meta_album_associative_synthetic_smoke_run(tmp_path: Path) -> None:
    meta = _load_module()
    matrix_path = tmp_path / "adjacency_unsigned.npz"
    out = tmp_path / "meta_album_assoc"
    sparse.save_npz(matrix_path, _toy_matrix().tocsr())

    code = meta.main(
        [
            "--dataset",
            "synthetic",
            "--matrix",
            str(matrix_path),
            "--output-dir",
            str(out),
            "--device",
            "cpu",
            "--models",
            "hemibrain_seeded",
            "gru",
            "nearest_support",
            "--seeds",
            "0",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--train-batches",
            "1",
            "--val-batches",
            "1",
            "--test-batches",
            "1",
            "--way",
            "3",
            "--shot",
            "1",
            "--queries-per-class",
            "1",
            "--synthetic-feature-dim",
            "6",
            "--synthetic-samples-per-class",
            "6",
            "--synthetic-train-classes",
            "8",
            "--synthetic-val-classes",
            "8",
            "--synthetic-test-classes",
            "8",
            "--gru-hidden",
            "8",
            "--log-every-seconds",
            "0",
        ]
    )

    assert code == 0
    metrics = pd.read_csv(out / "metrics_by_seed.csv")
    assert sorted(metrics["model"].tolist()) == ["gru", "hemibrain_seeded", "nearest_support"]
    assert (out / "metrics_summary.csv").exists()
    assert (out / "meta_album_associative_report.md").exists()
    assert (out / "meta_album_associative_accuracy.png").exists()
    assert (out / "run_manifest.json").exists()
