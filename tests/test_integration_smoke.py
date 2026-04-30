from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from src.config import build_paths
from src.connectome import prepare_connectome
from src.task import ensure_splits, load_split, validate_split_ids
from src.train import run_training, smoke_task_spec, smoke_train_config
from src.validate import run_validation


def write_mock_exports(root: Path) -> None:
    neurons = pd.DataFrame(
        {
            "bodyId": [101, 102, 103, 104, 105, 106],
            "type": ["ring_A", "EPG", "PEG", "CPU", "CPU", "PFL3"],
            "instance": ["ring_A", "EPG", "PEG", "CPU_a", "CPU_b", "PFL3_R"],
            "pre": [120, 110, 115, 105, 100, 130],
            "post": [130, 105, 110, 100, 100, 115],
            "predictedNt": ["ACh", "GABA", "Glu", "ACh", "GABA", "ACh"],
        }
    )
    roi_counts = pd.DataFrame(
        {
            "bodyId": [101, 102, 103, 104, 105, 106],
            "roi": ["EB", "EB", "PB", "FB", "NO", "FB"],
            "pre": [110, 100, 100, 95, 90, 65],
            "post": [70, 95, 100, 90, 90, 105],
        }
    )
    connections = pd.DataFrame(
        {
            "bodyId_pre": [101, 102, 103, 104, 105, 101, 102, 103, 104, 105, 106],
            "bodyId_post": [102, 103, 106, 105, 106, 101, 104, 105, 102, 103, 101],
            "weight": [8, 9, 11, 4, 10, 3, 5, 7, 6, 6, 2],
        }
    )
    neurons.to_csv(root / "neurons.csv", index=False)
    roi_counts.to_csv(root / "roi_counts.csv", index=False)
    connections.to_csv(root / "connections.csv", index=False)


def test_mocked_end_to_end_cpu(tmp_path: Path) -> None:
    paths = build_paths(tmp_path)
    write_mock_exports(tmp_path)
    graph = prepare_connectome(paths)
    assert graph.metadata["N"] == 6
    metrics = run_training(paths, smoke_train_config(), smoke_task_spec())
    assert not metrics.empty
    run_validation(paths)
    for name in [
        "neurons.csv",
        "roi_counts.csv",
        "connections.csv",
        "pool_assignments.csv",
        "graph_metadata.json",
        "adjacency_unsigned.npz",
        "data_validation.md",
        "bpu_validation.md",
        "control_validation.md",
        "summary.md",
        "metrics_by_seed.csv",
        "metrics_summary.csv",
        "loss_history.csv",
        "error_vs_sequence_length.png",
        "loss_curve.png",
    ]:
        assert (tmp_path / name).exists(), name
    with (tmp_path / "graph_metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["orientation"] == "W_rec[post_index, pre_index]"
    splits = ensure_splits(paths.sequence_dir, smoke_task_spec())
    validate_split_ids([split.path for split in splits])
    loaded = load_split(splits[0].path)
    loaded_again = load_split(splits[0].path)
    assert (loaded["inputs"] == loaded_again["inputs"]).all()
    noise_splits = [split for split in splits if split.name == "test_noise"]
    noise_clean = [load_split(split.path)["clean_inputs"] for split in noise_splits]
    assert all((noise_clean[0] == clean).all() for clean in noise_clean[1:])


def test_cuda_path_smoke_conditionally(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        return
    paths = build_paths(tmp_path)
    write_mock_exports(tmp_path)
    prepare_connectome(paths)
    cfg = smoke_train_config()
    cfg = type(cfg)(
        seeds=cfg.seeds,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        lr=cfg.lr,
        patience=cfg.patience,
        grad_clip=cfg.grad_clip,
        include_gru=cfg.include_gru,
        device="cuda",
    )
    metrics = run_training(paths, cfg, smoke_task_spec())
    assert not metrics.empty
