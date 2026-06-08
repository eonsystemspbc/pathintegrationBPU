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
    / "run_pruned_mb_associative_comparison.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("pruned_mb_assoc", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_bridge_matrix() -> sparse.coo_matrix:
    # Orientation is W_rec[post, pre], so each pair is pre -> post.
    rows = np.array([1, 4, 2, 4, 3, 5, 6], dtype=np.int64)
    cols = np.array([0, 1, 0, 2, 0, 3, 5], dtype=np.int64)
    data = np.array([0.2, 0.4, 0.3, 0.5, 0.1, 0.7, 0.8], dtype=np.float32)
    return sparse.coo_matrix((data, (rows, cols)), shape=(7, 7), dtype=np.float32)


def _toy_pools() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"index": 0, "pool": "sensory", "is_sensory": True, "is_internal": False, "is_output": False},
            {"index": 1, "pool": "internal", "is_sensory": False, "is_internal": True, "is_output": False},
            {"index": 2, "pool": "internal", "is_sensory": False, "is_internal": True, "is_output": False},
            {"index": 3, "pool": "internal", "is_sensory": False, "is_internal": True, "is_output": False},
            {"index": 4, "pool": "output", "is_sensory": False, "is_internal": False, "is_output": True},
            {"index": 5, "pool": "internal", "is_sensory": False, "is_internal": True, "is_output": False},
            {"index": 6, "pool": "output", "is_sensory": False, "is_internal": False, "is_output": True},
        ]
    )


def test_pruning_keeps_sensory_output_and_short_bridge_nodes() -> None:
    prune = _load_module()
    result = prune.prune_recurrent_matrix(
        _toy_bridge_matrix(),
        _toy_pools(),
        max_hops=1,
        max_internal_nodes=0,
    )

    assert result.metadata["original_N"] == 7
    assert result.metadata["pruned_N"] == 5
    assert set(result.keep_indices.tolist()) == {0, 1, 2, 4, 6}
    assert set(result.kept_internal_indices.tolist()) == {1, 2}
    assert result.matrix.shape == (5, 5)
    assert result.matrix.nnz == 4


def test_pruning_can_cap_internal_bridge_nodes() -> None:
    prune = _load_module()
    result = prune.prune_recurrent_matrix(
        _toy_bridge_matrix(),
        _toy_pools(),
        max_hops=1,
        max_internal_nodes=1,
    )

    assert result.metadata["candidate_internal_count"] == 2
    assert result.metadata["kept_internal_count"] == 1
    assert result.keep_indices.size == 4


def test_paired_pruned_delta_table() -> None:
    prune = _load_module()
    metrics = pd.DataFrame(
        [
            {"condition": "unpruned", "model": "hemibrain_seeded", "seed": 0, "test_query_accuracy": 0.7},
            {"condition": "unpruned", "model": "hemibrain_seeded", "seed": 1, "test_query_accuracy": 0.8},
            {"condition": "pruned", "model": "hemibrain_seeded", "seed": 0, "test_query_accuracy": 0.9},
            {"condition": "pruned", "model": "hemibrain_seeded", "seed": 1, "test_query_accuracy": 0.9},
        ]
    )

    deltas = prune.paired_pruned_deltas(metrics)
    row = deltas[
        (deltas["model"] == "hemibrain_seeded")
        & (deltas["metric"] == "test_query_accuracy")
    ].iloc[0]
    assert row["N"] == 2
    assert row["mean_delta_pruned_minus_unpruned"] == pytest.approx(0.15)


def test_pruned_comparison_smoke_run(tmp_path: Path) -> None:
    prune = _load_module()
    matrix_path = tmp_path / "adjacency_unsigned.npz"
    pool_path = tmp_path / "pool_assignments.csv"
    sparse.save_npz(matrix_path, _toy_bridge_matrix())
    _toy_pools().to_csv(pool_path, index=False)

    code = prune.main(
        [
            "--matrix",
            str(matrix_path),
            "--pool-assignments",
            str(pool_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--models",
            "hemibrain_seeded",
            "--seeds",
            "0",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--train-batches",
            "1",
            "--val-batches",
            "1",
            "--test-batches",
            "1",
            "--num-odors",
            "8",
            "--odor-dim",
            "6",
            "--odors-per-episode",
            "3",
            "--reversal-count",
            "2",
            "--prune-max-hops",
            "1",
            "--prune-max-internal-nodes",
            "0",
            "--log-every-seconds",
            "0",
        ]
    )

    out = tmp_path / "out"
    assert code == 0
    assert (out / "matrices" / "pruned_adjacency_unsigned.npz").exists()
    assert (out / "unpruned" / "metrics_by_seed.csv").exists()
    assert (out / "pruned" / "metrics_by_seed.csv").exists()
    assert (out / "pruned_vs_unpruned_summary.csv").exists()
    assert (out / "pruned_vs_unpruned_paired_deltas.csv").exists()
