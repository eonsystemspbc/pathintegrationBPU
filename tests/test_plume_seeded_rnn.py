from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse


pytestmark = pytest.mark.skipif(
    find_spec("gym") is None,
    reason="plumetracknets PPO module imports gym",
)

PPO_ROOT = Path(__file__).resolve().parents[1] / "plumetracknets" / "code" / "ppo"
sys.path.insert(0, str(PPO_ROOT))


def _load_plume_model_symbols():
    from a2c_ppo_acktr.model import (
        ConnectomeBPUBase,
        SeededRNNBase,
        _random_sparse_like,
    )

    return ConnectomeBPUBase, SeededRNNBase, _random_sparse_like


def _toy_matrix() -> sparse.coo_matrix:
    rows = np.array([0, 1, 2, 3, 4, 2], dtype=np.int64)
    cols = np.array([0, 0, 1, 2, 3, 4], dtype=np.int64)
    data = np.array([0.5, 0.2, -0.1, 0.4, 0.3, 0.7], dtype=np.float32)
    return sparse.coo_matrix((data, (rows, cols)), shape=(5, 5))


def _write_pool_csv(path: Path) -> None:
    pools = pd.DataFrame(
        {
            "bodyId": [10, 11, 12, 13, 14],
            "index": [0, 1, 2, 3, 4],
            "pool": ["sensory", "internal", "output", "output", "internal"],
        }
    )
    pools.to_csv(path, index=False)


def test_random_sparse_like_preserves_size_self_loops_and_weight_multiset() -> None:
    _, _, random_sparse_like = _load_plume_model_symbols()
    matrix = _toy_matrix()
    random_matrix = random_sparse_like(matrix, seed=7)
    assert random_matrix.shape == matrix.shape
    assert random_matrix.nnz == matrix.nnz
    assert int(np.sum(random_matrix.row == random_matrix.col)) == int(
        np.sum(matrix.row == matrix.col)
    )
    assert sorted(np.round(random_matrix.data, 6).tolist()) == sorted(
        np.round(matrix.data, 6).tolist()
    )


def test_connectome_and_random_seeded_plume_rnns_are_size_matched(tmp_path: Path) -> None:
    ConnectomeBPUBase, _, _ = _load_plume_model_symbols()
    matrix_path = tmp_path / "adjacency_unsigned.npz"
    pools_path = tmp_path / "pool_assignments.csv"
    sparse.save_npz(matrix_path, _toy_matrix().tocsr())
    _write_pool_csv(pools_path)

    common = {
        "num_inputs": 3,
        "hidden_size": 4,
        "matrix_path": str(matrix_path),
        "pools_path": str(pools_path),
        "bpu_k": 2,
        "bpu_train_recurrent": True,
        "bpu_state_clip": 5.0,
        "bpu_value_clip": 2.0,
    }
    connectome = ConnectomeBPUBase(**common, bpu_init="connectome", bpu_init_seed=7)
    random = ConnectomeBPUBase(**common, bpu_init="random_sparse", bpu_init_seed=7)

    assert connectome.N == random.N == 5
    assert connectome.W_rec_values.numel() == random.W_rec_values.numel()
    assert connectome.W_in.shape == random.W_in.shape
    assert connectome.actor1[0].weight.shape == random.actor1[0].weight.shape
    assert connectome.W_rec_values.requires_grad
    assert random.W_rec_values.requires_grad

    inputs = torch.randn(2, 3)
    hxs = torch.zeros(2, 5)
    masks = torch.ones(2, 1)
    value, actor_features, next_h, _ = connectome(inputs, hxs, masks)
    loss = value.mean() + actor_features.mean() + next_h.mean()
    loss.backward()
    assert connectome.W_rec_values.grad is not None
    assert value.shape == (2, 1)
    assert actor_features.shape == (2, 4)
    assert next_h.shape == (2, 5)


def test_dense_seeded_rnn_random_and_connectome_inits_are_size_matched(
    tmp_path: Path,
) -> None:
    _, SeededRNNBase, _ = _load_plume_model_symbols()
    matrix_path = tmp_path / "adjacency_unsigned.npz"
    sparse.save_npz(matrix_path, _toy_matrix().tocsr())

    common = {
        "num_inputs": 3,
        "hidden_size": 4,
        "matrix_path": str(matrix_path),
        "seeded_rnn_init_seed": 11,
        "seeded_rnn_state_clip": 5.0,
    }
    random = SeededRNNBase(**common, seeded_rnn_init="random")
    connectome = SeededRNNBase(**common, seeded_rnn_init="connectome")

    assert random.N == connectome.N == 5
    assert random.W_rec.shape == connectome.W_rec.shape == (5, 5)
    assert random.W_in.shape == connectome.W_in.shape == (5, 3)
    assert random.actor1[0].weight.shape == connectome.actor1[0].weight.shape
    assert random.W_rec.requires_grad
    assert connectome.W_rec.requires_grad

    dense = _toy_matrix().toarray()
    np.testing.assert_allclose(connectome.W_rec.detach().numpy(), dense, atol=1e-6)

    inputs = torch.randn(2, 3)
    hxs = torch.zeros(2, 5)
    masks = torch.ones(2, 1)
    value, actor_features, next_h, _ = connectome(inputs, hxs, masks)
    loss = value.mean() + actor_features.mean() + next_h.mean()
    loss.backward()
    assert connectome.W_rec.grad is not None
    assert value.shape == (2, 1)
    assert actor_features.shape == (2, 4)
    assert next_h.shape == (2, 5)
