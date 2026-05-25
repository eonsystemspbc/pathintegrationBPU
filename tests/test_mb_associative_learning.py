from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_mb_associative_learning.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("mb_assoc", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_matrix(n: int = 12) -> sparse.coo_matrix:
    rows = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 8], dtype=np.int64)
    cols = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2], dtype=np.int64)
    data = np.linspace(0.05, 0.7, rows.size, dtype=np.float32)
    return sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)


def test_episode_generator_requires_within_episode_association() -> None:
    mb = _load_module()
    episode = mb.EpisodeSpec(
        num_odors=10,
        odor_dim=8,
        odors_per_episode=4,
        reversal_count=2,
        reversal_repeats=1,
        odor_sparsity=0.5,
        odor_noise_std=0.0,
    )
    odor_bank = mb.make_odor_bank(episode, seed=3)
    batch = mb.generate_batch(odor_bank, episode, batch_size=5, rng=np.random.default_rng(4))

    assert batch.inputs.shape == (5, episode.timesteps, episode.input_dim)
    assert batch.targets.shape == (5, episode.timesteps)
    assert np.all(batch.query_mask.sum(axis=1) == 8)
    assert np.all(batch.initial_query_mask.sum(axis=1) == 4)
    assert np.all(batch.final_query_mask.sum(axis=1) == 4)

    query_rows = batch.query_mask.astype(bool)
    reward_col = episode.odor_dim
    punishment_col = episode.odor_dim + 1
    query_col = episode.odor_dim + 2
    assert np.all(batch.inputs[query_rows, reward_col] == 0.0)
    assert np.all(batch.inputs[query_rows, punishment_col] == 0.0)
    assert np.all(batch.inputs[query_rows, query_col] == 1.0)


def test_random_sparse_control_preserves_connectome_control_invariants() -> None:
    mb = _load_module()
    base = _toy_matrix()
    random_matrix = mb.random_sparse_like(base, seed=12)
    assert random_matrix.shape == base.shape
    assert random_matrix.nnz == base.nnz
    assert int(np.sum(random_matrix.row == random_matrix.col)) == int(
        np.sum(base.row == base.col)
    )
    np.testing.assert_allclose(
        np.sort(random_matrix.data),
        np.sort(base.data),
        rtol=0,
        atol=1e-7,
    )


def test_sparse_and_dense_associative_rnns_are_size_matched() -> None:
    mb = _load_module()
    recurrent = _toy_matrix()
    sparse_model = mb.AssociativeRNN(
        recurrent=recurrent,
        input_dim=7,
        runtime="sparse",
        state_clip=5.0,
        seed=1,
    )
    dense_model = mb.AssociativeRNN(
        recurrent=recurrent,
        input_dim=7,
        runtime="dense",
        state_clip=5.0,
        seed=1,
    )
    assert sparse_model.recurrent_parameter_count() == recurrent.nnz
    assert dense_model.recurrent_parameter_count() == recurrent.shape[0] ** 2
    assert dense_model.W_rec.requires_grad

    x = torch.randn(3, 5, 7)
    y = torch.randint(0, 2, (3, 5)).float()
    mask = torch.ones(3, 5)
    loss = mb.masked_bce_loss(dense_model(x), y, mask)
    loss.backward()
    assert dense_model.W_rec.grad is not None


def test_recurrent_orientation_is_post_by_pre() -> None:
    mb = _load_module()
    recurrent = sparse.coo_matrix(
        (
            np.array([2.0, 3.0, 4.0], dtype=np.float32),
            (
                np.array([1, 2, 0], dtype=np.int64),
                np.array([0, 1, 2], dtype=np.int64),
            ),
        ),
        shape=(3, 3),
        dtype=np.float32,
    )
    model = mb.AssociativeRNN(
        recurrent=recurrent,
        input_dim=2,
        runtime="sparse",
        state_clip=0.0,
        seed=5,
    )

    h = torch.tensor([[5.0, 7.0, 11.0]], dtype=torch.float32)
    expected = torch.tensor([[44.0, 10.0, 21.0]], dtype=torch.float32)
    torch.testing.assert_close(model._recurrent_step(h), expected)


def test_sparse_and_dense_recurrent_steps_are_equivalent() -> None:
    mb = _load_module()
    recurrent = _toy_matrix()
    sparse_model = mb.AssociativeRNN(
        recurrent=recurrent,
        input_dim=3,
        runtime="sparse",
        state_clip=0.0,
        seed=8,
    )
    dense_model = mb.AssociativeRNN(
        recurrent=recurrent,
        input_dim=3,
        runtime="dense",
        state_clip=0.0,
        seed=8,
    )

    h = torch.randn(4, recurrent.shape[0])
    torch.testing.assert_close(
        sparse_model._recurrent_step(h),
        dense_model._recurrent_step(h),
        rtol=1e-6,
        atol=1e-6,
    )


def test_associative_learning_smoke_run_writes_metrics_and_figures(tmp_path: Path) -> None:
    mb = _load_module()
    matrix_path = tmp_path / "adjacency_unsigned.npz"
    out = tmp_path / "assoc"
    sparse.save_npz(matrix_path, _toy_matrix().tocsr())

    code = mb.main(
        [
            "--matrix",
            str(matrix_path),
            "--output-dir",
            str(out),
            "--device",
            "cpu",
            "--models",
            "hemibrain_seeded",
            "random_sparse",
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
            "--num-odors",
            "8",
            "--odor-dim",
            "6",
            "--odors-per-episode",
            "3",
            "--reversal-count",
            "1",
            "--log-every-seconds",
            "0",
        ]
    )
    assert code == 0
    metrics = pd.read_csv(out / "metrics_by_seed.csv")
    assert sorted(metrics["model"].tolist()) == ["hemibrain_seeded", "random_sparse"]
    assert (out / "metrics_summary.csv").exists()
    assert (out / "loss_history.csv").exists()
    assert (out / "associative_accuracy.png").exists()
    assert (out / "associative_loss.png").exists()
    assert (out / "associative_learning_report.md").exists()
