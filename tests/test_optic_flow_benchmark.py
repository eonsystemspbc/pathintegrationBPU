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
    / "run_optic_flow_benchmark.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("optic_flow", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_matrix(n: int = 10) -> sparse.csr_matrix:
    rows = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 7], dtype=np.int64)
    cols = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2], dtype=np.int64)
    data = np.linspace(0.05, 0.6, rows.size, dtype=np.float32)
    return sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def test_hex_lattice_and_optic_flow_batch_have_expected_shapes() -> None:
    optic = _load_module()
    spec = optic.OpticFlowSpec(
        hex_rings=2,
        timesteps=5,
        panorama_width=48,
        panorama_height=24,
        blur_samples=2,
        acceptance_angle_deg=3.0,
        sensor_noise_std=0.0,
    )
    batch = optic.generate_optic_flow_batch(spec, batch_size=4, rng=np.random.default_rng(1))

    assert optic.hex_lattice(2).shape == (19, 2)
    assert batch.inputs.shape == (4, 5, 19)
    assert batch.targets.shape == (4, 5, 3)
    assert np.all(batch.inputs >= 0.0)
    assert np.all(batch.inputs <= 1.0)
    assert np.allclose(batch.targets[:, 0, :], batch.targets[:, -1, :])


def test_optic_flow_controls_are_size_matched() -> None:
    optic = _load_module()
    base = _toy_matrix()
    same_topology_random_weights = optic.model_matrix(
        base, optic.MODEL_RANDOM_WEIGHT_TOPOLOGY, seed=5
    )
    shuffled = optic.model_matrix(base, optic.MODEL_SHUFFLED, seed=5)
    random = optic.model_matrix(base, optic.MODEL_RANDOM, seed=5)

    assert same_topology_random_weights.shape == shuffled.shape == base.shape == random.shape
    assert same_topology_random_weights.nnz == shuffled.nnz == base.nnz == random.nnz
    np.testing.assert_array_equal(
        same_topology_random_weights.tocoo().row,
        base.tocoo().row,
    )
    np.testing.assert_array_equal(
        same_topology_random_weights.tocoo().col,
        base.tocoo().col,
    )
    np.testing.assert_allclose(
        np.sort(np.abs(shuffled.data)),
        np.sort(np.abs(base.data)),
        rtol=0,
        atol=1e-7,
    )
    assert not np.allclose(
        np.sort(same_topology_random_weights.data),
        np.sort(base.data),
    )
    assert not np.allclose(np.sort(random.data), np.sort(base.data))


def test_sparse_optic_flow_rnn_is_trainable_and_vector_valued() -> None:
    optic = _load_module()
    model = optic.SparseOpticFlowRNN(
        recurrent=_toy_matrix(),
        input_dim=7,
        output_dim=3,
        state_clip=3.0,
        seed=2,
    )
    x = torch.rand(3, 4, 7)
    y = torch.randn(3, 4, 3)
    pred = model(x)
    loss = torch.mean((pred - y) ** 2)
    loss.backward()

    assert pred.shape == (3, 4, 3)
    assert model.recurrent_parameter_count() == _toy_matrix().nnz
    assert model.W_rec_values.grad is not None


def test_prepare_optic_lobe_connectome_writes_expected_artifacts(tmp_path: Path) -> None:
    optic = _load_module()
    paths = optic.OutputPaths(tmp_path, tmp_path)
    neurons = pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4],
            "type": ["", "", "", ""],
            "instance": ["", "", "", ""],
            "pre": [12, 9, 5, 2],
            "post": [2, 5, 9, 12],
            "predictedNt": ["ACh", "ACh", "GABA", ""],
        }
    )
    roi_counts = pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4],
            "roi": ["ME_R", "ME_R", "ME_R", "ME_R"],
            "pre": [8, 8, 3, 1],
            "post": [1, 3, 8, 9],
        }
    )
    connections = pd.DataFrame(
        {
            "bodyId_pre": [1, 2, 3, 4, 2],
            "bodyId_post": [2, 3, 4, 1, 4],
            "weight": [4, 3, 2, 1, 2],
        }
    )
    neurons.to_csv(paths.neurons_csv, index=False)
    roi_counts.to_csv(paths.roi_counts_csv, index=False)
    connections.to_csv(paths.connections_csv, index=False)

    matrix = optic.prepare_optic_lobe_connectome(paths, optic_rois=("ME_R",))

    assert matrix.shape == (4, 4)
    assert paths.adjacency_unsigned_npz.exists()
    assert paths.graph_metadata_json.exists()
    assert paths.pool_assignments_csv.exists()
