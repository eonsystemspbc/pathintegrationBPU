from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.config import RHO_TARGET
from src.connectome import (
    build_raw_adjacency,
    build_signed_adjacency,
    choose_primary_matrix,
    control_invariants,
    degree_preserving_shuffle_matrix,
    estimate_k_from_support,
    random_control_matrix,
    scale_to_spectral_radius,
    sign_coverage,
    spectral_radius,
    weight_shuffled_control_matrix,
)
from src.models import CXBPU, assert_bpu_trainable_surface, count_trainable_parameters
from src.pools import assign_pools


def toy_neurons() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4],
            "type": ["ring_A", "CPU", "CPU", "PFL3"],
            "instance": ["ring_A", "int_1", "int_2", "PFL3_R"],
            "pre": [100, 100, 100, 100],
            "post": [100, 100, 100, 100],
            "predictedNt": ["ACh", "GABA", "Glu", "ACh"],
        }
    )


def toy_roi_counts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4],
            "roi": ["EB", "EB", "PB", "FB"],
            "pre": [95, 90, 90, 50],
            "post": [60, 90, 90, 95],
        }
    )


def toy_connections() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bodyId_pre": [1, 2, 3, 1, 2],
            "bodyId_post": [2, 3, 4, 1, 4],
            "weight": [5.0, 7.0, 11.0, 3.0, 13.0],
        }
    )


def test_w_rec_orientation_from_edge_table() -> None:
    matrix, body_to_index, _ = build_raw_adjacency(toy_neurons(), toy_connections())
    assert matrix[body_to_index[2], body_to_index[1]] == 5.0
    assert matrix[body_to_index[1], body_to_index[2]] == 0.0


def test_spectral_scaling_preserves_support_and_ratios() -> None:
    matrix, _, _ = build_raw_adjacency(toy_neurons(), toy_connections())
    scaled, _, raw_rho, scale = scale_to_spectral_radius(matrix, None, "unsigned")
    assert raw_rho > 0
    assert scale > 0
    assert np.array_equal(matrix.tocoo().row, scaled.tocoo().row)
    assert np.array_equal(matrix.tocoo().col, scaled.tocoo().col)
    original_ratio = matrix.data[0] / matrix.data[-1]
    scaled_ratio = scaled.data[0] / scaled.data[-1]
    assert scaled_ratio == pytest.approx(original_ratio)
    assert spectral_radius(scaled) == pytest.approx(RHO_TARGET, rel=1e-5)


def test_signed_policy_falls_back_when_coverage_is_low() -> None:
    matrix, _, _ = build_raw_adjacency(toy_neurons(), toy_connections())
    signs = {0: 1.0}
    signed = build_signed_adjacency(matrix, signs)
    coverage = sign_coverage(matrix, signs)
    assert coverage < 0.95
    assert choose_primary_matrix("auto", coverage, signed) == "unsigned"
    assert choose_primary_matrix("force_signed", coverage, signed) == "signed"


def test_pool_assignment_exhaustive_and_mutually_exclusive() -> None:
    pools = assign_pools(toy_neurons(), toy_roi_counts())
    assert set(pools["bodyId"]) == {1, 2, 3, 4}
    assert pools[["is_sensory", "is_internal", "is_output"]].sum(axis=1).eq(1).all()
    assert pools.set_index("bodyId").loc[1, "pool"] == "sensory"
    assert pools.set_index("bodyId").loc[4, "pool"] == "output"


def test_cx_bpu_trainable_surface_only_adapters() -> None:
    matrix, _, _ = build_raw_adjacency(toy_neurons(), toy_connections())
    model = CXBPU(matrix, sensory_indices=[0], output_indices=[3], K=3)
    assert_bpu_trainable_surface(model)
    assert count_trainable_parameters(model) == 1 * 2 + 1 + 4 * 1 + 4


def test_controls_preserve_required_invariants() -> None:
    matrix, _, _ = build_raw_adjacency(toy_neurons(), toy_connections())
    random_matrix = random_control_matrix(matrix, seed=1)
    degree_matrix = degree_preserving_shuffle_matrix(matrix, seed=2)
    weight_matrix = weight_shuffled_control_matrix(matrix, seed=3)
    base = control_invariants(matrix)
    for control in (random_matrix, degree_matrix, weight_matrix):
        inv = control_invariants(control)
        assert inv["N"] == base["N"]
        assert inv["edge_count"] == base["edge_count"]
        assert inv["self_loop_count"] == base["self_loop_count"]
        assert inv["weight_multiset"] == base["weight_multiset"]
    degree_inv = control_invariants(degree_matrix)
    assert degree_inv["in_degree"] == base["in_degree"]
    assert degree_inv["out_degree"] == base["out_degree"]
    assert np.array_equal(weight_matrix.nonzero()[0], matrix.nonzero()[0])
    assert np.array_equal(weight_matrix.nonzero()[1], matrix.nonzero()[1])


def test_k_estimation_uses_reachable_paths_and_clips() -> None:
    rows = np.arange(1, 10)
    cols = np.arange(0, 9)
    long_chain = sparse.coo_matrix((np.ones(9), (rows, cols)), shape=(10, 10)).tocsr()
    assert estimate_k_from_support(long_chain, [0], [9]) == 8
    short_chain = sparse.coo_matrix(([1.0], ([1], [0])), shape=(2, 2)).tocsr()
    assert estimate_k_from_support(short_chain, [0], [1]) == 3
    disconnected = sparse.csr_matrix((3, 3))
    with pytest.raises(RuntimeError):
        estimate_k_from_support(disconnected, [0], [2])
