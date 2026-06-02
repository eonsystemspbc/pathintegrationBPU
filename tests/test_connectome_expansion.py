from __future__ import annotations

import numpy as np
from scipy import sparse

from src.connectome_expansion import expand_connectome_dcsbm, infer_degree_blocks


def _weighted_signed_matrix() -> sparse.coo_matrix:
    rows = np.array([0, 1, 2, 3, 4, 5, 2, 4, 5, 1, 3, 0], dtype=np.int64)
    cols = np.array([1, 2, 3, 4, 5, 0, 0, 1, 2, 5, 0, 4], dtype=np.int64)
    data = np.array([8, 6, -5, 7, 9, 4, 3, -6, 5, 8, -4, 7], dtype=np.float32)
    return sparse.coo_matrix((data, (rows, cols)), shape=(6, 6), dtype=np.float32)


def test_degree_block_inference_returns_one_label_per_neuron() -> None:
    matrix = _weighted_signed_matrix()

    labels = infer_degree_blocks(matrix)

    assert labels.shape == (matrix.shape[0],)
    assert labels.dtype.kind in {"i", "u"}


def test_dcsbm_expansion_preserves_original_submatrix() -> None:
    base = _weighted_signed_matrix()

    result = expand_connectome_dcsbm(base, factor=2.0, seed=17)

    assert result.matrix.shape == (12, 12)
    original_submatrix = result.matrix.tocsr()[: base.shape[0], : base.shape[1]]
    diff = (original_submatrix - base.tocsr()).tocoo()
    diff.eliminate_zeros()
    assert diff.nnz == 0
    assert result.preserved_original_submatrix
    assert result.expanded_edges >= base.nnz
    assert result.metadata()["method"] == "directed_signed_degree_corrected_sbm"


def test_dcsbm_expansion_is_deterministic_for_seed() -> None:
    base = _weighted_signed_matrix()

    first = expand_connectome_dcsbm(base, factor=2.5, seed=23).matrix.tocsr()
    second = expand_connectome_dcsbm(base, factor=2.5, seed=23).matrix.tocsr()
    different = expand_connectome_dcsbm(base, factor=2.5, seed=24).matrix.tocsr()

    same_diff = (first - second).tocoo()
    same_diff.eliminate_zeros()
    assert same_diff.nnz == 0

    changed_diff = (first - different).tocoo()
    changed_diff.eliminate_zeros()
    assert changed_diff.nnz > 0
