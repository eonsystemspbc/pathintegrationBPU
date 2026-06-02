from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class ConnectomeExpansionResult:
    matrix: sparse.coo_matrix
    original_n: int
    target_n: int
    factor: float
    seed: int
    original_edges: int
    sampled_edge_events: int
    expanded_edges: int
    original_block_counts: tuple[int, ...]
    expanded_block_counts: tuple[int, ...]
    preserved_original_submatrix: bool

    def metadata(self) -> dict[str, object]:
        return {
            "method": "directed_signed_degree_corrected_sbm",
            "original_n": self.original_n,
            "target_n": self.target_n,
            "factor": self.factor,
            "seed": self.seed,
            "original_edges": self.original_edges,
            "sampled_edge_events": self.sampled_edge_events,
            "expanded_edges": self.expanded_edges,
            "original_block_counts": list(self.original_block_counts),
            "expanded_block_counts": list(self.expanded_block_counts),
            "preserved_original_submatrix": self.preserved_original_submatrix,
            "orientation": "matrix[row=postsynaptic, col=presynaptic]",
        }


def infer_degree_blocks(matrix: sparse.spmatrix) -> np.ndarray:
    """Infer source-like, internal-like, and sink-like blocks from degree imbalance."""
    matrix = matrix.astype(np.float32).tocoo()
    n = int(matrix.shape[0])
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got {matrix.shape}")
    if n == 0:
        raise ValueError("matrix must contain at least one neuron")
    if n < 3:
        return np.zeros(n, dtype=np.int64)

    abs_matrix = abs(matrix).tocsr()
    incoming = np.asarray(abs_matrix.sum(axis=1)).ravel()
    outgoing = np.asarray(abs_matrix.tocsc().sum(axis=0)).ravel()
    total = incoming + outgoing
    imbalance = (outgoing - incoming) / np.maximum(total, 1e-12)

    labels = np.ones(n, dtype=np.int64)
    block_size = max(1, int(round(n * 0.15)))
    order = np.argsort(imbalance)
    sink = order[:block_size]
    source = order[-block_size:]
    labels[source] = 0
    labels[sink] = 2
    return labels


def expand_connectome_dcsbm(
    matrix: sparse.spmatrix,
    factor: float = 1.0,
    target_neurons: int | None = None,
    block_labels: np.ndarray | None = None,
    seed: int = 0,
    min_strength: float = 1e-9,
) -> ConnectomeExpansionResult:
    """Expand a directed signed connectome with a degree-corrected block model.

    The input matrix is interpreted in the repo's recurrent orientation:
    rows are postsynaptic targets and columns are presynaptic sources. The
    original submatrix is restored exactly in the output.
    """
    base = matrix.astype(np.float32).tocoo()
    base.sum_duplicates()
    base.eliminate_zeros()
    if base.shape[0] != base.shape[1]:
        raise ValueError(f"matrix must be square, got {base.shape}")
    original_n = int(base.shape[0])
    if original_n == 0:
        raise ValueError("matrix must contain at least one neuron")
    if base.nnz == 0:
        raise ValueError("matrix must contain at least one edge")
    target_n = _target_neuron_count(original_n, factor, target_neurons)
    labels = _compact_labels(
        infer_degree_blocks(base) if block_labels is None else np.asarray(block_labels)
    )
    if labels.shape != (original_n,):
        raise ValueError(
            f"block_labels must have shape ({original_n},), got {labels.shape}"
        )

    if target_n <= original_n:
        result = base.copy().astype(np.float32).tocoo()
        counts = tuple(int(v) for v in np.bincount(labels))
        return ConnectomeExpansionResult(
            matrix=result,
            original_n=original_n,
            target_n=original_n,
            factor=1.0,
            seed=int(seed),
            original_edges=int(base.nnz),
            sampled_edge_events=0,
            expanded_edges=int(result.nnz),
            original_block_counts=counts,
            expanded_block_counts=counts,
            preserved_original_submatrix=True,
        )

    rng = np.random.default_rng(seed)
    incoming, outgoing = _strengths_by_orientation(base, min_strength)
    omega, positive_prob = _fit_block_parameters(base, labels, incoming, outgoing)
    all_labels = _sample_expanded_labels(labels, target_n, rng)
    all_incoming, all_outgoing = _bootstrap_strengths(
        labels,
        all_labels,
        incoming,
        outgoing,
        original_n,
        rng,
        min_strength,
    )

    rows, cols, data, sampled_events = _sample_expanded_edges(
        base,
        all_labels,
        all_incoming,
        all_outgoing,
        omega,
        positive_prob,
        original_n,
        target_n,
        rng,
    )
    expanded = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(target_n, target_n),
        dtype=np.float32,
    )
    expanded.sum_duplicates()
    expanded.eliminate_zeros()
    preserved = _submatrix_is_exact(expanded, base)
    return ConnectomeExpansionResult(
        matrix=expanded.tocoo(),
        original_n=original_n,
        target_n=target_n,
        factor=float(target_n / original_n),
        seed=int(seed),
        original_edges=int(base.nnz),
        sampled_edge_events=int(sampled_events),
        expanded_edges=int(expanded.nnz),
        original_block_counts=tuple(int(v) for v in np.bincount(labels)),
        expanded_block_counts=tuple(int(v) for v in np.bincount(all_labels)),
        preserved_original_submatrix=preserved,
    )


def _target_neuron_count(
    original_n: int,
    factor: float,
    target_neurons: int | None,
) -> int:
    if target_neurons is not None and target_neurons > 0:
        return int(target_neurons)
    if factor < 1.0:
        raise ValueError("factor must be at least 1.0")
    if factor == 1.0:
        return int(original_n)
    return max(original_n + 1, int(math.ceil(original_n * factor)))


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if labels.size == 0:
        raise ValueError("block_labels cannot be empty")
    unique = np.unique(labels)
    mapping = {int(value): idx for idx, value in enumerate(unique.tolist())}
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.int64)


def _strengths_by_orientation(
    matrix: sparse.coo_matrix,
    min_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    abs_matrix = abs(matrix).tocsr()
    incoming = np.asarray(abs_matrix.sum(axis=1)).ravel().astype(np.float64)
    outgoing = np.asarray(abs_matrix.tocsc().sum(axis=0)).ravel().astype(np.float64)
    active = (incoming + outgoing) > 0
    incoming = np.where(active, np.maximum(incoming, min_strength), 0.0)
    outgoing = np.where(active, np.maximum(outgoing, min_strength), 0.0)
    return incoming, outgoing


def _fit_block_parameters(
    matrix: sparse.coo_matrix,
    labels: np.ndarray,
    incoming: np.ndarray,
    outgoing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    block_count = int(labels.max()) + 1
    omega = np.zeros((block_count, block_count), dtype=np.float64)
    positive_prob = np.ones((block_count, block_count), dtype=np.float64)
    global_positive = float(np.mean(matrix.data > 0)) if matrix.nnz else 1.0

    incoming_sums = np.bincount(labels, weights=incoming, minlength=block_count)
    outgoing_sums = np.bincount(labels, weights=outgoing, minlength=block_count)
    for post_block in range(block_count):
        post_mask = labels[matrix.row] == post_block
        for pre_block in range(block_count):
            mask = post_mask & (labels[matrix.col] == pre_block)
            denom = incoming_sums[post_block] * outgoing_sums[pre_block]
            if denom > 0:
                omega[post_block, pre_block] = float(np.abs(matrix.data[mask]).sum() / denom)
            if np.any(mask):
                positive_prob[post_block, pre_block] = float(np.mean(matrix.data[mask] > 0))
            else:
                positive_prob[post_block, pre_block] = global_positive
    return omega, positive_prob


def _sample_expanded_labels(
    labels: np.ndarray,
    target_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    original_n = int(labels.size)
    block_count = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=block_count).astype(np.float64)
    probabilities = counts / counts.sum()
    new_labels = rng.choice(
        np.arange(block_count, dtype=np.int64),
        size=target_n - original_n,
        replace=True,
        p=probabilities,
    )
    return np.concatenate([labels, new_labels.astype(np.int64)])


def _bootstrap_strengths(
    labels: np.ndarray,
    all_labels: np.ndarray,
    incoming: np.ndarray,
    outgoing: np.ndarray,
    original_n: int,
    rng: np.random.Generator,
    min_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    target_n = int(all_labels.size)
    all_incoming = np.zeros(target_n, dtype=np.float64)
    all_outgoing = np.zeros(target_n, dtype=np.float64)
    all_incoming[:original_n] = incoming
    all_outgoing[:original_n] = outgoing

    for block in range(int(all_labels.max()) + 1):
        source_indices = np.flatnonzero(labels == block)
        new_indices = np.flatnonzero(all_labels[original_n:] == block) + original_n
        if new_indices.size == 0:
            continue
        if source_indices.size == 0:
            source_indices = np.arange(original_n, dtype=np.int64)
        choices = rng.choice(source_indices, size=new_indices.size, replace=True)
        all_incoming[new_indices] = incoming[choices]
        all_outgoing[new_indices] = outgoing[choices]

    active = (all_incoming + all_outgoing) > 0
    all_incoming = np.where(active, np.maximum(all_incoming, min_strength), 0.0)
    all_outgoing = np.where(active, np.maximum(all_outgoing, min_strength), 0.0)
    in_sum = float(all_incoming.sum())
    out_sum = float(all_outgoing.sum())
    if in_sum > 0 and out_sum > 0:
        all_outgoing *= in_sum / out_sum
    return all_incoming, all_outgoing


def _sample_expanded_edges(
    base: sparse.coo_matrix,
    labels: np.ndarray,
    incoming: np.ndarray,
    outgoing: np.ndarray,
    omega: np.ndarray,
    positive_prob: np.ndarray,
    original_n: int,
    target_n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    row_parts = [base.row.astype(np.int64)]
    col_parts = [base.col.astype(np.int64)]
    data_parts = [base.data.astype(np.float32)]
    sampled_events = 0

    for post_block in range(omega.shape[0]):
        post_indices = np.flatnonzero(labels == post_block)
        post_weights = incoming[post_indices]
        post_sum = float(post_weights.sum())
        if post_indices.size == 0 or post_sum <= 0:
            continue
        post_prob = post_weights / post_sum
        for pre_block in range(omega.shape[1]):
            pre_indices = np.flatnonzero(labels == pre_block)
            pre_weights = outgoing[pre_indices]
            pre_sum = float(pre_weights.sum())
            rate = float(omega[post_block, pre_block] * post_sum * pre_sum)
            if pre_indices.size == 0 or pre_sum <= 0 or rate <= 0:
                continue
            event_count = int(rng.poisson(rate))
            if event_count <= 0:
                continue
            pre_prob = pre_weights / pre_sum
            sampled_rows = rng.choice(post_indices, size=event_count, replace=True, p=post_prob)
            sampled_cols = rng.choice(pre_indices, size=event_count, replace=True, p=pre_prob)
            keep = (sampled_rows >= original_n) | (sampled_cols >= original_n)
            if not np.any(keep):
                continue
            kept_rows = sampled_rows[keep].astype(np.int64)
            kept_cols = sampled_cols[keep].astype(np.int64)
            signs = rng.random(kept_rows.size) < positive_prob[post_block, pre_block]
            kept_data = np.where(signs, 1.0, -1.0).astype(np.float32)
            row_parts.append(kept_rows)
            col_parts.append(kept_cols)
            data_parts.append(kept_data)
            sampled_events += int(kept_rows.size)

    rows = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int64)
    cols = np.concatenate(col_parts) if col_parts else np.empty(0, dtype=np.int64)
    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=np.float32)
    if rows.size and (int(rows.max()) >= target_n or int(cols.max()) >= target_n):
        raise AssertionError("sampled edge index exceeds expanded matrix size")
    return rows, cols, data, sampled_events


def _submatrix_is_exact(expanded: sparse.coo_matrix, base: sparse.coo_matrix) -> bool:
    original_n = int(base.shape[0])
    sub = expanded.tocsr()[:original_n, :original_n].tocoo()
    diff = (sub - base).tocoo()
    diff.eliminate_zeros()
    return diff.nnz == 0
