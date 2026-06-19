from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import hashlib

import networkx as nx
import numpy as np
import pandas as pd
from scipy import linalg as scipy_linalg
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .acquire import require_raw_exports
from .config import (
    CONNECTOME_FLYWIRE_WHOLE,
    CONNECTOME_FLYWIRE_MUSHROOM_BODY,
    CONNECTOME_HEMIBRAIN_CX,
    CONNECTOME_HEMIBRAIN_MUSHROOM_BODY,
    CX_ROI_LABELS,
    HEMIBRAIN_MB_ROI_LABELS,
    MB_ROI_LABELS,
    RHO_TARGET,
    SIGN_COVERAGE_THRESHOLD,
    OutputPaths,
)
from .pools import (
    validate_pool_assignments,
    write_pool_assignments,
    write_whole_brain_pool_assignments,
)


TRANSMITTER_SIGN = {
    "ach": 1.0,
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glu": -1.0,
    "glutamate": -1.0,
}


class ConnectomePreparationError(RuntimeError):
    """Raised when graph construction or validation fails."""


@dataclass(frozen=True)
class PreparedGraph:
    matrix: sparse.csr_matrix
    unsigned: sparse.csr_matrix
    signed: sparse.csr_matrix | None
    metadata: dict[str, object]
    pools: pd.DataFrame


def normalize_connections(connections: pd.DataFrame) -> pd.DataFrame:
    if connections.empty:
        return pd.DataFrame(columns=["bodyId_pre", "bodyId_post", "weight"])
    rename = {}
    candidates = {
        "bodyId_pre": ("bodyId_pre", "pre_bodyId", "pre", "bodyId_x"),
        "bodyId_post": ("bodyId_post", "post_bodyId", "post", "bodyId_y"),
        "weight": ("weight", "syn_count", "synapse_count", "count"),
    }
    for out_col, names in candidates.items():
        for name in names:
            if name in connections.columns:
                rename[name] = out_col
                break
    out = connections.rename(columns=rename).copy()
    missing = {"bodyId_pre", "bodyId_post", "weight"}.difference(out.columns)
    if missing:
        raise ConnectomePreparationError(
            f"connections.csv is missing required columns: {sorted(missing)}"
        )
    out["bodyId_pre"] = out["bodyId_pre"].astype("int64")
    out["bodyId_post"] = out["bodyId_post"].astype("int64")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    out = out[out["weight"] > 0].copy()
    return (
        out.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"]
        .sum()
        .sort_values(["bodyId_pre", "bodyId_post"])
    )


def build_raw_adjacency(
    neurons: pd.DataFrame,
    connections: pd.DataFrame,
) -> tuple[sparse.csr_matrix, dict[int, int], pd.DataFrame]:
    if "bodyId" not in neurons.columns:
        raise ConnectomePreparationError("neurons.csv must contain bodyId.")
    body_ids = neurons["bodyId"].astype("int64").drop_duplicates().sort_values().tolist()
    body_to_index = {body_id: idx for idx, body_id in enumerate(body_ids)}
    edges = normalize_connections(connections)
    edges = edges[
        edges["bodyId_pre"].isin(body_to_index) & edges["bodyId_post"].isin(body_to_index)
    ].copy()
    if edges.empty:
        raise ConnectomePreparationError("No within-CX edges remain after filtering.")
    rows = edges["bodyId_post"].map(body_to_index).to_numpy(dtype=np.int64)
    cols = edges["bodyId_pre"].map(body_to_index).to_numpy(dtype=np.int64)
    data = edges["weight"].to_numpy(dtype=np.float32)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(len(body_ids), len(body_ids)))
    matrix = matrix.tocsr()
    if matrix.nnz != len(edges):
        matrix.sum_duplicates()
    return matrix, body_to_index, edges


def _canonical_transmitter(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    lowered = raw.lower()
    if any(sep in lowered for sep in (",", ";", "|", "/", "+")):
        labels = [part.strip() for part in re.split(r"[,;|/+]", lowered) if part.strip()]
        mapped = [label for label in labels if label in TRANSMITTER_SIGN]
        if len(mapped) != len(labels) or len(set(mapped)) != 1:
            return None
        return mapped[0]
    if lowered in TRANSMITTER_SIGN:
        return lowered
    aliases = {
        "acetylcholine": "acetylcholine",
        "ach": "ach",
        "gaba": "gaba",
        "glutamate": "glutamate",
        "glu": "glu",
    }
    return aliases.get(lowered)


def assign_presynaptic_signs(neurons: pd.DataFrame, body_to_index: dict[int, int]) -> dict[int, float]:
    label_columns = [
        col
        for col in neurons.columns
        if col.lower()
        in {
            "predictednt",
            "predicted_nt",
            "nt",
            "transmitter",
            "predictedtransmitter",
            "predicted_transmitter",
        }
    ]
    signs: dict[int, float] = {}
    if not label_columns:
        return signs
    label_col = label_columns[0]
    for _, row in neurons.iterrows():
        body_id = int(row["bodyId"])
        if body_id not in body_to_index:
            continue
        label = _canonical_transmitter(row.get(label_col))
        if label is not None:
            signs[body_to_index[body_id]] = TRANSMITTER_SIGN[label]
    return signs


def build_signed_adjacency(
    unsigned: sparse.csr_matrix,
    signs_by_pre_index: dict[int, float],
) -> sparse.csr_matrix | None:
    if not signs_by_pre_index:
        return None
    csc = unsigned.tocsc(copy=True)
    for col, sign in signs_by_pre_index.items():
        start, end = csc.indptr[col], csc.indptr[col + 1]
        csc.data[start:end] *= sign
    return csc.tocsr()


def sign_coverage(unsigned: sparse.csr_matrix, signs_by_pre_index: dict[int, float]) -> float:
    total = float(unsigned.data.sum())
    if total <= 0:
        return 0.0
    csc = unsigned.tocsc()
    covered = 0.0
    for col in signs_by_pre_index:
        covered += float(csc.data[csc.indptr[col] : csc.indptr[col + 1]].sum())
    return covered / total


def spectral_radius(matrix: sparse.spmatrix) -> float:
    if matrix.nnz == 0:
        return 0.0
    n = matrix.shape[0]
    if n <= 256:
        vals = np.linalg.eigvals(matrix.toarray().astype(np.float64))
        return float(np.max(np.abs(vals)))
    if n > 20_000 or matrix.nnz > 2_000_000:
        return _power_iteration_radius(matrix, iters=160)
    try:
        vals = sparse_linalg.eigs(
            matrix.astype(np.float64),
            k=1,
            which="LM",
            return_eigenvectors=False,
            maxiter=max(1000, n * 4),
        )
        return float(np.max(np.abs(vals)))
    except Exception:
        return _power_iteration_radius(matrix)


def _power_iteration_radius(matrix: sparse.spmatrix, iters: int = 200) -> float:
    rng = np.random.default_rng(0)
    x = rng.normal(size=matrix.shape[1])
    x /= np.linalg.norm(x) + 1e-12
    last_norm = 0.0
    for _ in range(iters):
        y = matrix @ x
        norm = float(np.linalg.norm(y))
        if norm == 0:
            return 0.0
        x = y / norm
        last_norm = norm
    return last_norm


def power_iteration_radius(matrix: sparse.spmatrix, iters: int = 120) -> float:
    return _power_iteration_radius(matrix, iters=iters)


def scale_to_spectral_radius(
    unsigned: sparse.csr_matrix,
    signed: sparse.csr_matrix | None,
    primary_name: str,
    rho_target: float = RHO_TARGET,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix | None, float, float]:
    primary = signed if primary_name == "signed" else unsigned
    if primary is None:
        raise ConnectomePreparationError("signed primary requested but no signed matrix exists.")
    rho = spectral_radius(primary)
    if rho <= 0:
        raise ConnectomePreparationError("Primary adjacency has zero spectral radius.")
    scale = rho_target / rho
    scaled_unsigned = (unsigned * scale).astype(np.float32).tocsr()
    scaled_signed = (signed * scale).astype(np.float32).tocsr() if signed is not None else None
    return scaled_unsigned, scaled_signed, float(rho), float(scale)


def choose_primary_matrix(
    signed_policy: str,
    coverage: float,
    signed: sparse.csr_matrix | None,
) -> str:
    if signed_policy == "force_unsigned":
        return "unsigned"
    if signed_policy == "force_signed":
        if signed is None:
            raise ConnectomePreparationError(
                "--signed-policy force_signed requested, but no unambiguous transmitter signs were found."
            )
        return "signed"
    if signed is not None and coverage >= SIGN_COVERAGE_THRESHOLD:
        return "signed"
    return "unsigned"


def pool_indices(assignments: pd.DataFrame) -> dict[str, list[int]]:
    validate_pool_assignments(assignments)
    return {
        pool: assignments.loc[assignments["pool"] == pool, "index"].astype(int).tolist()
        for pool in ("sensory", "internal", "output")
    }


def estimate_k_from_support(
    matrix: sparse.spmatrix,
    sensory_indices: Iterable[int],
    output_indices: Iterable[int],
    min_k: int = 3,
    max_k: int = 8,
) -> int:
    sensory = list(map(int, sensory_indices))
    outputs = set(map(int, output_indices))
    if not sensory:
        raise ConnectomePreparationError("No sensory pool neurons are available for K estimation.")
    if not outputs:
        raise ConnectomePreparationError("No output pool neurons are available for K estimation.")
    coo = matrix.tocoo()
    graph = nx.DiGraph()
    graph.add_nodes_from(range(matrix.shape[0]))
    graph.add_edges_from((int(pre), int(post)) for post, pre in zip(coo.row, coo.col))
    distances: list[int] = []
    for source in sensory:
        lengths = nx.single_source_shortest_path_length(graph, source, cutoff=max_k * 4)
        distances.extend(int(dist) for node, dist in lengths.items() if node in outputs and dist > 0)
    if not distances:
        raise ConnectomePreparationError("No reachable sensory-to-output path exists in CX support.")
    median = int(round(float(np.median(distances))))
    return int(np.clip(median, min_k, max_k))


def estimate_k_from_support_sampled(
    matrix: sparse.spmatrix,
    sensory_indices: Iterable[int],
    output_indices: Iterable[int],
    min_k: int = 3,
    max_k: int = 8,
    max_sources: int = 256,
    seed: int = 0,
) -> int:
    sensory = np.array(list(map(int, sensory_indices)), dtype=np.int64)
    outputs = np.array(list(map(int, output_indices)), dtype=np.int64)
    if sensory.size == 0:
        raise ConnectomePreparationError("No sensory pool neurons are available for K estimation.")
    if outputs.size == 0:
        raise ConnectomePreparationError("No output pool neurons are available for K estimation.")
    rng = np.random.default_rng(seed)
    if sensory.size > max_sources:
        sensory = rng.choice(sensory, size=max_sources, replace=False)
    output_mask = np.zeros(matrix.shape[0], dtype=bool)
    output_mask[outputs] = True
    support_csc = matrix.tocsc(copy=True)
    support_csc.data = np.ones_like(support_csc.data, dtype=np.float32)
    distances: list[int] = []
    for source in sensory:
        visited = np.zeros(matrix.shape[0], dtype=bool)
        frontier = np.array([int(source)], dtype=np.int64)
        visited[frontier] = True
        for depth in range(1, max_k * 4 + 1):
            next_nodes = support_csc[:, frontier].nonzero()[0]
            if next_nodes.size == 0:
                break
            next_nodes = np.unique(next_nodes[~visited[next_nodes]])
            if next_nodes.size == 0:
                break
            if output_mask[next_nodes].any():
                distances.append(depth)
                break
            visited[next_nodes] = True
            frontier = next_nodes
    if not distances:
        raise ConnectomePreparationError(
            "No reachable sensory-to-output path exists in sampled whole-brain support."
        )
    median = int(round(float(np.median(distances))))
    return int(np.clip(median, min_k, max_k))


def _matrix_nonzero_triplets(matrix: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coo = matrix.tocoo()
    return coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data.astype(np.float32)


def random_control_matrix(matrix: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    rows, cols, weights = _matrix_nonzero_triplets(matrix)
    n = matrix.shape[0]
    rng = np.random.default_rng(seed)
    self_count = int(np.sum(rows == cols))
    self_nodes = rng.choice(n, size=self_count, replace=False) if self_count else np.array([], dtype=int)
    target_nonself = len(weights) - self_count
    total_nonself = n * (n - 1)
    if target_nonself > total_nonself:
        raise ConnectomePreparationError("Cannot sample requested non-self edge count.")
    if target_nonself > total_nonself // 3:
        code_iter = rng.choice(total_nonself, size=target_nonself, replace=False)
    else:
        selected: set[int] = set()
        chunks: list[np.ndarray] = []
        while len(selected) < target_nonself:
            remaining = target_nonself - len(selected)
            draw_count = max(remaining * 2, 4096)
            codes = rng.integers(0, total_nonself, size=draw_count, dtype=np.int64)
            keep: list[int] = []
            for code in codes:
                value = int(code)
                if value not in selected:
                    selected.add(value)
                    keep.append(value)
                    if len(selected) == target_nonself:
                        break
            if keep:
                chunks.append(np.array(keep, dtype=np.int64))
        code_iter = np.concatenate(chunks) if chunks else np.array([], dtype=np.int64)
    nonself_posts = (code_iter // (n - 1)).astype(np.int64)
    nonself_pres = (code_iter % (n - 1)).astype(np.int64)
    nonself_pres = nonself_pres + (nonself_pres >= nonself_posts)
    shuffled_weights = rng.permutation(weights)
    out_rows = np.concatenate([self_nodes.astype(np.int64), nonself_posts])
    out_cols = np.concatenate([self_nodes.astype(np.int64), nonself_pres])
    return sparse.coo_matrix(
        (shuffled_weights, (out_rows, out_cols)), shape=matrix.shape
    ).tocsr()


def weight_shuffled_control_matrix(matrix: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    rows, cols, weights = _matrix_nonzero_triplets(matrix)
    rng = np.random.default_rng(seed)
    return sparse.coo_matrix(
        (rng.permutation(weights), (rows, cols)), shape=matrix.shape
    ).tocsr()


def degree_preserving_shuffle_matrix(
    matrix: sparse.csr_matrix,
    seed: int,
    swap_multiplier: int = 10,
) -> sparse.csr_matrix:
    rows, cols, weights = _matrix_nonzero_triplets(matrix)
    rng = np.random.default_rng(seed)
    self_edges = {(int(c), int(r)) for r, c in zip(rows, cols) if r == c}
    edges = [(int(c), int(r)) for r, c in zip(rows, cols) if r != c]
    edge_set = set(edges).union(self_edges)
    if len(edges) < 2:
        return weight_shuffled_control_matrix(matrix, seed)
    attempts = max(len(edges) * swap_multiplier, 100)
    swaps = 0
    for _ in range(attempts):
        i, j = rng.choice(len(edges), size=2, replace=False)
        a, b = edges[i]
        c, d = edges[j]
        if len({a, b, c, d}) < 4:
            continue
        new1 = (a, d)
        new2 = (c, b)
        if new1[0] == new1[1] or new2[0] == new2[1]:
            continue
        if new1 in edge_set or new2 in edge_set:
            continue
        edge_set.remove(edges[i])
        edge_set.remove(edges[j])
        edges[i] = new1
        edges[j] = new2
        edge_set.add(new1)
        edge_set.add(new2)
        swaps += 1
    all_edges = list(self_edges) + edges
    shuffled_weights = rng.permutation(weights)
    out_cols = np.array([edge[0] for edge in all_edges], dtype=np.int64)
    out_rows = np.array([edge[1] for edge in all_edges], dtype=np.int64)
    return sparse.coo_matrix(
        (shuffled_weights, (out_rows, out_cols)), shape=matrix.shape
    ).tocsr()


def _random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed n x n orthogonal matrix (QR of a Gaussian, sign-corrected)."""
    gaussian = rng.standard_normal((n, n))
    q, r = np.linalg.qr(gaussian)
    diag_sign = np.sign(np.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    return q * diag_sign


def _matrix_fingerprint(matrix: sparse.csr_matrix) -> str:
    coo = matrix.tocoo()
    hasher = hashlib.sha1()
    hasher.update(np.ascontiguousarray(coo.row.astype(np.int64)))
    hasher.update(np.ascontiguousarray(coo.col.astype(np.int64)))
    hasher.update(np.ascontiguousarray(coo.data.astype(np.float64)))
    hasher.update(str(matrix.shape).encode())
    return hasher.hexdigest()[:16]


def _real_schur_cached(
    matrix: sparse.csr_matrix, schur_cache: "Path | None" = None
) -> np.ndarray:
    """Real Schur form T of the (densified) matrix: A = Z T Z^T, eigenvalues on T's
    quasi-diagonal. T is seed-independent, so we cache it to disk: the per-seed surrogate
    then only needs a cheap random rotation."""
    dense = (
        matrix.toarray().astype(np.float64)
        if sparse.issparse(matrix)
        else np.asarray(matrix, dtype=np.float64)
    )
    if schur_cache is not None:
        cache_file = schur_cache / f"schurT_{_matrix_fingerprint(matrix)}.npy"
        if cache_file.exists():
            return np.load(cache_file)
    t_mat, _ = scipy_linalg.schur(dense, output="real")
    if schur_cache is not None:
        schur_cache.mkdir(parents=True, exist_ok=True)
        np.save(schur_cache / f"schurT_{_matrix_fingerprint(matrix)}.npy", t_mat)
    return t_mat


def spectrum_matched_control_matrix(
    matrix: sparse.csr_matrix,
    seed: int,
    mode: str = "full",
    k: int = 16,
    rho_target: float = RHO_TARGET,
    max_dense_n: int = 20_000,
    schur_cache: "Path | None" = None,
) -> sparse.csr_matrix:
    """Eigenvalue-spectrum-matched ("dynamically matched") random surrogate.

    Unlike the topology nulls (random / degree_shuffle / weight_shuffle), this control
    matches the connectome's *dynamics* (its eigenvalue spectrum) while randomizing the
    eigenvectors (the specific wiring directions). It answers: how much of the connectome's
    advantage is captured by matching the spectrum alone?

    mode="full":  surrogate = V T V^T with V Haar-random orthogonal and T the connectome's
                  real Schur form -> EXACTLY the same full eigenvalue spectrum, random
                  eigenvectors. Dense.
    mode="topk":  keep the connectome's k largest-|lambda| eigenvalues exactly (the dominant
                  dynamical modes) and fill the remaining bulk with a Ginibre random spectrum
                  scaled to the connectome's bulk radius, then random-rotate. Matches SOME of
                  the eigenvalues.

    Requires a dense Schur (O(N^3)); guarded by max_dense_n. The result is returned as a
    (dense-content) CSR for API compatibility and rescaled to rho_target.
    """
    n = int(matrix.shape[0])
    if n > max_dense_n:
        raise ConnectomePreparationError(
            f"spectrum-matched control needs a dense Schur decomposition; N={n} exceeds "
            f"max_dense_n={max_dense_n}. Use a --max-neurons cap."
        )
    t_mat = _real_schur_cached(matrix, schur_cache=schur_cache)
    rng = np.random.default_rng(seed)
    if mode == "full":
        t_mix = t_mat
    elif mode == "topk":
        t_mix = _topk_mixed_schur(t_mat, k=k, rng=rng)
    else:
        raise ValueError(f"Unknown spectrum mode: {mode!r} (expected 'full' or 'topk').")
    v_mat = _random_orthogonal(n, rng).astype(np.float32)
    surrogate = (v_mat @ t_mix.astype(np.float32)) @ v_mat.T  # float32 matmul (~2x faster)
    # rho is preserved by construction (similarity transform); rescale defensively via a cheap
    # dense power iteration rather than a 54M-nnz sparse eig solve.
    rho = _dense_power_iteration_radius(surrogate, iters=80, seed=seed)
    if rho > 0:
        surrogate = surrogate * np.float32(rho_target / rho)
    return sparse.csr_matrix(surrogate)


def _dense_power_iteration_radius(mat: np.ndarray, iters: int = 80, seed: int = 0) -> float:
    rng = np.random.default_rng(seed + 1)
    x = rng.standard_normal(mat.shape[1]).astype(mat.dtype)
    x /= np.linalg.norm(x) + 1e-12
    last = 0.0
    for _ in range(iters):
        y = mat @ x
        norm = float(np.linalg.norm(y))
        if norm == 0:
            return 0.0
        x = y / norm
        last = norm
    return last


def _topk_mixed_schur(t_mat: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Block-diagonal quasi-triangular form whose leading block holds the connectome's
    k largest-|lambda| eigenvalues (exact) and whose trailing block is a Ginibre random
    bulk scaled to the connectome's (k+1)-th eigenvalue magnitude."""
    n = t_mat.shape[0]
    eigs = scipy_linalg.eigvals(t_mat)  # cheap: T is quasi-triangular
    order = np.argsort(-np.abs(eigs))
    k = int(max(1, min(k, n - 1)))
    threshold = float(np.abs(eigs[order[k - 1]]))  # magnitude of the k-th eigenvalue
    # bulk radius = magnitude of the (k+1)-th eigenvalue, so the random bulk sits strictly
    # below the preserved top-k (keeps the top-k as the clear leading modes).
    bulk_radius = float(np.abs(eigs[order[min(k, n - 1)]]))
    # reorder the real Schur form so the top-|lambda| eigenvalues occupy the leading block
    t_sorted, _, sdim = scipy_linalg.schur(
        t_mat, output="real", sort=lambda ev: np.abs(ev) >= threshold * (1.0 - 1e-9)
    )
    sdim = int(sdim)
    if sdim <= 0 or sdim >= n:
        return t_mat
    t_top = t_sorted[:sdim, :sdim]
    bulk_n = n - sdim
    bulk_radius = max(bulk_radius, 1e-6)
    ginibre = rng.standard_normal((bulk_n, bulk_n)) / np.sqrt(bulk_n)
    g_rho = float(np.max(np.abs(scipy_linalg.eigvals(ginibre)))) or 1.0
    ginibre *= bulk_radius / g_rho  # match bulk radius to the connectome's (k+1)-th |lambda|
    t_mix = np.zeros((n, n), dtype=np.float64)
    t_mix[:sdim, :sdim] = t_top
    t_mix[sdim:, sdim:] = ginibre
    return t_mix


def spectrum_summary(matrix: sparse.csr_matrix, k: int = 32) -> dict[str, object]:
    """Top-k eigenvalue magnitudes + simple spectral descriptors, for control validation."""
    n = int(matrix.shape[0])
    if n <= 2048:
        eigs = scipy_linalg.eigvals(matrix.toarray().astype(np.float64))
    else:
        eigs = sparse_linalg.eigs(
            matrix.astype(np.float64), k=min(k, n - 2), which="LM", return_eigenvectors=False
        )
    mags = np.sort(np.abs(eigs))[::-1]
    return {
        "spectral_radius": float(mags[0]) if mags.size else 0.0,
        "top_abs_eigs": [float(x) for x in mags[:k]],
    }


def control_invariants(
    matrix: sparse.csr_matrix,
    include_weight_multiset: bool = True,
) -> dict[str, object]:
    support = matrix.copy()
    support.data = np.ones_like(support.data)
    invariants = {
        "N": int(matrix.shape[0]),
        "edge_count": int(matrix.nnz),
        "self_loop_count": int(np.sum(matrix.tocoo().row == matrix.tocoo().col)),
        "in_degree": np.asarray(support.sum(axis=1)).ravel().astype(int).tolist(),
        "out_degree": np.asarray(support.sum(axis=0)).ravel().astype(int).tolist(),
        "negative_edge_fraction": float(np.mean(matrix.data < 0)) if matrix.nnz else 0.0,
    }
    if include_weight_multiset:
        invariants["weight_multiset"] = sorted(np.round(matrix.data.astype(float), 8).tolist())
    else:
        data = matrix.data.astype(float)
        invariants["weight_summary"] = {
            "min": float(np.min(data)) if data.size else 0.0,
            "max": float(np.max(data)) if data.size else 0.0,
            "mean": float(np.mean(data)) if data.size else 0.0,
            "sum": float(np.sum(data)) if data.size else 0.0,
        }
    return invariants


def prepare_connectome(
    paths: OutputPaths,
    signed_policy: str = "auto",
    connectome: str = CONNECTOME_HEMIBRAIN_CX,
    whole_brain_pool_fraction: float = 0.05,
) -> PreparedGraph:
    require_raw_exports(paths)
    neurons = pd.read_csv(paths.neurons_csv)
    connections = pd.read_csv(paths.connections_csv)
    unsigned_raw, body_to_index, aggregated_edges = build_raw_adjacency(neurons, connections)
    if connectome == CONNECTOME_FLYWIRE_WHOLE:
        primary_rois = ("whole_brain",)
        pools = write_whole_brain_pool_assignments(
            paths, pool_fraction=whole_brain_pool_fraction
        )
    elif connectome == CONNECTOME_FLYWIRE_MUSHROOM_BODY:
        primary_rois = MB_ROI_LABELS
        pools = write_pool_assignments(paths, primary_rois=primary_rois)
    elif connectome == CONNECTOME_HEMIBRAIN_MUSHROOM_BODY:
        primary_rois = HEMIBRAIN_MB_ROI_LABELS
        pools = write_pool_assignments(paths, primary_rois=primary_rois)
    elif connectome == CONNECTOME_HEMIBRAIN_CX:
        primary_rois = CX_ROI_LABELS
        pools = write_pool_assignments(paths, primary_rois=primary_rois)
    else:
        raise ConnectomePreparationError(f"Unknown connectome: {connectome}")
    indices = pool_indices(pools)
    signs = assign_presynaptic_signs(neurons, body_to_index)
    signed_raw = build_signed_adjacency(unsigned_raw, signs)
    coverage = sign_coverage(unsigned_raw, signs)
    primary_name = choose_primary_matrix(signed_policy, coverage, signed_raw)
    unsigned, signed, raw_rho, scale = scale_to_spectral_radius(
        unsigned_raw, signed_raw, primary_name=primary_name
    )
    primary = signed if primary_name == "signed" else unsigned
    if primary is None:
        raise ConnectomePreparationError("Primary adjacency could not be constructed.")
    if connectome == CONNECTOME_FLYWIRE_WHOLE:
        k = estimate_k_from_support_sampled(primary, indices["sensory"], indices["output"])
    else:
        k = estimate_k_from_support(primary, indices["sensory"], indices["output"])
    body_ids = [int(x) for x in neurons["bodyId"].astype("int64").drop_duplicates().sort_values()]
    include_full_invariants = primary.nnz <= 200_000 and primary.shape[0] <= 20_000
    metadata: dict[str, object] = {
        "connectome": connectome,
        "primary_rois": list(primary_rois),
        "N": int(unsigned.shape[0]),
        "body_ids": body_ids,
        "orientation": "W_rec[post_index, pre_index]",
        "unsigned_edge_count": int(unsigned.nnz),
        "raw_edge_count": int(len(aggregated_edges)),
        "self_loop_count": int(np.sum(unsigned.tocoo().row == unsigned.tocoo().col)),
        "signed_edge_count": int(signed.nnz) if signed is not None else 0,
        "signed_presynaptic_neuron_count": int(len(signs)),
        "sign_coverage": float(coverage),
        "signed_policy": signed_policy,
        "primary_matrix": primary_name,
        "raw_primary_spectral_radius": float(raw_rho),
        "spectral_scale": float(scale),
        "rho_target": float(RHO_TARGET),
        "estimated_K": int(k),
        "pool_counts": {pool: int(len(values)) for pool, values in indices.items()},
        "whole_brain_pool_fraction": float(whole_brain_pool_fraction),
        "control_invariants_primary": control_invariants(
            primary, include_weight_multiset=include_full_invariants
        ),
    }
    sparse.save_npz(paths.adjacency_unsigned_npz, unsigned)
    if signed is not None:
        sparse.save_npz(paths.adjacency_signed_npz, signed)
    with paths.graph_metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return PreparedGraph(
        matrix=primary,
        unsigned=unsigned,
        signed=signed,
        metadata=metadata,
        pools=pools,
    )


def load_prepared_graph(paths: OutputPaths) -> PreparedGraph:
    if not paths.graph_metadata_json.exists() or not paths.adjacency_unsigned_npz.exists():
        raise FileNotFoundError("Prepared graph artifacts are missing. Run --mode prepare first.")
    with paths.graph_metadata_json.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    unsigned = sparse.load_npz(paths.adjacency_unsigned_npz).astype(np.float32).tocsr()
    signed = (
        sparse.load_npz(paths.adjacency_signed_npz).astype(np.float32).tocsr()
        if paths.adjacency_signed_npz.exists()
        else None
    )
    primary_name = str(metadata["primary_matrix"])
    matrix = signed if primary_name == "signed" else unsigned
    if matrix is None:
        raise FileNotFoundError("Metadata selected signed primary, but adjacency_signed.npz is absent.")
    pools = pd.read_csv(paths.pool_assignments_csv)
    return PreparedGraph(matrix=matrix, unsigned=unsigned, signed=signed, metadata=metadata, pools=pools)
