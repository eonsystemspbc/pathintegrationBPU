#!/usr/bin/env python3
"""Validate the two Schur-based eigenvector controls on the real MB core, and stage the Schur cache.

Checks per surrogate: finite, real, dense, spectral radius == 0.95, and (for the shuffle) that the
eigenvalue SET is preserved while the matrix actually changed. Writes the seed-independent Schur
factors to substrate/schur_cache/ so the fleet never recomputes the O(N^3) decomposition.

Run:  uv run python scott/experiment_02_mb_core_pruning/eigvec_build_check.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

_spec = importlib.util.spec_from_file_location("exp2_engine", HERE / "run_experiment.py")
exp2 = importlib.util.module_from_spec(_spec)
sys.modules["exp2_engine"] = exp2
_spec.loader.exec_module(exp2)

sys.path.insert(0, str(HERE))
import eigvec_control as ev  # noqa: E402


def rho_dense(M, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(M.shape[0]).astype(np.float64)
    v /= np.linalg.norm(v)
    for _ in range(300):
        v = M @ v
        n = np.linalg.norm(v)
        if n == 0:
            return 0.0
        v /= n
    return float(np.linalg.norm(M @ v))


def main():
    cache = HERE / "substrate" / "schur_cache"
    base = exp2.mb.load_base_matrix(REPO / "connectomes/flywire_mushroom_body/adjacency_unsigned.npz", 0)
    core_idx = np.load(HERE / "substrate" / "core_indices.npy").astype(np.int64)
    target_rho = exp2.rho_of(base)
    C_coo, _, _ = exp2.build_run_matrix(base, base.tocsr(), core_idx, "core", 0, target_rho)
    core_csr = C_coo.tocsr()
    C = C_coo.toarray().astype(np.float64)
    N, E = C.shape[0], core_csr.nnz
    w_core = np.linalg.eigvals(C)
    print(f"core: N={N} edges={E} target_rho={target_rho:.4f} rho(C)~{np.max(np.abs(w_core)):.4f}")
    print("(building Schur cache on first call; O(N^3), ~1-2 min)\n", flush=True)

    for name, fn, base_seed in [
        ("eigvec_matched", ev.eigvec_matched_matrix, ev.EIGVEC_MATCHED_SEED_BASE),
        ("eigvec_shuffle", ev.eigvec_shuffle_matrix, ev.EIGVEC_SHUFFLE_SEED_BASE),
    ]:
        M_csr = fn(core_csr, seed=0, rho_target=0.95, schur_cache=cache)
        M = M_csr.toarray().astype(np.float64)
        finite = bool(np.all(np.isfinite(M)))
        dens = float((M != 0).mean())
        wM = np.linalg.eigvals(M)
        rho_pi = rho_dense(M)
        rho_eig = float(np.max(np.abs(wM)))
        # spectrum preservation (shuffle should keep the SET; matched should not)
        spec_diff = np.linalg.norm(np.sort_complex(wM) - np.sort_complex(w_core)) / np.linalg.norm(w_core)
        changed = np.linalg.norm(M - C) / np.linalg.norm(C)
        print(f"[{name}] seed_base={base_seed}")
        print(f"  finite={finite}  density={dens:.3f}  ||M-C||/||C||={changed:.3f}")
        print(f"  rho: max|eig(M)|={rho_eig:.4f}  power-iter={rho_pi:.4f}  (target 0.95)")
        print(f"  spectrum vs core: ||sort dW||/||w|| = {spec_diff:.3e}  "
              f"({'PRESERVED' if spec_diff < 1e-3 else 'changed'})")
        # second seed -> different matrix (independent draw)
        M2 = fn(core_csr, seed=1, rho_target=0.95, schur_cache=cache).toarray().astype(np.float64)
        seed_diff = np.linalg.norm(M2 - M) / (np.linalg.norm(M) + 1e-12)
        print(f"  seed0 vs seed1: ||dM||/||M|| = {seed_diff:.3f}  (independent draws)\n")

    print(f"Schur cache staged in {cache} (files: {[p.name for p in sorted(cache.glob('*.npy'))]})")


if __name__ == "__main__":
    main()
