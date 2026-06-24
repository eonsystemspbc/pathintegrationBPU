#!/usr/bin/env python3
"""Two checks before wiring eigvec controls into Exp 2:
(1) does eigvec_shuffle EXACTLY preserve the core's spectrum? (compare Schur diagonal BLOCKS, which
    are exact, not eig(M) which is unreliable on this near-defective matrix).
(2) how do the three 'gain' measures compare across core / matched / shuffle -- power-iteration rho
    (what every other Exp-2 condition is matched on), true spectral radius (max|block eig|), and
    spectral norm sigma_max? Decides which rho to match for a fair within-experiment comparison.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import svds

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("exp2_engine", HERE / "run_experiment.py")
exp2 = importlib.util.module_from_spec(_spec); sys.modules["exp2_engine"] = exp2
_spec.loader.exec_module(exp2)
sys.path.insert(0, str(HERE))
import eigvec_control as ev  # noqa: E402
from src.connectome import _real_schur_cached  # noqa: E402


def block_eigs(t_mat):
    n, out, i = t_mat.shape[0], [], 0
    while i < n:
        if i + 1 < n and abs(t_mat[i + 1, i]) > 1e-12:
            out.extend(np.linalg.eigvals(t_mat[i:i + 2, i:i + 2]).tolist()); i += 2
        else:
            out.append(complex(t_mat[i, i])); i += 1
    return np.array(out)


def main():
    cache = HERE / "substrate" / "schur_cache"
    base = exp2.mb.load_base_matrix(REPO / "connectomes/flywire_mushroom_body/adjacency_unsigned.npz", 0)
    core_idx = np.load(HERE / "substrate" / "core_indices.npy").astype(np.int64)
    target_rho = exp2.rho_of(base)
    C_coo, _, _ = exp2.build_run_matrix(base, base.tocsr(), core_idx, "core", 0, target_rho)
    core_csr = C_coo.tocsr()
    t_core, _ = _real_schur_cached(core_csr, schur_cache=cache, want_z=True)
    core_blocks = np.sort_complex(block_eigs(t_core))

    def sigma_max(M):
        return float(svds(M, k=1, return_singular_vectors=False)[0])

    print(f"{'matrix':16s} {'power_iter_rho':>14s} {'max|block_eig|':>14s} {'sigma_max':>10s}")
    print(f"{'core (conn)':16s} {exp2.rho_of(core_csr):14.4f} {np.max(np.abs(core_blocks)):14.4f} "
          f"{sigma_max(core_csr):10.3f}")

    # eigvec_shuffle: rebuild T_perm and compare its block spectrum to the core's (EXACT check)
    M_shuf = ev.eigvec_shuffle_matrix(core_csr, seed=0, rho_target=0.95, schur_cache=cache)
    # reconstruct T_perm's blocks: shuffle keeps the set, so block multiset must equal core's
    # (recompute via the generator's own logic by reading back Z^T M Z)
    _, z = _real_schur_cached(core_csr, schur_cache=cache, want_z=True)
    t_perm = z.T @ M_shuf.toarray().astype(np.float64) @ z
    shuf_blocks = np.sort_complex(block_eigs(t_perm))
    # scale-invariant set comparison (both rescaled to rho=0.95, so compare normalized)
    set_err = np.linalg.norm(shuf_blocks / np.max(np.abs(shuf_blocks))
                             - core_blocks / np.max(np.abs(core_blocks))) / np.linalg.norm(core_blocks / np.max(np.abs(core_blocks)))
    print(f"{'eigvec_shuffle':16s} {exp2.rho_of(M_shuf):14.4f} {np.max(np.abs(shuf_blocks)):14.4f} "
          f"{sigma_max(M_shuf):10.3f}")
    print(f"   -> shuffle block-spectrum vs core (normalized set err): {set_err:.3e} "
          f"({'EXACT match' if set_err < 1e-6 else 'DIFFERS'})")

    M_match = ev.eigvec_matched_matrix(core_csr, seed=0, rho_target=0.95, schur_cache=cache)
    t_m = z.T @ M_match.toarray().astype(np.float64) @ z
    print(f"{'eigvec_matched':16s} {exp2.rho_of(M_match):14.4f} {np.max(np.abs(block_eigs(t_m))):14.4f} "
          f"{sigma_max(M_match):10.3f}")
    print("\nIf power_iter_rho differs across rows, matching block-rho (CX) != matching power-iter-rho")
    print("(what core & all Exp-2 controls use). For a fair within-experiment gain control, re-rescale")
    print("the dense surrogates to power_iter_rho = 0.95 via rho_of, like every other condition.")


if __name__ == "__main__":
    main()
