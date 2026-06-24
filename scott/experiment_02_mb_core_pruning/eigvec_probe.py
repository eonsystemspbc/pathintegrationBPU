#!/usr/bin/env python3
"""Numerical feasibility probe for the eigenvector-matched (pairing-shuffled) control.

The control matrix is M = V diag(w') V^-1, where (w, V) are the eigenvalues/eigenvectors of the
rho-matched MB core C, and w' is w with its eigenvector<->eigenvalue PAIRING shuffled (same
eigenbasis V, same spectrum {w}, broken assignment). For M to stay REAL the shuffle must commute
with the conjugation involution S that pairs conjugate eigenvectors (real eigvecs map to
themselves; complex ones pair up). This probe measures:

  * cond(V)  -- if the core is strongly non-normal this can be huge, and then V^-1 (hence M) is
                numerically untrustworthy. THIS is the go/no-go number.
  * reconstruction error  || V diag(w) V^-1 - C || / || C ||   (sanity on the eig itself)
  * for a sample shuffle: the imaginary residual of M (should be ~roundoff if the shuffle
    commutes with S), and whether M's realized spectrum still matches C's.

Run:  uv run python scott/experiment_02_mb_core_pruning/eigvec_probe.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# load the Exp-2 engine as a library (gives mb, rho_of, rescale_to_rho, _induced, build_run_matrix)
_spec = importlib.util.spec_from_file_location("exp2_engine", HERE / "run_experiment.py")
exp2 = importlib.util.module_from_spec(_spec)
sys.modules["exp2_engine"] = exp2
_spec.loader.exec_module(exp2)


def conjugate_involution(w, V, tol=1e-6):
    """Return S: for each eigen-index i, the index of its conjugate partner (i itself if real).
    Matches on eigenvalue conjugacy, then disambiguates degenerate matches by eigenvector."""
    n = len(w)
    S = -np.ones(n, dtype=np.int64)
    taken = np.zeros(n, dtype=bool)
    for i in range(n):
        if S[i] != -1:
            continue
        if abs(w[i].imag) < tol * max(abs(w[i]), 1.0):
            S[i] = i  # real eigenvalue -> fixed point
            taken[i] = True
            continue
        # complex: find j!=i, not taken, with w[j]~conj(w[i]) and V[:,j]~conj(V[:,i])
        target = np.conj(w[i])
        cand = [j for j in range(n) if not taken[j] and j != i
                and abs(w[j] - target) < tol * max(abs(w[i]), 1.0)]
        if not cand:
            S[i] = i  # fallback: treat as (numerically) real
            taken[i] = True
            continue
        j = min(cand, key=lambda j: np.linalg.norm(V[:, j] - np.conj(V[:, i])))
        S[i] = j
        S[j] = i
        taken[i] = taken[j] = True
    return S


def shuffled_pairing_perm(S, rng):
    """A permutation pi of eigen-indices that commutes with S (so M=V diag(w[pi]) V^-1 is real):
    permute real slots among themselves; permute conjugate-PAIR blocks among themselves as units."""
    n = len(S)
    reals = [i for i in range(n) if S[i] == i]
    pairs = sorted({(i, S[i]) for i in range(n) if S[i] != i}, key=lambda t: (min(t), max(t)))
    pairs = [(i, j) for (i, j) in {(min(a, b), max(a, b)) for (a, b) in pairs}]
    pi = np.arange(n)
    # shuffle reals
    rp = rng.permutation(len(reals))
    for src, dst in zip(reals, [reals[k] for k in rp]):
        pi[src] = dst
    # shuffle pair-blocks as units (keep orientation: lo->lo, hi->hi of the destination block)
    pp = rng.permutation(len(pairs))
    for (lo, hi), k in zip(pairs, pp):
        dlo, dhi = pairs[k]
        pi[lo] = dlo
        pi[hi] = dhi
    return pi


def main():
    base = exp2.mb.load_base_matrix(REPO / "connectomes/flywire_mushroom_body/adjacency_unsigned.npz", 0)
    core_idx = np.load(HERE / "substrate" / "core_indices.npy").astype(np.int64)
    target_rho = exp2.rho_of(base)
    C_coo, rho_raw, scale = exp2.build_run_matrix(base, base.tocsr(), core_idx, "core", 0, target_rho)
    C = C_coo.toarray().astype(np.float64)
    N, E = C.shape[0], int((C != 0).sum())
    print(f"core: N={N}  edges(nnz)={E}  target_rho={target_rho:.4f}  realized_rho_raw={rho_raw:.4f}")

    print("eigendecomposing (dense, float64)...", flush=True)
    w, V = np.linalg.eig(C)
    kappa = np.linalg.cond(V)
    recon = V @ np.diag(w) @ np.linalg.inv(V)
    recon_err = np.linalg.norm(recon - C) / np.linalg.norm(C)
    n_real = int(np.sum(np.abs(w.imag) < 1e-6 * np.maximum(np.abs(w), 1.0)))
    print(f"cond(V) = {kappa:.3e}   <-- go/no-go (huge => V^-1 untrustworthy)")
    print(f"reconstruction ||VDV^-1 - C||/||C|| = {recon_err:.3e}")
    print(f"eigenvalues: {n_real} real, {N - n_real} complex  | rho(C)={np.max(np.abs(w)):.4f}")

    # build one sample shuffled-pairing M and check realness + spectrum preservation
    rng = np.random.default_rng(0)
    S = conjugate_involution(w, V)
    pi = shuffled_pairing_perm(S, rng)
    moved = int(np.sum(pi != np.arange(N)))
    Vinv = np.linalg.inv(V)
    M_c = V @ np.diag(w[pi]) @ Vinv
    imag_resid = np.max(np.abs(M_c.imag)) / (np.max(np.abs(M_c.real)) + 1e-12)
    M = M_c.real
    w_M = np.linalg.eigvals(M)
    spectrum_match = np.linalg.norm(np.sort_complex(w_M) - np.sort_complex(w)) / np.linalg.norm(w)
    rho_M = float(np.max(np.abs(w_M)))
    fro_ratio = np.linalg.norm(M) / np.linalg.norm(C)
    print(f"\nsample shuffle: moved {moved}/{N} eigen-slots")
    print(f"  imag residual of M (rel)            = {imag_resid:.3e}   (want ~roundoff)")
    print(f"  spectrum preserved ||sort dW||/||w|| = {spectrum_match:.3e}   (want ~roundoff)")
    print(f"  rho(M)={rho_M:.4f} vs rho(C)={np.max(np.abs(w)):.4f}")
    print(f"  ||M||_F / ||C||_F = {fro_ratio:.3f}   (non-normality => can be >> 1)")

    verdict = "FEASIBLE" if (kappa < 1e8 and imag_resid < 1e-3 and spectrum_match < 1e-3) else "PROBLEMATIC"
    print(f"\nVERDICT: {verdict}  (cond(V)={kappa:.1e}, imag={imag_resid:.1e}, spec={spectrum_match:.1e})")


if __name__ == "__main__":
    main()
