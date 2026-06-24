#!/usr/bin/env python3
"""Eigenvector-matched controls for the MB core, built on the connectome's REAL SCHUR basis.

Why Schur and not eigenvectors: the MB core is strongly non-normal -- its raw eigenvector matrix V
is numerically degenerate (cond(V) ~ 1e178; V diag(w') V^-1 overflows to inf). The CX
`cx_eigval_vs_eigvec` result hit the same wall and solved it with the real Schur factorization
A = Z T Z^T, where Z is ORTHOGONAL (the numerically stable stand-in for "directions") and T is
quasi-upper-triangular (eigenvalues in its diagonal blocks; strictly-upper part = non-normal
coupling). We reuse that machinery (src/connectome.py) here.

Two surrogates, both dense N x N, both rescaled to rho_target, both sharing the connectome's Z and
its strictly-upper coupling:

  eigvec_matched  -- Z T_rand Z^T : keep Z + coupling, RANDOMIZE the eigenvalues (diagonal blocks).
                     Unlike the CX generator, we rescale ONLY the eigenvalue blocks to rho_target and
                     leave the coupling at the connectome's scale (the CX version rescales the whole T
                     and silently inflates the coupling ~15x -> sigma_max 7.9; see eigvec_matched_matrix).
  eigvec_shuffle  -- Z T_perm Z^T : keep Z + coupling AND the EXACT spectrum; only REORDER which
                     eigenvalue block sits where (break the eigenvalue<->subspace pairing). A
                     tighter null -- same modes, same spectrum, scrambled assignment.

GAIN: rho-matching does NOT control gain for these non-normal dense matrices (rho and sigma_max are
decoupled ~8x). Both controls are gain-matched to the connectome's empirical init activation-RMS
(match_gain_to_activation_rms) -- the loudness a finite ReLU unroll actually sees.

TRAINING SURFACE: each is a dense frozen scaffold with E = nnz(connectome) random entries exposed as
a trainable sparse delta (DenseScaffoldDeltaRNN), so the trainable recurrent parameter count equals
the connectome's exactly. Built per substrate (5.6k core and 14k full).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.connectome import _real_schur_cached  # noqa: E402

EIGVEC_MATCHED_SEED_BASE = 50_000   # mirrors the CX seed offset for this family
EIGVEC_SHUFFLE_SEED_BASE = 55_000


def eigvec_matched_matrix(core_csr, seed, rho_target=0.95, schur_cache=None):
    """Z T_rand Z^T: keep the core's Schur basis + strictly-upper coupling AT THE CORE'S SCALE,
    randomize the eigenvalues. Returns dense CSR.

    IMPORTANT FIX vs the shared CX generator (src.connectome.eigenvector_matched_control_matrix):
    that generator draws random eigenvalues at scale=std(diag T) (tiny -- most eigenvalues are
    small) and then rescales the WHOLE T (coupling included) to rho_target, which silently
    multiplies the coupling up by a large factor (~15x on the MB core) -> sigma_max ~ 7.9 vs the
    core's 1.09. That 8x transient-gain inflation is a normalization artifact, not wiring, and it
    confounds the eigenvalue-vs-eigenvector comparison. Here we rescale ONLY the diagonal blocks
    (the eigenvalues) to radius rho_target and leave the coupling at the core's original scale, so
    the coupling-to-eigenvalue ratio matches the connectome. Final loudness is then equalized
    across conditions by match_gain_to_activation_rms (the RMS-based gain control)."""
    t_mat, z_mat = _real_schur_cached(core_csr, schur_cache=schur_cache, want_z=True)
    t_mat = t_mat.astype(np.float64)
    rng = np.random.default_rng(EIGVEC_MATCHED_SEED_BASE + int(seed))
    t_new = np.triu(t_mat).copy()  # core coupling, UNSCALED (the fix)

    drawn, mags = [], []
    for (s, sz) in _block_structure(t_mat):
        if sz == 1:
            v = rng.normal(0.0, 1.0)
            drawn.append((s, 1, (v,))); mags.append(abs(v))
        else:  # 2x2 real-Schur block [[p, q],[-r, p]] -> eigenvalues p +/- i*sqrt(qr)
            p, q, r = rng.normal(0.0, 1.0), rng.uniform(0.2, 1.0), rng.uniform(0.2, 1.0)
            drawn.append((s, 2, (p, q, -r, p))); mags.append(float(np.hypot(p, np.sqrt(q * r))))
    f = float(rho_target) / max(mags)  # scale ONLY the eigenvalues to radius rho_target
    for (s, sz, cells) in drawn:
        if sz == 1:
            t_new[s, s] = f * cells[0]
        else:
            t_new[s, s], t_new[s, s + 1] = f * cells[0], f * cells[1]
            t_new[s + 1, s], t_new[s + 1, s + 1] = f * cells[2], f * cells[3]
    z32 = z_mat.astype(np.float32)
    return sparse.csr_matrix((z32 @ t_new.astype(np.float32)) @ z32.T)


def _block_structure(t_mat, tol=1e-12):
    """Diagonal-block layout of a real quasi-triangular T: list of (start, size)."""
    n = t_mat.shape[0]
    blocks, i = [], 0
    while i < n:
        if i + 1 < n and abs(t_mat[i + 1, i]) > tol:
            blocks.append((i, 2))
            i += 2
        else:
            blocks.append((i, 1))
            i += 1
    return blocks


def _block_max_mag(t_mat, blocks):
    """Spectral radius from the diagonal blocks (exact for quasi-triangular T; power iteration
    overestimates it for the non-normal Z T Z^T)."""
    mags = []
    for (s, sz) in blocks:
        if sz == 1:
            mags.append(abs(float(t_mat[s, s])))
        else:
            mags.extend(np.abs(np.linalg.eigvals(t_mat[s:s + 2, s:s + 2])).tolist())
    return max(mags) if mags else 1e-12


def eigvec_shuffle_matrix(core_csr, seed, rho_target=0.95, schur_cache=None):
    """Z T_perm Z^T: keep Z + the strictly-upper coupling AND the exact spectrum; only REORDER the
    diagonal blocks (1x1 among 1x1 slots, 2x2 among 2x2 slots, moved verbatim so each block's
    eigenvalues are preserved). Same eigenvalue SET, broken eigenvalue<->subspace pairing. Real and
    numerically stable (Z orthogonal). Returns a (dense-content) CSR, rescaled to rho_target."""
    t_mat, z_mat = _real_schur_cached(core_csr, schur_cache=schur_cache, want_z=True)
    t_mat = t_mat.astype(np.float64)
    rng = np.random.default_rng(EIGVEC_SHUFFLE_SEED_BASE + int(seed))

    blocks = _block_structure(t_mat)
    ones = [b for b in blocks if b[1] == 1]
    twos = [b for b in blocks if b[1] == 2]

    # start from the strictly-upper coupling (triu zeros the 2x2 subdiagonal; block cells rewritten)
    t_new = np.triu(t_mat).copy()
    # permute 1x1 eigenvalues among 1x1 slots
    for src, dst in zip(ones, [ones[k] for k in rng.permutation(len(ones))]):
        t_new[dst[0], dst[0]] = t_mat[src[0], src[0]]
    # permute 2x2 blocks among 2x2 slots, moving all four cells verbatim (preserves the pair's eigs)
    for src, dst in zip(twos, [twos[k] for k in rng.permutation(len(twos))]):
        si, di = src[0], dst[0]
        t_new[di, di] = t_mat[si, si]
        t_new[di, di + 1] = t_mat[si, si + 1]
        t_new[di + 1, di] = t_mat[si + 1, si]
        t_new[di + 1, di + 1] = t_mat[si + 1, si + 1]

    t_new *= float(rho_target) / _block_max_mag(t_mat, blocks)  # exact rho via the (unchanged) spectrum
    z32 = z_mat.astype(np.float32)
    surrogate = (z32 @ t_new.astype(np.float32)) @ z32.T
    return sparse.csr_matrix(surrogate)


# --------------------------------------------------------------------------------------
# gain control by EMPIRICAL INIT ACTIVATION-RMS (not rho).
# For a finite ~16-step ReLU unroll, per-step amplification tracks the spectral norm / pseudo-
# spectrum, not the asymptotic spectral radius. On these non-normal matrices rho and sigma_max are
# decoupled ~8x, so NO scalar matches both -- but a scalar CAN match what the dynamics actually
# depend on: the loudness of the hidden state when real inputs are driven through the frozen
# recurrence at init. We measure the core's step-averaged activation RMS and rescale each control's
# W_rec to hit it. (The core itself is left at rho=0.95 so it stays consistent with the other
# rho-matched Exp-2 conditions; only the dense eigvec controls are rescaled, to the core's regime.)
# --------------------------------------------------------------------------------------
def activation_rms(matrix_csr, *, input_dim=35, T=16, batch=64, seed=0):
    """Step-averaged RMS of the hidden state under the MatrixEpisodicRNN init (frozen W_rec, ReLU,
    b=0, W_in ~ U(+/-1/sqrt(input_dim))) driven by representative one-hot(vocab)+role(3) tokens --
    the operational 'gain' a finite ReLU unroll sees. Also returns step-15 h_max and dead fraction."""
    N = matrix_csr.shape[0]
    rng = np.random.default_rng(seed)
    scale_in = 1.0 / np.sqrt(input_dim)
    w_in = rng.uniform(-scale_in, scale_in, size=(N, input_dim)).astype(np.float64)
    vocab = input_dim - 3
    X = np.zeros((T, batch, input_dim))
    tok = rng.integers(0, vocab, size=(T, batch))
    rol = rng.integers(0, 3, size=(T, batch))
    ar = np.arange(batch)
    for t in range(T):
        X[t, ar, tok[t]] = 1.0
        X[t, ar, vocab + rol[t]] = 1.0
    h = np.zeros((N, batch))
    rms, last_max, last_dead = [], 0.0, 0.0
    for t in range(T):
        h = np.maximum(0.0, matrix_csr @ h + w_in @ X[t].T)
        rms.append(float(np.sqrt(np.mean(h ** 2))))
        last_max, last_dead = float(h.max()), float((h <= 0).mean())
    return {"mean_rms": float(np.mean(rms)), "hmax": last_max, "dead_frac": last_dead}


def match_gain_to_activation_rms(matrix_csr, target_rms, *, tol=0.02, iters=44, **probe):
    """Scalar s so that activation_rms(s * matrix)['mean_rms'] ~ target_rms (monotone in s ->
    log-bisection). Returns s; multiply the matrix by it to put it in the target activation regime."""
    lo, hi = 1e-3, 1e3
    s = 1.0
    for _ in range(iters):
        s = float(np.sqrt(lo * hi))
        r = activation_rms(matrix_csr * s, **probe)["mean_rms"]
        if abs(r - target_rms) <= tol * target_rms:
            return s
        if r < target_rms:
            lo = s
        else:
            hi = s
    return s


def exposed_edges(n, n_edges, seed):
    """E = n_edges random off-... entries of an N x N dense scaffold to expose as trainable, so the
    trainable recurrent param count == the connectome's nnz. Returns (rows, cols) int64 arrays."""
    rng = np.random.default_rng(70_000 + int(seed))
    flat = rng.choice(n * n, size=int(n_edges), replace=False)
    return (flat // n).astype(np.int64), (flat % n).astype(np.int64)


# --------------------------------------------------------------------------------------
# model: dense FROZEN scaffold + sparse TRAINABLE delta on E exposed entries. I/O is initialized
# identically to MatrixEpisodicRNN so only the recurrent substrate differs. The scaffold is a
# non-persistent buffer (rebuilt from the staged Schur cache + seed on resume) so checkpoints stay
# small even for the 14k (788 MB dense). Lives here, not in the shared model file, to leave Exp 1
# untouched; the engine passes an instance into the (otherwise verbatim) train_one_run.
# --------------------------------------------------------------------------------------
def build_model(scaffold_csr, exposed_rc, input_dim, output_dim, state_clip, seed):
    import math

    import torch
    from torch import nn

    class DenseScaffoldDeltaRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.N = int(scaffold_csr.shape[0])
            self.input_dim, self.output_dim = int(input_dim), int(output_dim)
            self.state_clip = float(state_clip)
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            scale_in, scale_out = 1.0 / math.sqrt(max(input_dim, 1)), 1.0 / math.sqrt(max(self.N, 1))
            self.W_in = nn.Parameter(torch.empty(self.N, input_dim).uniform_(-scale_in, scale_in, generator=gen))
            self.b_rec = nn.Parameter(torch.zeros(self.N))
            self.readout = nn.Linear(self.N, output_dim)
            nn.init.uniform_(self.readout.weight, -scale_out, scale_out)
            nn.init.zeros_(self.readout.bias)
            dense = scaffold_csr.toarray().astype(np.float32)
            self.register_buffer("M", torch.from_numpy(dense), persistent=False)  # frozen, not checkpointed
            idx = np.vstack([exposed_rc[0], exposed_rc[1]]).astype(np.int64)
            self.register_buffer("delta_idx", torch.from_numpy(idx))
            self.delta_val = nn.Parameter(torch.zeros(idx.shape[1], dtype=torch.float32))  # init 0 -> dynamics == M

        def recurrent_parameter_count(self):
            return int(self.delta_val.numel())

        def trainable_parameter_count(self):
            return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

        def forward(self, inputs):
            batch, T, _ = inputs.shape
            h = inputs.new_zeros((batch, self.N))
            D = torch.sparse_coo_tensor(self.delta_idx, self.delta_val, size=(self.N, self.N),
                                        device=inputs.device).coalesce()
            outs = []
            for t in range(T):
                rec = h @ self.M.t() + torch.sparse.mm(D, h.t()).t()
                h = torch.relu(rec + inputs[:, t, :] @ self.W_in.t() + self.b_rec)
                if self.state_clip > 0:
                    h = torch.clamp(h, max=self.state_clip)
                outs.append(self.readout(h))
            return torch.stack(outs, dim=1)

    return DenseScaffoldDeltaRNN()
