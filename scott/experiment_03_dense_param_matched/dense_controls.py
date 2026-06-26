#!/usr/bin/env python3
"""Dense parameter-matched controls for Experiment 3.

Three dense controls per connectome substrate (core 5.6k / full 14k), all gain-matched by
empirical init activation-RMS to that substrate's connectome (rho is the wrong invariant for
dense non-normal matrices -- the Experiment-2 eigvec lesson: rho and sigma_max decouple ~8x, so
no scalar matches both, but a scalar CAN match the loudness a finite ReLU unroll actually sees):

  C1  dense, same N as the connectome, 100% trainable                 -> size-matched CEILING.
      Far MORE trainable params than the connectome (N^2 vs nnz); NOT param-matched. Answers:
      how much does the connectome give up by being a fixed sparse support vs a fully-free dense
      net of the same neuron count?
  C2  dense FROZEN random scaffold (same N) + E = nnz(connectome) random TRAINABLE delta edges
      -> trainable-param-matched. The random-directions dense reservoir: same neurons, same
      number of trainable knobs as the connectome, but a generic dense substrate instead of the
      connectome's wiring. C2's scaffold IS C1's init matrix, frozen except E entries. This is the
      matched-param topology test, and the complement to Exp 2's eigvec controls (which kept the
      connectome's Schur directions; C2 uses random directions).
  C3  smaller dense network (N' < N), 100% trainable, sized so TOTAL trainable params
      (recurrent + I/O) match the connectome -> param-matched, params concentrated in fewer
      neurons. Answers: is a fixed budget better spent densely-over-few or sparsely-over-many?

The activation-RMS gain match (`activation_rms`, `match_gain_to_activation_rms`), the random
exposed-edge picker (`exposed_edges`), and the dense-scaffold+delta model
(`build_scaffold_delta_model`) are adapted -- math identical -- from Experiment 2's
`eigvec_control.py`, copied here (not imported) so Experiment 3 is a self-contained record and is
not coupled to Exp 2's frozen code. C1/C3 use the shared `MatrixEpisodicRNN` (dense runtime,
recurrent NOT frozen), built by the engine.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

EXPOSED_EDGE_SEED_BASE = 70_000   # mirrors eigvec_control.exposed_edges (comparable random support)
DENSE_INIT_SEED_BASE = 30_000     # C1/C2/C3 random dense init matrices


# --------------------------------------------------------------------------------------
# parameter accounting (must match MatrixEpisodicRNN / DenseScaffoldDeltaRNN exactly)
# --------------------------------------------------------------------------------------
def io_param_count(n: int, input_dim: int, output_dim: int) -> int:
    """Trainable I/O params: W_in (n*input_dim) + b_rec (n) + readout (output_dim*n + output_dim)."""
    return n * input_dim + n + output_dim * n + output_dim


def connectome_total_trainable(n: int, edges: int, input_dim: int, output_dim: int) -> int:
    """The connectome's total trainable params = recurrent nnz + I/O (sparse-trainable model)."""
    return int(edges) + io_param_count(n, input_dim, output_dim)


def c3_neuron_count(target_total: int, input_dim: int, output_dim: int) -> int:
    """Smallest dense N' whose TOTAL trainable params (N'^2 + I/O(N')) best match target_total.

    Dense recurrent = N'^2; total = N'^2 + (input_dim+1+output_dim)*N' + output_dim. Solve the
    quadratic for the real root, then pick the integer N' minimizing |total(N') - target|."""
    a, b, c = 1.0, float(input_dim + 1 + output_dim), float(output_dim - target_total)
    root = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    cand = [int(np.floor(root)), int(np.ceil(root))]
    def total(n):
        return n * n + io_param_count(n, input_dim, output_dim)
    return min(cand, key=lambda n: abs(total(n) - target_total))


def dense_total_trainable(n: int, input_dim: int, output_dim: int) -> int:
    """Total trainable params of a fully-trainable dense net of N neurons (C1/C3)."""
    return n * n + io_param_count(n, input_dim, output_dim)


# --------------------------------------------------------------------------------------
# random dense substrate (shared by C1 and C2; C2 freezes all but E entries of it)
# --------------------------------------------------------------------------------------
def dense_random_matrix(n: int, seed: int) -> np.ndarray:
    """An N x N iid-Gaussian dense matrix (unscaled). The gain scalar from
    match_gain_to_activation_rms then puts it in the connectome's activation regime."""
    rng = np.random.default_rng(DENSE_INIT_SEED_BASE + int(seed))
    return rng.standard_normal((n, n)).astype(np.float64)


# --------------------------------------------------------------------------------------
# gain control by EMPIRICAL INIT ACTIVATION-RMS (copied from eigvec_control.py; works for a dense
# ndarray or a scipy sparse matrix -- both support `@`). The connectome substrate (sparse) defines
# the target; each dense control is rescaled to hit it.
# --------------------------------------------------------------------------------------
def activation_rms(matrix, *, input_dim=35, T=16, batch=64, seed=0):
    """Step-averaged RMS of the hidden state under the MatrixEpisodicRNN init (frozen W_rec, ReLU,
    b=0, W_in ~ U(+/-1/sqrt(input_dim))) driven by representative one-hot(vocab)+role(3) tokens --
    the operational 'gain' a finite ReLU unroll sees. `matrix` may be dense (ndarray) or sparse."""
    n = matrix.shape[0]
    rng = np.random.default_rng(seed)
    scale_in = 1.0 / np.sqrt(input_dim)
    w_in = rng.uniform(-scale_in, scale_in, size=(n, input_dim)).astype(np.float64)
    vocab = input_dim - 3
    X = np.zeros((T, batch, input_dim))
    tok = rng.integers(0, vocab, size=(T, batch))
    rol = rng.integers(0, 3, size=(T, batch))
    ar = np.arange(batch)
    for t in range(T):
        X[t, ar, tok[t]] = 1.0
        X[t, ar, vocab + rol[t]] = 1.0
    h = np.zeros((n, batch))
    rms = []
    for t in range(T):
        h = np.maximum(0.0, np.asarray(matrix @ h) + w_in @ X[t].T)
        rms.append(float(np.sqrt(np.mean(h ** 2))))
    return {"mean_rms": float(np.mean(rms))}


def match_gain_to_activation_rms(matrix, target_rms, *, tol=0.02, iters=44, **probe):
    """Scalar s so activation_rms(s*matrix)['mean_rms'] ~ target_rms (monotone in s -> log-bisection)."""
    lo, hi, s = 1e-3, 1e3, 1.0
    for _ in range(iters):
        s = float(np.sqrt(lo * hi))
        r = activation_rms(matrix * s, **probe)["mean_rms"]
        if abs(r - target_rms) <= tol * target_rms:
            return s
        if r < target_rms:
            lo = s
        else:
            hi = s
    return s


def exposed_edges(n, n_edges, seed):
    """E = n_edges random entries of an N x N dense scaffold to expose as trainable, so the trainable
    recurrent param count == the connectome's nnz. Returns (rows, cols) int64. (Identical to
    eigvec_control.exposed_edges.)"""
    rng = np.random.default_rng(EXPOSED_EDGE_SEED_BASE + int(seed))
    flat = rng.choice(n * n, size=int(n_edges), replace=False)
    return (flat // n).astype(np.int64), (flat % n).astype(np.int64)


# --------------------------------------------------------------------------------------
# C2 model: dense FROZEN scaffold + sparse TRAINABLE delta on E exposed entries. I/O initialized
# identically to MatrixEpisodicRNN so only the recurrent substrate differs. Copied (math identical)
# from eigvec_control.build_model so Exp 3 is self-contained; the scaffold is a non-persistent
# buffer so checkpoints stay small even for the 14k.
# --------------------------------------------------------------------------------------
def build_scaffold_delta_model(scaffold_csr, exposed_rc, input_dim, output_dim, state_clip, seed):
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
