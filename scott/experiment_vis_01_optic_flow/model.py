#!/usr/bin/env python3
"""Experiment vis-01 -- the sparse-trainable RNN with GENERIC all-neuron I/O for the optic-flow task.

Self-contained (copy-adapted from the Exp-1/5/6 ``MatrixEpisodicRNN`` in
``scripts/associative/run_omniglot_associative_benchmark.py``) so the vision branch's frozen record
does not depend on the MB engine's model class, and so it can add the two things the flow task needs
that the MB model lacks:
  1. a REGRESSION readout (linear, ``output_dim`` = the scored channel count, 7) with a per-timestep
     MSE objective, and
  2. ``microsteps`` -- >=1 recurrence sub-iterations per input frame, so the network has temporal
     DEPTH within a frame (motion is a temporal computation; a single matmul per frame is shallow), and
  3. a memory-safe sparse recurrence: ``_SparseEdgeMatmul`` (custom ``autograd.Function``) whose backward
     computes the edge-VALUE gradient ONLY on the existing edges (gather pre/post states at endpoints),
     never a dense N×N gradient. ``torch.sparse.mm``'s native backward w.r.t. the 4.2M trainable values
     materializes a dense N×N (~8.9 GB at N=48,894) and OOMs at batch 4 on 16 GB; this trains the real
     substrate at ~0.5 GB/batch-of-4. Numerically identical forward + gradient.

Everything else is byte-for-byte the MB engine's construction so connectome-vs-control numbers stay
comparable to the MB experiments:
  * GENERIC all-neuron I/O: dense trainable ``W_in`` (input_dim -> all N), dense trainable readout
    (all N -> 5). NOT biological ports (a later vision experiment).
  * sparse TRAINABLE recurrence on the FIXED connectome support (edge VALUES trainable,
    ``freeze_recurrent=False``); the support (which edges exist) is the connectome and never changes.
  * the recurrence operator passed in is already rescaled to rho=0.95 (done in common.py); the model
    never rescales it.

ACTIVATION CHOICE (deliberate -- for the reviewer). Default = ReLU, matching the entire MB engine
family and, critically, the activation-RMS-match derivation reused from Exp-6 (``_preact_rms`` there
measures the pre-ReLU activation; using a different nonlinearity here would invalidate that matched
control). Signed 5-DOF velocities are NOT a problem for non-negative ReLU states because the readout
is a signed linear map (``nn.Linear``) -- it freely produces negative outputs from non-negative
hidden activity, exactly as a rate-coded population with opponent readout weights would. ``tanh`` is
available via the ``activation`` arg for a robustness subrun, but then the RMS-match probe in
common.py must switch nonlinearity too (guarded there).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import scipy.sparse as sp

_ACTS = {"relu": torch.relu, "tanh": torch.tanh}


class _SparseEdgeMatmul(torch.autograd.Function):
    """rec = W @ h^T  (returned [B, N]) for a sparse W with FIXED support (edge_index) and TRAINABLE
    edge VALUES, with a memory-safe backward.

    The whole point (MUST-FIX 1): the backward of ``torch.sparse.mm`` w.r.t. the sparse VALUES
    materializes a DENSE N×N gradient (~8.9 GB at N=48,894) and OOMs. Here the value-gradient is
    computed ONLY on the existing edges by gathering the pre/post states at each edge's endpoints:

        rec[b, i] = Σ_{e: row_e=i} value_e · h[b, col_e]                        (post ← pre)
        dL/dvalue_e = Σ_b  gradrec[b, row_e] · h[b, col_e]                       (edge-local; [E])
        dL/dh[b, j] = Σ_{e: col_e=j} value_e · gradrec[b, row_e]                 (sparse W^T @ gradrec)

    No N×N tensor is ever formed; the only [B, E] temporaries live inside backward and are freed
    immediately. Forward reuses cuSPARSE (``torch.sparse.mm``) for speed -- only its dense-materializing
    backward is replaced. Numerically identical forward + gradient to the naive version.

    Orientation: edge_index is [2, E] = (row=post, col=pre) so W[row, col] and rec = W @ h^T flows
    pre→post, matching the substrate's post×pre storage.
    """

    @staticmethod
    def forward(ctx, values, edge_index, h, N):
        W = torch.sparse_coo_tensor(edge_index, values, size=(N, N))
        rec = torch.sparse.mm(W, h.t()).t().contiguous()          # [B, N]
        ctx.save_for_backward(values, edge_index, h)
        ctx.N = N
        return rec

    @staticmethod
    def backward(ctx, grad_rec):
        values, edge_index, h = ctx.saved_tensors
        row, col = edge_index[0], edge_index[1]
        grad_rec = grad_rec.contiguous()
        gr = grad_rec.index_select(1, row)                        # [B, E]  gradrec gathered at post
        grad_values = grad_h = None
        if ctx.needs_input_grad[0]:
            hc = h.index_select(1, col)                           # [B, E]  states gathered at pre
            grad_values = (gr * hc).sum(dim=0)                    # [E]     edge-local value gradient
            del hc
        if ctx.needs_input_grad[2]:
            contrib = values.unsqueeze(0) * gr                    # [B, E]
            grad_h = torch.zeros_like(h)
            grad_h.index_add_(1, col, contrib)                    # sparse W^T @ gradrec (pre ← post)
            del contrib
        return grad_values, None, grad_h, None


class FlowRNN(nn.Module):
    def __init__(self, recurrent: sp.spmatrix, input_dim: int, output_dim: int = 5,
                 seed: int = 0, state_clip: float = 0.0, microsteps: int = 1,
                 activation: str = "relu", freeze_recurrent: bool = False,
                 normalize: bool = True, norm_gain: float = 1.0, norm_learnable: bool = True,
                 norm_eps: float = 1e-5, w_in_gain: float = 1.0) -> None:
        super().__init__()
        if activation not in _ACTS:
            raise ValueError(f"activation must be one of {tuple(_ACTS)}")
        recurrent = recurrent.astype(np.float32).tocoo()
        recurrent.sum_duplicates()
        if recurrent.shape[0] != recurrent.shape[1]:
            raise ValueError("recurrent matrix must be square.")
        self.N = int(recurrent.shape[0])
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.state_clip = float(state_clip)
        self.microsteps = int(max(1, microsteps))
        self.act_name = activation
        self.act = _ACTS[activation]

        # --- ACTIVITY NORMALIZATION (the optic lobe's biological gain control) ----------------------
        # A divisive gain-control / RMS-norm applied to the recurrent hidden state at EVERY microstep
        # (see forward): h <- h / (rms(h) + eps) * g, with rms(h) = sqrt(mean(h**2)) over the neuron
        # dimension and g a small scalar gain. Real optic-lobe neurons do exactly this (gain control /
        # brightness adaptation). It keeps activity bounded regardless of the operator's sigma_max, so
        # the connectome and the degree-matched control run in a COMPARABLE dynamic regime -- which is
        # what lets the connectome-vs-control comparison isolate the wiring SHAPE rather than which
        # operator's activity happens to blow up. Applied IDENTICALLY to both arms; the gain g is a
        # single shared scalar (learnable by default). Toggle with normalize=False.
        self.normalize = bool(normalize)
        self.norm_eps = float(norm_eps)
        g0 = torch.tensor(float(norm_gain))
        if norm_learnable:
            self.norm_gain = nn.Parameter(g0)
        else:
            self.register_buffer("norm_gain", g0)

        gen = torch.Generator(device="cpu"); gen.manual_seed(int(seed))
        # w_in_gain scales the INPUT-pathway init (default 1.0 = unchanged). A larger gain makes the
        # movie re-perturb the recurrent state harder each frame -- the "stronger W_in" anti-fixed-point
        # lever (dyn-01 found the state collapses because the recurrence out-contracts a weak input drive).
        scale_in = float(w_in_gain) / math.sqrt(max(input_dim, 1))
        scale_out = 1.0 / math.sqrt(max(self.N, 1))
        self.W_in = nn.Parameter(torch.empty(self.N, input_dim).uniform_(-scale_in, scale_in, generator=gen))
        self.b_rec = nn.Parameter(torch.zeros(self.N))
        self.readout = nn.Linear(self.N, self.output_dim)
        nn.init.uniform_(self.readout.weight, -scale_out, scale_out)
        nn.init.zeros_(self.readout.bias)

        indices = np.vstack([recurrent.row, recurrent.col]).astype(np.int64)
        self.register_buffer("edge_indices", torch.from_numpy(indices))
        values = recurrent.data.astype(np.float32)
        self.W_rec_values = nn.Parameter(torch.from_numpy(values))
        self.register_buffer("W_rec_initial_values", torch.from_numpy(values.copy()))
        if freeze_recurrent:
            self.W_rec_values.requires_grad_(False)

    def recurrent_parameter_count(self) -> int:
        return int(self.W_rec_values.numel())

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs [B, T, input_dim] -> per-timestep 5-DOF regression outputs [B, T, 5]. The recurrence
        is applied ``microsteps`` times per frame (input drive re-injected each microstep) to give the
        network temporal depth within a frame; the readout is taken once, after the last microstep."""
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"inputs must be [batch, T, {self.input_dim}], got {tuple(inputs.shape)}")
        B, T, _ = inputs.shape
        h = inputs.new_zeros((B, self.N))
        outs = []
        for t in range(T):
            drive = inputs[:, t, :] @ self.W_in.t() + self.b_rec       # frame drive (constant over microsteps)
            for _ in range(self.microsteps):
                rec = _SparseEdgeMatmul.apply(self.W_rec_values, self.edge_indices, h, self.N)
                h = self.act(rec + drive)
                if self.state_clip > 0:                                # symmetric state clip (bounds
                    h = torch.clamp(h, min=-self.state_clip, max=self.state_clip)  # non-normal transients
                if self.normalize:                                     # biological gain control (see __init__):
                    rms = h.pow(2).mean(dim=-1, keepdim=True).sqrt()   # RMS over the neuron dimension
                    # Detach the denominator: the state is still renormalized to a fixed overall magnitude
                    # each microstep (forward unchanged), but the unstable d/dh(1/rms) term is not propagated
                    # -- that 1/rms backward blows up on sparse ReLU states (small rms) and diverged training.
                    h = h / (rms + self.norm_eps).detach() * self.norm_gain
            outs.append(self.readout(h))
        return torch.stack(outs, dim=1)                                # [B, T, output_dim]
