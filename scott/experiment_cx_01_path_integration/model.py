#!/usr/bin/env python3
"""Experiment cx-01 -- the sparse-trainable RNN with GENERIC all-neuron I/O for the CX path-integration
task.

Self-contained (copy-adapted from vis-01's ``FlowRNN``, itself copy-adapted from the Exp-1/5/6
``MatrixEpisodicRNN``) so this branch's frozen record does not depend on another experiment's model
class. It is the SAME construction as vis-01's FlowRNN -- only ``output_dim`` differs by default
(35 = 32 bump bins + 3 home-vector channels rather than 7 flow DOF). Keeping the class identical is
deliberate: it makes the cx-01 vs vis-01 contrast (does the connectome help on a TRACKING task in the
region whose computation IS its topology?) a comparison of substrate + task, not of model code.

WHY NOT THE MB ENGINE'S ``MatrixEpisodicRNN``: it is a CLASSIFIER (categorical readout + masked
cross-entropy at query steps). ``cx_polar_bump`` is per-timestep 35-D REGRESSION, so it needs a
regression readout at every step -- hence the vis-01 lineage.

WHY NOT THE PRIOR CX ``CXBPU`` (src/models.py): that model's default is a FROZEN reservoir (only I/O
trains; ``train_recurrent`` in {frozen, observed, dense}), and its frozen mode is what produced the
prior CX results. This experiment is specifically the TRAINABLE-EDGES, generic-I/O regime that the MB
experiments used -- i.e. the ``observed`` analogue -- so the connectome's edge VALUES are retuned by
gradient descent on a FIXED connectome support.

  * GENERIC all-neuron I/O: dense trainable ``W_in`` (2 -> all N), dense trainable readout (all N -> 35).
    NOT biological ports (PFN/PEN input, PFL/PFR output) -- that is a later experiment (cx-02); the
    pools are already built and shipped in substrate/celltype_pools.npz.
  * sparse TRAINABLE recurrence on the FIXED connectome support (edge VALUES trainable); the support
    never changes.
  * the operator passed in is already rescaled to rho=0.95 (common.py); the model never rescales it.
  * ``microsteps`` defaults to 3 -- the prior CX work's estimated K for this substrate (its
    graph_metadata recorded ``estimated_K: 3``), i.e. recurrence sub-iterations per input step, giving
    the network temporal depth within a step. (vis-01 used 2 for flow; 3 is the CX-native value.)

``_SparseEdgeMatmul`` (the memory-safe sparse backward) is carried over unchanged: ``torch.sparse.mm``'s
native backward w.r.t. the trainable edge values materializes a dense N x N gradient. At N = 6,195
that is ~154 MB and would not OOM, but the edge-local backward is strictly cheaper and keeps this
model numerically identical to vis-01's, so the two branches' results stay comparable.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import scipy.sparse as sp

_ACTS = {"relu": torch.relu, "tanh": torch.tanh}


class _SparseEdgeMatmul(torch.autograd.Function):
    """rec = W @ h^T (returned [B, N]) for sparse W with FIXED support and TRAINABLE edge VALUES.

        rec[b, i]   = sum_{e: row_e=i} value_e * h[b, col_e]              (post <- pre)
        dL/dvalue_e = sum_b gradrec[b, row_e] * h[b, col_e]               (edge-local; [E])
        dL/dh[b, j] = sum_{e: col_e=j} value_e * gradrec[b, row_e]        (sparse W^T @ gradrec)

    No N x N tensor is ever formed. Forward reuses cuSPARSE; only the dense-materializing backward is
    replaced. Numerically identical forward + gradient.

    Orientation: edge_index is [2, E] = (row=post, col=pre) so rec = W @ h^T flows pre->post, matching
    the substrate's post x pre storage.
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
        gr = grad_rec.index_select(1, row)                        # [B, E] gradrec gathered at post
        grad_values = grad_h = None
        if ctx.needs_input_grad[0]:
            hc = h.index_select(1, col)                           # [B, E] states gathered at pre
            grad_values = (gr * hc).sum(dim=0)                    # [E]    edge-local value gradient
            del hc
        if ctx.needs_input_grad[2]:
            contrib = values.unsqueeze(0) * gr                    # [B, E]
            grad_h = torch.zeros_like(h)
            grad_h.index_add_(1, col, contrib)                    # sparse W^T @ gradrec (pre <- post)
            del contrib
        return grad_values, None, grad_h, None


class CXRNN(nn.Module):
    def __init__(self, recurrent: sp.spmatrix, input_dim: int = 2, output_dim: int = 35,
                 seed: int = 0, state_clip: float = 0.0, microsteps: int = 3,
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

        # --- ACTIVITY NORMALIZATION (divisive gain control), identical to vis-01's FlowRNN ---------
        # h <- h / (rms(h) + eps) * g at every microstep. It bounds activity regardless of the
        # operator's sigma_max, so connectome and control run in a comparable dynamic regime.
        # NOTE (the vis-01 lesson): with normalize=True every substrate vis-01 tried FLOORED on
        # continuous regression -- dyn-01 then showed the normalization is the DOMINANT contraction
        # lever (dwarfing rho), collapsing the state to a fixed point so the readout emits the
        # per-episode mean. vis-01 subrun 06 broke that floor with normalize=False + a stronger W_in
        # drive. `cx_polar_bump` is the same task CLASS (track a moving bump), so this toggle is the
        # first thing to move if cx-01 floors -- but it is left at the historical default here so the
        # floor question is ASKED rather than assumed (the user's call: find out whether the CX
        # behaves like the optic lobe before applying the optic lobe's fix).
        self.normalize = bool(normalize)
        self.norm_eps = float(norm_eps)
        g0 = torch.tensor(float(norm_gain))
        if norm_learnable:
            self.norm_gain = nn.Parameter(g0)
        else:
            self.register_buffer("norm_gain", g0)

        gen = torch.Generator(device="cpu"); gen.manual_seed(int(seed))
        # w_in_gain scales the INPUT-pathway init (default 1.0). Larger => the self-motion stream
        # re-perturbs the recurrent state harder each step (the anti-fixed-point lever from vis-01).
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
        """inputs [B, T, 2] -> per-timestep 35-D regression outputs [B, T, 35]. The recurrence runs
        ``microsteps`` times per input step (drive re-injected each microstep); the readout is taken
        once, after the last microstep. The bump head is squashed by the LOSS/metric (sigmoid), not
        here, so these are logits for the first 32 channels and raw values for the home vector."""
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"inputs must be [batch, T, {self.input_dim}], got {tuple(inputs.shape)}")
        B, T, _ = inputs.shape
        h = inputs.new_zeros((B, self.N))
        outs = []
        for t in range(T):
            drive = inputs[:, t, :] @ self.W_in.t() + self.b_rec      # step drive (const over microsteps)
            for _ in range(self.microsteps):
                rec = _SparseEdgeMatmul.apply(self.W_rec_values, self.edge_indices, h, self.N)
                h = self.act(rec + drive)
                if self.state_clip > 0:
                    h = torch.clamp(h, min=-self.state_clip, max=self.state_clip)
                if self.normalize:
                    rms = h.pow(2).mean(dim=-1, keepdim=True).sqrt()
                    # Detach the denominator: forward is unchanged, but the unstable d/dh(1/rms) term
                    # is not propagated -- it blows up on sparse ReLU states and diverged training.
                    h = h / (rms + self.norm_eps).detach() * self.norm_gain
            outs.append(self.readout(h))
        return torch.stack(outs, dim=1)                              # [B, T, 35]
