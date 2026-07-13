"""BioFlowRNN -- a port-gated sparse trainable-recurrence RNN for the optic-flow task, built to LET A
DEEP BIOLOGICAL READOUT TRAIN.

The plain SparseOpticFlowRNN stalls under biological I/O (inject into R1-6 photoreceptors, read from the
~11 HS/VS LPTCs) because HS/VS sit ~4-5 synapses deep: the yaw-encoding temporal signal decays to
out_tstd ~= 1e-4 by the time it reaches them (~30x weaker than a degree-matched control that manufactures
short shortcuts), so the readout gradient is starved and W_rec never learns to route the pathway.

Knobs added here to unstall it (all default OFF so it reduces to the plain model):
  * microsteps K        -- run K recurrence substeps per input frame, so signal propagates deep WITHIN a
                           frame and the deep readout keeps per-frame temporal structure.
  * readout_norm        -- rescale the OUTPUT-pool activations by their DETACHED per-neuron batch std
                           before the linear readout (identical in train & eval -- no BatchNorm
                           running-stat mismatch). This lifts the tiny (~1e-4) deep-neuron signal to
                           O(1), which restores a well-scaled gradient back into W_rec (amplified by
                           1/std) -- the key lever for the starved readout. Detached denominator keeps
                           the backward clean.
  * state_norm          -- keep the recurrent state healthy / prevent fixed-point collapse:
                             "global_rms" = divide the state by its detached RMS each substep (vis_01's
                                            biological gain-control, stable via the detached denominator);
                             "none"       = off.
  * input_gain          -- scalar on W_in (more drive into the photoreceptor pool).
State update per substep: h = clamp(relu(rho-scaled W @ h + inject_into_input_pool + b), max=state_clip),
then optional state normalization. Readout each FRAME: (optional BN)(h[output_pool]) -> Linear -> R^3.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sparse
import torch
from torch import nn


class BioFlowRNN(nn.Module):
    def __init__(self, recurrent: sparse.spmatrix, input_dim: int, output_dim: int,
                 input_indices, output_indices, *, microsteps: int = 1, state_clip: float = 5.0,
                 readout_norm: bool = False, state_norm: str = "none", input_gain: float = 1.0,
                 leak: float = 1.0, seed: int = 0) -> None:
        super().__init__()
        coo = recurrent.astype(np.float32).tocoo(); coo.sum_duplicates()
        if coo.shape[0] != coo.shape[1]:
            raise ValueError("recurrent matrix must be square.")
        self.N = int(coo.shape[0])
        self.input_dim = int(input_dim); self.output_dim = int(output_dim)
        self.microsteps = int(microsteps); self.state_clip = float(state_clip)
        self.state_norm = str(state_norm); self.input_gain = float(input_gain)
        self.leak = float(leak)  # h <- (1-leak)*h + leak*update; leak<1 = leaky integrator (gradient highway)
        self.pool_gated = input_indices is not None and output_indices is not None
        if self.pool_gated:
            self.register_buffer("input_indices", torch.as_tensor(np.asarray(input_indices), dtype=torch.long))
            self.register_buffer("output_indices", torch.as_tensor(np.asarray(output_indices), dtype=torch.long))
            n_in, n_out = int(self.input_indices.numel()), int(self.output_indices.numel())
        else:
            n_in = n_out = self.N
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        scale_in = 1.0 / math.sqrt(max(self.input_dim, 1))
        self.W_in = nn.Parameter(torch.empty(n_in, self.input_dim).uniform_(-scale_in, scale_in, generator=g))
        self.b_rec = nn.Parameter(torch.zeros(self.N))
        self.readout_norm = bool(readout_norm)
        self.readout = nn.Linear(n_out, self.output_dim)
        nn.init.uniform_(self.readout.weight, -1.0 / math.sqrt(max(n_out, 1)), 1.0 / math.sqrt(max(n_out, 1)))
        nn.init.zeros_(self.readout.bias)
        idx = np.vstack([coo.row, coo.col]).astype(np.int64)
        self.register_buffer("edge_indices", torch.from_numpy(idx))
        self.W_rec_values = nn.Parameter(torch.from_numpy(coo.data.astype(np.float32)))

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def _normalize_state(self, h: torch.Tensor) -> torch.Tensor:
        if self.state_norm == "global_rms":
            rms = h.pow(2).mean(dim=1, keepdim=True).add(1e-6).sqrt().detach()  # detached -> stable backward
            return h / rms
        if self.state_norm == "per_neuron":
            # divide EACH neuron by its detached batch std -> every neuron (incl. quiet deep HS/VS)
            # operates at unit scale, so the deep readout participates in the dynamics and the W_rec
            # gradient is no longer starved. Applied inside the recurrence, so it is NOT absorbed by the
            # linear readout (unlike a readout-side rescale). Cross-sample ordering is preserved -> yaw
            # signal kept. eps floors near-dead neurons so noise-only units aren't blown up.
            sd = h.detach().std(dim=0, keepdim=True)
            return h / (sd + 1e-2)
        return h

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"inputs must be [batch, T, {self.input_dim}], got {tuple(inputs.shape)}")
        batch, T, _ = inputs.shape
        W = torch.sparse_coo_tensor(self.edge_indices, self.W_rec_values, size=(self.N, self.N),
                                    device=inputs.device).coalesce()
        h = inputs.new_zeros((batch, self.N))
        outs = []
        for t in range(T):
            inj = (inputs[:, t, :] @ self.W_in.t()) * self.input_gain
            for _ in range(self.microsteps):
                rec = torch.sparse.mm(W, h.t()).t() + self.b_rec
                if self.pool_gated:
                    rec = rec.index_add(1, self.input_indices, inj)
                else:
                    rec = rec + inj
                upd = torch.relu(rec)
                if self.state_clip > 0:
                    upd = torch.clamp(upd, max=self.state_clip)
                h = (1.0 - self.leak) * h + self.leak * upd if self.leak < 1.0 else upd
                h = self._normalize_state(h)
            read = h.index_select(1, self.output_indices) if self.pool_gated else h
            if self.readout_norm:
                # Scale the output pool by a SINGLE detached scalar (its global RMS at this frame) so
                # the tiny (~1e-4) deep signal becomes O(1) and the starved gradient into W_rec is
                # amplified by 1/rms -- WITHOUT touching per-neuron or cross-sample structure (the
                # cross-sample variance IS the yaw signal, so per-sample/per-neuron norm would erase it).
                scale = read.detach().pow(2).mean().sqrt()
                read = read / (scale + 1e-8)
            outs.append(self.readout(read))
        return torch.stack(outs, dim=1)
