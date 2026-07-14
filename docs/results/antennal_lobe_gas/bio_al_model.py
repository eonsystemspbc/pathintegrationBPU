"""BioALRNN -- a leaky-tanh antennal-lobe RNN with biological I/O.

Implements the biology-spec model exactly:

    h_{t+1} = (1-alpha) h_t + alpha * tanh( (M_AL ⊙ W) h_t + B_ORN A x_t )
    y_hat   = C_PN h_T                                 (readout from projection neurons only)

  * M_AL ⊙ W : recurrence whose SPARSITY PATTERN is the connectome (or a matched control graph),
               with TRAINABLE values initialised at the signed synapse-count weights (rho-scaled).
               Sparse for the param-matched sparse arms; dense for the spectral/dense controls.
  * A         : small NONNEGATIVE sensor->glomerulus adapter (softplus-parameterised), shared
               identically by the connectome and every control. Olfactory glomeruli are driven
               only by the 8 MOX sensors; thermo/hygro glomeruli only by [T, RH].
  * B_ORN     : FIXED 0/1 broadcast of each glomerular drive onto its receptor neurons (ORNs for
               olfactory glomeruli, TRN/HRN for thermo/hygro).
  * C_PN      : linear readout from the projection-neuron pool (optionally RMS-normalised so the
               deep PN signal is well-scaled for the gradient into W).

Generic (free-I/O) arm: input injected into ALL neurons via a trainable W_in, readout from ALL
neurons -- the all-neuron reference the prior experiments used.

Graded local neurons (`graded_ln`): ALLN units use a linear (graded, non-spiking) activation
instead of tanh -- the compartmentalised-LN robustness the spec asks for.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sparse
import torch
from torch import nn

DENSE_FRACTION = 0.2   # operators denser than this are stored/multiplied as dense matrices


class BioALRNN(nn.Module):
    def __init__(self, recurrent: sparse.spmatrix, *, input_dim: int = 10,
                 pn_indices=None, receptor_indices=None, broadcast=None,
                 n_glom_olf: int = 0, n_glom_thr: int = 0, n_sensor: int = 8,
                 bio_io: bool = True, leak: float = 0.3, readout_norm: bool = True,
                 graded_ln: bool = False, ln_indices=None, output_dim: int = 1, seed: int = 0) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        coo = recurrent.astype(np.float32).tocoo(); coo.sum_duplicates()
        if coo.shape[0] != coo.shape[1]:
            raise ValueError("recurrent must be square")
        self.N = int(coo.shape[0])
        self.input_dim = int(input_dim); self.n_sensor = int(n_sensor)
        self.leak = float(leak); self.bio_io = bool(bio_io)
        self.readout_norm = bool(readout_norm); self.graded_ln = bool(graded_ln)
        g = torch.Generator(device="cpu").manual_seed(int(seed))

        # --- recurrence: sparse (param-matched) or dense (spectral/dense controls) ---
        density = coo.nnz / float(self.N * self.N)
        self.dense = density > DENSE_FRACTION
        if self.dense:
            W0 = torch.zeros(self.N, self.N)
            W0[torch.from_numpy(coo.row).long(), torch.from_numpy(coo.col).long()] = \
                torch.from_numpy(coo.data)
            self.W_dense = nn.Parameter(W0)
        else:
            self.register_buffer("edge_idx", torch.from_numpy(
                np.vstack([coo.row, coo.col]).astype(np.int64)))
            self.W_val = nn.Parameter(torch.from_numpy(coo.data.astype(np.float32)))

        # --- graded-LN activation mask ---
        if graded_ln and ln_indices is not None and len(ln_indices):
            m = torch.zeros(self.N); m[torch.as_tensor(np.asarray(ln_indices)).long()] = 1.0
            self.register_buffer("ln_mask", m)               # 1 where linear (graded LN)
        else:
            self.ln_mask = None

        self.b_rec = nn.Parameter(torch.zeros(self.N))

        # --- input path ---
        if bio_io:
            self.n_glom_olf = int(n_glom_olf); self.n_glom_thr = int(n_glom_thr)
            # nonnegative adapters (softplus): olfactory <- 8 sensors ; thermo/hygro <- [T,RH]
            self.A_olf = nn.Parameter(torch.empty(n_glom_olf, n_sensor).uniform_(-2.0, -0.5, generator=g))
            self.A_thr = nn.Parameter(torch.empty(max(n_glom_thr, 1), max(input_dim - n_sensor, 1))
                                      .uniform_(-2.0, -0.5, generator=g))
            self.register_buffer("broadcast", torch.as_tensor(broadcast, dtype=torch.float32))  # [N, G]
            self.in_gain = nn.Parameter(torch.ones(1))
        else:
            scale = 1.0 / math.sqrt(input_dim)
            self.W_in = nn.Parameter(torch.empty(self.N, input_dim).uniform_(-scale, scale, generator=g))

        # --- readout ---
        if bio_io:
            self.register_buffer("pn_idx", torch.as_tensor(np.asarray(pn_indices)).long())
            n_read = int(self.pn_idx.numel())
        else:
            n_read = self.N
        self.readout = nn.Linear(n_read, self.output_dim)
        nn.init.uniform_(self.readout.weight, -1.0 / math.sqrt(n_read), 1.0 / math.sqrt(n_read))
        nn.init.zeros_(self.readout.bias)

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def _rec(self, h: torch.Tensor) -> torch.Tensor:
        if self.dense:
            return h @ self.W_dense.t()
        W = torch.sparse_coo_tensor(self.edge_idx, self.W_val, size=(self.N, self.N),
                                    device=h.device).coalesce()
        return torch.sparse.mm(W, h.t()).t()

    def _inject(self, x_t: torch.Tensor) -> torch.Tensor:
        if self.bio_io:
            sens = x_t[:, :self.n_sensor]
            olf = sens @ torch.nn.functional.softplus(self.A_olf).t()          # [B, G_olf]
            drive = olf
            if self.n_glom_thr > 0 and self.input_dim > self.n_sensor:
                thr = x_t[:, self.n_sensor:] @ torch.nn.functional.softplus(self.A_thr).t()
                drive = torch.cat([olf, thr], dim=1)                            # [B, G_olf+G_thr]
            return (drive @ self.broadcast.t()) * self.in_gain                  # [B, N]
        return x_t @ self.W_in.t()

    def _activate(self, pre: torch.Tensor) -> torch.Tensor:
        a = torch.tanh(pre)
        if self.ln_mask is not None:
            a = a * (1.0 - self.ln_mask) + pre * self.ln_mask                    # graded (linear) LNs
        return a

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs [B, T, input_dim] -> logits [B]
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"inputs must be [B,T,{self.input_dim}], got {tuple(inputs.shape)}")
        B, T, _ = inputs.shape
        h = inputs.new_zeros((B, self.N))
        for t in range(T):
            pre = self._rec(h) + self._inject(inputs[:, t, :]) + self.b_rec
            upd = self._activate(pre)
            h = (1.0 - self.leak) * h + self.leak * upd
        read = h.index_select(1, self.pn_idx) if self.bio_io else h
        if self.readout_norm:
            scale = read.detach().pow(2).mean().sqrt()
            read = read / (scale + 1e-8)
        out = self.readout(read)
        return out.squeeze(-1) if self.output_dim == 1 else out


class AdapterOnly(nn.Module):
    """Floor baseline: the nonnegative sensor->glomerulus adapter feeding a mean-pooled linear
    readout, with NO recurrent AL circuit. Proves the circuit -- not the adapter -- does the work."""
    def __init__(self, *, input_dim: int = 10, n_glom_olf: int = 0, n_glom_thr: int = 0,
                 n_sensor: int = 8, seed: int = 0) -> None:
        super().__init__()
        self.input_dim = int(input_dim); self.n_sensor = int(n_sensor)
        self.n_glom_thr = int(n_glom_thr)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.A_olf = nn.Parameter(torch.empty(n_glom_olf, n_sensor).uniform_(-2.0, -0.5, generator=g))
        self.A_thr = nn.Parameter(torch.empty(max(n_glom_thr, 1), max(input_dim - n_sensor, 1))
                                  .uniform_(-2.0, -0.5, generator=g))
        self.readout = nn.Linear(n_glom_olf + n_glom_thr, 1)

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sens = inputs[:, :, :self.n_sensor]
        olf = torch.nn.functional.softplus(sens @ torch.nn.functional.softplus(self.A_olf).t())
        drive = olf
        if self.n_glom_thr > 0 and self.input_dim > self.n_sensor:
            thr = inputs[:, :, self.n_sensor:] @ torch.nn.functional.softplus(self.A_thr).t()
            drive = torch.cat([olf, thr], dim=2)
        pooled = drive.mean(dim=1)                     # mean over time
        return self.readout(pooled).squeeze(-1)
