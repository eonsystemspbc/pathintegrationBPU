"""ALRNN -- the house connectome-as-RNN, applied to windowed gas detection.

DYNAMICS (mb-01..06 / cx-01 lineage, NOT the prior AL study's leaky-tanh):

    for each input step t:
        drive = W_in x_t + b_rec                    # constant across microsteps
        repeat K times:  h <- relu(M h + drive)     # M = rho-scaled connectome or control
    y_hat = readout(h_T)                            # single logit, read at the FINAL step

There is no leak term and no dt: this is a full-replacement map, exactly as in
`MatrixEpisodicRNN` (mb-01..06) and `CXRNN` (cx-01). K=2 microsteps per input step matches
mb-05/06 and reflects the AL's ~2-hop depth (receptor -> local -> projection).

TRAINABLE vs FROZEN -- the house regime:
  trainable : W_rec_values [E] (one scalar per EXISTING edge), W_in [N x 10], b_rec [N], readout
  frozen    : edge_indices [2, E]  -- the WIRING itself never changes
So the connectome and its control differ ONLY in which edges exist and their initial values; both
arms get identical parameter counts, identical initialization scheme, and identical optimization.
Signs are free to flip during training (no Dale projection) -- same as every prior experiment.

I/O is GENERIC: input projects to all N neurons, readout reads all N. This is the mb-01/02/06 and
cx-01 convention. (The prior AL study also ran a biological-port variant; that is deliberately out
of scope here so this experiment tests wiring alone.)

NORMALIZATION is available but OFF by default. cx-01/vis-01 default it ON; mb-01..06 -- the
CLASSIFICATION lineage this task belongs to -- ran without it, and dyn-01 showed normalization is
the dominant contraction lever that collapsed vis-01's regression task to a fixed point. Since
this is a settle-to-an-answer detection task, we follow the mb lineage. The flag exists so that a
floor can be diagnosed rather than assumed.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

_ACTS = {"relu": torch.relu, "tanh": torch.tanh}


class _SparseEdgeMatmul(torch.autograd.Function):
    """rec = M @ h with M defined by (values, indices), gradient computed EDGE-LOCALLY.

    Copied from cx-01's model.py. The point is dL/dvalue_e = sum_b grad[b, row_e] * h[b, col_e],
    which never materializes a dense N x N gradient (N=4,947 -> 24M floats per step otherwise).
    Numerically identical to torch.sparse.mm's gradient.
    """

    @staticmethod
    def forward(ctx, values, indices, h, N):
        ctx.save_for_backward(values, indices, h)
        ctx.N = N
        W = torch.sparse_coo_tensor(indices, values, size=(N, N), device=h.device).coalesce()
        return torch.sparse.mm(W, h.t()).t()

    @staticmethod
    def backward(ctx, grad_out):
        values, indices, h = ctx.saved_tensors
        N = ctx.N
        rows, cols = indices[0], indices[1]
        grad_values = grad_h = None
        if ctx.needs_input_grad[0]:
            grad_values = (grad_out[:, rows] * h[:, cols]).sum(dim=0)
        if ctx.needs_input_grad[2]:
            Wt = torch.sparse_coo_tensor(torch.stack([cols, rows]), values,
                                         size=(N, N), device=h.device).coalesce()
            grad_h = torch.sparse.mm(Wt, grad_out.t()).t()
        return grad_values, None, grad_h, None


class ALRNN(nn.Module):
    def __init__(self, recurrent: sp.spmatrix, input_dim: int = 10, output_dim: int = 1,
                 seed: int = 0, microsteps: int = 2, activation: str = "relu",
                 normalize: bool = False, norm_gain: float = 1.0, norm_eps: float = 1e-5,
                 freeze_recurrent: bool = False) -> None:
        super().__init__()
        if activation not in _ACTS:
            raise ValueError(f"activation must be one of {tuple(_ACTS)}")
        coo = recurrent.astype(np.float32).tocoo()
        coo.sum_duplicates()
        if coo.shape[0] != coo.shape[1]:
            raise ValueError("recurrent matrix must be square")
        self.N = int(coo.shape[0])
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.microsteps = int(max(1, microsteps))
        self.act = _ACTS[activation]
        self.act_name = activation
        self.normalize = bool(normalize)
        self.norm_eps = float(norm_eps)
        self.register_buffer("norm_gain", torch.tensor(float(norm_gain)))

        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        scale_in = 1.0 / math.sqrt(max(input_dim, 1))
        scale_out = 1.0 / math.sqrt(max(self.N, 1))
        self.W_in = nn.Parameter(
            torch.empty(self.N, input_dim).uniform_(-scale_in, scale_in, generator=gen))
        self.b_rec = nn.Parameter(torch.zeros(self.N))
        self.readout = nn.Linear(self.N, self.output_dim)
        nn.init.uniform_(self.readout.weight, -scale_out, scale_out)
        nn.init.zeros_(self.readout.bias)

        indices = np.vstack([coo.row, coo.col]).astype(np.int64)
        self.register_buffer("edge_indices", torch.from_numpy(indices))
        values = coo.data.astype(np.float32)
        self.W_rec_values = nn.Parameter(torch.from_numpy(values))
        self.register_buffer("W_rec_initial_values", torch.from_numpy(values.copy()))
        if freeze_recurrent:
            self.W_rec_values.requires_grad_(False)

    def recurrent_parameter_count(self) -> int:
        return int(self.W_rec_values.numel())

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs [B, T, 10] -> [B] logits (single readout at the final timestep)."""
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"inputs must be [B, T, {self.input_dim}], got {tuple(inputs.shape)}")
        B, T, _ = inputs.shape
        h = inputs.new_zeros((B, self.N))
        for t in range(T):
            drive = inputs[:, t, :] @ self.W_in.t() + self.b_rec
            for _ in range(self.microsteps):
                rec = _SparseEdgeMatmul.apply(self.W_rec_values, self.edge_indices, h, self.N)
                h = self.act(rec + drive)
                if self.normalize:
                    rms = h.pow(2).mean(dim=-1, keepdim=True).sqrt()
                    h = h / (rms + self.norm_eps).detach() * self.norm_gain
        out = self.readout(h)
        return out.squeeze(-1) if self.output_dim == 1 else out


class GRUCeiling(nn.Module):
    """Dense GRU learnability gate (cx-01's convention).

    Not a graph control -- a check that the TASK is learnable at all under this data budget and
    supervision. If every connectome/control arm sits at a floor, this says whether the floor is
    the task or the substrate. Interpreting a null without it is unsafe.
    """

    def __init__(self, input_dim: int = 10, hidden: int = 256, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, 1)

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def recurrent_parameter_count(self) -> int:
        return int(sum(p.numel() for n, p in self.named_parameters() if n.startswith("gru")))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(inputs)
        return self.readout(out[:, -1, :]).squeeze(-1)
