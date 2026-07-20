"""ALRNN / BioALRNN -- the house connectome-as-RNN, applied to windowed gas detection.

al-02 ADDS `BioALRNN` alongside al-01's `ALRNN`. `ALRNN` is kept UNCHANGED as the generic-I/O
reference arm (and because al-01's frozen record depends on its behaviour). Read the `BioALRNN`
docstring for what al-02 changes; everything else -- ReLU full-replacement map, K=2 microsteps, no
leak, trainable edge values on frozen wiring, rho=0.95 -- is identical between the two.

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


class BioALRNN(nn.Module):
    """al-02's BIOLOGICAL-I/O variant of ALRNN. Same dynamics, different ports.

    WHAT CHANGES vs al-01's `ALRNN` (this is the whole hypothesis under test):

      al-01 generic I/O                        al-02 biological I/O
      -------------------------------------    ---------------------------------------------
      W_in [N, 10] = 49,470 free params        glomerulus-tied fan-out, 53*8 + 8*2 = 440 params
      every neuron gets its own input weight   every ORN in a glomerulus gets the SAME drive
      readout over all N = 4,947 neurons       readout over the PN pool only (683 neurons)

    THE INPUT ADAPTER.  The task hands us 10 channels, `[T, RH, s1..s8]` (gas_task.py:95 --
    `feats = arr[:, 1:11]`, i.e. columns T, RH, then the 8 metal-oxide sensors). They are routed
    the way an antenna routes odour and micro-climate:

        8 chemical sensors --[53 x 8 mixing]--> 53 olfactory glomeruli --> every ORN in that glom
        T, RH              --[ 8 x 2 mixing]-->  8 thermo/hygro gloms  --> every thermo/hygro neuron

    A glomerulus is a labelled line: all ~40 ORNs expressing one receptor converge on one glomerulus
    and carry one channel of information. So the model gets ONE scalar drive per glomerulus, and
    neurons sharing a glomerulus are driven IDENTICALLY. That glomerular quantisation is not a
    convenience -- it IS the structure al-01's generic W_in destroyed, and restoring it is the
    single variable al-02 changes (al-01 notebook, "Next").

    Neurons outside the receptor pools (LNs, PNs, halo) receive NO direct sensor input. They are
    reached only through the recurrence, which is the point: the circuit, not the input layer, has
    to do the routing.

    NON-NEGATIVE MIXING via SOFTPLUS.  The prior AL study used a non-negative input adapter, and it
    is the biologically right constraint -- a receptor's response to an odorant is an excitatory
    firing rate, so a glomerulus cannot be driven "negatively" by a sensor. We use `softplus(raw)`
    rather than `abs(raw)`: softplus is smooth everywhere and strictly positive, whereas `abs` has a
    gradient sign-flip at zero that lets a weight tunnel through the origin and reappear with the
    same effective value, which makes an "off" channel unstable under Adam. Softplus instead lets a
    channel decay smoothly toward (but never through) zero, with a vanishing gradient as it does.
    `raw` is initialised by inverting softplus on effective weights drawn U(0, 1/sqrt(8)), so the
    initial drive magnitude matches al-01's `W_in ~ U(-1/sqrt(10), 1/sqrt(10))` to within ~10%.

    THE READOUT.  A single `Linear(683 -> 1)` over `pn_idx`, evaluated on h at the FINAL timestep.
    PNs are the antennal lobe's only output channel; reading anywhere else is reading a wire the
    animal cannot read.

    THE INPUT GAIN.  `input_gain` is a non-trainable scalar buffer multiplying the adapter's drive
    and nothing else. It exists to carry the mb-06 readout-pool activation-RMS match (fitted in
    `common.fit_readout_rms_gain`). It is deliberately OUTSIDE the recurrence, so it cannot change
    `M` or its spectral radius -- rho stays exactly 0.95 in every arm after matching.

    TRAINABLE vs FROZEN -- unchanged from al-01:
      trainable : W_rec_values [E] (one scalar per EXISTING edge), mix_olf_raw [53,8],
                  mix_thermo_raw [8,2], b_rec [N], readout weight+bias [683+1]
      frozen    : edge_indices [2,E] (the WIRING), input_gain (a buffer, fitted not learned)
    So connectome and controls differ ONLY in which edges exist and their initial values. Parameter
    counts are identical across arms by construction (E is preserved by every control here); this
    is asserted at run time, not assumed.

    `b_rec` stays on all N, as in al-01. It is an intrinsic-excitability term belonging to the
    recurrent unit, not a sensor port -- neurons outside the receptor pools still receive no
    SENSOR input, which is the claim being made. It is identical in shape and count across arms.
    """

    CHEM_CHANNELS = (2, 3, 4, 5, 6, 7, 8, 9)   # s1..s8  (gas_task feature order [T, RH, s1..s8])
    TRH_CHANNELS = (0, 1)                      # T, RH

    def __init__(self, recurrent: sp.spmatrix, ports: dict, input_dim: int = 10,
                 output_dim: int = 1, seed: int = 0, microsteps: int = 2,
                 activation: str = "relu", normalize: bool = False, norm_gain: float = 1.0,
                 norm_eps: float = 1e-5, freeze_recurrent: bool = False,
                 input_gain: float = 1.0) -> None:
        super().__init__()
        if activation not in _ACTS:
            raise ValueError(f"activation must be one of {tuple(_ACTS)}")
        if input_dim != 10:
            raise ValueError("BioALRNN's port map assumes the 10-channel [T, RH, s1..s8] layout")
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
        self.register_buffer("input_gain", torch.tensor(float(input_gain)))

        # ---- ports (schema: build_ports.py -> substrate/ports.npz, index order == root_ids.npy)
        glom_id = np.asarray(ports["glom_id"], dtype=np.int64)
        thermo_glom_id = np.asarray(ports["thermo_glom_id"], dtype=np.int64)
        orn_idx = np.asarray(ports["orn_idx"], dtype=np.int64)
        thermo_idx = np.asarray(ports["thermo_idx"], dtype=np.int64)
        pn_idx = np.asarray(ports["pn_idx"], dtype=np.int64)
        if len(glom_id) != self.N or len(thermo_glom_id) != self.N:
            raise ValueError("ports arrays must be per-neuron, length N, in root_ids order")
        self.n_glom = int(len(ports["glom_names"]))
        self.n_thermo_glom = int(len(ports["thermo_glom_names"]))

        # An ORN with no glomerulus assignment (glom_id == -1) has no labelled line to be driven
        # through, so it gets no direct input. Counted and exposed, never silently dropped.
        okg = glom_id[orn_idx] >= 0
        okt = thermo_glom_id[thermo_idx] >= 0
        self.n_orn_total, self.n_orn_driven = int(len(orn_idx)), int(okg.sum())
        self.n_thermo_total, self.n_thermo_driven = int(len(thermo_idx)), int(okt.sum())
        self.register_buffer("orn_pos", torch.from_numpy(orn_idx[okg]))
        self.register_buffer("orn_glom", torch.from_numpy(glom_id[orn_idx][okg]))
        self.register_buffer("thermo_pos", torch.from_numpy(thermo_idx[okt]))
        self.register_buffer("thermo_glom", torch.from_numpy(thermo_glom_id[thermo_idx][okt]))
        self.register_buffer("pn_idx", torch.from_numpy(pn_idx))
        self.n_pn = int(len(pn_idx))

        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        n_chem, n_trh = len(self.CHEM_CHANNELS), len(self.TRH_CHANNELS)
        self.mix_olf_raw = nn.Parameter(_inv_softplus(
            torch.empty(self.n_glom, n_chem).uniform_(0.0, 1.0 / math.sqrt(n_chem), generator=gen)))
        self.mix_thermo_raw = nn.Parameter(_inv_softplus(
            torch.empty(self.n_thermo_glom, n_trh).uniform_(
                0.0, 1.0 / math.sqrt(n_trh), generator=gen)))
        self.b_rec = nn.Parameter(torch.zeros(self.N))
        self.readout = nn.Linear(self.n_pn, self.output_dim)
        scale_out = 1.0 / math.sqrt(max(self.n_pn, 1))
        nn.init.uniform_(self.readout.weight, -scale_out, scale_out)
        nn.init.zeros_(self.readout.bias)

        indices = np.vstack([coo.row, coo.col]).astype(np.int64)
        self.register_buffer("edge_indices", torch.from_numpy(indices))
        values = coo.data.astype(np.float32)
        self.W_rec_values = nn.Parameter(torch.from_numpy(values))
        self.register_buffer("W_rec_initial_values", torch.from_numpy(values.copy()))
        if freeze_recurrent:
            self.W_rec_values.requires_grad_(False)

    # -------------------------------------------------------------------------------- utilities
    def recurrent_parameter_count(self) -> int:
        return int(self.W_rec_values.numel())

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def adapter_parameter_count(self) -> int:
        return int(self.mix_olf_raw.numel() + self.mix_thermo_raw.numel())

    def set_input_gain(self, g: float) -> None:
        """Set the non-recurrent input gain. Cannot touch M, so rho is provably unaffected."""
        with torch.no_grad():
            self.input_gain.fill_(float(g))

    def input_drive(self, x_t: torch.Tensor) -> torch.Tensor:
        """[B, 10] -> [B, N]. Glomerulus-tied fan-out; zero outside the receptor pools."""
        w_olf = torch.nn.functional.softplus(self.mix_olf_raw)        # [53, 8], >= 0
        w_the = torch.nn.functional.softplus(self.mix_thermo_raw)     # [ 8, 2], >= 0
        g_olf = x_t[:, list(self.CHEM_CHANNELS)] @ w_olf.t()          # [B, 53] per-glomerulus drive
        g_the = x_t[:, list(self.TRH_CHANNELS)] @ w_the.t()           # [B,  8]
        d = x_t.new_zeros(x_t.shape[0], self.N)
        d = d.index_add(1, self.orn_pos, g_olf[:, self.orn_glom])     # every ORN <- its glom's drive
        d = d.index_add(1, self.thermo_pos, g_the[:, self.thermo_glom])
        return d * self.input_gain

    def forward(self, inputs: torch.Tensor, return_state: bool = False):
        """inputs [B, T, 10] -> [B] logits (single readout over the PN pool at the final step)."""
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"inputs must be [B, T, {self.input_dim}], got {tuple(inputs.shape)}")
        B, T, _ = inputs.shape
        h = inputs.new_zeros((B, self.N))
        for t in range(T):
            drive = self.input_drive(inputs[:, t, :]) + self.b_rec
            for _ in range(self.microsteps):
                rec = _SparseEdgeMatmul.apply(self.W_rec_values, self.edge_indices, h, self.N)
                h = self.act(rec + drive)
                if self.normalize:
                    rms = h.pow(2).mean(dim=-1, keepdim=True).sqrt()
                    h = h / (rms + self.norm_eps).detach() * self.norm_gain
        h_pn = h[:, self.pn_idx]                       # the ONLY thing the readout ever sees
        out = self.readout(h_pn)
        out = out.squeeze(-1) if self.output_dim == 1 else out
        return (out, h, h_pn) if return_state else out


def _inv_softplus(w: torch.Tensor, floor: float = 1e-4) -> torch.Tensor:
    """raw such that softplus(raw) == w. Used so the adapter is INITIALISED in effective space."""
    return torch.log(torch.expm1(w.clamp_min(floor)))


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
