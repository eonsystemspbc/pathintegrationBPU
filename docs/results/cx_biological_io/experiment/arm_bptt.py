#!/usr/bin/env python3
"""Experiment 4, Arm A -- backprop, port-gated MB I/O (SPEC.md section 3.1).

`PortGatedMatrixRNN` is `MatrixEpisodicRNN`
(scripts/associative/run_omniglot_associative_benchmark.py:151-252) restricted so a
trainable readout can no longer route around the wiring:
  * input is gated through two small blocks -- `W_in_alpn` injects the MQAR cue/query
    symbol into ALPN rows only, `W_in_dan` injects the value/teaching symbol into DAN
    rows only; every other neuron (KC, MBON, MBIN) gets zero external drive;
  * `MICROSTEPS` recurrence steps run per input token so a signal entering at ALPN can
    reach MBON via KC within the same token slot (ALPN->KC->MBON is 2 hops; see
    SPEC.md section 3.1);
  * the readout reads MBON rows only.

The recurrent operator is built by the caller via `common.build_condition_operator`
(sparse, biologically-forward = M itself since the adjacency is stored post x pre, and
already rescaled to rho=0.95). This module never transposes or rescales it again
(SPEC.md section 1). Sparse machinery
(edge_indices buffer + W_rec_values Parameter, one `torch.sparse_coo_tensor` built per
forward, `torch.sparse.mm`) mirrors `MatrixEpisodicRNN` verbatim so the recurrent
dynamics stay identical to the rest of Exp 1-4 -- only the I/O ports differ. Recurrent
weights are always trainable on the fixed support (freeze_recurrent=False), same regime
as Exp 1-3.

`run_condition` is Arm A's implementation of the arm interface (SPEC.md section 6): train
one (condition, unit, hp) triple and return `train_one_run`'s result dict.
  * 'generic_io' reuses the stock all-neuron `MatrixEpisodicRNN` (train_one_run's own
    `model=None` path) on the SAME forward operator as 'connectome' -- the internal reference
    for "does restricting I/O to biological ports help or hurt?".
  * 'connectome' / 'degree_matched' build a `PortGatedMatrixRNN` on the condition's
    operator.

Do NOT redefine common.py's helpers here -- import and reuse them.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    ROLE_DIMS,
    TARGET_RHO,
    build_condition_operator,
    split_roles,
    train_one_run,
)

CONDITIONS = ("connectome", "degree_matched", "generic_io")
DEFAULT_MICROSTEPS = 2
ARM_NAME = "bptt"  # must match run_experiment.py's build_plan()/dispatch() arm tag


class PortGatedMatrixRNN(nn.Module):
    """Port-restricted `MatrixEpisodicRNN`: I/O gated to the biological ALPN (input) /
    DAN (teaching) / MBON (readout) ports; recurrence is trainable on the fixed sparse
    support, exactly like `MatrixEpisodicRNN` with `freeze_recurrent=False`."""

    def __init__(
        self,
        recurrent,                  # scipy sparse (coo/csr) -- forward operator (=M), rho-rescaled
        ports: dict,                # {'alpn','dan','mbon',...} -> int64 index arrays (substrate space)
        vocab_size: int,
        microsteps: int = DEFAULT_MICROSTEPS,
        state_clip: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        recurrent = recurrent.astype(np.float32).tocoo()
        recurrent.sum_duplicates()
        if recurrent.shape[0] != recurrent.shape[1]:
            raise ValueError("recurrent matrix must be square.")
        for key in ("alpn", "dan", "mbon"):
            if key not in ports:
                raise KeyError(f"ports must include {key!r} (have {list(ports)})")

        self.N = int(recurrent.shape[0])
        self.vocab_size = int(vocab_size)
        self.microsteps = int(microsteps)
        if self.microsteps < 1:
            raise ValueError("microsteps must be >= 1")
        self.state_clip = float(state_clip)

        alpn_idx = np.asarray(ports["alpn"], dtype=np.int64)
        dan_idx = np.asarray(ports["dan"], dtype=np.int64)
        mbon_idx = np.asarray(ports["mbon"], dtype=np.int64)
        if (np.intersect1d(alpn_idx, dan_idx).size or np.intersect1d(alpn_idx, mbon_idx).size
                or np.intersect1d(dan_idx, mbon_idx).size):
            raise ValueError("alpn/dan/mbon ports must be disjoint (they gate different rows).")
        self.n_alpn, self.n_dan, self.n_mbon = int(alpn_idx.size), int(dan_idx.size), int(mbon_idx.size)
        self.register_buffer("alpn_idx", torch.from_numpy(alpn_idx))
        self.register_buffer("dan_idx", torch.from_numpy(dan_idx))
        self.register_buffer("mbon_idx", torch.from_numpy(mbon_idx))

        # Init mirrors MatrixEpisodicRNN's own scheme: a local generator (seeded explicitly)
        # draws the port-gated W_in blocks; the readout uses nn.Linear's default construction
        # then an explicit uniform overwrite on the GLOBAL torch RNG, exactly as
        # MatrixEpisodicRNN does -- callers seed torch.manual_seed(seed) before constructing
        # this model (see run_condition) so that piece is reproducible too.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        scale_in = 1.0 / math.sqrt(max(self.vocab_size, 1))
        self.W_in_alpn = nn.Parameter(
            torch.empty(self.n_alpn, self.vocab_size, dtype=torch.float32).uniform_(
                -scale_in, scale_in, generator=generator
            )
        )
        self.W_in_dan = nn.Parameter(
            torch.empty(self.n_dan, self.vocab_size, dtype=torch.float32).uniform_(
                -scale_in, scale_in, generator=generator
            )
        )
        self.b_rec = nn.Parameter(torch.zeros(self.N, dtype=torch.float32))

        scale_out = 1.0 / math.sqrt(max(self.n_mbon, 1))
        self.readout = nn.Linear(self.n_mbon, self.vocab_size)
        nn.init.uniform_(self.readout.weight, -scale_out, scale_out)
        nn.init.zeros_(self.readout.bias)

        # sparse recurrent operator -- identical machinery to MatrixEpisodicRNN's "sparse"
        # runtime (edge_indices buffer + W_rec_values Parameter; rebuilt as a
        # torch.sparse_coo_tensor once per forward, not once per step). Trainable on the
        # fixed support (freeze_recurrent=False, always, for this arm).
        indices = np.vstack([recurrent.row, recurrent.col]).astype(np.int64)
        self.register_buffer("edge_indices", torch.from_numpy(indices))
        values = recurrent.data.astype(np.float32)
        self.W_rec_values = nn.Parameter(torch.from_numpy(values))
        self.register_buffer("W_rec_initial_values", torch.from_numpy(values.copy()))

    def recurrent_parameter_count(self) -> int:
        return int(self.W_rec_values.numel())

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def recurrent_prior_loss(self) -> torch.Tensor:
        return nn.functional.mse_loss(self.W_rec_values, self.W_rec_initial_values)

    def _external_drive(self, cue_t: torch.Tensor, value_t: torch.Tensor) -> torch.Tensor:
        """[B, vocab] cue/value at one timestep -> [B, N] drive, zero outside ALPN/DAN rows."""
        batch = cue_t.shape[0]
        drive = cue_t.new_zeros((batch, self.N))
        drive[:, self.alpn_idx] = cue_t @ self.W_in_alpn.t()
        drive[:, self.dan_idx] = value_t @ self.W_in_dan.t()
        return drive

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"inputs must be [batch, T, vocab+{ROLE_DIMS}], got {tuple(inputs.shape)}")
        vocab = inputs.shape[-1] - ROLE_DIMS
        if vocab != self.vocab_size:
            raise ValueError(f"expected vocab_size={self.vocab_size}, inputs imply {vocab}")
        batch, T, _ = inputs.shape
        cue, value, _gate = split_roles(inputs)  # [B,T,vocab] each; routing per SPEC section 2

        h = inputs.new_zeros((batch, self.N))
        # Build the sparse recurrent operator ONCE per forward (W_rec_values is constant within
        # a forward) -- mirrors MatrixEpisodicRNN.forward's optimization.
        W_sparse = torch.sparse_coo_tensor(
            self.edge_indices, self.W_rec_values, size=(self.N, self.N), device=inputs.device
        ).coalesce()

        outputs: list[torch.Tensor] = []
        for t in range(T):
            drive = self._external_drive(cue[:, t, :], value[:, t, :])
            for _ in range(self.microsteps):  # drive held constant across the token's microsteps
                rec = torch.sparse.mm(W_sparse, h.t()).t()
                h = torch.relu(rec + drive + self.b_rec)
                if self.state_clip > 0:
                    h = torch.clamp(h, max=self.state_clip)
            outputs.append(self.readout(h[:, self.mbon_idx]))
        return torch.stack(outputs, dim=1)


def _run_id(condition: str, unit: int, hp: float) -> str:
    """<arm>_<condition>_u<unit>_hp<hp> -- MUST match run_experiment.py's build_plan()/add()
    (f"{arm}_{condition}_u{unit:02d}_hp{hp:g}") and _parse_run_id(), since dispatch() computes
    spec['run_id'] independently and pre-checks out_dir/runs/<run_id>/result.json BEFORE
    calling this module; if the two derivations diverge, runs land in different directories
    and analyze()'s run_id parser silently drops them. See the "ambiguities" note in the
    handoff for why this differs from the SPEC snippet's illustrative `armA_...` example."""
    return f"{ARM_NAME}_{condition}_u{int(unit):02d}_hp{float(hp):g}"


def run_condition(cfg, sub, ports: dict, condition: str, unit: int, hp: float,
                  device, out_dir: Path) -> dict:
    """Train/evaluate ONE unit for Arm A (SPEC.md section 6 arm interface).

    cfg: SimpleNamespace from common.make_args(...) (task/optim config; `cfg.microsteps`
         read if present, else DEFAULT_MICROSTEPS).
    sub: NATIVE csr sub-adjacency (M[i,j] = weight(j->i), post x pre) for this substrate, from
         common.load_substrate / common.synthetic_substrate.
    ports: {'alpn','kc','mbon','dan','mbin'} -> int64 index arrays (substrate space).
    condition: 'connectome' | 'degree_matched' | 'generic_io'.
    unit: training-seed index (connectome/generic_io, one real graph) or graph index
          (degree_matched, an independent rewiring per unit) -- also used as the training seed.
    hp: learning rate.
    device: torch device (or device string) train_one_run runs on.
    out_dir: this arm's output root; writes to out_dir/runs/<run_id>/{metrics_epochs.csv,
             checkpoint.pt, result.json}.

    Idempotent: if runs/<run_id>/result.json already exists, returns it without retraining
    (train_one_run's own checkpoint resume additionally covers partially-trained runs, e.g.
    if dispatch() ever calls in without this shortcut)."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    if isinstance(device, str):
        # SPEC.md section 6 types `device` as `str` (run_experiment.py's dispatch() passes
        # cfg.device straight through) but common.train_one_run (reused verbatim from Exp 1)
        # needs a real torch.device -- it reads `device.type` and calls
        # torch.cuda.get_rng_state(device). Canonicalize here (same cuda-availability
        # fallback every experiment's main() uses) so this arm works with either a bare
        # device string or an already-built torch.device (e.g. from the smoke script).
        if device == "cuda":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
    out_dir = Path(out_dir)
    run_id = _run_id(condition, unit, hp)
    run_dir = out_dir / "runs" / run_id
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    op = build_condition_operator(sub, condition, seed=unit)
    microsteps = int(getattr(cfg, "microsteps", DEFAULT_MICROSTEPS))
    meta = {
        "condition": condition, "arm": ARM_NAME, "run_id": run_id,
        "unit": int(unit), "graph_seed": int(unit), "train_seed": int(unit),
        "hp": float(hp), "lr": float(hp), "microsteps": microsteps,
        "N": int(op.shape[0]), "edges": int(op.nnz), "rho_target": TARGET_RHO,
    }

    if condition == "generic_io":
        # Exp-1-3's all-neuron I/O reference, on the identical forward operator -- isolates the
        # effect of restricting I/O to the biological ports (SPEC.md section 3.1).
        return train_one_run(run_dir, op, cfg, unit, device, meta, hp, model=None)

    # connectome / degree_matched: port-gated model. Seed torch BEFORE construction so the
    # readout's global-RNG-dependent init is reproducible (mirrors
    # scott/experiment_03_dense_param_matched/run_experiment.py's custom-model pattern).
    torch.manual_seed(cfg.init_seed + unit)
    model = PortGatedMatrixRNN(
        recurrent=op, ports=ports, vocab_size=cfg.vocab_size,
        microsteps=microsteps, state_clip=cfg.state_clip, seed=cfg.init_seed + unit,
    )
    return train_one_run(run_dir, None, cfg, unit, device, meta, hp, model=model)


__all__ = ["PortGatedMatrixRNN", "run_condition", "CONDITIONS", "DEFAULT_MICROSTEPS", "ARM_NAME"]
