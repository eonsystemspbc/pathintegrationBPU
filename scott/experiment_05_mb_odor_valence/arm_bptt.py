#!/usr/bin/env python3
"""Experiment 5, Arm A -- backprop, port-gated MB I/O, odor->valence task (Phase 2).

The odor->valence port of Experiment 4's Arm A. `PortGatedMatrixRNN` is the all-neuron
`MatrixEpisodicRNN` restricted so a trainable readout can no longer route around the wiring:
  * input is gated through two small blocks -- `W_in_alpn` injects the ODOR into ALPN rows
    only, `W_in_dan` injects the reward/punishment TEACHING signal into DAN rows only; every
    other neuron (KC, MBON, MBIN) gets zero external drive;
  * `microsteps` recurrence steps run per token so a signal entering at ALPN can reach MBON
    via KC within one token (ALPN->KC->MBON is 2 hops);
  * the readout reads MBON rows only, into `n_valence` logits.

Unlike the plasticity arm there is NO per-episode fast weight and NO state reset: the whole
association (learn -> query -> reversal -> query) must be held in the recurrent hidden-state
dynamics that BPTT learns. Learn vs query steps are distinguished by the presence of the DAN
teaching drive (reinforcement present = learn/reversal; absent = query); the query gate itself
is a task-bookkeeping bit and is deliberately NOT injected into any biological port (matching
Arm A's port discipline). Only query-step outputs are scored (masked loss).

The recurrent operator is built by the caller via `common.build_condition_operator` (sparse,
biologically-forward = M, rho=0.95). This module never transposes or rescales it again.
Recurrent weights are trainable on the fixed support (freeze_recurrent=False), same regime as
Exp 1-4. Training uses `common.train_one_run_ov` (odor->valence loop).
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

import common  # noqa: E402
import odor_valence_task as ov  # noqa: E402

CONDITIONS = ("connectome", "degree_matched", "generic_io")
DEFAULT_MICROSTEPS = 2
ARM_NAME = "bptt"
N_TEACH = ov.N_VALENCE  # reward/punishment 2-bit -> DAN teaching channels


class PortGatedMatrixRNN(nn.Module):
    """Port-restricted MatrixEpisodicRNN: I/O gated to the biological ALPN (odor input) /
    DAN (reward-punishment teaching) / MBON (valence readout) ports; recurrence trainable on
    the fixed sparse support."""

    def __init__(self, recurrent, ports: dict, cue_dim: int, n_valence: int,
                 microsteps: int = DEFAULT_MICROSTEPS, state_clip: float = 0.0,
                 seed: int = 0) -> None:
        super().__init__()
        recurrent = recurrent.astype(np.float32).tocoo(); recurrent.sum_duplicates()
        if recurrent.shape[0] != recurrent.shape[1]:
            raise ValueError("recurrent matrix must be square.")
        for key in ("alpn", "dan", "mbon"):
            if key not in ports:
                raise KeyError(f"ports must include {key!r} (have {list(ports)})")

        self.N = int(recurrent.shape[0])
        self.cue_dim = int(cue_dim)
        self.n_valence = int(n_valence)
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

        generator = torch.Generator(device="cpu"); generator.manual_seed(int(seed))
        scale_in = 1.0 / math.sqrt(max(self.cue_dim, 1))
        scale_teach = 1.0 / math.sqrt(max(N_TEACH, 1))
        self.W_in_alpn = nn.Parameter(
            torch.empty(self.n_alpn, self.cue_dim, dtype=torch.float32).uniform_(
                -scale_in, scale_in, generator=generator))
        self.W_in_dan = nn.Parameter(
            torch.empty(self.n_dan, N_TEACH, dtype=torch.float32).uniform_(
                -scale_teach, scale_teach, generator=generator))
        self.b_rec = nn.Parameter(torch.zeros(self.N, dtype=torch.float32))

        scale_out = 1.0 / math.sqrt(max(self.n_mbon, 1))
        self.readout = nn.Linear(self.n_mbon, self.n_valence)
        nn.init.uniform_(self.readout.weight, -scale_out, scale_out)
        nn.init.zeros_(self.readout.bias)

        indices = np.vstack([recurrent.row, recurrent.col]).astype(np.int64)
        self.register_buffer("edge_indices", torch.from_numpy(indices))
        values = recurrent.data.astype(np.float32)
        self.W_rec_values = nn.Parameter(torch.from_numpy(values))
        self.register_buffer("W_rec_initial_values", torch.from_numpy(values.copy()))

    def recurrent_parameter_count(self) -> int:
        return int(self.W_rec_values.numel())

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def _external_drive(self, odor_t: torch.Tensor, teach_t: torch.Tensor) -> torch.Tensor:
        """[B,cue_dim] odor + [B,N_TEACH] reward/punish at one timestep -> [B,N] drive,
        zero outside ALPN/DAN rows."""
        batch = odor_t.shape[0]
        drive = odor_t.new_zeros((batch, self.N))
        drive[:, self.alpn_idx] = odor_t @ self.W_in_alpn.t()
        drive[:, self.dan_idx] = teach_t @ self.W_in_dan.t()
        return drive

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"inputs must be [batch, T, cue_dim+{ov.ROLE_DIMS}], got {tuple(inputs.shape)}")
        cd = self.cue_dim
        if inputs.shape[-1] != cd + ov.ROLE_DIMS:
            raise ValueError(f"expected last dim {cd + ov.ROLE_DIMS}, got {inputs.shape[-1]}")
        odor = inputs[..., :cd]
        teach = inputs[..., cd:cd + N_TEACH]                  # [reward, punish] -> DAN
        batch, T, _ = inputs.shape

        h = inputs.new_zeros((batch, self.N))
        W_sparse = torch.sparse_coo_tensor(
            self.edge_indices, self.W_rec_values, size=(self.N, self.N),
            device=inputs.device).coalesce()

        outputs: list[torch.Tensor] = []
        for t in range(T):
            drive = self._external_drive(odor[:, t, :], teach[:, t, :])
            for _ in range(self.microsteps):                  # drive held across the token's microsteps
                rec = torch.sparse.mm(W_sparse, h.t()).t()
                h = torch.relu(rec + drive + self.b_rec)
                if self.state_clip > 0:
                    h = torch.clamp(h, max=self.state_clip)
            outputs.append(self.readout(h[:, self.mbon_idx]))
        return torch.stack(outputs, dim=1)                    # [B, T, n_valence]


def _run_id(condition: str, unit: int, hp: float) -> str:
    return f"{ARM_NAME}_{condition}_u{int(unit):02d}_hp{float(hp):g}"


def run_condition(cfg, sub, ports: dict, condition: str, unit: int, hp: float,
                  device, out_dir: Path) -> dict:
    """Train/evaluate ONE unit for Arm A (odor->valence).

    condition: 'connectome' | 'degree_matched' | 'generic_io'.
    unit: training-seed index (connectome/generic_io) or graph index (degree_matched).
    hp: learning rate. Idempotent (cached result.json short-circuits; train_one_run_ov resumes).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    if isinstance(device, str):
        device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available())
                              else (device if device != "cuda" else "cpu"))
    out_dir = Path(out_dir)
    run_id = _run_id(condition, unit, hp)
    run_dir = out_dir / "runs" / run_id
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    op = common.build_condition_operator(sub, condition, seed=unit)
    microsteps = int(getattr(cfg, "microsteps", DEFAULT_MICROSTEPS))
    meta = {
        "condition": condition, "arm": ARM_NAME, "run_id": run_id,
        "unit": int(unit), "graph_seed": int(unit), "train_seed": int(unit),
        "hp": float(hp), "lr": float(hp), "microsteps": microsteps,
        "substrate": getattr(cfg, "substrate", "unknown"),
        "N": int(op.shape[0]), "edges": int(op.nnz), "rho_target": common.TARGET_RHO,
    }

    if condition == "generic_io":
        # All-neuron I/O reference on the identical forward operator: does restricting I/O to
        # biological ports still bottleneck backprop on the ALIGNED task (as it did on MQAR)?
        torch.manual_seed(cfg.init_seed + unit)
        model = common.MatrixEpisodicRNN(
            recurrent=op, input_dim=cfg.odor_dim + ov.ROLE_DIMS, output_dim=cfg.n_valence,
            runtime="sparse", state_clip=cfg.state_clip, seed=cfg.init_seed + unit,
            freeze_recurrent=False)
        return common.train_one_run_ov(run_dir, model, cfg, unit, device, meta, hp)

    # connectome / degree_matched: port-gated model (seed torch BEFORE construction so the
    # readout's global-RNG-dependent init is reproducible).
    torch.manual_seed(cfg.init_seed + unit)
    model = PortGatedMatrixRNN(
        recurrent=op, ports=ports, cue_dim=cfg.odor_dim, n_valence=cfg.n_valence,
        microsteps=microsteps, state_clip=cfg.state_clip, seed=cfg.init_seed + unit)
    return common.train_one_run_ov(run_dir, model, cfg, unit, device, meta, hp)


__all__ = ["PortGatedMatrixRNN", "run_condition", "CONDITIONS", "DEFAULT_MICROSTEPS", "ARM_NAME"]
