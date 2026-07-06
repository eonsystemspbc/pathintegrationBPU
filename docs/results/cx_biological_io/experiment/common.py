#!/usr/bin/env python3
"""Shared scaffolding for Experiment 4 (biological-I/O MB on MQAR).

This module is the stable interface every arm builds against. It:
  * bootstraps sys.path and loads the Exp-1 engine (verbatim reuse of train_one_run,
    _empirical_null, MQAR task, rho/rescale, MatrixEpisodicRNN) so cross-experiment
    numbers stay comparable -- exactly as Exp 2 and Exp 3 did;
  * loads the biological substrates + port indices built by build_mb_ports.py;
  * enforces the Exp-4 orientation convention (biologically-forward recurrence: the
    adjacency is stored post x pre so M[i,j]=weight(j->i), and the forward operator is
    M ITSELF, not M^T -- see SPEC.md section 1);
  * provides MQAR->port routing, the value<->MBON codebook, degree-matched controls
    with the port sets preserved, and rho-matching.

Do NOT redefine these helpers in the arm modules; import them from here.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in [HERE, *HERE.parents] if (p / "pyproject.toml").exists())
SUBSTRATE_NPZ = HERE / "substrate" / "port_indices.npz"
DEFAULT_ADJ = REPO_ROOT / "connectomes/cx_structure_polar_observed/adjacency_unsigned.npz"

PORT_KEYS = ("alpn", "kc", "mbon", "dan", "mbin")
TARGET_RHO = 0.95  # every condition rescaled to this (Exp 1-3 convention)

# --- sys.path bootstrap so the topic-scripts cross-import (mirrors Exp 1-3) -------------
for _sub in (REPO_ROOT / "scripts").iterdir():
    if _sub.is_dir() and str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- load the Exp-1 engine as a module (verbatim reuse; identical to Exp 2/3) ----------
_EXP1 = REPO_ROOT / "scott/experiment_01_mb_mqar_degree_matched/run_experiment.py"
_spec = importlib.util.spec_from_file_location("exp1_engine", _EXP1)
exp1 = importlib.util.module_from_spec(_spec)
sys.modules["exp1_engine"] = exp1
_spec.loader.exec_module(exp1)

# reused primitives (single source of truth)
mb = exp1.mb                                   # run_mb_associative_learning (control generator)
rho_of = exp1.rho_of
rescale_to_rho = exp1.rescale_to_rho           # (coo, target) -> (coo, raw_rho, scale)
train_one_run = exp1.train_one_run             # BPTT loop w/ checkpoint/resume/grok
synthetic_matrix = exp1.synthetic_matrix
GROK_THRESHOLDS = exp1.GROK_THRESHOLDS
empirical_null = exp1._empirical_null          # permutation-null (rank primary) + MWU
ROLE_DIMS = exp1.ROLE_DIMS                      # 3: is_key, is_value, is_query
MatrixEpisodicRNN = exp1.MatrixEpisodicRNN
make_batch = exp1.make_batch
to_torch = exp1.to_torch
accuracy = exp1.accuracy
masked_ce = exp1.masked_ce
power_iteration_radius = exp1.power_iteration_radius


# --------------------------------------------------------------------------------------
# substrate + ports
# --------------------------------------------------------------------------------------
def load_substrate(name: str, adjacency: Path = DEFAULT_ADJ,
                   npz: Path = SUBSTRATE_NPZ) -> tuple[sp.csr_matrix, dict]:
    """Return (M, ports) for substrate `name` in {'core_alpn','full'}.
    M is the sub-adjacency in NATIVE orientation M[i,j] = weight(j->i) (post x pre, csr).
    ports maps each PORT_KEY -> int64 index array into the substrate's own 0..n-1 space.
    Use forward_operator(M) to get the operator the model consumes (see SPEC section 1)."""
    d = np.load(npz)
    sub_rows = d[f"{name}__sub_rows"]
    M14 = sp.load_npz(adjacency).tocsr()
    M = M14[np.ix_(sub_rows, sub_rows)].tocsr().astype(np.float32)
    ports = {k: d[f"{name}__{k}"].astype(np.int64) for k in PORT_KEYS}
    return M, ports


def synthetic_substrate(n: int = 400, seed: int = 0,
                        density: float = 0.03) -> tuple[sp.csr_matrix, dict]:
    """Small labeled substrate for CPU smoke tests (no FlyWire download).
    Partitions n neurons into the five ports with MB-like proportions."""
    M = synthetic_matrix(n, seed=seed, density=density).tocsr().astype(np.float32)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    # proportions loosely mirror the real MB core+ALPN
    n_alpn = max(4, n // 15); n_mbon = max(2, n // 60); n_dan = max(3, n // 18); n_mbin = 2
    n_kc = n - n_alpn - n_mbon - n_dan - n_mbin
    cuts = np.cumsum([n_alpn, n_kc, n_mbon, n_dan, n_mbin])
    a, k, mo, da, mi = np.split(perm, cuts[:-1])
    ports = {"alpn": np.sort(a), "kc": np.sort(k), "mbon": np.sort(mo),
             "dan": np.sort(da), "mbin": np.sort(mi)}
    return M, ports


def forward_operator(M: sp.spmatrix) -> sp.coo_matrix:
    """Biologically-forward recurrence operator (SPEC section 1).

    The adjacency is stored POST x PRE: empirically M[i,j] = weight of the synapse j->i
    (src/connectome.py builds coo((data,(post,pre))); verified against connections.csv,
    100% of edges land at M[post,pre]). The model computes rec[i] = sum_j W[i,j] h[j], so
    to drive neuron i from its PRESYNAPTIC partners j we need W[i,j] = weight(j->i) = M[i,j].
    Hence the forward operator is M ITSELF (no transpose) -- exactly what Exp 1-3 passed.
    (Transposing here would make activity flow BACKWARD along edges -- the bug caught in
    review 2026-07-02.)"""
    return M.tocoo().astype(np.float32)


def degree_matched(M: sp.spmatrix, seed: int) -> sp.coo_matrix:
    """Degree-preserving random rewiring (same in/out degree + weight multiset). Node
    identity/order is preserved, so the port index sets remain valid (fairness: controls
    share the exact ALPN/KC/MBON/DAN/MBIN ports; only the wiring differs)."""
    return mb.degree_preserving_random_like(M.tocoo(), seed=seed)


def build_condition_operator(M: sp.csr_matrix, condition: str, seed: int,
                             target_rho: float = TARGET_RHO) -> sp.coo_matrix:
    """The rho-matched, biologically-forward operator for one condition/unit.
      * 'connectome'/'generic_io' -> forward_operator(M) rescaled to target_rho.
      * 'degree_matched'          -> forward_operator(degree_matched(M, seed)) rescaled.
    Returns a coo float32 ready to hand to a model as `recurrent`."""
    if condition in ("connectome", "generic_io"):
        base = forward_operator(M)
    elif condition == "degree_matched":
        base = forward_operator(degree_matched(M, seed))
    else:
        raise ValueError(f"unknown condition {condition!r}")
    op, _raw, _scale = rescale_to_rho(base, target_rho)
    return op


# --------------------------------------------------------------------------------------
# MQAR -> port routing (SPEC section 2)
# --------------------------------------------------------------------------------------
def split_roles(inputs):
    """Split MQAR input [B,T,vocab+ROLE_DIMS] into routed drives (torch tensors):
      cue   = symbol * (is_key OR is_query)   -> goes to ALPN
      value = symbol * is_value               -> goes to DAN (teaching signal)
      is_value = the DAN/dopamine gate [B,T,1]
    vocab is inferred as inputs.shape[-1] - ROLE_DIMS."""
    vocab = inputs.shape[-1] - ROLE_DIMS
    symbol = inputs[..., :vocab]
    is_key = inputs[..., vocab + 0:vocab + 1]
    is_value = inputs[..., vocab + 1:vocab + 2]
    is_query = inputs[..., vocab + 2:vocab + 3]
    cue = symbol * (is_key + is_query)
    value = symbol * is_value
    return cue, value, is_value


def make_codebook(n_mbon: int, vocab: int, seed: int = 0):
    """Fixed random value<->MBON codebook C [n_mbon, vocab] (unit-norm columns) for the
    pure-plasticity arms: target for value v is C[:,v]; decode via matched filter
    argmax_v (C^T y)_v. Torch float32."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    C = torch.randn(n_mbon, vocab, generator=g)
    C = C / C.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return C


# --------------------------------------------------------------------------------------
# args namespace for train_one_run (Exp 1-3 defaults; override as needed)
# --------------------------------------------------------------------------------------
def make_args(**overrides) -> SimpleNamespace:
    """Build the args namespace train_one_run expects, with Exp 1-3 defaults."""
    base = dict(
        vocab_size=32, num_pairs=8, num_queries=8, reversal_pairs=0,
        epochs=300, patience=300, converge_acc=0.995,          # patience off (Exp 2-3)
        train_batches=200, val_batches=40, test_batches=100, batch_size=64,
        lr=1e-3, lr_schedule="constant", lr_min=1e-5, grad_clip=1.0,
        state_clip=0.0, init_seed=0, device="cuda",
        # plasticity-arm defaults. eta = fixed plastic rate (delta + hybrid inner; hebbian is
        # eta-invariant). elig_lambda = eligibility-trace decay: the pinned value for HYBRID; the
        # pure rules (hebbian/delta) SWEEP lambda via the hp grid, overriding this per run. 0.3 is
        # near-optimal (review 2026-07-02: lambda=0.9 roughly halved pure-arm recall).
        eta=0.3, elig_lambda=0.3,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


__all__ = [
    "REPO_ROOT", "HERE", "SUBSTRATE_NPZ", "PORT_KEYS", "TARGET_RHO",
    "mb", "rho_of", "rescale_to_rho", "train_one_run", "synthetic_matrix",
    "GROK_THRESHOLDS", "empirical_null", "ROLE_DIMS", "MatrixEpisodicRNN",
    "make_batch", "to_torch", "accuracy", "masked_ce", "power_iteration_radius",
    "load_substrate", "synthetic_substrate", "forward_operator", "degree_matched",
    "build_condition_operator", "split_roles", "make_codebook", "make_args",
]
