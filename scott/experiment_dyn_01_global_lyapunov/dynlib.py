#!/usr/bin/env python3
"""dynlib.py -- shared scaffolding for Experiment dyn-01 (global Lyapunov / phase-space of the
connectome-as-RNN).

Reuses the SHARED numerical primitives from the concluded Exp-1 engine (the same bootstrap vis-01 and
Exp 2-6 use) so dyn-01's spectral rescale and its degree-matched control are BYTE-IDENTICAL to what the
task experiments ran:
  * rescale_to_rho -- power-iteration spectral rescale to rho (the connectome-vs-control matching knob)
  * mb.degree_preserving_random_like -- the genuine directed degree-preserving shuffle (the PRIMARY
    control across the whole program)

Everything dyn-01-specific (substrate registry, operator build per condition, sigma_max diagnostic)
lives here so the frozen record is self-contained. Substrates are loaded from THIS experiment's
substrate/ dir (built by build_substrates.py); dyn-01 never reads another experiment's folder at run
time. Orientation is the program-wide convention: adjacency stored POST x PRE (M[i,j]=weight j->i), so
the biologically-forward recurrence operator is M itself and rec = M @ h flows pre->post.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SUBSTRATE_DIR = HERE / "substrate"

# --- bootstrap the concluded Exp-1 engine for shared primitives (identical pattern to vis-01) --------
_EXP1 = REPO_ROOT / "scott" / "experiment_01_mb_mqar_degree_matched" / "run_experiment.py"
_spec = importlib.util.spec_from_file_location("exp1_engine", _EXP1)
exp1 = importlib.util.module_from_spec(_spec)
sys.modules["exp1_engine"] = exp1
_spec.loader.exec_module(exp1)

mb = exp1.mb                          # degree-preserving control lives here
rho_of = exp1.rho_of                 # power-iteration spectral radius
rescale_to_rho = exp1.rescale_to_rho  # (coo, target) -> (coo, raw_rho, scale)

# substrate name -> file basename under substrate/ (built by build_substrates.py)
SUBSTRATE_REGISTRY = {
    "mb_full": "mb_full_substrate.npz",
    "mb_core_alpn": "mb_core_alpn_substrate.npz",
    "ol_left": "ol_substrate.npz",
}


def load_substrate(name: str) -> sp.csr_matrix:
    """Return the raw signed/unsigned sub-adjacency M (post x pre, csr, float32) for `name`, from THIS
    experiment's substrate/ dir. Not yet rho-rescaled (that is build_operator's job)."""
    if name not in SUBSTRATE_REGISTRY:
        raise ValueError(f"unknown substrate {name!r}; known: {tuple(SUBSTRATE_REGISTRY)}")
    npz = SUBSTRATE_DIR / SUBSTRATE_REGISTRY[name]
    if not npz.exists():
        raise FileNotFoundError(
            f"substrate '{name}' not built: {npz}. Run build_substrates.py first "
            f"(uv run python scott/experiment_dyn_01_global_lyapunov/build_substrates.py).")
    return sp.load_npz(npz).tocsr().astype(np.float32)


def sigma_max_of(op: sp.spmatrix, iters: int = 120, seed: int = 0) -> float:
    """Largest singular value via power iteration on op^T op. sigma_max >> rho => a NON-NORMAL operator
    (transient growth even when asymptotically contracting) -- the diagnostic that explains an early
    bump in the Lyapunov convergence curve. Same computation as vis-01/common.sigma_max_of."""
    A = op.tocsr().astype(np.float32)
    AT = A.T.tocsr()
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(A.shape[1]).astype(np.float32)
    x /= np.linalg.norm(x) + 1e-12
    s = 0.0
    for _ in range(iters):
        y = AT @ (A @ x)
        n = float(np.linalg.norm(y))
        if n == 0:
            return 0.0
        x = y / n
        s = n
    return float(np.sqrt(s))


def degree_matched_control(M: sp.spmatrix, seed: int) -> sp.coo_matrix:
    """The PRIMARY control: genuine directed degree-preserving random rewiring (same in/out degree
    sequence + weight multiset incl. signs), via the shared Exp-1 primitive -- identical to
    vis-01/common.degree_matched and the mb-* arc."""
    return mb.degree_preserving_random_like(M.tocoo(), seed=seed)


def build_operator(M: sp.csr_matrix, condition: str, seed: int, rho: float
                   ) -> tuple[sp.coo_matrix, dict]:
    """Recurrence operator for one arm, rescaled to spectral radius `rho` (the connectome-vs-control
    matching constraint -- BOTH arms get the same rho). Returns (operator_coo, diagnostics).

    condition == 'connectome' -> M itself (rescaled).
    condition == 'degree_matched' -> a degree-preserving shuffle of M (rescaled), seeded by `seed`.
    """
    if condition == "connectome":
        base = M.tocoo().astype(np.float32)
    elif condition == "degree_matched":
        base = degree_matched_control(M, seed)
    else:
        raise ValueError(f"unknown condition {condition!r}")
    op, raw_rho, _scale = rescale_to_rho(base, rho)
    op = op.tocoo().astype(np.float32)
    diag = {
        "N": int(op.shape[0]),
        "edges": int(op.nnz),
        "raw_rho": round(float(raw_rho), 4),
        "rho_after": round(float(rho_of(op)), 4),
        "sigma_max_after": round(sigma_max_of(op), 4),
    }
    return op, diag
