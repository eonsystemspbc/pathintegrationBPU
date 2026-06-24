#!/usr/bin/env python3
"""One-time: compute & cache the real Schur of the full 14k substrate at rho=0.95, so the fleet's
full-eigvec controls never recompute the O(N^3) decomposition. Writes substrate/schur_cache/."""
import importlib.util, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("exp2", HERE/"run_experiment.py")
exp2 = importlib.util.module_from_spec(spec); sys.modules["exp2"] = exp2; spec.loader.exec_module(exp2)
from src.connectome import _real_schur_cached, _matrix_fingerprint
base = exp2.mb.load_base_matrix(REPO/"connectomes/flywire_mushroom_body/adjacency_unsigned.npz", 0)
tr = exp2.rho_of(base)
full = exp2.build_run_matrix(base, base.tocsr(), np.array([0]), "full", 0, tr)[0].tocsr()
print(f"full N={full.shape[0]} nnz={full.nnz} rho={tr:.4f} fingerprint={_matrix_fingerprint(full)}", flush=True)
t0 = time.time()
t, z = _real_schur_cached(full, schur_cache=HERE/"substrate"/"schur_cache", want_z=True)
print(f"Schur done in {(time.time()-t0)/60:.1f} min; T{t.shape} Z{z.shape} staged.", flush=True)
