#!/usr/bin/env python3
"""Show the fix works: construction fix (coupling not inflated) + activation-RMS gain match put the
eigvec controls in the CORE's operational regime, where rho-matching could not.

Run: uv run python scott/experiment_02_mb_core_pruning/eigvec_solve_check.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import svds

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("exp2_engine", HERE / "run_experiment.py")
exp2 = importlib.util.module_from_spec(_spec); sys.modules["exp2_engine"] = exp2
_spec.loader.exec_module(exp2)
sys.path.insert(0, str(HERE))
import eigvec_control as ev  # noqa: E402


def sigma_max(M):
    return float(svds(M, k=1, return_singular_vectors=False)[0])


def main():
    cache = HERE / "substrate" / "schur_cache"
    base = exp2.mb.load_base_matrix(REPO / "connectomes/flywire_mushroom_body/adjacency_unsigned.npz", 0)
    core_idx = np.load(HERE / "substrate" / "core_indices.npy").astype(np.int64)
    tr = exp2.rho_of(base)
    core = exp2.build_run_matrix(base, base.tocsr(), core_idx, "core", 0, tr)[0].tocsr()

    matched = ev.eigvec_matched_matrix(core, 0, 0.95, cache)
    shuffle = ev.eigvec_shuffle_matrix(core, 0, 0.95, cache)

    core_rms = ev.activation_rms(core)
    target = core_rms["mean_rms"]
    print(f"core target mean activation-RMS = {target:.4f}\n")

    print(f"{'condition':22s} {'sigma_max':>9s} {'mean_rms':>9s} {'hmax':>7s} {'dead':>6s}  {'gain_s':>7s}")
    print(f"{'core (ref, rho=.95)':22s} {sigma_max(core):9.3f} {core_rms['mean_rms']:9.4f} "
          f"{core_rms['hmax']:7.2f} {core_rms['dead_frac']:6.2f}  {'-':>7s}")

    for name, M in [("eigvec_matched", matched), ("eigvec_shuffle", shuffle)]:
        raw = ev.activation_rms(M)
        print(f"{name+' (raw)':22s} {sigma_max(M):9.3f} {raw['mean_rms']:9.4f} "
              f"{raw['hmax']:7.2f} {raw['dead_frac']:6.2f}  {'-':>7s}")
        s = ev.match_gain_to_activation_rms(M, target)
        Ms = M * s
        got = ev.activation_rms(Ms)
        print(f"{name+' (gain-matched)':22s} {sigma_max(Ms):9.3f} {got['mean_rms']:9.4f} "
              f"{got['hmax']:7.2f} {got['dead_frac']:6.2f}  {s:7.3f}")

    print("\nGOAL: after the construction fix, eigvec_matched sigma_max is ~core-scale (not ~7.9);")
    print("after the activation-RMS match, both controls sit at the core's mean_rms / hmax / dead")
    print("regime, so a later accuracy gap reflects structure, not loudness.")


if __name__ == "__main__":
    main()
