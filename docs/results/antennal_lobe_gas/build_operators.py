#!/usr/bin/env python3
"""Precompute the recurrence operators (connectome + matched controls), rescaled to rho=0.95.

Precomputing locally and staging them guarantees EVERY fleet worker and every arm/seed uses the
identical graphs, and keeps the expensive dense controls (spectrum-matched needs an O(N^3) Schur)
off the workers. Saved into substrate/operators/:

  sparse arms (edge-count / trainable-param matched):
    connectome.npz                     -- the AL connectome (signed synapse counts, W[post,pre])
    degree_s{0,1,2}.npz                -- degree-preserving rewire (SAME in/out degree sequence)
    random_s{0,1,2}.npz                -- Erdos-Renyi with the same edge count + weight multiset
  dense arms (density / dynamics reference; float16 to keep staging small):
    spectrum_s{0,1,2}.npy              -- eigenvalue-spectrum-matched (connectome spectrum, random dirs)
    dense_s{0,1,2}.npy                 -- dense Gaussian init (density confound; no connectome structure)

All are rescaled so their |.| spectral radius == 0.95 (fair dynamical conditioning). index.json
records which arms are sparse vs dense and the seed list.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / "pyproject.toml").exists()), HERE.parents[-1])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import connectome as C  # noqa: E402

SUB = HERE / "substrate"
OPS = SUB / "operators"
RHO = 0.95
SEEDS = (0, 1, 2)


def rho_of(m: sp.spmatrix) -> float:
    return C.power_iteration_radius(sp.csr_matrix(np.abs(m.astype(np.float64))), iters=200)


def rescale_sparse(m: sp.csr_matrix, target: float = RHO) -> sp.csr_matrix:
    r = rho_of(m)
    if r > 1e-12:
        m = (m * (target / r)).astype(np.float32)
    return sp.csr_matrix(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--include-eigenvector", action="store_true", help="also build eigenvector-matched control")
    a = ap.parse_args()
    OPS.mkdir(parents=True, exist_ok=True)
    A = sp.load_npz(SUB / "al_signed.npz").tocsr().astype(np.float32)
    print(f"connectome N={A.shape[0]} nnz={A.nnz} raw_rho={rho_of(A):.2f}")

    index = {"rho": RHO, "seeds": a.seeds, "sparse_arms": [], "dense_arms": [], "N": int(A.shape[0])}

    # connectome (single graph)
    con = rescale_sparse(A)
    sp.save_npz(OPS / "connectome.npz", con)
    index["sparse_arms"].append("connectome")
    print(f"  connectome rescaled rho={rho_of(con):.3f} nnz={con.nnz}")

    for s in a.seeds:
        deg = rescale_sparse(C.degree_preserving_shuffle_matrix(A, seed=s, swap_multiplier=10))
        sp.save_npz(OPS / f"degree_s{s}.npz", deg)
        rnd = rescale_sparse(C.random_control_matrix(A, seed=s))
        sp.save_npz(OPS / f"random_s{s}.npz", rnd)
        print(f"  seed {s}: degree rho={rho_of(deg):.3f} nnz={deg.nnz} | random rho={rho_of(rnd):.3f} nnz={rnd.nnz}")
    index["sparse_arms"] += ["degree", "random"]

    for s in a.seeds:
        spec = C.spectrum_matched_control_matrix(A, seed=s, mode="full", rho_target=RHO).toarray().astype(np.float16)
        np.save(OPS / f"spectrum_s{s}.npy", spec)
        dns = C.dense_random_control_matrix(A, seed=s, rho_target=RHO).toarray().astype(np.float16)
        np.save(OPS / f"dense_s{s}.npy", dns)
        print(f"  seed {s}: spectrum saved {spec.shape} | dense saved {dns.shape}")
    index["dense_arms"] += ["spectrum", "dense"]

    if a.include_eigenvector:
        for s in a.seeds:
            ev = C.eigenvector_matched_control_matrix(A, seed=s, rho_target=RHO).toarray().astype(np.float16)
            np.save(OPS / f"eigenvector_s{s}.npy", ev)
            print(f"  seed {s}: eigenvector saved {ev.shape}")
        index["dense_arms"].append("eigenvector")

    (OPS / "index.json").write_text(json.dumps(index, indent=2))
    total = sum(f.stat().st_size for f in OPS.iterdir()) / 1e6
    print(f"operators written to {OPS}  ({total:.0f} MB)  index={index}")


if __name__ == "__main__":
    main()
