#!/usr/bin/env python3
"""Size-matched operators for the gas column of the region x task matrix.

Each region (AL / MB / CX / OL) is capped to a common N (default 3499 = the antennal lobe's native
size) by keeping the highest-total-degree neurons and taking the induced subgraph, so the 4 regions
are compared at EQUAL size -- otherwise the bigger connectome wins on capacity, not structure
(the prior grid's MQAR result showed exactly that confound). For each region we build the
connectome plus degree-preserving and edge-random controls, all rescaled to rho = 0.95.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
from src import connectome as C  # noqa: E402

CONN = ROOT / "connectomes"
AL = ROOT / "docs/results/antennal_lobe_gas/substrate"
REGIONS = {
    "AL": AL / "al_unsigned.npz",
    "MB": CONN / "flywire_mushroom_body/adjacency_unsigned.npz",
    "CX": CONN / "cx_polar_bump_seed0/adjacency_unsigned.npz",
    "OL": CONN / "flywire_optic_lobe_bpu/adjacency_unsigned.npz",
}
RHO = 0.95


def rho_of(m):
    return C.power_iteration_radius(sp.csr_matrix(np.abs(m.astype(np.float64))), iters=150)


def rescale(m, target=RHO):
    r = rho_of(m)
    return sp.csr_matrix((m * (target / r)).astype(np.float32)) if r > 1e-12 else sp.csr_matrix(m)


def cap(A, n):
    """Keep the n highest-total-degree neurons; return the induced subgraph."""
    if A.shape[0] <= n:
        return A.tocsr()
    deg = np.asarray((A != 0).sum(0)).ravel() + np.asarray((A != 0).sum(1)).ravel()
    keep = np.sort(np.argsort(-deg)[:n])
    return A.tocsr()[keep][:, keep].tocsr()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3499)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--regions", nargs="+", default=list(REGIONS))
    a = ap.parse_args()
    manifest = {}
    for rk in a.regions:
        out = HERE / "operators" / rk
        out.mkdir(parents=True, exist_ok=True)
        A = sp.load_npz(REGIONS[rk]).tocsr().astype(np.float32)
        n0, e0 = A.shape[0], A.nnz
        A = cap(A, a.n); A.eliminate_zeros()
        con = rescale(A)
        sp.save_npz(out / "connectome.npz", con)
        for s in a.seeds:
            sp.save_npz(out / f"degree_s{s}.npz",
                        rescale(C.degree_preserving_shuffle_matrix(A, seed=s, swap_multiplier=10)))
            sp.save_npz(out / f"random_s{s}.npz", rescale(C.random_control_matrix(A, seed=s)))
        manifest[rk] = {"native_N": int(n0), "native_edges": int(e0),
                        "capped_N": int(A.shape[0]), "capped_edges": int(A.nnz),
                        "rho": RHO}
        print(f"{rk}: {n0}->{A.shape[0]} neurons, {e0}->{A.nnz} edges", flush=True)
    (HERE / "operators" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
