#!/usr/bin/env python3
"""Port-preserving, size-matched operators for the PROPER-I/O gas comparison.

The generic-I/O gas column asked "does any region's wiring help when input/output are free?"
(answer: no). This asks the sharper question: **give every region its OWN biological interface and
then compare.** MB gets ALPN->MBON, OL gets R1-6->HS/VS, CX gets its compass-in/steering-out pools,
AL gets ORN->PN — each region wired the way its brain actually wires it.

Two things have to be true for that comparison to be fair:
  1. SIZE-MATCHED. Regions differ 3.5k..97k neurons, and the prior grid showed raw capacity, not
     biology, drove its MQAR "alignment". So every region is capped to a common N.
  2. PORTS SURVIVE THE CAP. A plain top-degree cap would delete the very neurons that make the
     interface biological. So we keep ALL port neurons first, then fill the remaining budget with
     the highest-degree non-port neurons, and remap the port indices into the capped space.

Emits, per region, into operators_bioio/<REGION>/:
    connectome.npz, degree_s{k}.npz, random_s{k}.npz   (all rescaled to rho = 0.95)
    ports.json  -> {"input": [...], "output": [...]} remapped into the capped index space
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
from src import connectome as C  # noqa: E402

ADJ = {
    "AL": ROOT / "docs/results/antennal_lobe_gas/substrate/al_unsigned.npz",
    "MB": ROOT / "connectomes/flywire_mushroom_body/adjacency_unsigned.npz",
    "CX": ROOT / "connectomes/cx_polar_bump_seed0/adjacency_unsigned.npz",
    "OL": ROOT / "connectomes/flywire_optic_lobe_bpu/adjacency_unsigned.npz",
}
PORTS_DIR = HERE / "ports"
OUT = HERE / "operators_bioio"
RHO = 0.95


def rho_of(m):
    return C.power_iteration_radius(sp.csr_matrix(np.abs(m.astype(np.float64))), iters=150)


def rescale(m, target=RHO):
    r = rho_of(m)
    return sp.csr_matrix((m * (target / r)).astype(np.float32)) if r > 1e-12 else sp.csr_matrix(m)


def al_ports() -> dict:
    """AL ports come from the antennal-lobe substrate: ORNs (+ thermo/hygro) in, PNs out."""
    p = json.loads((ROOT / "docs/results/antennal_lobe_gas/substrate/ports.json").read_text())
    return {"input": sorted(set(p["orn_all"]) | set(p["trn_all"])), "output": sorted(p["pn_all"]),
            "input_pool": "ORN+TRN/HRN", "output_pool": "ALPN"}


def load_ports(region: str) -> dict:
    if region == "AL":
        return al_ports()
    f = PORTS_DIR / f"{region}.json"
    if not f.exists():
        raise FileNotFoundError(f"missing biological ports for {region}: {f}")
    d = json.loads(f.read_text())
    return {"input": sorted(d["input"]), "output": sorted(d["output"]),
            "input_pool": d.get("input_pool", "?"), "output_pool": d.get("output_pool", "?")}


def cap_preserving_ports(A: sp.csr_matrix, ports: dict, n_target: int, max_input_frac: float = 0.4):
    """Keep every port neuron, then top-degree non-port neurons up to n_target.

    If the INPUT pool alone would blow the budget (the optic lobe has 7,931 R1-6 photoreceptors vs a
    3,499 budget), subsample it by degree down to `max_input_frac` of the budget. Outputs are always
    kept in full — they are the scarce side (OL has only 22 HS/VS) and subsampling them would change
    the interface. Subsampling a large retinotopically-redundant receptor population is the standard
    concession; it is recorded in the manifest."""
    N = A.shape[0]
    deg = np.asarray((A != 0).sum(0)).ravel() + np.asarray((A != 0).sum(1)).ravel()
    inp = np.asarray(ports["input"], int); out = np.asarray(ports["output"], int)
    inp = inp[(inp >= 0) & (inp < N)]; out = out[(out >= 0) & (out < N)]
    n_in_cap = int(max_input_frac * n_target)
    subsampled = False
    if N > n_target and len(inp) > n_in_cap:
        inp = inp[np.argsort(-deg[inp])][:n_in_cap]
        subsampled = True
    ports = {**ports, "input": sorted(int(i) for i in inp)}
    keep_must = np.unique(np.concatenate([inp, out]))
    if N <= n_target:
        keep = np.arange(N)
    else:
        budget = n_target - len(keep_must)
        if budget < 0:
            raise ValueError(f"ports ({len(keep_must)}) exceed target N ({n_target})")
        deg = np.asarray((A != 0).sum(0)).ravel() + np.asarray((A != 0).sum(1)).ravel()
        others = np.setdiff1d(np.arange(N), keep_must, assume_unique=False)
        others = others[np.argsort(-deg[others])][:budget]
        keep = np.sort(np.concatenate([keep_must, others]))
    old2new = -np.ones(N, dtype=np.int64)
    old2new[keep] = np.arange(len(keep))
    sub = A.tocsr()[keep][:, keep].tocsr(); sub.eliminate_zeros()
    remap = lambda idx: sorted(int(old2new[i]) for i in idx if 0 <= i < N and old2new[i] >= 0)
    return sub, {"input": remap(ports["input"]), "output": remap(ports["output"]),
                 "input_pool": ports["input_pool"], "output_pool": ports["output_pool"],
                 "input_subsampled": bool(subsampled)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3499)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--regions", nargs="+", default=list(ADJ))
    a = ap.parse_args()
    manifest = {}
    for rk in a.regions:
        ports = load_ports(rk)
        A = sp.load_npz(ADJ[rk]).tocsr().astype(np.float32)
        n0, e0 = A.shape[0], A.nnz
        A, rp = cap_preserving_ports(A, ports, a.n)
        d = OUT / rk; d.mkdir(parents=True, exist_ok=True)
        sp.save_npz(d / "connectome.npz", rescale(A))
        for s in a.seeds:
            sp.save_npz(d / f"degree_s{s}.npz",
                        rescale(C.degree_preserving_shuffle_matrix(A, seed=s, swap_multiplier=10)))
            sp.save_npz(d / f"random_s{s}.npz", rescale(C.random_control_matrix(A, seed=s)))
        (d / "ports.json").write_text(json.dumps(rp))
        manifest[rk] = {"native_N": int(n0), "native_edges": int(e0), "capped_N": int(A.shape[0]),
                        "capped_edges": int(A.nnz), "n_input": len(rp["input"]),
                        "n_output": len(rp["output"]), "input_pool": rp["input_pool"], "input_subsampled": rp.get("input_subsampled", False),
                        "output_pool": rp["output_pool"]}
        print(f"{rk}: {n0}->{A.shape[0]} N, {e0}->{A.nnz} edges | "
              f"in={len(rp['input'])} ({rp['input_pool']}) out={len(rp['output'])} ({rp['output_pool']})",
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
