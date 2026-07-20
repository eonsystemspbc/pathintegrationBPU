#!/usr/bin/env python3
"""Independent reachability audit of the pathway-capped operators.

Orientation is W[post, pre]; a forward step along pre->post is  frontier_next = (A != 0) @ frontier.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp

ROOT = Path("/home/ec2-user/pathintegrationBPU/.claude/worktrees/diagonal-writeup")
OPS = ROOT / "docs/results/proper_io_matrix/operators_pathway"
PORTS = ROOT / "docs/results/region_task_4x4/ports"

FULL_ADJ = {
    "AL": ROOT / "docs/results/antennal_lobe_gas/substrate/al_unsigned.npz",
    "MB": ROOT / "connectomes/flywire_mushroom_body/adjacency_unsigned.npz",
    "CX": ROOT / "connectomes/cx_polar_bump_seed0/adjacency_unsigned.npz",
    "OL": ROOT / "connectomes/flywire_optic_lobe_bpu/adjacency_unsigned.npz",
}


def full_ports(region):
    if region == "AL":
        p = json.loads((ROOT / "docs/results/antennal_lobe_gas/substrate/ports.json").read_text())
        return sorted(set(p["orn_all"]) | set(p["trn_all"])), sorted(p["pn_all"])
    d = json.loads((PORTS / f"{region}.json").read_text())
    return sorted(d["input"]), sorted(d["output"])


def bfs_hops(A, inp, out, max_hop=8):
    """Return hop array (BFS distance from input pool, -1 = unreached) restricted to `out`."""
    N = A.shape[0]
    B = (A != 0).astype(np.int8).tocsr()
    frontier = np.zeros(N, bool)
    frontier[np.asarray(inp, int)] = True
    reached = frontier.copy()
    hop = np.full(N, -1, np.int64)
    hop[np.asarray(inp, int)] = 0
    for k in range(1, max_hop + 1):
        nxt = (B @ frontier.astype(np.int8)) > 0
        new = nxt & ~reached
        if not new.any():
            break
        hop[new] = k
        reached |= new
        frontier = new
    return hop


def summarize(A, inp, out, max_hop=8):
    out = np.asarray(out, int)
    hop = bfs_hops(A, inp, out, max_hop)
    ho = hop[out]
    # outputs that are themselves in the input pool count as hop 0; treat separately
    cum = {k: int(((ho >= 0) & (ho <= k)).sum()) for k in range(1, max_hop + 1)}
    pos = ho[ho > 0]
    return {
        "N": int(A.shape[0]), "nnz": int(A.nnz), "n_in": int(len(inp)), "n_out": int(len(out)),
        "reached_total": int((ho > 0).sum()),
        "cum_by_hop": cum,
        "min_hop": int(pos.min()) if pos.size else None,
        "median_hop": float(np.median(pos)) if pos.size else None,
        "max_hop_reached": int(pos.max()) if pos.size else None,
        "frac_all_reached": float((hop >= 0).mean()),
    }


def main():
    regions = sys.argv[1:] or ["AL", "MB", "CX", "OL"]
    res = {}
    for rk in regions:
        res[rk] = {}
        # FULL
        A = sp.load_npz(FULL_ADJ[rk]).tocsr()
        fi, fo = full_ports(rk)
        res[rk]["full"] = summarize(A, fi, fo)
        print(rk, "full", json.dumps(res[rk]["full"]), flush=True)
        del A
        # CAPPED
        d = OPS / rk
        pj = d / "ports.json"
        if not pj.exists():
            print(rk, "capped: NOT BUILT YET", flush=True)
            continue
        p = json.loads(pj.read_text())
        ci, co = p["input"], p["output"]
        for name in ["connectome"] + [f"{c}_s{s}" for c in ("degree", "random") for s in range(6)]:
            f = d / f"{name}.npz"
            if not f.exists():
                continue
            M = sp.load_npz(f).tocsr()
            res[rk][name] = summarize(M, ci, co)
            print(rk, name, json.dumps(res[rk][name]), flush=True)
        res[rk]["_ports_meta"] = {k: v for k, v in p.items() if k not in ("input", "output")}
    (OPS / "reach_audit.json").write_text(json.dumps(res, indent=2))
    print("WROTE", OPS / "reach_audit.json")


if __name__ == "__main__":
    main()
