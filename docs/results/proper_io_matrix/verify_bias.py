#!/usr/bin/env python3
"""Quantify the CONVERSE risk: do the controls have systematically shorter input->output paths
than the connectome in the capped graph? Metrics per operator:
  - direct edges input->output (1 hop) : count and weight mass
  - mean/median hop of the output pool
  - fraction of the output pool reachable at <=1, <=2, <=3 hops
  - total signal mass reaching the output pool after 1,2,3 propagation steps of |W| from a
    uniform unit input (a linear proxy for how much drive the readout actually receives)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp

ROOT = Path("/home/ec2-user/pathintegrationBPU/.claude/worktrees/diagonal-writeup")
OPS = ROOT / "docs/results/proper_io_matrix/operators_pathway"


def hops(Amat, inp, max_hop=8):
    N = Amat.shape[0]
    B = (Amat != 0).astype(np.int8).tocsr()
    f = np.zeros(N, bool); f[np.asarray(inp, int)] = True
    r = f.copy(); h = np.full(N, -1, np.int64); h[np.asarray(inp, int)] = 0
    for k in range(1, max_hop + 1):
        nxt = (B @ f.astype(np.int8)) > 0
        new = nxt & ~r
        if not new.any(): break
        h[new] = k; r |= new; f = new
    return h


def metrics(Amat, inp, out):
    inp = np.asarray(inp, int); out = np.asarray(out, int)
    A = Amat.tocsr()
    h = hops(A, inp); ho = h[out]; pos = ho[ho > 0]
    direct = A[out][:, inp]
    x = np.zeros(A.shape[0]); x[inp] = 1.0 / len(inp)
    Aa = abs(A)
    mass = []
    v = x.copy()
    for _ in range(3):
        v = Aa @ v
        mass.append(float(v[out].sum()))
    return dict(
        direct_edges=int((direct != 0).nnz), direct_mass=float(abs(direct).sum()),
        reached=int((ho > 0).sum()), n_out=int(len(out)),
        le1=int(((ho > 0) & (ho <= 1)).sum()), le2=int(((ho > 0) & (ho <= 2)).sum()),
        le3=int(((ho > 0) & (ho <= 3)).sum()),
        mean_hop=round(float(pos.mean()), 3) if pos.size else None,
        median_hop=float(np.median(pos)) if pos.size else None,
        mass1=round(mass[0], 6), mass2=round(mass[1], 6), mass3=round(mass[2], 6),
    )


def main():
    regions = sys.argv[1:] or ["AL", "MB", "CX", "OL"]
    res = {}
    for rk in regions:
        d = OPS / rk
        if not (d / "ports.json").exists():
            print(rk, "NOT BUILT"); continue
        p = json.loads((d / "ports.json").read_text())
        inp, out = p["input"], p["output"]
        res[rk] = {"n_in": len(inp), "n_out": len(out), "input_subsampled": p.get("input_subsampled")}
        for name in ["connectome"] + [f"{c}_s{s}" for c in ("degree", "random") for s in range(6)]:
            f = d / f"{name}.npz"
            if not f.exists(): continue
            res[rk][name] = metrics(sp.load_npz(f), inp, out)
        print(f"\n=== {rk}  n_in={len(inp)} n_out={len(out)} subsampled={p.get('input_subsampled')}")
        hdr = f"{'op':<12}{'direct_e':>9}{'dmass':>10}{'<=1':>6}{'<=2':>6}{'<=3':>6}{'meanhop':>9}{'mass1':>10}{'mass2':>10}{'mass3':>10}"
        print(hdr)
        for k, v in res[rk].items():
            if not isinstance(v, dict): continue
            print(f"{k:<12}{v['direct_edges']:>9}{v['direct_mass']:>10.4f}{v['le1']:>6}{v['le2']:>6}"
                  f"{v['le3']:>6}{(v['mean_hop'] if v['mean_hop'] is not None else -1):>9.3f}"
                  f"{v['mass1']:>10.5f}{v['mass2']:>10.5f}{v['mass3']:>10.5f}")
    (OPS / "bias_audit.json").write_text(json.dumps(res, indent=2))
    print("\nWROTE", OPS / "bias_audit.json")


if __name__ == "__main__":
    main()
