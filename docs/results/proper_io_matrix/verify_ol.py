#!/usr/bin/env python3
"""OL-specific: full graph, OLD degree cap, NEW pathway cap -- reachability side by side."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp

ROOT = Path("/home/ec2-user/pathintegrationBPU/.claude/worktrees/diagonal-writeup")
sys.path.insert(0, str(ROOT))
from src import connectome as C  # noqa: E402

A = sp.load_npz(ROOT / "connectomes/flywire_optic_lobe_bpu/adjacency_unsigned.npz").tocsr().astype(np.float32)
p = json.loads((ROOT / "docs/results/region_task_4x4/ports/OL.json").read_text())
INP = np.array(sorted(p["input"]), int)
OUT = np.array(sorted(p["output"]), int)
N_TARGET = 3499
print(f"FULL: N={A.shape[0]} nnz={A.nnz} n_in={len(INP)} n_out={len(OUT)}", flush=True)


def bfs(Amat, seed, max_hop=8, transpose=False):
    N = Amat.shape[0]
    B = (Amat != 0).astype(np.int8)
    B = B.T.tocsr() if transpose else B.tocsr()
    f = np.zeros(N, bool); f[np.asarray(seed, int)] = True
    r = f.copy(); h = np.full(N, -1, np.int64); h[np.asarray(seed, int)] = 0
    for k in range(1, max_hop + 1):
        nxt = (B @ f.astype(np.int8)) > 0
        new = nxt & ~r
        if not new.any(): break
        h[new] = k; r |= new; f = new
    return h


def summ(tag, Amat, inp, out, max_hop=8):
    out = np.asarray(out, int)
    h = bfs(Amat, inp, max_hop)
    ho = h[out]
    cum = {k: int(((ho >= 0) & (ho <= k)).sum()) for k in range(1, max_hop + 1)}
    pos = ho[ho > 0]
    d = dict(tag=tag, N=int(Amat.shape[0]), nnz=int(Amat.nnz), n_in=int(len(inp)), n_out=int(len(out)),
             reached=int((ho > 0).sum()), cum_by_hop=cum,
             min_hop=int(pos.min()) if pos.size else None,
             median_hop=float(np.median(pos)) if pos.size else None,
             max_hop=int(pos.max()) if pos.size else None,
             frac_graph_reached=round(float((h >= 0).mean()), 4))
    print(json.dumps(d), flush=True)
    return d


res = {}
res["full"] = summ("OL full", A, INP, OUT)

deg = np.asarray((A != 0).sum(0)).ravel() + np.asarray((A != 0).sum(1)).ravel()

# ---- OLD cap: keep highest-degree N_TARGET neurons (the bug) ----
keep_old = np.sort(np.argsort(-deg)[:N_TARGET])
o2n = -np.ones(A.shape[0], np.int64); o2n[keep_old] = np.arange(len(keep_old))
sub_old = A[keep_old][:, keep_old].tocsr()
oi = np.array([o2n[i] for i in INP if o2n[i] >= 0], int)
oo = np.array([o2n[i] for i in OUT if o2n[i] >= 0], int)
print(f"OLD degree cap: inputs surviving {len(oi)}/{len(INP)}  outputs surviving {len(oo)}/{len(OUT)}", flush=True)
res["old_degree_cap"] = summ("OL old-degree-cap", sub_old, oi, oo) if len(oi) and len(oo) else {"note": "port empty"}

# ---- NEW pathway cap (replicating build_pathway_operators.pathway_cap) ----
max_in_frac = 0.4
cap_in = int(max_in_frac * N_TARGET)
inp2 = INP[np.argsort(-deg[INP])][:cap_in] if len(INP) > cap_in else INP
print(f"input subsampled: {len(INP)} -> {len(inp2)}", flush=True)
B = (A != 0).astype(np.int8).tocsr()
Bt = B.T.tocsr()


def bfs_d(seed, mat, k=6):
    Nn = mat.shape[0]
    f = np.zeros(Nn, bool); f[np.asarray(seed, int)] = True
    r = f.copy(); d = np.full(Nn, 99); d[np.asarray(seed, int)] = 0
    for h in range(1, k + 1):
        nxt = (mat @ f.astype(np.int8)) > 0
        new = nxt & ~r
        if not new.any(): break
        d[new] = h; r |= new; f = new
    return d


d_in = bfs_d(inp2, B)
d_out = bfs_d(OUT, Bt)
onpath = d_in + d_out
must = np.unique(np.concatenate([inp2, OUT]))
budget = N_TARGET - len(must)
others = np.setdiff1d(np.arange(A.shape[0]), must)
order = np.lexsort((-deg[others], onpath[others]))
keep = np.sort(np.concatenate([must, others[order][:max(budget, 0)]]))
o2n = -np.ones(A.shape[0], np.int64); o2n[keep] = np.arange(len(keep))
sub = A[keep][:, keep].tocsr(); sub.eliminate_zeros()
ni = np.array(sorted(int(o2n[i]) for i in inp2 if o2n[i] >= 0), int)
no = np.array(sorted(int(o2n[i]) for i in OUT if o2n[i] >= 0), int)
print(f"NEW pathway cap: keep={len(keep)} in={len(ni)} out={len(no)}  "
      f"onpath distribution of kept: {np.bincount(np.clip(onpath[keep],0,20)).tolist()}", flush=True)
res["pathway_cap_connectome"] = summ("OL pathway-cap connectome", sub, ni, no)

# controls on the capped graph
for s in [0, 1]:
    dm = C.degree_preserving_shuffle_matrix(sub, seed=s, swap_multiplier=10)
    res[f"pathway_cap_degree_s{s}"] = summ(f"OL pathway-cap degree_s{s}", sp.csr_matrix(dm), ni, no)
    rm = C.random_control_matrix(sub, seed=s)
    res[f"pathway_cap_random_s{s}"] = summ(f"OL pathway-cap random_s{s}", sp.csr_matrix(rm), ni, no)

(ROOT / "docs/results/proper_io_matrix/operators_pathway/ol_reach_check.json").write_text(json.dumps(res, indent=2, default=str))
print("done")
