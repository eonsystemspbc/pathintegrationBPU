#!/usr/bin/env python3
"""Size-matched operators that PRESERVE THE BIOLOGICAL PATHWAY, not just the port neurons.

Why this exists: the first 4x4 attempt capped every region to N=3499 by keeping the highest-degree
neurons. For the optic lobe that DELETED the R1-6 -> HS/VS route (the relay cells are numerous but
individually low-degree, ~3.4 partners each), leaving the readout completely unreachable (0/22 within
6 hops) and forcing AUROC=0.500. That cell was an artifact, not biology.

Here the cap is built along the actual pathway: BFS forward from the input pool and backward from the
output pool, keeping the neurons that lie ON short input->output paths first, then filling any
remaining budget by degree. Ports are always kept in full.

Emits per region into operators_pathway/<REGION>/:
    connectome.npz, degree_s{0..5}.npz, random_s{0..5}.npz  (all rescaled to rho=0.95)
    ports.json   {"input":[...],"output":[...], ...}  remapped into the capped index space
plus a manifest recording input->output reachability BEFORE and AFTER the cap (the check that the
first attempt lacked).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
from src import connectome as C  # noqa: E402

ADJ = {"AL": ROOT/"docs/results/antennal_lobe_gas/substrate/al_unsigned.npz",
       "MB": ROOT/"connectomes/flywire_mushroom_body/adjacency_unsigned.npz",
       "CX": ROOT/"connectomes/cx_polar_bump_seed0/adjacency_unsigned.npz",
       "OL": ROOT/"connectomes/flywire_optic_lobe_bpu/adjacency_unsigned.npz"}
PORTS = ROOT/"docs/results/region_task_4x4/ports"
OUT = HERE/"operators_pathway"
RHO = 0.95


def rho_of(m): return C.power_iteration_radius(sp.csr_matrix(np.abs(m.astype(np.float64))), iters=150)
def rescale(m, t=RHO):
    r = rho_of(m); return sp.csr_matrix((m*(t/r)).astype(np.float32)) if r > 1e-12 else sp.csr_matrix(m)


def load_ports(region):
    if region == "AL":
        p = json.loads((ROOT/"docs/results/antennal_lobe_gas/substrate/ports.json").read_text())
        return (sorted(set(p["orn_all"]) | set(p["trn_all"])), sorted(p["pn_all"]),
                "ORN+TRN/HRN", "ALPN")
    d = json.loads((PORTS/f"{region}.json").read_text())
    return sorted(d["input"]), sorted(d["output"]), d.get("input_pool","?"), d.get("output_pool","?")


def hops_to_output(A, inp, out, max_hop=8):
    """BFS forward along pre->post (A is W[post,pre]); returns hop at which each output is reached."""
    N = A.shape[0]; B = (A != 0).astype(np.int8)
    frontier = np.zeros(N, bool); frontier[inp] = True
    reached = frontier.copy(); hop = np.full(N, -1); hop[inp] = 0
    for k in range(1, max_hop+1):
        nxt = (B @ frontier.astype(np.int8)) > 0
        new = nxt & ~reached
        if not new.any(): break
        hop[new] = k; reached |= new; frontier = new
    return hop, int((hop[out] > 0).sum())



def _bfs_dist(mat, seed, N, k=6):
    """Hop distance from `seed` following `mat` (already oriented for the direction wanted)."""
    import numpy as _np
    f = _np.zeros(N, bool); f[seed] = True; r = f.copy(); d = _np.full(N, 99); d[seed] = 0
    for h in range(1, k + 1):
        nxt = (mat @ f.astype(_np.int8)) > 0
        new = nxt & ~r
        if not new.any(): break
        d[new] = h; r |= new; f = new
    return d


def pathway_cap(A, inp, out, n_target, max_in_frac=0.4):
    """Keep ports, then neurons ON short input->output paths, then top-degree filler."""
    N = A.shape[0]
    deg = np.asarray((A != 0).sum(0)).ravel() + np.asarray((A != 0).sum(1)).ravel()
    inp = np.asarray(inp, int); out = np.asarray(out, int)
    sub_flag = False
    cap_in = int(max_in_frac * n_target)
    if N > n_target and len(inp) > cap_in:
        # Subsample the input pool by USEFULNESS, not raw degree. Ranking by degree kept R1-6 cells
        # that are dead ends after the cap: 52% of retained inputs had out-degree 0, only 12% of
        # R1-6 out-edges survived, and the drive reaching the readout was 3.8e-8 vs 7.8e-3 for the
        # degree control — topologically reachable but numerically dead, i.e. the same artifact as
        # the original bug wearing a different mask. Rank instead by out-degree (an input that
        # cannot send is useless) and prefer inputs close to the output pool.
        Bm = (A != 0).astype(np.int8)
        outdeg_in = np.asarray(Bm.sum(0)).ravel()            # col sum = out-degree (W[post,pre])
        d_out_all = _bfs_dist(Bm.T.tocsr(), out, N, k=6)     # hops from each node TO the output pool
        alive = inp[outdeg_in[inp] > 0]
        if len(alive) >= cap_in:
            inp = alive[np.lexsort((-outdeg_in[alive], d_out_all[alive]))][:cap_in]
        else:                                                 # keep all live ones, top up by degree
            rest = np.setdiff1d(inp, alive)
            inp = np.concatenate([alive, rest[np.argsort(-deg[rest])][:cap_in - len(alive)]])
        sub_flag = True
    if N <= n_target:
        keep = np.arange(N)
    else:
        B = (A != 0).astype(np.int8)
        # forward reachable from input, and backward reachable from output (ancestors of the readout)
        def bfs(seed, mat, k=6):
            f = np.zeros(N, bool); f[seed] = True; r = f.copy(); d = np.full(N, 99); d[seed] = 0
            for h in range(1, k+1):
                nxt = (mat @ f.astype(np.int8)) > 0
                new = nxt & ~r
                if not new.any(): break
                d[new] = h; r |= new; f = new
            return d
        d_in = bfs(inp, B)             # distance from input pool (forward)
        d_out = bfs(out, B.T.tocsr())  # distance to output pool (backward)
        onpath = d_in + d_out          # small = lies on a short input->output route
        must = np.unique(np.concatenate([inp, out]))
        budget = n_target - len(must)
        others = np.setdiff1d(np.arange(N), must)
        # rank by path-centrality first, break ties by degree
        order = np.lexsort((-deg[others], onpath[others]))
        keep = np.sort(np.concatenate([must, others[order][:max(budget, 0)]]))
    old2new = -np.ones(N, np.int64); old2new[keep] = np.arange(len(keep))
    sub = A.tocsr()[keep][:, keep].tocsr(); sub.eliminate_zeros()
    rm = lambda idx: sorted(int(old2new[i]) for i in idx if 0 <= i < N and old2new[i] >= 0)
    return sub, rm(inp), rm(out), sub_flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3499)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4,5])
    ap.add_argument("--regions", nargs="+", default=["AL","MB","CX","OL"])
    a = ap.parse_args()
    man = {}
    for rk in a.regions:
        inp, out, ip, op = load_ports(rk)
        A = sp.load_npz(ADJ[rk]).tocsr().astype(np.float32)
        n0, e0 = A.shape[0], A.nnz
        _, reach_before = hops_to_output(A, inp, out)
        sub, ni, no, subf = pathway_cap(A, inp, out, a.n)
        hop_after, reach_after = hops_to_output(sub, ni, no)
        d = OUT/rk; d.mkdir(parents=True, exist_ok=True)
        sp.save_npz(d/"connectome.npz", rescale(sub))
        for s in a.seeds:
            sp.save_npz(d/f"degree_s{s}.npz", rescale(C.degree_preserving_shuffle_matrix(sub, seed=s, swap_multiplier=10)))
            sp.save_npz(d/f"random_s{s}.npz", rescale(C.random_control_matrix(sub, seed=s)))
        (d/"ports.json").write_text(json.dumps({"input": ni, "output": no,
                                                "input_pool": ip, "output_pool": op,
                                                "input_subsampled": subf}))
        # acceptance criterion must be DELIVERY, not topology: propagate unit drive from the input
        # pool and measure what actually arrives at the readout, connectome vs its own controls.
        def delivery(M, ii, oo, steps=4):
            x = np.zeros((M.shape[0], 8), np.float64); rng2 = np.random.default_rng(0)
            x[ii] = rng2.standard_normal((len(ii), 8)); x /= (np.linalg.norm(x) + 1e-12)
            best = 0.0
            for _ in range(steps):
                x = M @ x; best = max(best, float(np.abs(x[oo]).mean()))
            return best
        deliv_c = delivery(rescale(sub), ni, no)
        deliv_d = delivery(sp.load_npz(d/"degree_s0.npz").tocsr(), ni, no)
        dead_in = int((np.asarray((sub != 0).sum(0)).ravel()[np.asarray(ni, int)] == 0).sum())
        hv = hop_after[np.asarray(no, int)]; hv = hv[hv > 0]
        man[rk] = {"native_N": int(n0), "capped_N": int(sub.shape[0]), "capped_edges": int(sub.nnz),
                   "n_in": len(ni), "n_out": len(no), "input_pool": ip, "output_pool": op,
                   "outputs_reachable_full": f"{reach_before}/{len(out)}",
                   "outputs_reachable_capped": f"{reach_after}/{len(no)}",
                   "median_hops_capped": int(np.median(hv)) if len(hv) else None,
                   "readout_delivery_connectome": float(f"{deliv_c:.3e}"),
                   "readout_delivery_degree_ctrl": float(f"{deliv_d:.3e}"),
                   "delivery_ratio_con_over_deg": round(deliv_c / max(deliv_d, 1e-30), 4),
                   "dead_input_neurons": dead_in, "n_input_total": len(ni)}
        print(f"{rk}: N {n0}->{sub.shape[0]}  in={len(ni)} out={len(no)}  "
              f"reach {reach_before}/{len(out)}->{reach_after}/{len(no)} hops={man[rk]['median_hops_capped']} "
              f"| DELIVERY con={deliv_c:.2e} deg={deliv_d:.2e} ratio={deliv_c/max(deliv_d,1e-30):.3f} "
              f"dead_inputs={dead_in}/{len(ni)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
