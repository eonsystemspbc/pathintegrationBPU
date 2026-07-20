#!/usr/bin/env python3
"""A control that is degree-matched AND shortcut-matched.

WHY. Degree-preserving rewiring of a LAYERED circuit manufactures direct input->output edges the
real wiring forbids. Measured on the mushroom body (ALPN->MBON, real route ALPN->KC->MBON):
real 32 direct edges vs ~2,680 in a degree shuffle (~84x). The control gets a one-hop express lane
that skips the Kenyon-cell layer -- i.e. it skips the computation the circuit exists to do. The
handicap scales with pathway depth: AL 1.2x (1 hop), CX 21.7x (1.8 hops), MB 83.8x (2.1 hops),
OL inf (3 hops). That alone can produce "the connectome only wins on the shallow region".

THIS CONTROL removes that confound: start from a standard degree-preserving shuffle, then repair it
with degree-preserving swaps that specifically delete surplus direct input->output edges, until the
count matches the connectome's own. Degrees are preserved exactly throughout (every operation is a
double-edge swap), so it is still a strict degree control -- it just no longer gets free shortcuts.

Emits degree_sm_s{seed}.npz alongside the existing arms.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
from src import connectome as C  # noqa: E402
OPS = HERE / "operators_pathway"
RHO = 0.95


def rescale(m, t=RHO):
    r = C.power_iteration_radius(sp.csr_matrix(np.abs(m.astype(np.float64))), iters=150)
    return sp.csr_matrix((m * (t / r)).astype(np.float32)) if r > 1e-12 else sp.csr_matrix(m)


def n_direct(B, inp_arr, out_arr):
    """# edges pre in inp -> post in out, for a boolean COO.

    NB: pass numpy ARRAYS, never python sets. np.isin(x, some_set) silently returns all-False (it
    wraps the set in a 0-d object array). That bug made this return 0 for every region, so the
    'shortcut-matched' control was quietly just a plain degree shuffle."""
    inp_arr = np.asarray(sorted(inp_arr) if isinstance(inp_arr, set) else inp_arr, dtype=np.int64)
    out_arr = np.asarray(sorted(out_arr) if isinstance(out_arr, set) else out_arr, dtype=np.int64)
    r, c = np.asarray(B.row), np.asarray(B.col)
    return int(np.count_nonzero(np.isin(c, inp_arr) & np.isin(r, out_arr)))


def shortcut_match(A, inp, out, target, seed, max_iter=400000):
    """Degree-preserving double-edge swaps that reduce direct input->output edges to `target`."""
    rng = np.random.default_rng(seed)
    coo = A.tocoo()
    rows = coo.row.astype(np.int64).copy(); cols = coo.col.astype(np.int64).copy()
    data = coo.data.astype(np.float32).copy()
    E = len(rows)
    eset = set(zip(cols.tolist(), rows.tolist()))          # (pre, post)
    inp_m = np.zeros(A.shape[0], bool); inp_m[inp] = True
    out_m = np.zeros(A.shape[0], bool); out_m[out] = True
    is_sc = inp_m[cols] & out_m[rows]                       # edge is a direct in->out shortcut
    sc_idx = list(np.nonzero(is_sc)[0])
    n_sc = len(sc_idx)
    it = 0
    while n_sc > target and it < max_iter and sc_idx:
        it += 1
        a = sc_idx[rng.integers(len(sc_idx))]
        b = int(rng.integers(E))
        if b == a or is_sc[b]:
            continue
        pa, qa = cols[a], rows[a]      # pre_a -> post_a  (the shortcut)
        pb, qb = cols[b], rows[b]
        if len({pa, qa, pb, qb}) < 4:
            continue
        # swap targets: pa->qb and pb->qa  (degrees preserved exactly)
        if (pa, qb) in eset or (pb, qa) in eset:
            continue
        new_a_sc = inp_m[pa] and out_m[qb]
        new_b_sc = inp_m[pb] and out_m[qa]
        if new_a_sc:                    # would stay a shortcut; no progress
            continue
        eset.discard((pa, qa)); eset.discard((pb, qb))
        rows[a], rows[b] = qb, qa
        eset.add((pa, qb)); eset.add((pb, qa))
        is_sc[a] = new_a_sc; is_sc[b] = new_b_sc
        sc_idx.remove(a)
        if new_b_sc: sc_idx.append(b)
        n_sc = int(is_sc.sum())
    M = sp.coo_matrix((data, (rows, cols)), shape=A.shape).tocsr()
    M.eliminate_zeros()
    return M, n_sc, it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=["MB", "CX", "AL"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    a = ap.parse_args()
    rep = {}
    for rk in a.regions:
        p = json.loads((OPS / rk / "ports.json").read_text())
        inp = np.asarray(p["input"], int); out = np.asarray(p["output"], int)
        A = sp.load_npz(OPS / rk / "connectome.npz").tocsr()
        Bc = (A != 0).tocoo()
        tgt = n_direct(Bc, inp, out)
        got = []
        for s in a.seeds:
            D = C.degree_preserving_shuffle_matrix(A, seed=s, swap_multiplier=10).tocsr()
            before = n_direct((D != 0).tocoo(), inp, out)
            M, after, iters = shortcut_match(D, inp, out, tgt, seed=s)
            sp.save_npz(OPS / rk / f"degree_sm_s{s}.npz", rescale(M))
            # verify degrees are still preserved exactly
            din = np.asarray((M != 0).sum(1)).ravel(); dinD = np.asarray((D != 0).sum(1)).ravel()
            dou = np.asarray((M != 0).sum(0)).ravel(); douD = np.asarray((D != 0).sum(0)).ravel()
            ok = bool((din == dinD).all() and (dou == douD).all())
            got.append((before, after, ok))
            print(f"{rk} s{s}: direct in->out {before} -> {after} (connectome target {tgt}) "
                  f"degrees_preserved={ok} iters={iters}", flush=True)
        rep[rk] = {"connectome_direct": tgt, "per_seed": got}
    (OPS / "shortcut_match_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
