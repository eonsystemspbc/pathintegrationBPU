#!/usr/bin/env python3
"""Build reciprocity-matched and degree-matched controls for the AL connectome.

Source: docs/results/region_task_4x4/al_prepared_unsigned.npz (N=3499, rho=0.95, reciprocity 0.450).

  recipmatched_sK.npz : reciprocal edges KEPT FIXED, non-reciprocal edges degree-preserving
                        double-edge-swapped (Maslov-Sneppen). Matches degree AND reciprocity.
  degreematched_sK.npz: ALL edges degree-preserving double-edge-swapped. Matches degree,
                        destroys reciprocity.

Both keep the connectome's exact multiset of synapse weights and are rescaled to rho(|M|)=0.95.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
SRC = ROOT / "docs/results/region_task_4x4/al_prepared_unsigned.npz"


def rewire(rows, cols, vals, seed, mult=6):
    """Maslov-Sneppen double-edge swap; preserves in-degree and out-degree exactly."""
    rng = np.random.default_rng(seed)
    r = rows.copy().astype(np.int64)
    c = cols.copy().astype(np.int64)
    es = set(zip(r.tolist(), c.tolist()))
    m = len(r)
    n_try = int(mult * m)
    ii = rng.integers(0, m, n_try)
    jj = rng.integers(0, m, n_try)
    accepted = 0
    for k in range(n_try):
        i = int(ii[k]); j = int(jj[k])
        if i == j:
            continue
        r1 = int(r[i]); c1 = int(c[i]); r2 = int(r[j]); c2 = int(c[j])
        if len({r1, c1, r2, c2}) < 4:
            continue
        if (r1, c2) in es or (r2, c1) in es:
            continue
        es.discard((r1, c1)); es.discard((r2, c2))
        r[i] = r1; c[i] = c2; r[j] = r2; c[j] = c1
        es.add((r1, c2)); es.add((r2, c1))
        accepted += 1
    return r, c, vals, accepted


def rescale(M, rho=0.95, iters=300):
    n = M.shape[0]
    x = np.random.default_rng(0).standard_normal(n); x /= np.linalg.norm(x)
    Ma = abs(M.astype(np.float64))
    nn = 1.0
    for _ in range(iters):
        y = Ma @ x
        nn = np.linalg.norm(y)
        if nn < 1e-30:
            break
        x = y / nn
    return sp.csr_matrix((M * (rho / nn)).astype(np.float32)), nn


def stats(M):
    B = (M != 0)
    return dict(nnz=int(M.nnz), reciprocity=float(B.multiply(B.T).nnz) / max(B.nnz, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--mult", type=int, default=6)
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()

    A = sp.load_npz(SRC).tocsr().astype(np.float32)
    N = A.shape[0]
    B = (A != 0)
    coo = A.tocoo()
    is_rec = np.asarray(B.multiply(B.T)[coo.row, coo.col]).ravel() > 0
    d_out0 = np.asarray(B.sum(0)).ravel()
    d_in0 = np.asarray(B.sum(1)).ravel()
    base = stats(A)
    print(f"AL connectome: N={N} nnz={base['nnz']} reciprocity={base['reciprocity']:.4f}")

    report = {"source": str(SRC.relative_to(ROOT)), "N": N, "connectome": base, "controls": []}
    for seed in a.seeds:
        for kind in ("recipmatched", "degreematched"):
            if kind == "recipmatched":
                keep = is_rec
            else:
                keep = np.zeros(len(is_rec), dtype=bool)
            r2, c2, v2, acc = rewire(coo.row[~keep], coo.col[~keep], coo.data[~keep], seed, a.mult)
            R = np.concatenate([coo.row[keep], r2])
            C = np.concatenate([coo.col[keep], c2])
            V = np.concatenate([coo.data[keep], v2])
            M = sp.coo_matrix((V, (R, C)), shape=(N, N)).tocsr()
            M.sum_duplicates()
            M, rho_raw = rescale(M)
            Bm = (M != 0)
            d_out = np.asarray(Bm.sum(0)).ravel()
            d_in = np.asarray(Bm.sum(1)).ravel()
            st = stats(M)
            st.update(
                kind=kind, seed=seed, swaps_accepted=acc, swaps_attempted=int(a.mult * int((~keep).sum())),
                edge_ratio=st["nnz"] / base["nnz"],
                degree_corr_out=float(np.corrcoef(d_out0, d_out)[0, 1]),
                degree_corr_in=float(np.corrcoef(d_in0, d_in)[0, 1]),
                selfloops=int((M.diagonal() != 0).sum()),
                rho_before_rescale=float(rho_raw),
            )
            p = a.out / f"al_{kind}_s{seed}.npz"
            sp.save_npz(p, M)
            report["controls"].append(st)
            print(f"  {kind} s{seed}: nnz={st['nnz']} ({st['edge_ratio']:.4f}x) "
                  f"recip={st['reciprocity']:.4f} deg_corr_out={st['degree_corr_out']:.5f} "
                  f"deg_corr_in={st['degree_corr_in']:.5f} accepted={acc} -> {p.name}", flush=True)
    (a.out / "control_build_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
