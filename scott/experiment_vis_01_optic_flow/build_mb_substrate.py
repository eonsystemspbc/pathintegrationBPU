#!/usr/bin/env python3
"""Build the MUSHROOM-BODY signed substrates for Experiment vis-01 subrun 04 -- the substrate-swap
companion to subrun 03. Subrun 04 asks the SAME yaw-only learnability question, but swaps the optic
lobe for the mushroom body: can a NON-visual connectome learn instantaneous yaw from the fly-eye movie
at all, vs the SAME GRU ceiling? (Read alongside subrun 03; the GRU ceiling is substrate-independent
and shared.)

WHAT IT BUILDS (two arms, matching the prior MB experiments' node definitions exactly)
-------------------------------------------------------------------------------------
  * mb_full       : the whole 14,025-neuron FlyWire-783 mushroom-body graph
                    (connectomes/flywire_mushroom_body/adjacency_signed.npz) -- the same 14k graph the
                    mb-* arc used, taken verbatim.
  * mb_core_alpn  : the ~6,014-neuron MB core + ALPN sub-graph = the SAME node set exp-04/05/06 used
                    (KC/MBON/DAN/MBIN + ALPN), sliced out of the 14k adjacency by the row indices in
                    experiment_04_mb_biological_io/substrate/port_indices.npz (key core_alpn__sub_rows).

WHY UNSIGNED (the pinned choice): the mb-* experiments (exp-02/04/05/06) all loaded the UNSIGNED 14k
adjacency -- that is the mushroom body's VERSION OF RECORD. subrun 04 uses the SAME unsigned MB so it is
apples-to-apples with the mb-* arc (user decision 2026-07-10). This does mean the MB arm is unsigned
while subrun 03's optic lobe is signed -- a substrate difference to note when reading the two together,
but the mb-* continuity was judged the more important axis. rho is recorded RAW; the model rescales to
0.95 at run time (like the OL). (--signed rebuilds from adjacency_signed.npz instead; node sets are
identical either way.)

ORIENTATION: the MB adjacency is stored post x pre (W_rec[post_index, pre_index], rec = M @ h flows
pre->post) -- the SAME convention as build_ol_substrate.py, so no transpose is needed.

OUTPUTS (substrate/):
  * mb_full_substrate.npz        + mb_full_manifest.json
  * mb_core_alpn_substrate.npz   + mb_core_alpn_manifest.json

Usage:
  uv run python scott/experiment_vis_01_optic_flow/build_mb_substrate.py            # UNSIGNED (pinned)
  uv run python scott/experiment_vis_01_optic_flow/build_mb_substrate.py --signed   # signed variant
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MB_DIR = REPO_ROOT / "connectomes" / "flywire_mushroom_body"
PORT_NPZ = REPO_ROOT / "scott" / "experiment_04_mb_biological_io" / "substrate" / "port_indices.npz"
SUBSTRATE_DIR = HERE / "substrate"


def _power_iteration_rho(M: sp.csr_matrix, iters: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(M.shape[1]).astype(np.float32)
    x /= np.linalg.norm(x) + 1e-12
    last = 0.0
    for _ in range(iters):
        y = M @ x
        nrm = float(np.linalg.norm(y))
        if nrm == 0:
            return 0.0
        x = y / nrm
        last = nrm
    return last


def _write(name: str, M: sp.csr_matrix, body_ids: np.ndarray, provenance: dict) -> dict:
    M = M.tocsr().astype(np.float32)
    M.sum_duplicates()
    raw_rho = _power_iteration_rho(M)
    neg_frac = float((M.data < 0).mean()) if M.nnz else 0.0
    SUBSTRATE_DIR.mkdir(parents=True, exist_ok=True)
    sp.save_npz(SUBSTRATE_DIR / f"{name}_substrate.npz", M)
    np.save(SUBSTRATE_DIR / f"{name}_root_ids.npy", body_ids)
    manifest = {
        "substrate": name,
        "source": "flywire_mushroom_body (FlyWire release 783)",
        "N": int(M.shape[0]),
        "edges": int(M.nnz),
        "orientation": "M[post_index, pre_index]  (rec = M @ h flows pre->post)",
        "signed": provenance["signed"],
        "inhibitory_edge_fraction": round(neg_frac, 4),
        "raw_spectral_radius_signed": round(float(raw_rho), 4),
        "rho_target_at_runtime": 0.95,
        "weight": "summed syn_count, signed per presynaptic dominant fast NT (mb-* build)",
        "provenance": provenance,
        "pools": {},   # cell-type analysis-lens pools not used by the vis-01 learnability run
    }
    (SUBSTRATE_DIR / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[build] {name}: N={M.shape[0]:,} edges={M.nnz:,} neg_frac={neg_frac:.3%} "
          f"raw_rho={raw_rho:.4f} -> wrote {name}_substrate.npz", flush=True)
    return manifest


def build(unsigned: bool) -> None:
    adj_name = "adjacency_unsigned.npz" if unsigned else "adjacency_signed.npz"
    adj_path = MB_DIR / adj_name
    if not adj_path.exists():
        sys.exit(f"missing MB adjacency: {adj_path}")
    if not PORT_NPZ.exists():
        sys.exit(f"missing port indices: {PORT_NPZ} (from experiment_04_mb_biological_io)")

    M14 = sp.load_npz(adj_path).tocsr().astype(np.float32)
    meta = json.loads((MB_DIR / "graph_metadata.json").read_text())
    body_ids = np.asarray(meta["body_ids"], dtype=np.int64)
    assert M14.shape == (len(body_ids), len(body_ids)), "adjacency / body_ids length mismatch"
    print(f"[build] loaded 14k MB {adj_name}: {M14.shape}, nnz={M14.nnz:,}", flush=True)

    prov_common = {"signed": (not unsigned), "adjacency": adj_name}

    # arm 1: the full 14k graph, verbatim
    _write("mb_full", M14, body_ids,
           {**prov_common, "node_set": "all 14,025 MB-neuropil-anchored neurons (verbatim)"})

    # arm 2: the ~6k core+ALPN sub-graph -- the SAME node set as exp-04/05/06
    d = np.load(PORT_NPZ)
    rows = np.sort(d["core_alpn__sub_rows"].astype(np.int64))
    Msub = M14[np.ix_(rows, rows)]
    _write("mb_core_alpn", Msub, body_ids[rows],
           {**prov_common,
            "node_set": "MB core (KC/MBON/DAN/MBIN) + ALPN = core_alpn__sub_rows",
            "port_indices_source": str(PORT_NPZ.relative_to(REPO_ROOT)),
            "n_sub_rows": int(len(rows))})

    print("[build] done. Two MB substrates written to substrate/ "
          "(mb_full, mb_core_alpn).", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the vis-01 mushroom-body substrates (subrun 04).")
    ap.add_argument("--signed", action="store_true",
                    help="build from the SIGNED 14k adjacency (default is UNSIGNED, the mb-* version of record)")
    args = ap.parse_args(argv)
    build(unsigned=not args.signed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
