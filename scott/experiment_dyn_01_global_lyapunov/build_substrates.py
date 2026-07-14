#!/usr/bin/env python3
"""build_substrates.py -- build the dyn-01 substrates into THIS experiment's substrate/ dir, so the
experiment is self-contained (the probe reads only dyn-01/substrate/, never another experiment folder).

  * mb_full       : the whole 14,025-neuron FlyWire-783 mushroom-body graph (UNSIGNED -- the mb-* version
                    of record), read from the SHARED source connectomes/flywire_mushroom_body/.
  * mb_core_alpn  : the ~6,014-neuron MB core + ALPN sub-graph = the SAME node set exp-04/05/06 used
                    (KC/MBON/DAN/MBIN + ALPN), sliced by the row indices in the VENDORED
                    substrate/port_indices.npz (key core_alpn__sub_rows, copied once from exp-04 -- the
                    one-time data copy the build-experiment rule sanctions, so dyn-01 stays decoupled).
  * ol_left       : built via the copied build_ol_substrate.py (reads only the shared 783 release).
                    Heavy (reads the release feather); build with --ol.

Orientation: MB adjacency is stored POST x PRE (rec = M @ h flows pre->post) -- the program-wide
convention; no transpose. rho is left RAW here; dynlib.build_operator rescales to rho at run time.

Usage:
  uv run python scott/experiment_dyn_01_global_lyapunov/build_substrates.py         # MB (mb_full + core)
  uv run python scott/experiment_dyn_01_global_lyapunov/build_substrates.py --ol    # also build ol_left
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MB_DIR = REPO_ROOT / "connectomes" / "flywire_mushroom_body"
SUBSTRATE_DIR = HERE / "substrate"
PORT_NPZ = SUBSTRATE_DIR / "port_indices.npz"        # VENDORED (copied from exp-04), not a cross-ref


def _write(name: str, M: sp.csr_matrix, provenance: dict) -> None:
    M = M.tocsr().astype(np.float32)
    M.sum_duplicates()
    neg_frac = float((M.data < 0).mean()) if M.nnz else 0.0
    SUBSTRATE_DIR.mkdir(parents=True, exist_ok=True)
    sp.save_npz(SUBSTRATE_DIR / f"{name}_substrate.npz", M)
    manifest = {
        "substrate": name, "N": int(M.shape[0]), "edges": int(M.nnz),
        "orientation": "M[post, pre]  (rec = M @ h flows pre->post)",
        "inhibitory_edge_fraction": round(neg_frac, 4),
        "rho_target_at_runtime": "set by dynlib.build_operator (default 0.95)",
        "provenance": provenance,
    }
    (SUBSTRATE_DIR / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[build] {name}: N={M.shape[0]:,} edges={M.nnz:,} neg_frac={neg_frac:.3%} -> {name}_substrate.npz")


def build_mb() -> None:
    adj_path = MB_DIR / "adjacency_unsigned.npz"   # the mb-* version of record
    if not adj_path.exists():
        sys.exit(f"missing MB adjacency: {adj_path}")
    if not PORT_NPZ.exists():
        sys.exit(f"missing vendored port indices: {PORT_NPZ} (copy from exp-04's substrate/)")
    M14 = sp.load_npz(adj_path).tocsr().astype(np.float32)
    print(f"[build] loaded 14k MB adjacency_unsigned.npz: {M14.shape}, nnz={M14.nnz:,}")
    _write("mb_full", M14, {"node_set": "all 14,025 MB neurons (verbatim)", "signed": False,
                            "source": "connectomes/flywire_mushroom_body/adjacency_unsigned.npz"})
    rows = np.sort(np.load(PORT_NPZ)["core_alpn__sub_rows"].astype(np.int64))
    Msub = M14[np.ix_(rows, rows)]
    _write("mb_core_alpn", Msub, {"node_set": "MB core (KC/MBON/DAN/MBIN) + ALPN = core_alpn__sub_rows",
                                  "signed": False, "n_sub_rows": int(len(rows)),
                                  "port_indices_source": "vendored substrate/port_indices.npz (from exp-04)"})


def build_ol() -> int:
    print("[build] ol_left via build_ol_substrate.py (reads the shared 783 release; heavy) ...")
    return subprocess.run(["uv", "run", "python", str(HERE / "build_ol_substrate.py")],
                          cwd=str(REPO_ROOT)).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build dyn-01 substrates into substrate/.")
    ap.add_argument("--ol", action="store_true", help="also build ol_left (heavy; reads the 783 release)")
    args = ap.parse_args(argv)
    build_mb()
    if args.ol:
        return build_ol()
    print("[build] MB substrates done. Add --ol to also build the optic lobe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
