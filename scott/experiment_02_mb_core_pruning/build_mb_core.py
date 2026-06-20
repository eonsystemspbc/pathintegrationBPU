#!/usr/bin/env python3
"""Build the MB-core substrate definition for Experiment 2.

The Exp-1 FlyWire "mushroom_body" substrate (connectomes/flywire_mushroom_body/, 14,025
neurons) is an MB-neuropil-anchored subgraph: every neuron with >=1 synapse in an MB
neuropil, with NO synapse threshold. Cell-type analysis (Schlegel et al. 2024 FlyWire
annotations, release 783) shows it is a strongly-attached ~5.6k MB core (Kenyon cells,
MBONs, DANs, MBINs/APL) embedded in an ~8.4k weakly-attached halo (central-complex
neurons, unlabeled fragments, passing fibers) whose neurons spend a median ~1.5% of their
synapses in the MB -- i.e. boundary leakage, not MB membership.

This script identifies the canonical MB core by joining the substrate's bodyIds against the
FlyWire annotation table and selecting cell_class in {Kenyon_Cell, MBON, DAN, MBIN} (APL is
annotated as MBIN, so it is included; ALPN -- the antennal-lobe olfactory *input* -- is
deliberately EXCLUDED, it is not MB-intrinsic and is reserved for the later I/O experiment).

Outputs (tracked, staged with the code to the fleet):
  substrate/core_indices.npy   int64 row indices into the 14,025-node adjacency that are MB core
  substrate/core_manifest.json human-readable definition + composition + provenance

The engine (run_experiment.py) loads the full adjacency + core_indices.npy and never needs
the annotation TSV; only this one-time prep step does.

Annotation source (Schlegel et al., Nature 2024; v2.1.0 == FlyWire materialization 783):
  https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv
Join key: annotation `root_id` == substrate `bodyId` (both are FlyWire 783 root ids).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_MATRIX = REPO_ROOT / "connectomes/flywire_mushroom_body/adjacency_unsigned.npz"
DEFAULT_META = REPO_ROOT / "connectomes/flywire_mushroom_body/graph_metadata.json"
ANNOT_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)
# MB-core cell classes (FlyWire annotation `cell_class`). APL is annotated MBIN -> included.
# ALPN (olfactory PN input) is intentionally excluded -- not MB-intrinsic.
CORE_CELL_CLASSES = ("Kenyon_Cell", "MBON", "DAN", "MBIN")


def power_iteration_rho(matrix: sp.spmatrix, iters: int = 200) -> float:
    m = matrix.tocsr().astype(float)
    if m.shape[0] == 0 or m.nnz == 0:
        return 0.0
    rng = np.random.default_rng(0)
    x = rng.random(m.shape[0])
    for _ in range(iters):
        y = m @ x
        n = float(np.linalg.norm(y))
        if n == 0:
            return 0.0
        x = y / n
    return float(np.linalg.norm(m @ x))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--annotations", type=Path, default=Path("/tmp/fw_annot.tsv"),
                    help="local path to the FlyWire annotation TSV; downloaded from ANNOT_URL if missing.")
    ap.add_argument("--out-dir", type=Path, default=HERE / "substrate")
    args = ap.parse_args(argv)

    if not args.annotations.exists():
        print(f"annotation TSV not found at {args.annotations}; downloading from\n  {ANNOT_URL}")
        args.annotations.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ANNOT_URL, args.annotations)
    print(f"annotations: {args.annotations} ({args.annotations.stat().st_size/1e6:.1f} MB)")

    meta = json.loads(args.meta.read_text())
    body_ids = np.asarray(meta["body_ids"], dtype=np.int64)   # row i of the adjacency == body_ids[i]
    N = len(body_ids)
    M = sp.load_npz(args.matrix).tocsr()
    assert M.shape == (N, N), f"matrix {M.shape} != metadata N {N}"

    ann = pd.read_csv(args.annotations, sep="\t", low_memory=False,
                      usecols=["root_id", "super_class", "cell_class", "cell_type"])
    lut = ann.set_index("root_id")
    body = pd.DataFrame({"bodyId": body_ids, "row": np.arange(N)})
    j = body.join(lut, on="bodyId")
    matched = int(j["cell_class"].notna().sum() + j["cell_class"].isna().sum())  # all rows present
    n_in_annot = int(j["super_class"].notna().sum())
    print(f"substrate N={N}; matched to annotation: {n_in_annot} ({100*n_in_annot/N:.1f}%)")

    core_mask = j["cell_class"].isin(CORE_CELL_CLASSES).to_numpy()
    core_idx = np.sort(j.loc[core_mask, "row"].to_numpy().astype(np.int64))
    ncore = len(core_idx)

    # composition + induced-subgraph stats (verification)
    comp = j.loc[core_mask, "cell_class"].value_counts().to_dict()
    sub = M[np.ix_(core_idx, core_idx)]
    nc, lab = connected_components(sub + sub.T, directed=False)
    largest = int(np.bincount(lab).max()) if ncore else 0
    rho_full = power_iteration_rho(M)
    rho_core = power_iteration_rho(sub)

    print("\n=== MB CORE ===")
    print(f"  cell classes: {CORE_CELL_CLASSES} (ALPN excluded)")
    print(f"  composition : {comp}")
    print(f"  N_core      : {ncore} / {N}  ({100*ncore/N:.1f}% of nodes)")
    print(f"  edges       : {sub.nnz} / {M.nnz}  ({100*sub.nnz/M.nnz:.1f}% of edges)")
    print(f"  components  : {nc} (largest WCC {largest}/{ncore})")
    print(f"  raw rho     : core {rho_core:.4f}  vs  full {rho_full:.4f}")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "core_indices.npy", core_idx)
    manifest = {
        "description": "MB-core node indices into connectomes/flywire_mushroom_body adjacency (row order = graph_metadata body_ids).",
        "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "annotation_source": ANNOT_URL,
        "annotation_release": "FlyWire 783 / Schlegel et al. 2024 (flywire_annotations v2.1.0)",
        "join_key": "annotation root_id == substrate bodyId",
        "core_cell_classes": list(CORE_CELL_CLASSES),
        "alpn_excluded": True,
        "n_full": N,
        "n_core": ncore,
        "core_composition": comp,
        "edges_full": int(M.nnz),
        "edges_core": int(sub.nnz),
        "edge_retention": round(sub.nnz / M.nnz, 4),
        "core_components": int(nc),
        "core_largest_wcc": largest,
        "rho_core_raw": round(rho_core, 4),
        "rho_full_raw": round(rho_full, 4),
    }
    (out / "core_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out/'core_indices.npy'} ({ncore} indices) and {out/'core_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
