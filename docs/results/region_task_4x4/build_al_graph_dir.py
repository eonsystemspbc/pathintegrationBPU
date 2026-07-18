#!/usr/bin/env python3
"""Synthesise a src/train-compatible GRAPH DIRECTORY for the antennal lobe.

The shared harnesses (scripts/path/run_path_offdiagonal.py, etc.) take `--regions NAME:DIR`
where DIR is a prepared-connectome directory, not a bare .npz. This builds that directory for
the AL from the gas-experiment substrate:

  adjacency_unsigned.npz  <- docs/results/region_task_4x4/al_prepared_unsigned.npz  (rho = 0.95)
  pool_assignments.csv    <- substrate/ports.json  (sensory = ORN+TRN, output = ALPN, internal = LLN)
  graph_metadata.json     <- substrate/manifest.json + estimated_K from sensory->output BFS

NOTE: the RAW AL matrix (substrate/al_unsigned.npz) has rho ~2852 and NaNs immediately; only the
prepared (rho=0.95) matrix is written here.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
from src.connectome import estimate_k_from_support_sampled  # noqa: E402

SUB = ROOT / "docs/results/antennal_lobe_gas/substrate"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, default=HERE / "al_prepared_unsigned.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "connectomes/flywire_antennal_lobe")
    a = ap.parse_args()

    A = sp.load_npz(a.matrix).tocsr().astype(np.float32)
    n = int(A.shape[0])
    ports = json.loads((SUB / "ports.json").read_text())
    manifest = json.loads((SUB / "manifest.json").read_text())
    root_ids = np.load(SUB / "root_ids.npy")
    assert len(root_ids) == n, (len(root_ids), n)

    sensory = sorted(set(ports["orn_all"]) | set(ports["trn_all"]))
    output = sorted(set(ports["pn_all"]) - set(sensory))
    internal = sorted(set(range(n)) - set(sensory) - set(output))
    pool = np.empty(n, dtype=object)
    for idx, name in ((sensory, "sensory"), (output, "output"), (internal, "internal")):
        pool[np.asarray(idx, dtype=np.int64)] = name

    pools = pd.DataFrame({
        "index": np.arange(n, dtype=np.int64),
        "bodyId": root_ids.astype(np.int64),
        "pool": pool,
        "is_sensory": pool == "sensory",
        "is_internal": pool == "internal",
        "is_output": pool == "output",
    })
    k = estimate_k_from_support_sampled(A, sensory, output, max_sources=256, seed=0)

    a.out.mkdir(parents=True, exist_ok=True)
    sp.save_npz(a.out / "adjacency_unsigned.npz", A)
    pools.to_csv(a.out / "pool_assignments.csv", index=False)
    meta = {
        "N": n,
        "connectome": "flywire_antennal_lobe",
        "primary_matrix": "unsigned",
        "estimated_K": int(k),
        "orientation": manifest["orientation"],
        "rho_target": 0.95,
        "raw_primary_spectral_radius": manifest["raw_spectral_radius_unsigned"],
        "unsigned_edge_count": int(A.nnz),
        "raw_edge_count": int(manifest["edges_unsigned"]),
        "self_loop_count": int((A.diagonal() != 0).sum()),
        "pool_counts": {"sensory": len(sensory), "internal": len(internal), "output": len(output)},
        "source_matrix": str(a.matrix.relative_to(ROOT)),
        "source_substrate": manifest["source"],
        "note": "AL graph dir for the shared src/train harnesses; matrix already rescaled to rho=0.95.",
    }
    (a.out / "graph_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
