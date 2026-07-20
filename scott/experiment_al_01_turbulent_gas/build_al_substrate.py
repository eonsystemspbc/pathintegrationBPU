#!/usr/bin/env python3
"""Build the ANTENNAL-LOBE substrate from FlyWire 783 -- self-contained, no imports outside scott/.

REGION.  The antennal lobe is the fly's first olfactory relay: receptor axons arrive, local neurons
impose lateral inhibition / gain control across glomeruli, projection neurons carry the result to the
mushroom body and lateral horn. We take every proofread neuron with at least one synapse in the
AL_L / AL_R neuropils and the induced subgraph over that set -- exactly the ROI-anchored recipe
mb-01 used for the mushroom body and cx-01 used for the central complex.

WHY ROI-ANCHORED AND NOT CELL-CLASS-ANCHORED.  A prior AL study (docs/results/antennal_lobe_gas)
selected neurons by the Schlegel-2024 `cell_class` annotation (olfactory / ALLN / ALPN / thermo /
hygro) because it needed ORN-vs-LN-vs-PN identity to wire biological input and output ports. This
experiment runs GENERIC all-neuron I/O -- input to every neuron, readout from every neuron, the
regime mb-01/02/06 and cx-01 all used -- so cell identity is not needed, and the ROI recipe keeps us
on the house pattern with zero external annotation dependency. The substrate is therefore buildable
entirely from data already on disk.

SIGNS (Dale).  Per-PRESYNAPTIC-neuron, from the syn-count-weighted dominant fast transmitter across
that neuron's edges: ACh -> +1, GABA/Glu -> -1 (NT_SIGN below). Modulatory-dominant or unlabelled
presynapses default to +1. This is cx-01's `build_cx_substrate.py` logic verbatim, so the two
substrates are directly comparable. A neuron's outgoing edges therefore share one sign
(Dale-consistent), though signs are NOT constrained during training -- same as every prior
experiment.

ORIENTATION.  Stored POST x PRE: M[i, j] = weight(j -> i), so the recurrence is `rec = M @ h`
with no transpose. Matches mb-* and cx-01.

SCALING.  The matrix is saved RAW. Rescaling to rho=0.95 happens at run time in common.py, so the
same file serves the connectome arm and every control.

Outputs (into substrate/):
  al_substrate.npz   -- signed adjacency, CSR float32, M[post, pre]
  root_ids.npy       -- FlyWire root_ids in matrix-row order
  manifest.json      -- N, edges, raw rho (signed + unsigned), sign coverage, ROI provenance
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as fa
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
SUB = HERE / "substrate"

# The 783 whole-brain proofread tables already on disk (folder is named for the MB experiment that
# first downloaded them, but the feather is whole-brain: 16.8M rows, every neuropil).
FLYWIRE_DIR = REPO_ROOT / "connectomes" / "flywire_mushroom_body" / "flywire_release_783"
CONN_FEATHER = FLYWIRE_DIR / "proofread_connections_783.feather"
ROOT_IDS_NPY = FLYWIRE_DIR / "proofread_root_ids_783.npy"

AL_ROIS = ("AL_L", "AL_R")
NT_COLS = {"ach": "ach_avg", "gaba": "gaba_avg", "glut": "glut_avg"}
NT_SIGN = {"ach": 1.0, "gaba": -1.0, "glut": -1.0}


def power_iteration_rho(m: sp.spmatrix, iters: int = 200, seed: int = 0) -> float:
    """Spectral radius of |m| by power iteration (the house estimator, 200 iters)."""
    A = sp.csr_matrix(np.abs(m.astype(np.float64)))
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[0])
    v /= np.linalg.norm(v) + 1e-12
    lam = 0.0
    for _ in range(iters):
        w = A @ v
        nw = float(np.linalg.norm(w))
        if nw < 1e-30:
            return 0.0
        v = w / nw
        lam = nw
    return float(lam)


def build() -> dict:
    if not CONN_FEATHER.exists():
        raise SystemExit(
            f"missing {CONN_FEATHER}\n"
            "This is the FlyWire 783 whole-brain proofread connections table (Zenodo 10676866).\n"
            "It is the same file mb-01 and cx-01 build from; it is untracked by git."
        )
    print(f"[build] reading {CONN_FEATHER.name} (ROIs {AL_ROIS}) ...", flush=True)
    cols = ["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count",
            "ach_avg", "gaba_avg", "glut_avg"]
    with pa.memory_map(str(CONN_FEATHER), "r") as src:
        df = fa.read_table(src, columns=cols, memory_map=True).to_pandas()
    df = df[df["neuropil"].isin(AL_ROIS)].copy()
    print(f"[build] {len(df):,} synaptic connections in the AL neuropils", flush=True)

    # --- node set: neurons appearing as pre OR post on an AL edge, restricted to proofread ---
    proof = set(np.load(ROOT_IDS_NPY).astype(np.int64).tolist())
    nodes = np.union1d(df["pre_pt_root_id"].to_numpy(), df["post_pt_root_id"].to_numpy())
    nodes = np.sort(np.array([n for n in nodes.tolist() if int(n) in proof], dtype=np.int64))
    idx = {int(r): i for i, r in enumerate(nodes.tolist())}
    N = len(nodes)
    print(f"[build] N = {N:,} AL-anchored proofread neurons", flush=True)

    df = df[df["pre_pt_root_id"].isin(idx) & df["post_pt_root_id"].isin(idx)].copy()

    # --- aggregate pre->post weight = summed syn_count ---
    agg = df.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False).agg(
        weight=("syn_count", "sum"))
    pre = agg["pre_pt_root_id"].map(idx).to_numpy(np.int64)
    post = agg["post_pt_root_id"].map(idx).to_numpy(np.int64)
    w = agg["weight"].to_numpy(np.float32)
    print(f"[build] {len(w):,} aggregated pre->post edges", flush=True)

    # --- per-presynaptic-neuron sign from syn-count-weighted dominant fast NT (cx-01 logic) ---
    nt_sums = {k: df.groupby("pre_pt_root_id").apply(
        lambda g, c=NT_COLS[k]: float((g[c] * g["syn_count"]).sum()), include_groups=False)
        for k in ("ach", "gaba", "glut")}
    nt = pd.DataFrame(nt_sums)
    sign_by_pre: dict[int, float] = {}
    for root, row in nt.iterrows():
        scores = row.to_numpy(dtype=float)
        if not np.all(np.isfinite(scores)) or scores.sum() <= 0:
            continue
        sign_by_pre[int(root)] = NT_SIGN[("ach", "gaba", "glut")[int(np.argmax(scores))]]
    pre_roots = agg["pre_pt_root_id"].to_numpy()
    edge_sign = np.array([sign_by_pre.get(int(r), 1.0) for r in pre_roots], dtype=np.float32)
    sign_cov_edges = float(np.mean([int(r) in sign_by_pre for r in pre_roots]))
    neg_frac = float(np.mean(edge_sign < 0))
    print(f"[build] sign: {len(sign_by_pre):,} pre-neurons NT-labelled; {sign_cov_edges:.1%} of "
          f"edges sign-covered; {neg_frac:.1%} inhibitory", flush=True)

    # --- SIGNED adjacency, POST x PRE ---
    M = sp.coo_matrix((w * edge_sign, (post, pre)), shape=(N, N)).tocsr().astype(np.float32)
    M.sum_duplicates()
    M.eliminate_zeros()
    M_uns = sp.coo_matrix((w, (post, pre)), shape=(N, N)).tocsr().astype(np.float32)

    rho_signed = power_iteration_rho(M)
    rho_unsigned = power_iteration_rho(M_uns)

    SUB.mkdir(parents=True, exist_ok=True)
    sp.save_npz(SUB / "al_substrate.npz", M)
    np.save(SUB / "root_ids.npy", nodes)

    manifest = {
        "substrate": "antennal_lobe_flywire783",
        "selection": "ROI-anchored: proofread neurons with >=1 synapse in AL_L/AL_R, induced subgraph",
        "rois": list(AL_ROIS),
        "release": "783",
        "source_feather": str(CONN_FEATHER.relative_to(REPO_ROOT)),
        "zenodo_record": "10676866",
        "N": int(N),
        "edges": int(M.nnz),
        "total_synapses": int(w.sum()),
        "orientation": "M[post, pre] (rec = M @ h)",
        "stored": "RAW (unscaled); rho rescale happens at run time in common.py",
        "raw_spectral_radius_signed": round(rho_signed, 4),
        "raw_spectral_radius_unsigned": round(rho_unsigned, 4),
        "sign_coverage_edges": round(sign_cov_edges, 4),
        "frac_inhibitory_edges": round(neg_frac, 4),
        "n_pre_neurons_nt_labelled": int(len(sign_by_pre)),
    }
    (SUB / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    build()
