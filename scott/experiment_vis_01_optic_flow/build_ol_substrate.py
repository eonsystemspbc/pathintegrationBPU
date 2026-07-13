#!/usr/bin/env python3
"""Build the SINGLE (left) optic-lobe SIGNED substrate for Experiment vis-01 from the FlyWire 783
release, plus a cell-type join used both to label the analysis-lens pools (T4/T5, photoreceptors,
HS/VS) and (optionally) to refine the substrate. Self-contained: reads only the shared 783 data files
under connectomes/flywire_mushroom_body/flywire_release_783/ and writes artifacts into this
experiment's substrate/. Does NOT import anything under scripts/flow/ or
scripts/connectome/assign_optic_lobe_io.py.

WHAT IT BUILDS
--------------
  * NODE SET     : every neuron with >=1 synapse in the LEFT optic ROIs {LA_L, ME_L, LO_L, LOP_L,
                   AME_L} (a single hemisphere -- the decision locked with the user). ~48.7k neurons.
  * EDGES        : all synapses BETWEEN those nodes that fall in the left optic ROIs, aggregated to a
                   pre->post weight = summed syn_count. ~4.2M edges.
  * SIGN         : per-PRESYNAPTIC-neuron sign from the release's per-connection neurotransmitter
                   probabilities (ACh -> +1 excitatory; GABA/Glut -> -1 inhibitory), assigned by the
                   pre neuron's syn-count-weighted dominant fast transmitter. Modulatory-dominant
                   neurons (oct/ser/da) default to +1. -> SIGNED adjacency.
  * ORIENTATION  : stored POST x PRE  (M[i,j] = weight of synapse j->i), so rec = M @ h is
                   biologically forward -- the SAME convention as Exp 4-6.
  * RHO          : the signed adjacency's raw spectral radius is recorded; the model rescales to
                   rho=0.95 at run time (common.build_condition_operator), so the saved matrix is the
                   RAW signed adjacency (not pre-rescaled) -- keeping the substrate reusable.
  * CELL-TYPE LENS: joins a FlyWire 783 cell-type annotation TSV (key `cell_type`, root_id==bodyId)
                   if one is present locally or downloadable; labels T4/T5 (motion detectors),
                   photoreceptors (R1-8), and HS/VS (lobula-plate tangential cells) into the manifest
                   pools for later analysis. If the annotation is absent AND cannot be downloaded, the
                   substrate still builds (labels = empty, 0% -- recoverable later via the join), and
                   the manifest records that cleanly.

OUTPUTS (substrate/):
  * ol_substrate.npz  -- the signed CSR adjacency (post x pre), float32.
  * root_ids.npy      -- the node root_ids in matrix-row order (the join key back to FlyWire).
  * manifest.json     -- N, edges, raw rho, ROI set, sign coverage, cell-type pool indices + counts.

Usage:
  uv run python scott/experiment_vis_01_optic_flow/build_ol_substrate.py            # build (local data)
  uv run python scott/experiment_vis_01_optic_flow/build_ol_substrate.py --annotation-tsv PATH
  uv run python scott/experiment_vis_01_optic_flow/build_ol_substrate.py --report-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.feather as fa
import pyarrow as pa
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RELEASE = REPO_ROOT / "connectomes/flywire_mushroom_body/flywire_release_783"
CONN_FEATHER = RELEASE / "proofread_connections_783.feather"
SUBSTRATE_DIR = HERE / "substrate"

LEFT_OPTIC_ROIS = ("LA_L", "ME_L", "LO_L", "LOP_L", "AME_L")   # a SINGLE hemisphere (locked decision)

# fast-transmission sign map (per presynaptic neuron); modulatory NTs default to +1.
NT_SIGN = {"ach": +1.0, "gaba": -1.0, "glut": -1.0}
NT_COLS = {"ach": "ach_avg", "gaba": "gaba_avg", "glut": "glut_avg"}

# cell-type substring patterns for the analysis-lens pools (matched case-insensitively on cell_type).
CELLTYPE_PATTERNS = {
    "T4": ("t4",),
    "T5": ("t5",),
    "photoreceptor": ("r1-6", "r7", "r8", "r1", "r2", "r3", "r4", "r5", "r6"),
    "HS": ("hs",),      # horizontal-system lobula-plate tangential cells
    "VS": ("vs",),      # vertical-system lobula-plate tangential cells
}

# candidate local locations / cache for a FlyWire 783 cell-type annotation TSV (key: root_id, cell_type)
ANNOTATION_CANDIDATES = (
    RELEASE / "cell_types_783.tsv",
    REPO_ROOT / "connectomes/flywire_mushroom_body/cell_types_783.tsv",
    Path("/tmp/flywire_cell_types_783.tsv"),
)


def _load_connections():
    """Load only the columns we need from the 783 connections feather (pre/post root_id, neuropil,
    syn_count, and the 3 fast-NT average scores), filtered to the LEFT optic ROIs."""
    cols = ["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count",
            "ach_avg", "gaba_avg", "glut_avg"]
    with pa.memory_map(str(CONN_FEATHER), "r") as src:
        tbl = fa.read_table(src, columns=cols, memory_map=True)
    df = tbl.to_pandas()
    df = df[df["neuropil"].isin(LEFT_OPTIC_ROIS)].copy()
    return df


def build(annotation_tsv: Path | None) -> dict:
    print(f"[build] reading {CONN_FEATHER.name} (left optic ROIs {LEFT_OPTIC_ROIS}) ...", flush=True)
    df = _load_connections()
    print(f"[build] {len(df):,} synaptic connections in the left optic ROIs", flush=True)

    # --- node set: every neuron appearing as pre OR post on a left-optic edge ---
    nodes = np.union1d(df["pre_pt_root_id"].to_numpy(), df["post_pt_root_id"].to_numpy())
    nodes = np.sort(nodes)
    idx = {int(r): i for i, r in enumerate(nodes.tolist())}
    N = len(nodes)
    print(f"[build] N = {N:,} neurons", flush=True)

    # --- aggregate pre->post weight = summed syn_count ---
    agg = df.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False).agg(
        weight=("syn_count", "sum"))
    pre = agg["pre_pt_root_id"].map(idx).to_numpy(np.int64)
    post = agg["post_pt_root_id"].map(idx).to_numpy(np.int64)
    w = agg["weight"].to_numpy(np.float32)
    print(f"[build] {len(w):,} aggregated pre->post edges", flush=True)

    # --- per-presynaptic-neuron sign from syn-count-weighted dominant fast NT ---
    nt = df.groupby("pre_pt_root_id").apply(
        lambda g: np.array([(g[NT_COLS[k]] * g["syn_count"]).sum() for k in ("ach", "gaba", "glut")]),
        include_groups=False)
    sign_by_pre = {}
    covered = 0
    for root, scores in nt.items():
        if not np.all(np.isfinite(scores)) or scores.sum() <= 0:
            continue
        dom = ("ach", "gaba", "glut")[int(np.argmax(scores))]
        sign_by_pre[int(root)] = NT_SIGN[dom]
        covered += 1
    # map each edge's sign from its pre neuron (default +1 if no NT info)
    pre_roots = agg["pre_pt_root_id"].to_numpy()
    edge_sign = np.array([sign_by_pre.get(int(r), 1.0) for r in pre_roots], dtype=np.float32)
    w_signed = w * edge_sign
    sign_coverage = covered / max(len(nodes), 1)
    neg_frac = float(np.mean(edge_sign < 0))
    print(f"[build] sign: {covered:,}/{N:,} pre-neurons NT-labelled ({sign_coverage:.1%}); "
          f"{neg_frac:.1%} of edges inhibitory", flush=True)

    # --- SIGNED adjacency stored POST x PRE (M[post, pre]) so rec = M @ h flows pre->post ---
    M = sp.coo_matrix((w_signed, (post, pre)), shape=(N, N)).tocsr().astype(np.float32)
    M.sum_duplicates()

    # --- raw spectral radius (power iteration; the run rescales to 0.95, so store RAW) ---
    raw_rho = _power_iteration_rho(M)
    print(f"[build] raw signed spectral radius ~= {raw_rho:.4f} (model rescales to 0.95 at run time)",
          flush=True)

    # --- cell-type analysis-lens pools ---
    pools, celltype_status = _celltype_pools(nodes, idx, annotation_tsv)

    SUBSTRATE_DIR.mkdir(parents=True, exist_ok=True)
    sp.save_npz(SUBSTRATE_DIR / "ol_substrate.npz", M)
    np.save(SUBSTRATE_DIR / "root_ids.npy", nodes)
    manifest = {
        "substrate": "ol_left",
        "release": "783",
        "left_optic_rois": list(LEFT_OPTIC_ROIS),
        "N": int(N),
        "edges": int(M.nnz),
        "orientation": "M[post_index, pre_index]  (rec = M @ h flows pre->post)",
        "signed": True,
        "sign_coverage_neurons": round(float(sign_coverage), 4),
        "inhibitory_edge_fraction": round(neg_frac, 4),
        "raw_spectral_radius_signed": round(float(raw_rho), 4),
        "rho_target_at_runtime": 0.95,
        "weight": "summed syn_count, signed by presynaptic dominant fast NT (ACh +, GABA/Glut -)",
        "celltype_join": celltype_status,
        "pools": {k: {"n": len(v)} for k, v in pools.items()},
    }
    np.savez(SUBSTRATE_DIR / "celltype_pools.npz",
             **{k: np.asarray(v, dtype=np.int64) for k, v in pools.items()})
    (SUBSTRATE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[build] wrote {SUBSTRATE_DIR/'ol_substrate.npz'} + manifest.json", flush=True)
    print(json.dumps(manifest, indent=2))
    return manifest


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


def _load_annotation(annotation_tsv: Path | None):
    """Return a DataFrame with columns [root_id, cell_type] or None. Tries the given path, then local
    candidates, then a best-effort download (skipped silently if offline)."""
    import pandas as pd
    paths = ([annotation_tsv] if annotation_tsv else []) + list(ANNOTATION_CANDIDATES)
    for p in paths:
        if p and Path(p).exists():
            try:
                df = pd.read_csv(p, sep="\t")
                col = _find_celltype_col(df)
                rid = _find_rootid_col(df)
                if col and rid:
                    print(f"[celltype] using annotation {p} (root_id={rid}, cell_type={col})", flush=True)
                    return df.rename(columns={rid: "root_id", col: "cell_type"})[["root_id", "cell_type"]]
            except Exception as e:
                print(f"[celltype] could not parse {p}: {type(e).__name__}: {e}", flush=True)
    # best-effort download (Codex/Zenodo FlyWire 783 classification). Offline -> skip cleanly.
    url = os.environ.get("FLYWIRE_CELLTYPE_TSV_URL",
                         "https://github.com/murthylab/flywire-annotations/raw/main/"
                         "supplemental_files/Supplemental_file1_neuron_annotations.tsv")
    try:
        import urllib.request
        dest = Path("/tmp/flywire_cell_types_783.tsv")
        print(f"[celltype] attempting download {url} -> {dest} ...", flush=True)
        urllib.request.urlretrieve(url, dest)
        df = pd.read_csv(dest, sep="\t")
        col = _find_celltype_col(df); rid = _find_rootid_col(df)
        if col and rid:
            return df.rename(columns={rid: "root_id", col: "cell_type"})[["root_id", "cell_type"]]
    except Exception as e:
        print(f"[celltype] download unavailable ({type(e).__name__}); building without cell-type labels",
              flush=True)
    return None


def _find_celltype_col(df):
    for c in ("cell_type", "cell_type_783", "type", "hemibrain_type", "cell_class", "class"):
        if c in df.columns:
            return c
    return None


def _find_rootid_col(df):
    for c in ("root_id", "root_id_783", "pt_root_id", "bodyId", "root_783"):
        if c in df.columns:
            return c
    return None


def _celltype_pools(nodes: np.ndarray, idx: dict, annotation_tsv: Path | None):
    ann = _load_annotation(annotation_tsv)
    pools = {k: [] for k in CELLTYPE_PATTERNS}
    if ann is None:
        return pools, {"status": "unavailable",
                       "note": "no cell-type TSV present/downloadable; substrate built by ROI only, "
                               "labels recoverable later via a root_id join (0% labelled now)."}
    ann = ann.dropna(subset=["cell_type"])
    ann["root_id"] = ann["root_id"].astype("int64", errors="ignore")
    node_set = set(int(r) for r in nodes.tolist())
    sub = ann[ann["root_id"].isin(node_set)].copy()
    sub["ct_l"] = sub["cell_type"].astype(str).str.lower()
    labelled = 0
    for pool, pats in CELLTYPE_PATTERNS.items():
        mask = sub["ct_l"].apply(lambda s: any(s.startswith(p) or s == p for p in pats))
        rids = sub.loc[mask, "root_id"].astype(int).tolist()
        pools[pool] = sorted({idx[r] for r in rids if r in idx})
        labelled += len(pools[pool])
    status = {"status": "joined", "n_annotated_nodes": int(len(sub)),
              "n_pool_labelled": int(labelled),
              "pool_counts": {k: len(v) for k, v in pools.items()}}
    print(f"[celltype] joined: {len(sub):,} annotated nodes; pools "
          f"{status['pool_counts']}", flush=True)
    return pools, status


def report_only():
    """Cheap: just report the left-optic ROI edge/neuron counts without building the adjacency."""
    df = _load_connections()
    nodes = np.union1d(df["pre_pt_root_id"].to_numpy(), df["post_pt_root_id"].to_numpy())
    print(f"left optic ROIs {LEFT_OPTIC_ROIS}: {len(df):,} connections, "
          f"{len(nodes):,} neurons, {df['syn_count'].sum():,} synapses")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the single-left-optic-lobe signed substrate.")
    ap.add_argument("--annotation-tsv", type=Path, default=None,
                    help="path to a FlyWire 783 cell-type TSV (key root_id, cell_type)")
    ap.add_argument("--report-only", action="store_true", help="print ROI counts and exit (no build)")
    args = ap.parse_args(argv)
    if not CONN_FEATHER.exists():
        sys.exit(f"missing release data: {CONN_FEATHER}")
    if args.report_only:
        report_only(); return 0
    build(args.annotation_tsv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
