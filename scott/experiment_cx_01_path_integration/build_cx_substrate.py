#!/usr/bin/env python3
"""Build the CENTRAL-COMPLEX substrate for Experiment cx-01 from the FlyWire 783 release.

Self-contained: reads only the shared 783 data files under
connectomes/flywire_mushroom_body/flywire_release_783/ and writes artifacts into this experiment's
substrate/. Does NOT import anything under src/, scripts/path/, or scripts/connectome/ -- the prior
CX work (hemibrain via neuPrint) is a separate lineage and is deliberately not reused.

WHY NOT THE EXISTING CX SUBSTRATE. The repo's prior CX graph came from hemibrain v1.2.1 over the
neuPrint API: it needs live credentials, is not pinned to a snapshot, and its metadata records
`sign_coverage: 0.0` -- hemibrain's neuPrint export carries no neurotransmitter prediction, so every
CX edge entered that model as excitatory. Building from FlyWire 783 instead gives (a) pinned local
data with no credentials, (b) real NT predictions -> a genuinely SIGNED option, and (c) the exact
same provenance + conventions as the MB (Exp 1-6) and OL (vis-01) substrates, so cross-region
numbers are comparable.

WHAT IT BUILDS
--------------
  * NODE SET   : every neuron with >=1 synapse in the CX neuropils {EB, PB, FB, NO}. N = 6,195.
                 (The CX is a midline structure, so these ROIs are unpaired -- there is no _L/_R
                 split to make, unlike vis-01's single-hemisphere optic-lobe decision.)
  * EDGES      : all synapses BETWEEN those nodes that fall in the CX neuropils, aggregated to a
                 pre->post weight = summed syn_count. ~304k edges. (Same ROI-restricted convention
                 as build_ol_substrate.py: the in-region computation, not the neurons' whole-brain
                 traffic.)
  * SIGN       : per-PRESYNAPTIC-neuron sign from the release's per-connection NT probabilities
                 (ACh -> +1; GABA/Glut -> -1), by the pre neuron's syn-count-weighted dominant fast
                 transmitter. Modulatory-dominant neurons default to +1. Identical rule to
                 build_ol_substrate.py.
  * CORE       : `cell_class == "CX"` in the Schlegel et al. 2024 FlyWire annotations -> 2,874
                 neurons. This is the direct analogue of Exp-2's MB-core prune.
  * ORIENTATION: stored POST x PRE (M[i,j] = weight of synapse j->i), so rec = M @ h is
                 biologically forward -- the SAME convention as Exp 4-6 and vis-01.
  * RHO        : the raw spectral radius is recorded; the model rescales to rho=0.95 at run time
                 (common.build_condition_operator), so the saved matrix is RAW -- keeping the
                 substrate reusable across the signed/unsigned x core/full options.

THE HALO (the Exp-2 lesson, applied up front). ROI-anchoring with no synapse threshold pulls in
passing fibres, exactly as it did for the MB. The CX-anchored 6,195 is sharply bimodal: the median
anchored neuron spends only ~3.6% of its synapses in the CX (p25 ~ 0.4%), while p75 ~ 94%. Two
independent cuts agree on where the real circuit is -- `cell_class == "CX"` gives 2,874 neurons, and
a >10%-of-synapses-in-CX threshold gives 2,978. We therefore ship BOTH the full anchored graph and
the annotation-defined core, selectable at run time. For the record, the halo is the mirror image of
Exp-2's finding: 454 Kenyon cells, 80 DAN, 20 MBON and 2,483 unlabelled fragments sit in the
CX-anchored graph, just as 639 CX neurons sat in the MB substrate.

FOUR VARIANTS FROM ONE BUILD. We save the SIGNED, FULL adjacency plus the core index vector; the
loader (common.load_substrate) derives all four signed/unsigned x core/full combinations from those
two artifacts (unsigned = |M|; core = M[core][:, core]). Nothing is rebuilt per variant, and the
saved matrix stays the single source of truth.

OUTPUTS (substrate/):
  * cx_substrate.npz   -- the SIGNED FULL CSR adjacency (post x pre), float32, N=6,195.
  * root_ids.npy       -- node root_ids in matrix-row order (the join key back to FlyWire).
  * core_indices.npy   -- int64 rows into the full matrix for the cell_class=="CX" core.
  * manifest.json      -- N, edges, raw rho (signed + unsigned, full + core), sign coverage,
                          halo composition, cell-type pool counts.
  * celltype_pools.npz -- canonical CX cell-type index pools (EPG/PEN/PFN/PFL/ER/Delta7/...) for
                          later analysis and a future biological-I/O experiment (cx-02).

Usage:
  uv run python scott/experiment_cx_01_path_integration/build_cx_substrate.py
  uv run python scott/experiment_cx_01_path_integration/build_cx_substrate.py --annotation-tsv PATH
  uv run python scott/experiment_cx_01_path_integration/build_cx_substrate.py --report-only
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as fa
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RELEASE = REPO_ROOT / "connectomes/flywire_mushroom_body/flywire_release_783"
CONN_FEATHER = RELEASE / "proofread_connections_783.feather"
ROOT_IDS_NPY = RELEASE / "proofread_root_ids_783.npy"
SUBSTRATE_DIR = HERE / "substrate"

CX_ROIS = ("EB", "PB", "FB", "NO")          # the central complex; midline/unpaired -> no _L/_R split

# fast-transmission sign map (per presynaptic neuron); modulatory NTs default to +1.
NT_SIGN = {"ach": +1.0, "gaba": -1.0, "glut": -1.0}
NT_COLS = {"ach": "ach_avg", "gaba": "gaba_avg", "glut": "glut_avg"}

CORE_CELL_CLASS = "CX"                      # Schlegel 2024 cell_class marking canonical CX neurons

# canonical CX cell-type families, for analysis pools + a later biological-I/O experiment.
CELLTYPE_PATTERNS = {
    "EPG": ("epg",),          # compass / heading bump (EB<->PB)
    "PEN": ("pen",),          # angular-velocity bump shifters
    "PFN": ("pfn",),          # translational-velocity input to the FB
    "PFL": ("pfl",),          # premotor steering output
    "PFR": ("pfr",),          # home-vector-related output
    "ER": ("er",),            # ring neurons (visual/inhibitory ring)
    "ExR": ("exr",),
    "Delta7": ("delta7",),    # global inhibition across the PB
    "hDelta": ("hdelta",),
    "vDelta": ("vdelta",),
    "FC": ("fc",),
    "FS": ("fs",),
}

# Annotation source (Schlegel et al., Nature 2024; v2.1.0 == FlyWire materialization 783):
ANNOTATION_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
                  "supplemental_files/Supplemental_file1_neuron_annotations.tsv")
ANNOTATION_CANDIDATES = (
    RELEASE / "cell_types_783.tsv",
    REPO_ROOT / "connectomes/flywire_mushroom_body/cell_types_783.tsv",
    Path("/tmp/flywire_cell_types_783.tsv"),
)


def _power_iteration_rho(M: sp.spmatrix, iters: int = 200, seed: int = 0) -> float:
    """Spectral radius by power iteration (same primitive the run-time rescale uses)."""
    A = M.tocsr().astype(np.float32)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(A.shape[0]).astype(np.float32)
    x /= np.linalg.norm(x) + 1e-12
    lam = 0.0
    for _ in range(iters):
        y = A @ x
        n = float(np.linalg.norm(y))
        if n == 0.0:
            return 0.0
        x = y / n
        lam = n
    return float(lam)


def _load_connections() -> pd.DataFrame:
    """Load only the needed columns from the 783 connections feather, filtered to the CX neuropils."""
    cols = ["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count",
            "ach_avg", "gaba_avg", "glut_avg"]
    with pa.memory_map(str(CONN_FEATHER), "r") as src:
        tbl = fa.read_table(src, columns=cols, memory_map=True)
    df = tbl.to_pandas()
    return df[df["neuropil"].isin(CX_ROIS)].copy()


def _resolve_annotation(annotation_tsv: Path | None) -> Path | None:
    """Find a local annotation TSV, else download it once to /tmp. Returns None if unavailable."""
    if annotation_tsv is not None and Path(annotation_tsv).exists():
        return Path(annotation_tsv)
    for cand in ANNOTATION_CANDIDATES:
        if cand.exists():
            return cand
    dest = Path("/tmp/flywire_cell_types_783.tsv")
    try:
        print(f"[build] downloading cell-type annotations -> {dest}", flush=True)
        urllib.request.urlretrieve(ANNOTATION_URL, dest)
        return dest
    except Exception as e:                                   # offline -> caller decides
        print(f"[build] annotation download failed ({type(e).__name__}: {e})", flush=True)
        return None


def _annotate(nodes: np.ndarray, annotation_tsv: Path | None) -> tuple[pd.DataFrame | None, dict]:
    path = _resolve_annotation(annotation_tsv)
    if path is None:
        return None, {"joined": False, "reason": "annotation TSV unavailable (offline)"}
    a = pd.read_csv(path, sep="\t", low_memory=False,
                    usecols=["root_id", "super_class", "cell_class", "cell_type", "top_nt"])
    a = a[a["root_id"].isin(set(nodes.tolist()))].drop_duplicates("root_id")
    return a, {"joined": True, "source": str(path), "url": ANNOTATION_URL,
               "matched": int(len(a)), "matched_frac": round(len(a) / max(len(nodes), 1), 4)}


def build(annotation_tsv: Path | None) -> dict:
    print(f"[build] reading {CONN_FEATHER.name} (CX ROIs {CX_ROIS}) ...", flush=True)
    df = _load_connections()
    print(f"[build] {len(df):,} synaptic connections in the CX neuropils", flush=True)

    # --- node set: every neuron appearing as pre OR post on a CX edge, restricted to proofread ---
    proof = set(np.load(ROOT_IDS_NPY).astype(np.int64).tolist())
    nodes = np.union1d(df["pre_pt_root_id"].to_numpy(), df["post_pt_root_id"].to_numpy())
    nodes = np.sort(np.array([n for n in nodes.tolist() if int(n) in proof], dtype=np.int64))
    idx = {int(r): i for i, r in enumerate(nodes.tolist())}
    N = len(nodes)
    print(f"[build] N = {N:,} CX-anchored proofread neurons", flush=True)

    df = df[df["pre_pt_root_id"].isin(idx) & df["post_pt_root_id"].isin(idx)].copy()

    # --- aggregate pre->post weight = summed syn_count ---
    agg = df.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False).agg(
        weight=("syn_count", "sum"))
    pre = agg["pre_pt_root_id"].map(idx).to_numpy(np.int64)
    post = agg["post_pt_root_id"].map(idx).to_numpy(np.int64)
    w = agg["weight"].to_numpy(np.float32)
    print(f"[build] {len(w):,} aggregated pre->post edges", flush=True)

    # --- per-presynaptic-neuron sign from syn-count-weighted dominant fast NT ---
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
    w_signed = w * edge_sign
    covered = len(sign_by_pre)
    sign_cov_edges = float(np.mean([int(r) in sign_by_pre for r in pre_roots]))
    neg_frac = float(np.mean(edge_sign < 0))
    print(f"[build] sign: {covered:,} pre-neurons NT-labelled; {sign_cov_edges:.1%} of edges "
          f"sign-covered; {neg_frac:.1%} of edges inhibitory", flush=True)

    # --- SIGNED adjacency stored POST x PRE (M[post, pre]) so rec = M @ h flows pre->post ---
    M = sp.coo_matrix((w_signed, (post, pre)), shape=(N, N)).tocsr().astype(np.float32)
    M.sum_duplicates()

    # --- cell-type join: the CX core + the canonical type pools ---
    a, join_status = _annotate(nodes, annotation_tsv)
    core_idx = np.array([], dtype=np.int64)
    pools: dict[str, list[int]] = {}
    halo: dict[str, int] = {}
    if a is not None:
        cls = a.set_index("root_id")["cell_class"]
        core_roots = cls[cls == CORE_CELL_CLASS].index.to_numpy()
        core_idx = np.sort(np.array([idx[int(r)] for r in core_roots if int(r) in idx], dtype=np.int64))
        ct = a.set_index("root_id")["cell_type"].fillna("").astype(str).str.lower()
        for pool, pats in CELLTYPE_PATTERNS.items():
            roots = ct[ct.str.startswith(pats)].index.to_numpy()
            pools[pool] = sorted(idx[int(r)] for r in roots if int(r) in idx)
        vc = a["cell_class"].value_counts(dropna=False)
        halo = {("unlabelled" if pd.isna(k) else str(k)): int(v) for k, v in vc.items()}
        print(f"[build] core (cell_class=='{CORE_CELL_CLASS}'): {len(core_idx):,} neurons", flush=True)
    else:
        print("[build] WARNING: no annotation -> core_indices empty; only full/* variants usable",
              flush=True)

    Mu = M.copy(); Mu.data = np.abs(Mu.data)                    # the unsigned variant's matrix
    rho = {"full_signed": round(_power_iteration_rho(M), 4),
           "full_unsigned": round(_power_iteration_rho(Mu), 4)}
    if core_idx.size:
        Mc = M[core_idx][:, core_idx]
        Mcu = Mc.copy(); Mcu.data = np.abs(Mcu.data)
        rho["core_signed"] = round(_power_iteration_rho(Mc), 4)
        rho["core_unsigned"] = round(_power_iteration_rho(Mcu), 4)
        core_edges = int(Mc.nnz)
    else:
        core_edges = 0
    print(f"[build] raw spectral radii (model rescales to 0.95 at run time): {rho}", flush=True)

    SUBSTRATE_DIR.mkdir(parents=True, exist_ok=True)
    sp.save_npz(SUBSTRATE_DIR / "cx_substrate.npz", M)
    np.save(SUBSTRATE_DIR / "root_ids.npy", nodes)
    np.save(SUBSTRATE_DIR / "core_indices.npy", core_idx)
    if pools:
        np.savez(SUBSTRATE_DIR / "celltype_pools.npz",
                 **{k: np.asarray(v, dtype=np.int64) for k, v in pools.items()})
    manifest = {
        "substrate": "cx_flywire783",
        "release": "783",
        "source": "FlyWire 783 proofread connections (local, pinned) -- NOT hemibrain/neuPrint",
        "cx_rois": list(CX_ROIS),
        "N_full": int(N),
        "edges_full": int(M.nnz),
        "N_core": int(core_idx.size),
        "edges_core": core_edges,
        "core_rule": f"Schlegel-2024 cell_class == '{CORE_CELL_CLASS}'",
        "orientation": "M[post_index, pre_index]  (rec = M @ h flows pre->post)",
        "stored_matrix": "SIGNED, FULL. unsigned = |M|; core = M[core][:, core] (derived at load).",
        "sign_coverage_edges": round(sign_cov_edges, 4),
        "sign_labelled_pre_neurons": int(covered),
        "inhibitory_edge_fraction": round(neg_frac, 4),
        "raw_spectral_radius": rho,
        "rho_target_at_runtime": 0.95,
        "weight": "summed syn_count, signed by presynaptic dominant fast NT (ACh +, GABA/Glut -)",
        "celltype_join": join_status,
        "halo_composition_cell_class": halo,
        "pools": {k: {"n": len(v)} for k, v in pools.items()},
    }
    (SUBSTRATE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[build] wrote {SUBSTRATE_DIR/'cx_substrate.npz'} + manifest.json", flush=True)
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotation-tsv", type=Path, default=None,
                    help="local Schlegel-2024 FlyWire annotation TSV (else auto-resolve/download)")
    ap.add_argument("--report-only", action="store_true",
                    help="print the existing manifest without rebuilding")
    args = ap.parse_args()
    if args.report_only:
        mf = SUBSTRATE_DIR / "manifest.json"
        if not mf.exists():
            print("no manifest; build first.", file=sys.stderr)
            return 1
        print(mf.read_text())
        return 0
    if not CONN_FEATHER.exists():
        print(f"missing {CONN_FEATHER} (the shared FlyWire 783 release data)", file=sys.stderr)
        return 1
    build(args.annotation_tsv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
