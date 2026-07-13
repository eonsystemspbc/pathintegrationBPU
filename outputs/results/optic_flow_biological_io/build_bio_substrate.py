#!/usr/bin/env python3
"""Build the LEFT optic-lobe substrate + BIOLOGICALLY-CORRECT I/O ports for vis-02.

Unlike vis-01's connectivity-heuristic `assign_optic_lobe_io.py` (which INFERS ports from
graph source/sink + ROI structure), this assigns the input/output ports from the ACTUAL
FlyWire 783 cell-type identities (Schlegel et al. 2024 whole-brain annotations + Matsliah
et al. 2024 optic-lobe visual typing), joined by root_id.

  INPUT  (primary) : R1-6 photoreceptors  -- the achromatic luminance channel that drives the
                     motion pathway (R7/R8 are chromatic and excluded).
  INPUT  (fallback): L1+L2+L3 lamina monopolar cells -- first-order motion-pathway interneurons.
  OUTPUT (primary) : HS + VS lobula-plate tangential cells (LPTCs) -- the wide-field self-motion
                     / optic-flow "matched-filter" readout (Krapp & Hengstenberg 1996).
  OUTPUT (wide)    : + H2/CH/VST tangential cells.
  OUTPUT (dense)   : T4 + T5 elementary motion detectors (retinotopic local-motion field).

Substrate = the LEFT optic lobe induced subgraph of connectomes/flywire_optic_lobe_bpu
(unsigned adjacency, the exact representation the +12% data-efficiency win used), restricted
to side=='left' neurons. "Full OL" = one full optic lobe (the two lobes are ~99% independent).

Outputs (into substrate/):
  * ol_left_unsigned.npz  -- induced left-OL adjacency (CSR, same orientation as source), float32
  * root_ids_left.npy     -- left-OL root_ids in matrix-row order (join key back to FlyWire)
  * ports.json            -- {pool_name: [matrix indices]} for every candidate I/O pool
  * manifest.json         -- N, edges, raw rho, port counts + provenance
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "scripts").is_dir() and (p / "connectomes").is_dir())
CONN = ROOT / "connectomes" / "flywire_optic_lobe_bpu"
SUB = HERE / "substrate"
SUB.mkdir(parents=True, exist_ok=True)

# --- biological port definitions (exact FlyWire cell_type strings; see biology spec) -----------
# Each pool is (super_class predicate, cell_type regex). None super_class = any.
PORTS = {
    # inputs
    "in_R16":   ("sensory", r"R1-6"),
    "in_L123":  ("optic",   r"L[123]"),
    # outputs
    "out_HSVS":     ("visual_projection", r"(HS[NES]|VS[0-9]+)"),                 # textbook wide-field (strict)
    "out_LPTCwide": ("visual_projection", r"(HS[NEST]|VS[0-9]*|VSm|VST[0-9]*|H[12]|dCH|vCH|DCH|VCH|Hx)"),
    "out_T4T5":     ("optic",             r"T[45][a-d]"),                          # dense retinotopic EMD field
}


def main() -> None:
    A = sp.load_npz(CONN / "adjacency_unsigned.npz").tocsr().astype(np.float32)
    N0 = A.shape[0]
    pa = pd.read_csv(CONN / "pool_assignments.csv", usecols=["bodyId", "index"]).astype(
        {"bodyId": "int64", "index": "int64"}
    )
    ct = pd.read_csv(SUB / "celltypes_783_OL.csv").astype({"root_id": "int64"})

    # bodyId -> (matrix index, cell_type, super_class, side)
    meta = pa.merge(ct, left_on="bodyId", right_on="root_id", how="left").sort_values("index")
    assert len(meta) == N0 and (meta["index"].to_numpy() == np.arange(N0)).all(), "index misalignment"

    left_mask = (meta["side"] == "left").to_numpy()
    left_idx = np.nonzero(left_mask)[0].astype(np.int64)          # matrix rows on the left side
    A_left = A[left_idx][:, left_idx].tocsr()
    A_left.eliminate_zeros()
    old2new = -np.ones(N0, dtype=np.int64)
    old2new[left_idx] = np.arange(left_idx.size)

    root_ids_left = meta["bodyId"].to_numpy()[left_idx]
    ctype = meta["cell_type"].fillna("").astype(str).to_numpy()
    sclass = meta["super_class"].fillna("").astype(str).to_numpy()

    # degree in the induced subgraph (row = post/incoming, col = pre/outgoing)
    indeg = np.asarray((A_left != 0).sum(axis=1)).ravel()
    outdeg = np.asarray((A_left != 0).sum(axis=0)).ravel()

    ports: dict[str, list[int]] = {}
    port_report: dict[str, dict] = {}
    for name, (sc, rx) in PORTS.items():
        m = np.array([bool(re.fullmatch(rx, c)) for c in ctype])
        if sc is not None:
            m &= (sclass == sc)
        sel_old = np.nonzero(m & left_mask)[0]
        sel_new = old2new[sel_old]
        sel_new = sel_new[sel_new >= 0]
        # keep only connected port neurons (nonzero total degree) so the port can actually carry signal
        connected = sel_new[(indeg[sel_new] + outdeg[sel_new]) > 0]
        ports[name] = np.sort(connected).astype(int).tolist()
        types = sorted(set(ctype[left_idx][connected].tolist()))
        port_report[name] = {
            "super_class": sc, "regex": rx,
            "n_matched": int(sel_new.size), "n_connected": int(connected.size),
            "cell_types": types,
        }

    # raw spectral radius (power iteration on |A|) for provenance / rescale target at runtime
    rho = power_iteration_rho(A_left)

    sp.save_npz(SUB / "ol_left_unsigned.npz", A_left)
    np.save(SUB / "root_ids_left.npy", root_ids_left)
    (SUB / "ports.json").write_text(json.dumps(ports))
    manifest = {
        "substrate": "ol_left_bio_io",
        "source": "connectomes/flywire_optic_lobe_bpu/adjacency_unsigned.npz (side==left subset)",
        "release": "783", "signed": False,
        "N": int(A_left.shape[0]), "edges": int(A_left.nnz),
        "raw_spectral_radius": round(float(rho), 4),
        "orientation": "as-loaded from flywire_optic_lobe_bpu (same as the +12% data-efficiency run)",
        "celltype_source": "FlyWire 783 Schlegel-2024 annotations + Matsliah-2024 OL typing (root_id join)",
        "ports": port_report,
    }
    (SUB / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def power_iteration_rho(matrix: sp.spmatrix, iters: int = 150, seed: int = 0) -> float:
    A = sp.csr_matrix(np.abs(matrix.astype(np.float64)))
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v) + 1e-12
    lam = 0.0
    for _ in range(iters):
        w = A @ v
        nw = np.linalg.norm(w)
        if nw < 1e-30:
            return 0.0
        v = w / nw
        lam = nw
    return float(lam)


if __name__ == "__main__":
    main()
