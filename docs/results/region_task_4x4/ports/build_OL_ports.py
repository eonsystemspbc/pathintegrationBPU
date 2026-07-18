#!/usr/bin/env python3
"""Canonical BIOLOGICAL I/O ports for the OL region, as indices into the FULL
connectomes/flywire_optic_lobe_bpu/adjacency_unsigned.npz (both optic lobes).

INPUT  = R1-6 photoreceptors      (achromatic luminance channel driving the motion pathway)
OUTPUT = HS + VS lobula-plate tangential cells (wide-field optic-flow matched filters)
FALLBACK OUTPUT = wider LPTC pool (HS/VS/VSm/VST + H1/H2 + DCH/VCH + named LPT tangentials)

This is the FULL-matrix analogue of docs/results/optic_flow_biological_io/build_bio_substrate.py,
which built the same ports restricted to the LEFT lobe only.

Join chain: adjacency row index -> pool_assignments.csv(bodyId,index) -> celltypes_783_OL.csv(root_id).
Writes OL.json next to this script.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "connectomes").is_dir())
CONN = ROOT / "connectomes" / "flywire_optic_lobe_bpu"
CT = ROOT / "docs" / "results" / "optic_flow_biological_io" / "substrate" / "celltypes_783_OL.csv"
OUT = HERE / "OL.json"

# (super_class predicate or None, full-match cell_type regex)
PORTS = {
    "in_R16":       ("sensory", r"R1-6"),
    "out_HSVS":     ("visual_projection", r"HS[NES]|VS[1-9][0-9]*"),
    # wider LPTC pool: HS/VS + VSm/VST + H1/H2 + CH centrifugals + named LPT tangentials.
    # LPi (lobula-plate INTRINSIC) and LPTe are deliberately excluded -- not projection outputs.
    "out_LPTCwide": (None, r"HS[NES]|VS[1-9][0-9]*|VSm|VST[0-9]+|H[12]|DCH|VCH|LPT[0-9]+(_.*)?"),
}


def main() -> None:
    A = sp.load_npz(CONN / "adjacency_unsigned.npz").tocsr()
    N = int(A.shape[0])

    pa = pd.read_csv(CONN / "pool_assignments.csv", usecols=["bodyId", "index"]).astype(
        {"bodyId": "int64", "index": "int64"}
    )
    ct = pd.read_csv(CT).astype({"root_id": "int64"})
    meta = pa.merge(ct, left_on="bodyId", right_on="root_id", how="left").sort_values("index")
    assert len(meta) == N and (meta["index"].to_numpy() == np.arange(N)).all(), "index misalignment"

    ctype = meta["cell_type"].fillna("").astype(str).to_numpy()
    sclass = meta["super_class"].fillna("").astype(str).to_numpy()
    side = meta["side"].fillna("").astype(str).to_numpy()

    indeg = np.asarray((A != 0).sum(axis=1)).ravel()
    outdeg = np.asarray((A != 0).sum(axis=0)).ravel()

    ports, report = {}, {}
    for name, (sc, rx) in PORTS.items():
        pat = re.compile(rx)
        m = np.array([bool(pat.fullmatch(c)) for c in ctype])
        if sc is not None:
            m &= sclass == sc
        sel = np.nonzero(m)[0]
        # keep only neurons that can actually carry signal in this matrix
        conn = sel[(indeg[sel] + outdeg[sel]) > 0]
        ports[name] = np.sort(conn).astype(int).tolist()
        report[name] = {
            "super_class": sc,
            "regex": rx,
            "n_matched": int(sel.size),
            "n_connected": int(conn.size),
            "n_dropped_unconnected": int(sel.size - conn.size),
            "cell_types": sorted(set(ctype[conn].tolist())),
            "by_side": {k: int(v) for k, v in pd.Series(side[conn]).value_counts().items()},
        }

    doc = {
        "region": "OL",
        "n": N,
        "input": ports["in_R16"],
        "output": ports["out_HSVS"],
        "input_pool": "R1-6 photoreceptors (FlyWire cell_type == 'R1-6', super_class == 'sensory')",
        "output_pool": "HS/VS lobula-plate tangential cells (HSN/HSE/HSS + VS1-VS8, super_class == 'visual_projection')",
        "output_fallback": ports["out_LPTCwide"],
        "output_fallback_pool": (
            "wide LPTC pool: HS/VS + VSm/VST1-2 + H1/H2 + DCH/VCH + named LPT## tangentials "
            "(lobula-plate INTRINSIC LPi and LPTe excluded)"
        ),
        "provenance": (
            "Built by docs/results/region_task_4x4/ports/build_OL_ports.py. Indices are rows of "
            "connectomes/flywire_optic_lobe_bpu/adjacency_unsigned.npz (N=%d, BOTH optic lobes). "
            "Row order taken from connectomes/flywire_optic_lobe_bpu/pool_assignments.csv "
            "(bodyId,index), joined bodyId -> root_id against "
            "docs/results/optic_flow_biological_io/substrate/celltypes_783_OL.csv "
            "(FlyWire 783: Schlegel et al. 2024 whole-brain annotations + Matsliah et al. 2024 "
            "optic-lobe visual typing). Pools selected by exact cell_type full-match regex + "
            "super_class, then filtered to neurons with nonzero in+out degree in this adjacency. "
            "The generic is_sensory/is_output ROI-flow heuristic in pool_assignments.csv was NOT "
            "used -- these are true cell-type-identified afferents (photoreceptors) and efferents "
            "(lobula-plate tangential projection neurons). Same port definitions as the LEFT-lobe "
            "builder docs/results/optic_flow_biological_io/build_bio_substrate.py, lifted to the "
            "full matrix." % N
        ),
        "caveats": (
            "(1) Severe fan-in/fan-out asymmetry: ~%d R1-6 inputs vs only %d HS/VS outputs. This is "
            "biologically correct but is exactly the readout bottleneck implicated in the OL "
            "biological-I/O stall (docs/results/optic_flow_biological_io/README.md), where the full "
            "OL connectome floored on optic flow under these ports because gradient reaching a "
            "22-cell deep readout is ~30x weaker. Use output_fallback (n=%d) if the strict pool "
            "starves training. "
            "(2) Indices span BOTH lobes; the two optic lobes are ~99%% independent, so the graph is "
            "near-block-diagonal and left/right ports are largely separate subnetworks. Prior work "
            "used the LEFT lobe only (N=48749). "
            "(3) R7/R8 (chromatic) photoreceptors and HBeyelet are excluded from the input pool by "
            "design -- R1-6 is the achromatic motion channel. "
            "(4) Adjacency orientation is as-loaded and is row=post / col=pre, confirmed empirically: "
            "HS/VS mean row-degree 887 vs col-degree 259 (integrator, as expected for an LPTC "
            "pooling T4/T5), and R1-6 col-degree 3.4 > row-degree 2.4 (source). Note a plain "
            "reachability test does NOT disambiguate orientation here -- R1-6 reaches all 22 HS/VS "
            "in 3 hops under BOTH A and A.T, because the OL graph is densely recurrent. Weights are "
            "unsigned. "
            "(5) R1-6 neurons are extremely sparsely connected in this matrix (~3.4 downstream "
            "partners each, i.e. the L1/L2/L3 lamina targets) against a global mean degree of 90.5. "
            "Input drive is therefore highly local/retinotopic and does not broadcast; any task "
            "driving this port must respect that fan-out. "
            "(6) 22 celltype rows have NaN side/super_class; none fall in these pools. "
            "(7) output_fallback / output_fallback_pool are extra keys beyond the requested schema, "
            "added per the request to record a wider LPTC fallback."
            % (len(ports["in_R16"]), len(ports["out_HSVS"]), len(ports["out_LPTCwide"]))
        ),
    }
    OUT.write_text(json.dumps(doc, indent=2))
    (HERE / "OL_build_report.json").write_text(json.dumps({"N": N, "ports": report}, indent=2))
    print(json.dumps({"N": N, "ports": report}, indent=2))


if __name__ == "__main__":
    main()
