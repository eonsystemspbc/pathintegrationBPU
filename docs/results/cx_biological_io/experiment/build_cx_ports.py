#!/usr/bin/env python3
"""Build the biological-I/O port definition for Experiment 6 (CX analogue of Exp 4).

Exp 4 restricted MB I/O to biological cell types (ALPN/KC/MBON/DAN/MBIN) via a FlyWire
cell_class join. Exp 6 does the SAME experiment on the central complex (CX). The CX
connectome is hemibrain (neuPrint), so cell types ship directly in neurons.csv['type']
(no FlyWire annotation download needed; the FlyWire root_id join does NOT apply to
hemibrain body IDs).

We reuse the MB engine (common/arm_bptt/arm_plasticity) VERBATIM, so the CX ports are
emitted under the SAME npz key names as the MB ports (alpn/kc/mbon/dan/mbin). The CX->MB
role analogy (documented in the manifest and the lab notebook):

    MB role (Exp 4)            CX analogue (Exp 6)                     hemibrain type(s)
    ----------------------     ----------------------------------     -----------------------------
    alpn  INPUT (odor cue)     heading + landmark/ring SENSORY input  EPG, ER*, ExR, TuBu, LNO*,
                                                                       GLNO, SpsP, IbSpsP
    kc    HIDDEN (sparse code) FB/PB integration substrate            hDelta*, vDelta*, FC*, PEN*,
                                                                       PEG*, FB*, FR*, EL*, P6*, PFG*
    mbon  OUTPUT (readout)     steering / FB premotor output          PFL*, PFR*, FS*
    dan   TEACHING (dopamine)  self-motion INSTRUCTIVE signal *       PFN*
    mbin  GAIN (APL)           global inhibition                      Delta7

* IMPORTANT DISANALOGY: the CX has NO canonical dopaminergic teaching population like the
  MB's DAN. The CX is a path-integration / vector-navigation circuit, not a dopamine-gated
  associative-learning circuit. We use PFN as the "teaching" port because PFN neurons carry
  the instructive self-motion (angular/translational velocity) signal that biologically
  drives the FB's heading-vector update -- the closest FUNCTIONAL analogue of an instructive
  signal in the CX. In the plasticity arms this port only supplies the scalar write-gate
  (is_value); its neuron identity matters only for the backprop arm's value injection. This
  substitution is itself a finding: the MB experiment's structure does not map cleanly onto
  the CX (see the lab-notebook Interpretation).

Regex families follow the repo's canonical CX roster
(scripts/connectome/report_pool_fidelity.py) and the cell-type-routed CX steering task
(scripts/path/run_cx_steering.py: EPG=heading input, FC2=goal input, PFL3=steering output).

Substrate emitted:
  * cx_full : the whole 7,349-node hemibrain CX substrate (EB/PB/FB/NO). Ports are the
              biologically-typed subsets; untyped + unassigned-typed neurons are pure
              recurrent context (no I/O, not read as the KC code) -- the Exp-4 `full`-substrate
              flavour. 40.5% of nodes are hemibrain-typed.

Outputs (staged with the code, like Exp 4):
  substrate/port_indices.npz    cx_full__sub_rows + cx_full__{alpn,kc,mbon,dan,mbin}
  substrate/port_manifest.json  human-readable roster + provenance
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
HERE = Path(__file__).resolve().parent
CX_DIR = REPO_ROOT / "connectomes/cx_structure_polar_observed"   # raw CX adj (identical across cx_* dirs)
DEFAULT_MATRIX = CX_DIR / "adjacency_unsigned.npz"
DEFAULT_META = CX_DIR / "graph_metadata.json"
DEFAULT_NEURONS = CX_DIR / "neurons.csv"

# CX cell-type -> biological role, matched on the hemibrain `type` string (checked in ORDER;
# first match wins, so ports are disjoint by construction).
ROLE_PATTERNS = [
    # OUTPUT first (PFR must beat PFN etc. is not an issue, but keep steering explicit)
    ("mbon", re.compile(r"^(PFL|PFR|FS)\d", re.I)),                 # steering + FB premotor output
    ("dan",  re.compile(r"^PFN", re.I)),                            # instructive self-motion (teaching)
    ("mbin", re.compile(r"^Delta7", re.I)),                         # global inhibition (APL analogue)
    ("alpn", re.compile(r"^(EPG|ER\d|ExR|TuBu|LNO|LCNO|GLNO|SpsP|IbSpsP)", re.I)),  # sensory/heading input
    ("kc",   re.compile(r"^(hDelta|vDelta|FC\d|PEN|PEG|FB|FR\d|EL|P6|PFG|PFGs)", re.I)),  # integration hidden
]
PORT_KEYS = ("alpn", "kc", "mbon", "dan", "mbin")


def assign_roles(types: np.ndarray) -> np.ndarray:
    roles = np.array(["none"] * len(types), dtype=object)
    for i, t in enumerate(types):
        if not isinstance(t, str):
            continue
        for role, pat in ROLE_PATTERNS:
            if pat.match(t):
                roles[i] = role
                break
    return roles


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


def main() -> int:
    meta = json.loads(DEFAULT_META.read_text())
    body_ids = np.asarray(meta["body_ids"], dtype=np.int64)
    N = len(body_ids)
    M = sp.load_npz(DEFAULT_MATRIX).tocsr()
    assert M.shape == (N, N), f"matrix {M.shape} != metadata N {N}"

    ndf = pd.read_csv(DEFAULT_NEURONS)
    assert len(ndf) == N, f"neurons.csv {len(ndf)} != N {N}"
    # neurons.csv row order == body_ids order (verified: same file provenance)
    assert np.array_equal(ndf["bodyId"].to_numpy(), body_ids), "neurons.csv row order != body_ids"
    types = ndf["type"].to_numpy()
    roles = assign_roles(types)

    ports = {k: np.sort(np.where(roles == k)[0]).astype(np.int64) for k in PORT_KEYS}
    n_typed = int(np.sum([isinstance(t, str) for t in types]))
    n_ported = int(sum(len(v) for v in ports.values()))

    # forward-pathway edge counts in the POST x PRE store (M[post,pre] = weight pre->post)
    def block_nnz(pre, post):
        return int((M[np.ix_(ports[post], ports[pre])] != 0).nnz)

    kc_mbon_edges = block_nnz("kc", "mbon")     # the plastic KC->MBON analogue (hidden->output)
    alpn_kc_edges = block_nnz("alpn", "kc")
    dan_kc_edges = block_nnz("dan", "kc")

    # weakly-connected component of the whole substrate
    nc, lab = connected_components(M + M.T, directed=False)
    largest = int(np.bincount(lab).max())

    print(f"CX substrate N={N} edges={M.nnz} rho_raw={power_iteration_rho(M):.3f} "
          f"typed={n_typed} ({100*n_typed/N:.1f}%) ported={n_ported}")
    for k in PORT_KEYS:
        ex = pd.Series(types[ports[k]]).value_counts().head(6).to_dict()
        print(f"  {k:5s} n={len(ports[k]):5d}  e.g. {ex}")
    print(f"  forward edges: alpn->kc={alpn_kc_edges}  dan->kc={dan_kc_edges}  "
          f"kc->mbon(plastic)={kc_mbon_edges}")
    print(f"  WCC largest={largest}/{N} ({100*largest/N:.1f}%)")

    # ---- validation gates (fail loudly) ----
    errs = []
    for k in ("alpn", "kc", "mbon", "dan"):
        if len(ports[k]) < 20:
            errs.append(f"port {k} too small ({len(ports[k])})")
    if len(ports["mbon"]) < 32:
        errs.append(f"output port {len(ports['mbon'])} < vocab 32 (need >= 32 for readout rank)")
    if kc_mbon_edges < 1000:
        errs.append(f"hidden->output plastic support only {kc_mbon_edges} edges")
    # disjointness (Arm A requires alpn/dan/mbon disjoint; all ports disjoint by construction)
    allidx = np.concatenate([ports[k] for k in PORT_KEYS])
    if len(allidx) != len(np.unique(allidx)):
        errs.append("ports overlap (should be disjoint)")
    if errs:
        raise SystemExit("VALIDATION FAILED:\n  " + "\n  ".join(errs))
    print("validation: OK")

    # ---- save (same npz schema as Exp 4: <substrate>__sub_rows + <substrate>__<port>) ----
    out = HERE / "substrate"
    out.mkdir(parents=True, exist_ok=True)
    name = "cx_full"
    npz = {f"{name}__sub_rows": np.arange(N, dtype=np.int64)}
    for k in PORT_KEYS:
        npz[f"{name}__{k}"] = ports[k]
    np.savez(out / "port_indices.npz", **npz)

    manifest = {
        "description": "Biological-I/O port indices for Experiment 6 (CX analogue of Exp 4). "
                       "npz keys 'cx_full__sub_rows' index the 7349-row CX adjacency; "
                       "'cx_full__<port>' index the substrate's own 0..N-1 space. Port names reuse "
                       "the MB engine's keys (alpn/kc/mbon/dan/mbin); CX role mapping below.",
        "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "connectome": "hemibrain_cx (neuPrint hemibrain:v1.2.1), ROIs EB/PB/FB/NO",
        "adjacency": str(DEFAULT_MATRIX.relative_to(REPO_ROOT)),
        "orientation": meta.get("orientation"),
        "N": int(N), "typed": n_typed, "ported": n_ported,
        "role_map": {
            "alpn(input)": "heading+landmark/ring sensory: EPG, ER*, ExR, TuBu, LNO/GLNO, SpsP, IbSpsP",
            "kc(hidden)": "FB/PB integration: hDelta*, vDelta*, FC*, PEN*, PEG*, FB*, FR*, EL, P6, PFG",
            "mbon(output)": "steering/FB premotor output: PFL*, PFR*, FS*",
            "dan(teaching)": "PFN* -- instructive self-motion; DISANALOGY: no dopaminergic teaching "
                             "population exists in the CX (see build_cx_ports docstring / lab notebook)",
            "mbin(gain)": "Delta7 -- global inhibition (APL analogue)",
        },
        "port_counts": {k: int(len(ports[k])) for k in PORT_KEYS},
        "forward_edges": {"alpn_to_kc": alpn_kc_edges, "dan_to_kc": dan_kc_edges,
                          "kc_to_mbon_plastic": kc_mbon_edges},
        "wcc_largest": largest,
        "notes": [
            "CX is hemibrain -> cell types from neurons.csv['type'] (FlyWire root_id join does NOT apply).",
            "All three connectomes/cx_* adjacency_unsigned.npz are byte-identical raw CX wiring.",
            "Prior repo CX I/O precedent: run_cx_steering.py (EPG in, PFL3 out); report_pool_fidelity.py regexes.",
        ],
    }
    (out / "port_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out/'port_indices.npz'} and {out/'port_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
