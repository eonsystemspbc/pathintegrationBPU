#!/usr/bin/env python3
"""Build the biological-I/O port definition for Experiment 4.

Experiments 1-3 injected task input into, and read output from, ALL neurons (generic
all-neuron I/O), so a trainable readout could route around the wiring. Experiment 4
restricts I/O to the biologically-correct mushroom-body neurons, identified by the
FlyWire/Schlegel-2024 cell-type annotation join (the same table Exp 2's build_mb_core.py
uses; join key annotation root_id == substrate bodyId, 100% matched):

    INPUT  (odor / CS)      = ALPN  (antennal-lobe projection neurons)   406
    HIDDEN (sparse code)    = Kenyon_Cell                                5177
    OUTPUT (readout)        = MBON  (mushroom-body output neurons)         96
    LEARNING (teaching)     = DAN   (dopaminergic neurons)               331
    GAIN CONTROL            = MBIN  (incl. APL)                            4

Why cell_class and not the alternatives (see the Exp-4 lab-notebook methods):
  * predictedNt (neurons.csv) has ZERO dopamine labels -> cannot identify DANs.
  * The native ROI-flow pools (src/pools.py) put ALPN and DAN both in "sensory"
    (cannot separate odor input from the teaching signal) and contaminate the
    "output" pool with ~1112 Kenyon cells. Inadequate for biological I/O.

Substrates emitted (row order = graph_metadata body_ids):
  * core_alpn : the Exp-2 MB core (KC/MBON/DAN/MBIN) PLUS ALPN = 6014 neurons.
                This is Exp 4's PRIMARY substrate -- it adds the biological input
                population the Exp-2 core was missing (all 406 ALPN are in the halo,
                0 in the core). 99.2% weakly-connected.
  * full      : the whole 14,025-node Exp-1 substrate (robustness arm).

For each substrate we save the row indices into the 14k adjacency AND each port's
indices re-expressed in the substrate's own 0..(n-1) index space (what the engine uses
after it slices the sub-adjacency).

Outputs (tracked, staged with the code to the fleet):
  substrate/port_indices.npz    arrays per substrate (see KEYS below)
  substrate/port_manifest.json  human-readable definition + composition + provenance

Annotation source (Schlegel et al., Nature 2024; v2.1.0 == FlyWire materialization 783):
  https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv
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
DEFAULT_ROI = REPO_ROOT / "connectomes/flywire_mushroom_body/roi_counts.csv"
ANNOT_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)

# FlyWire annotation cell_class -> biological role. APL is annotated MBIN (gain control).
PORT_CLASSES = {
    "alpn": ("ALPN",),          # input  (olfactory projection neurons)
    "kc": ("Kenyon_Cell",),     # hidden (sparse code)
    "mbon": ("MBON",),          # output (readout)
    "dan": ("DAN",),            # learning (dopaminergic teaching signal)
    "mbin": ("MBIN",),          # gain control (incl. APL)
}
# The MB core (Exp 2) + ALPN. Order of the union does not matter; indices are sorted.
CORE_ALPN_CLASSES = ("Kenyon_Cell", "MBON", "DAN", "MBIN", "ALPN")


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


def _roi_group(r: str) -> str:
    if "_CA_" in r:
        return "calyx"
    if "_ML_" in r or "_VL_" in r:
        return "lobes"
    if "_PED_" in r:
        return "ped"
    return "other"


def compartment_profiles(roi_csv: Path, cell_class: pd.Series, body_ids: np.ndarray) -> dict:
    """Annotation-free cross-check: where does each population make its synapses?
    Textbook expectation: ALPN presynaptic in calyx (onto KC), MBON postsynaptic in
    lobes (from KC), KC post in calyx / pre in lobes."""
    roi = pd.read_csv(roi_csv)
    roi["grp"] = roi["roi"].map(_roi_group)
    g = roi.groupby(["bodyId", "grp"])[["pre", "post"]].sum().unstack(fill_value=0.0)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reindex(body_ids).fillna(0.0)
    out = {}
    cc = cell_class.to_numpy()
    for key, classes in PORT_CLASSES.items():
        m = np.isin(cc, classes)
        sub = g.loc[m]
        out[key] = {
            "n": int(m.sum()),
            "mean_pre_calyx": round(float(sub.get("pre_calyx", pd.Series(0.0)).mean()), 2),
            "mean_post_calyx": round(float(sub.get("post_calyx", pd.Series(0.0)).mean()), 2),
            "mean_pre_lobes": round(float(sub.get("pre_lobes", pd.Series(0.0)).mean()), 2),
            "mean_post_lobes": round(float(sub.get("post_lobes", pd.Series(0.0)).mean()), 2),
        }
    return out


def build_substrate(name: str, sub_rows: np.ndarray, cell_class: np.ndarray,
                    M: sp.csr_matrix) -> dict:
    """Given the 14k-row indices of a substrate, return its port arrays (in substrate
    index space), connectivity stats, and raw rho."""
    sub_rows = np.sort(sub_rows.astype(np.int64))
    n = len(sub_rows)
    sub = M[np.ix_(sub_rows, sub_rows)].tocsr()
    # map 14k-row -> substrate-position via searchsorted (sub_rows is sorted)
    ports = {}
    cc_sub = cell_class[sub_rows]
    for key, classes in PORT_CLASSES.items():
        rows14k = sub_rows[np.isin(cc_sub, classes)]
        ports[key] = np.searchsorted(sub_rows, rows14k).astype(np.int64)
    nc, lab = connected_components(sub + sub.T, directed=False)
    largest = int(np.bincount(lab).max()) if n else 0
    return {
        "name": name,
        "sub_rows": sub_rows,                 # indices into the 14k adjacency
        "ports": ports,                        # indices into 0..n-1 substrate space
        "n": n,
        "edges": int(sub.nnz),
        "wcc": int(nc),
        "largest_wcc": largest,
        "rho_raw": round(power_iteration_rho(sub), 4),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--roi", type=Path, default=DEFAULT_ROI)
    ap.add_argument("--annotations", type=Path, default=Path("/tmp/fw_annot.tsv"),
                    help="local path to FlyWire annotation TSV; downloaded from ANNOT_URL if missing.")
    ap.add_argument("--out-dir", type=Path, default=HERE / "substrate")
    args = ap.parse_args(argv)

    if not args.annotations.exists():
        print(f"annotation TSV not found; downloading from\n  {ANNOT_URL}")
        args.annotations.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ANNOT_URL, args.annotations)
    print(f"annotations: {args.annotations} ({args.annotations.stat().st_size/1e6:.1f} MB)")

    meta = json.loads(args.meta.read_text())
    body_ids = np.asarray(meta["body_ids"], dtype=np.int64)
    N = len(body_ids)
    M = sp.load_npz(args.matrix).tocsr()
    assert M.shape == (N, N), f"matrix {M.shape} != metadata N {N}"

    ann = pd.read_csv(args.annotations, sep="\t", low_memory=False,
                      usecols=["root_id", "super_class", "cell_class", "cell_type"])
    lut = ann.drop_duplicates("root_id").set_index("root_id")
    body = pd.DataFrame({"bodyId": body_ids, "row": np.arange(N)})
    j = body.join(lut, on="bodyId")
    n_matched = int(j["super_class"].notna().sum())
    print(f"substrate N={N}; matched to annotation: {n_matched} ({100*n_matched/N:.1f}%)")
    cell_class = j["cell_class"].to_numpy()  # object array, NaN where unlabeled

    # substrate row-sets
    core_alpn_rows = j.index[j["cell_class"].isin(CORE_ALPN_CLASSES)].to_numpy().astype(np.int64)
    full_rows = np.arange(N, dtype=np.int64)
    subs = {
        "core_alpn": build_substrate("core_alpn", core_alpn_rows, cell_class, M),
        "full": build_substrate("full", full_rows, cell_class, M),
    }

    # composition report + biological sanity
    comp = j["cell_class"].value_counts(dropna=False).to_dict()
    comp = {("NaN" if pd.isna(k) else str(k)): int(v) for k, v in comp.items()}
    profiles = compartment_profiles(args.roi, j["cell_class"], body_ids)

    for name, s in subs.items():
        pc = {k: len(v) for k, v in s["ports"].items()}
        print(f"\n=== {name} ===  N={s['n']} edges={s['edges']} "
              f"rho_raw={s['rho_raw']} WCC={s['wcc']} largest={s['largest_wcc']} "
              f"({100*s['largest_wcc']/s['n']:.1f}%)")
        print(f"  ports (in-substrate): {pc}")

    print("\n=== compartment cross-check (mean synapses / neuron; textbook in comments) ===")
    for k, p in profiles.items():
        print(f"  {k:5s} n={p['n']:5d}  pre_calyx={p['mean_pre_calyx']:8.1f} "
              f"post_lobes={p['mean_post_lobes']:9.1f}  (ALPN->pre_calyx high; MBON->post_lobes high)")

    # ---- validation gates (fail loudly if biology is wrong) ----
    errs = []
    ca = subs["core_alpn"]
    exp_counts = {"alpn": 406, "kc": 5177, "mbon": 96, "dan": 331, "mbin": 4}
    for k, want in exp_counts.items():
        got = len(ca["ports"][k])
        if got != want:
            errs.append(f"core_alpn port {k}: expected {want}, got {got}")
    if ca["n"] != 6014:
        errs.append(f"core_alpn N: expected 6014, got {ca['n']}")
    # biological sanity: ALPN should be presynaptic-dominant in calyx; MBON postsynaptic in lobes
    if profiles["alpn"]["mean_pre_calyx"] < profiles["alpn"]["mean_post_calyx"]:
        errs.append("ALPN not presynaptic-dominant in calyx (expected PN axons -> KC)")
    if profiles["mbon"]["mean_post_lobes"] < profiles["mbon"]["mean_pre_lobes"]:
        errs.append("MBON not postsynaptic-dominant in lobes (expected KC -> MBON dendrites)")
    if errs:
        raise SystemExit("VALIDATION FAILED:\n  " + "\n  ".join(errs))
    print("\nvalidation: OK (port counts + biological compartment sanity)")

    # ---- save ----
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    npz = {}
    for name, s in subs.items():
        npz[f"{name}__sub_rows"] = s["sub_rows"]
        for pk, pv in s["ports"].items():
            npz[f"{name}__{pk}"] = pv
    np.savez(out / "port_indices.npz", **npz)

    manifest = {
        "description": "Biological-I/O port indices for Experiment 4. For each substrate, "
                       "'<substrate>__sub_rows' are indices into the 14k adjacency; "
                       "'<substrate>__<port>' are indices into the substrate's own 0..n-1 space.",
        "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "annotation_source": ANNOT_URL,
        "annotation_release": "FlyWire 783 / Schlegel et al. 2024 (flywire_annotations v2.1.0)",
        "join_key": "annotation root_id == substrate bodyId",
        "n_full": int(N),
        "annotation_matched": n_matched,
        "port_cell_classes": {k: list(v) for k, v in PORT_CLASSES.items()},
        "core_alpn_classes": list(CORE_ALPN_CLASSES),
        "substrate_composition_full": comp,
        "substrates": {
            name: {
                "n": s["n"], "edges": s["edges"], "rho_raw": s["rho_raw"],
                "wcc": s["wcc"], "largest_wcc": s["largest_wcc"],
                "port_counts": {k: int(len(v)) for k, v in s["ports"].items()},
            } for name, s in subs.items()
        },
        "compartment_profiles": profiles,
        "notes": [
            "core_alpn is the PRIMARY Exp-4 substrate (MB core + the ALPN input layer).",
            "ALPN (biological input) is entirely in the halo: 0 of 406 in the Exp-2 core.",
            "predictedNt has no DA labels and native ROI-flow pools cannot separate ALPN/DAN;"
            " cell_class is the only signal that cleanly resolves all five roles.",
        ],
    }
    (out / "port_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out/'port_indices.npz'} and {out/'port_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
