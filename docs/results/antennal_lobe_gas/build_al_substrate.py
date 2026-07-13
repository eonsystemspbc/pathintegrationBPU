#!/usr/bin/env python3
"""Build the ANTENNAL-LOBE substrate + biologically-correct I/O ports from FlyWire 783.

Region:  the fly's first olfactory relay. We take the induced subgraph over the
bilateral antennal-lobe cell populations, identified by the FlyWire/Schlegel-2024
whole-brain annotation `cell_class`:

    ORN  (olfactory receptor neurons)   cell_class == "olfactory"      ~2282   INPUT
    TRN/HRN (thermo/hygro receptors)    cell_class in {thermo,hygro}    ~103   INPUT (T/RH)
    ALLN (antennal-lobe local neurons)  cell_class == "ALLN"            ~429   lateral inhibition
    ALPN (projection neurons)           cell_class == "ALPN"            ~685   OUTPUT (readout)

Edges = the induced subgraph over those neurons in the FlyWire 783 proofread synapse
table (pre & post both in the AL set), aggregated to synapse counts per ordered pair.
Two representations are saved:

  * UNSIGNED : |syn_count|  (the exact representation the prior +12% data-efficiency wins used)
  * SIGNED   : sign(pre transmitter) * syn_count, Dale's law by the PRESYNAPTIC neuron's
               top_nt (ACh -> +1, GABA/Glu -> -1). Sensory receptors (ORN/TRN/HRN) are
               forced excitatory (+1): they are cholinergic; the per-neuron top_nt predictor
               mislabels a chunk of ORNs as serotonergic. Modulatory (DA/5HT/OA/unknown)
               presynapses default to +1. The user asked to "preserve signs where possible";
               SIGNED is the biologically-faithful primary, UNSIGNED the robustness arm.

Orientation: W[post, pre] (a recurrence operator: h <- W h injects pre onto post), matching
the "operator = M, post x pre" convention of the MB biological-I/O experiment.

Glomeruli (the 51 olfactory + 7 thermo/hygro "channels") are parsed from cell_type:
    ORN_DA1 -> DA1 ;  TRN_VP2 -> VP2 ;  HRN_VP4 -> VP4 ;  uniglomerular PN DA3_adPN -> DA3.

Outputs (into substrate/):
  * al_unsigned.npz / al_signed.npz  -- induced AL adjacency, CSR float32, W[post,pre]
  * root_ids.npy                     -- root_ids in matrix-row order (FlyWire join key)
  * ports.json                       -- {port_name: [matrix indices]} + glomerulus groupings
  * manifest.json                    -- N, edges, rho, sign coverage, port + glomerulus report
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / "pyproject.toml").exists()), HERE.parents[-1])
SUB = HERE / "substrate"
SUB.mkdir(parents=True, exist_ok=True)

ANNOT = ROOT / "flywire_cache" / "fw_annotations_783.tsv"
CONN = ROOT / "flywire_cache" / "proofread_connections_783.feather"

AL_CLASSES = ["olfactory", "thermosensory", "hygrosensory", "ALLN", "ALPN"]
SENSORY_CLASSES = {"olfactory", "thermosensory", "hygrosensory"}
TRANSMITTER_SIGN = {"acetylcholine": 1.0, "ach": 1.0, "gaba": -1.0, "glu": -1.0, "glutamate": -1.0}


def glomerulus_of(row) -> str | None:
    """Parse the glomerulus identity from a neuron's cell_type. None if not glomerulus-specific
    (multiglomerular PNs, LNs)."""
    ct = str(row["cell_type"]) if pd.notna(row["cell_type"]) else ""
    cc = row["cell_class"]
    if cc == "olfactory":
        m = re.fullmatch(r"ORN_(.+)", ct)
        return m.group(1) if m else None
    if cc in ("thermosensory", "hygrosensory"):
        m = re.fullmatch(r"[TH]RN_(.+)", ct)
        return m.group(1) if m else None
    if cc == "ALPN" and row["cell_sub_class"] == "uniglomerular":
        # DA3_adPN -> DA3 ; DL2v_adPN -> DL2v ; VM5v_adPN -> VM5v
        m = re.match(r"([A-Za-z0-9]+)_[a-z]*PN", ct)
        return m.group(1) if m else None
    return None


def power_iteration_rho(matrix: sp.spmatrix, iters: int = 200, seed: int = 0) -> float:
    A = sp.csr_matrix(np.abs(matrix.astype(np.float64)))
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[0]); v /= np.linalg.norm(v) + 1e-12
    lam = 0.0
    for _ in range(iters):
        w = A @ v
        nw = np.linalg.norm(w)
        if nw < 1e-30:
            return 0.0
        v = w / nw; lam = nw
    return float(lam)


def main() -> None:
    print(f"annotations: {ANNOT}")
    ann = pd.read_csv(ANNOT, sep="\t", low_memory=False,
                      usecols=["root_id", "super_class", "cell_class", "cell_sub_class",
                               "cell_type", "top_nt", "side"])
    ann = ann.dropna(subset=["root_id"]).copy()
    ann["root_id"] = ann["root_id"].astype("int64")
    al = ann[ann["cell_class"].isin(AL_CLASSES)].drop_duplicates("root_id").copy()
    al = al.sort_values("root_id").reset_index(drop=True)
    N = len(al)
    root_ids = al["root_id"].to_numpy()
    id2idx = {int(r): i for i, r in enumerate(root_ids)}
    print(f"AL neurons: N={N}  by class: {dict(al['cell_class'].value_counts())}")

    # --- induced subgraph from the proofread synapse table ---------------------------------
    print("loading FlyWire connections (feather) ...")
    df = feather.read_table(CONN, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"]).to_pandas()
    al_ids = set(id2idx)
    m = df["pre_pt_root_id"].isin(al_ids) & df["post_pt_root_id"].isin(al_ids)
    sub = df[m].groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False)["syn_count"].sum()
    pre = sub["pre_pt_root_id"].map(id2idx).to_numpy()
    post = sub["post_pt_root_id"].map(id2idx).to_numpy()
    w = sub["syn_count"].to_numpy(dtype=np.float32)
    print(f"induced edges: {len(sub)}  total synapses: {int(w.sum())}")

    # presynaptic sign (Dale): sensory forced +1; else transmitter map; modulatory/unknown -> +1
    top_nt = al["top_nt"].fillna("").astype(str).to_numpy()
    is_sensory = al["cell_class"].isin(SENSORY_CLASSES).to_numpy()
    sign = np.array([1.0 if is_sensory[i] else TRANSMITTER_SIGN.get(top_nt[i].lower(), 1.0)
                     for i in range(N)], dtype=np.float32)
    pre_sign = sign[pre]

    # W[post, pre]  (recurrence operator: post row, pre col)
    A_uns = sp.coo_matrix((w, (post, pre)), shape=(N, N)).tocsr()
    A_sig = sp.coo_matrix((w * pre_sign, (post, pre)), shape=(N, N)).tocsr()
    A_uns.eliminate_zeros(); A_sig.eliminate_zeros()

    n_inhib_pre = int((sign < 0).sum())
    frac_inhib_edges = float((pre_sign < 0).sum()) / max(len(pre), 1)

    # --- glomeruli + ports ----------------------------------------------------------------
    al["glom"] = al.apply(glomerulus_of, axis=1)
    cc = al["cell_class"].to_numpy()
    idx_all = np.arange(N)
    orn_idx = idx_all[cc == "olfactory"]
    trn_idx = idx_all[np.isin(cc, ["thermosensory", "hygrosensory"])]
    lln_idx = idx_all[cc == "ALLN"]
    pn_idx = idx_all[cc == "ALPN"]
    pn_uni_idx = idx_all[(cc == "ALPN") & (al["cell_sub_class"].to_numpy() == "uniglomerular")]

    # input port: ORN indices grouped by olfactory glomerulus (adapter target)
    orn_by_glom: dict[str, list[int]] = {}
    for i in orn_idx:
        g = al["glom"].iloc[i]
        if g:
            orn_by_glom.setdefault(str(g), []).append(int(i))
    # thermo/hygro receptors grouped by glomerulus (T/RH channels)
    thr_by_glom: dict[str, list[int]] = {}
    for i in trn_idx:
        g = al["glom"].iloc[i]
        if g:
            thr_by_glom.setdefault(str(g), []).append(int(i))

    ports = {
        "orn_all": orn_idx.astype(int).tolist(),
        "trn_all": trn_idx.astype(int).tolist(),
        "lln_all": lln_idx.astype(int).tolist(),
        "pn_all": pn_idx.astype(int).tolist(),
        "pn_uni": pn_uni_idx.astype(int).tolist(),
        "orn_by_glom": {k: v for k, v in sorted(orn_by_glom.items())},
        "thr_by_glom": {k: v for k, v in sorted(thr_by_glom.items())},
    }

    rho_uns = power_iteration_rho(A_uns)
    rho_sig = power_iteration_rho(A_sig)

    sp.save_npz(SUB / "al_unsigned.npz", A_uns)
    sp.save_npz(SUB / "al_signed.npz", A_sig)
    np.save(SUB / "root_ids.npy", root_ids)
    (SUB / "ports.json").write_text(json.dumps(ports))

    manifest = {
        "substrate": "antennal_lobe_flywire783",
        "source": "flywire_cache/proofread_connections_783.feather (induced subgraph)",
        "annotation_source": "flyconnectome/flywire_annotations Supplemental_file1 (Schlegel 2024, 783)",
        "release": "783",
        "N": int(N),
        "edges_unsigned": int(A_uns.nnz),
        "edges_signed": int(A_sig.nnz),
        "total_synapses": int(w.sum()),
        "orientation": "W[post, pre] (recurrence operator M = post x pre)",
        "raw_spectral_radius_unsigned": round(rho_uns, 4),
        "raw_spectral_radius_signed": round(rho_sig, 4),
        "n_inhibitory_presynaptic_neurons": n_inhib_pre,
        "frac_inhibitory_edges": round(frac_inhib_edges, 4),
        "class_counts": {k: int(v) for k, v in al["cell_class"].value_counts().items()},
        "port_counts": {
            "orn_all": len(ports["orn_all"]), "trn_all": len(ports["trn_all"]),
            "lln_all": len(ports["lln_all"]), "pn_all": len(ports["pn_all"]),
            "pn_uni": len(ports["pn_uni"]),
        },
        "n_olfactory_glomeruli": len(orn_by_glom),
        "n_thermo_hygro_glomeruli": len(thr_by_glom),
        "olfactory_glomeruli": sorted(orn_by_glom),
        "thermo_hygro_glomeruli": sorted(thr_by_glom),
    }
    (SUB / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in (
        "N", "edges_signed", "total_synapses", "raw_spectral_radius_signed",
        "frac_inhibitory_edges", "port_counts", "n_olfactory_glomeruli",
        "n_thermo_hygro_glomeruli")}, indent=2))
    print(f"olfactory glomeruli ({len(orn_by_glom)}): {sorted(orn_by_glom)}")
    print(f"thermo/hygro glomeruli ({len(thr_by_glom)}): {sorted(thr_by_glom)}")


if __name__ == "__main__":
    main()
