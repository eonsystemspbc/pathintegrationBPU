#!/usr/bin/env python3
"""Label the ANTENNAL-LOBE substrate with biological cell classes and glomeruli -> substrate/ports.npz.

WHAT THIS IS FOR.  al-01 ran GENERIC all-neuron I/O: the gas signal was injected into every neuron and
the readout pooled every neuron. al-02 asks whether the AL connectome's advantage survives -- or grows
-- when the I/O is BIOLOGICAL instead: sensor drive enters only the olfactory receptor neurons (ORNs),
and the readout reads only the projection neurons (PNs), which is where the real animal's antennal
lobe hands its output to the mushroom body and lateral horn. That requires knowing which of the 4,947
substrate neurons are ORNs, which are PNs, which are local neurons, and which glomerulus each belongs
to. This script answers that and nothing else -- it does NOT rebuild the substrate, it labels it.

SOURCE OF IDENTITY.  The Schlegel-2024 FlyWire annotation table (see ANNOTATION_URL below), which
joins to substrate/root_ids.npy on `root_id` at 100%. We read four columns: `cell_class` for the
port assignment, `cell_sub_class` to isolate uniglomerular PNs, `cell_type` for the glomerulus
regexes, and `super_class` for provenance on the halo. The TSV lives in the SHARED connectome folder,
not in this experiment -- cx-01 and mb-02 expect it at the same path -- and is untracked by git
(`connectomes/` is in .gitignore).

THE FOUR BLOCKS.  block_id partitions all N neurons:
  0 RN   -- olfactory (2,279) + thermosensory (29) + hygrosensory (74) = the sensory input port
  1 LN   -- ALLN (429), the lateral-inhibition interior
  2 PN   -- ALPN (683), THE READOUT POOL
  3 halo -- everything else (1,453): the pass-through central / ascending / descending / endocrine
            neurons that the ROI-anchored recipe swept in because they touch AL_L/AL_R. Of these,
            1,195 carry no `cell_class` at all and 258 carry a non-AL class (AN, mechanosensory,
            ALIN, CX, mAL, ALON, MBON, gustatory, DAN, ...). See HALO NOTE below.

HALO NOTE (a deliberate deviation, recorded here and in the manifest).  The spec this script was
written to expected halo == 1,195, i.e. only the neurons whose `cell_class` is null. That count is
exactly right as a count of NULLS -- we reproduce it -- but it does not partition the substrate:
3,494 AL-class + 1,195 null = 4,689, leaving 258 neurons with a non-AL `cell_class` in no block at
all. Their `super_class` breakdown (86 central, 70 ascending, 60 sensory, 33 endocrine, 7
sensory_ascending, 2 optic) is precisely the "pass-through central/ascending/descending" population
the spec described as halo, so they are halo. block_id is therefore a complete partition and
halo_idx has 1,453 entries. The manifest records BOTH numbers (`n_cell_class_null` = 1195,
`n_halo` = 1453) plus the full breakdown, and `cell_class_raw` preserves the original annotation
string for every neuron so nothing is thrown away.

GLOMERULI.  Parsed from `cell_type` by regex (all three verified against the canonical structure):
  ORN_(.+)                    -> 53 olfactory glomeruli, 2,275/2,279 = 99.8% parsed
  [TH]RN_(.+)                 -> 8 thermo/hygro glomeruli (VP1d/VP1l/VP1m/VP2/VP3a/VP3b/VP4/VP5), 100%
  ([A-Za-z0-9]+)_[a-z]*PN     -> uniglomerular ALPNs, 261/277 parsed across 53 glomeruli
`glom_id` indexes the 53 OLFACTORY glomeruli only; `thermo_glom_id` indexes the 8 thermo/hygro ones.
A few uniglomerular PNs name a glomerulus in neither list (VM6, and the VP2/VP4 thermo targets); the
`glomerulus` string keeps their label while both id arrays are -1 for them.

SANITY CHECK.  The labeled line: for each glomerulus with both ORNs and a uniglomerular PN, what
fraction of that PN's receptor fan-in comes from its OWN glomerulus? This is the quantitative
statement of the structure al-02 exists to test -- if it were near chance there would be no
biological I/O worth wiring. Measured and asserted here, reported into the manifest.

Outputs (into substrate/):
  ports.npz           -- per-neuron class / glomerulus / block labels + the five index pools
  ports_manifest.json -- provenance: source URL, sha256, join rate, every count, block edge matrix,
                         glomerulus tables, regexes, parse rates, labeled-line strength
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
SUB = HERE / "substrate"

# Schlegel et al. 2024 FlyWire neuron annotations (139,244 rows, 32.6 MB). Download with:
#   curl -sSL -o connectomes/flywire_mushroom_body/flywire_release_783/cell_types_783.tsv \
#     https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv
ANNOTATION_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
                  "supplemental_files/Supplemental_file1_neuron_annotations.tsv")
ANNOTATION_TSV = (REPO_ROOT / "connectomes" / "flywire_mushroom_body" / "flywire_release_783"
                  / "cell_types_783.tsv")

# --- the glomerulus regexes (verified; see docstring) ---
RE_ORN = r"^ORN_(.+)$"
RE_TRN = r"^[TH]RN_(.+)$"
RE_UPN = r"^([A-Za-z0-9]+)_[a-z]*PN"

RN_CLASSES = ("olfactory", "thermosensory", "hygrosensory")
AL_CLASSES = ("olfactory", "ALLN", "ALPN", "thermosensory", "hygrosensory")
BLOCK_NAMES = ("RN", "LN", "PN", "halo")

# Expected counts, computed by a prior investigation against al-01's exact 4,947 root_ids. These are
# assertions, not defaults: if the data stops matching them something upstream changed and the run
# must fail loudly rather than quietly relabel the substrate.
EXPECT = {
    "N": 4947,
    "olfactory": 2279, "ALPN": 683, "ALLN": 429, "hygrosensory": 74, "thermosensory": 29,
    "al_subtotal": 3494, "cell_class_null": 1195,
    "n_olf_glom": 53, "n_thermo_glom": 8,
    "orn_parsed": 2275, "trn_parsed": 103, "upn_total": 277, "upn_parsed": 261, "upn_gloms": 53,
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_annotations(root_ids: np.ndarray) -> pd.DataFrame:
    """Read the annotation TSV and reindex it onto substrate row order. Asserts a 100% join."""
    if not ANNOTATION_TSV.exists():
        raise SystemExit(
            f"missing {ANNOTATION_TSV}\n"
            "This is the Schlegel-2024 FlyWire neuron annotation table (cell_class / cell_type).\n"
            "It lives in the SHARED connectome folder (cx-01 and mb-02 expect it there too) and is\n"
            "untracked by git. Fetch it with:\n"
            f"  mkdir -p {ANNOTATION_TSV.parent}\n"
            f"  curl -sSL -o {ANNOTATION_TSV} \\\n    {ANNOTATION_URL}"
        )
    cols = ["root_id", "super_class", "cell_class", "cell_sub_class", "cell_type", "side", "top_nt"]
    ann = pd.read_csv(ANNOTATION_TSV, sep="\t", usecols=cols, low_memory=False)
    assert not ann["root_id"].duplicated().any(), "annotation table has duplicate root_ids"
    ann = ann.set_index("root_id")

    n_hit = int(np.isin(root_ids, ann.index.to_numpy()).sum())
    match_rate = n_hit / len(root_ids)
    print(f"[ports] annotation join: {n_hit:,}/{len(root_ids):,} = {match_rate:.2%}", flush=True)
    assert match_rate == 1.0, f"expected a 100% join, got {match_rate:.4%} ({n_hit}/{len(root_ids)})"
    return ann.reindex(root_ids)


def assign_classes(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """cell_class -> the 6-value port vocabulary + block_id. Asserts every expected count."""
    raw = sub["cell_class"]
    N = len(sub)

    cell_class = np.full(N, "unlabeled", dtype="<U16")
    for c in AL_CLASSES:
        cell_class[(raw == c).to_numpy()] = c

    counts = {c: int((raw == c).sum()) for c in AL_CLASSES}
    for c in AL_CLASSES:
        assert counts[c] == EXPECT[c], f"cell_class {c}: got {counts[c]}, expected {EXPECT[c]}"
    al_subtotal = sum(counts.values())
    n_null = int(raw.isna().sum())
    assert al_subtotal == EXPECT["al_subtotal"], f"AL subtotal {al_subtotal}"
    assert n_null == EXPECT["cell_class_null"], f"cell_class nulls {n_null}"

    # The 258 non-AL-class neurons: halo by super_class, see HALO NOTE in the module docstring.
    other = raw[raw.notna() & ~raw.isin(AL_CLASSES)]
    other_classes = other.value_counts().to_dict()
    n_other = len(other)
    assert al_subtotal + n_null + n_other == N, "class partition does not cover N"

    block_id = np.full(N, 3, dtype=np.int32)
    block_id[np.isin(cell_class, RN_CLASSES)] = 0
    block_id[cell_class == "ALLN"] = 1
    block_id[cell_class == "ALPN"] = 2

    info = {
        "counts_by_cell_class": counts,
        "al_class_subtotal": al_subtotal,
        "n_cell_class_null": n_null,
        "n_non_al_cell_class": n_other,
        "non_al_cell_class_breakdown": {str(k): int(v) for k, v in other_classes.items()},
        "non_al_super_class_breakdown": {
            str(k): int(v) for k, v in sub.loc[other.index, "super_class"]
            .value_counts(dropna=False).to_dict().items()},
        "null_super_class_breakdown": {
            str(k): int(v) for k, v in sub.loc[raw.isna(), "super_class"]
            .value_counts(dropna=False).to_dict().items()},
    }
    print(f"[ports] classes: " + ", ".join(f"{c}={counts[c]:,}" for c in AL_CLASSES)
          + f" | AL subtotal={al_subtotal:,} | null={n_null:,} | non-AL labelled={n_other:,}",
          flush=True)
    return cell_class, block_id, info


def assign_glomeruli(sub: pd.DataFrame, cell_class: np.ndarray) -> tuple:
    """Parse glomerulus identity from cell_type for ORNs, TRN/HRNs and uniglomerular ALPNs."""
    N = len(sub)
    ct = sub["cell_type"]
    glomerulus = np.full(N, "", dtype="<U16")

    is_orn = cell_class == "olfactory"
    is_trn = np.isin(cell_class, ("thermosensory", "hygrosensory"))
    is_upn = ((cell_class == "ALPN") & (sub["cell_sub_class"] == "uniglomerular").to_numpy())

    orn_g = ct[is_orn].str.extract(RE_ORN)[0]
    trn_g = ct[is_trn].str.extract(RE_TRN)[0]
    upn_g = ct[is_upn].str.extract(RE_UPN)[0]

    glomerulus[is_orn] = orn_g.fillna("").to_numpy()
    glomerulus[is_trn] = trn_g.fillna("").to_numpy()
    glomerulus[is_upn] = upn_g.fillna("").to_numpy()

    n_orn_p, n_trn_p, n_upn_p = int(orn_g.notna().sum()), int(trn_g.notna().sum()), int(upn_g.notna().sum())
    assert n_orn_p == EXPECT["orn_parsed"], f"ORN parse {n_orn_p}"
    assert n_trn_p == EXPECT["trn_parsed"], f"TRN/HRN parse {n_trn_p}"
    assert int(is_upn.sum()) == EXPECT["upn_total"], f"uniglomerular ALPN count {int(is_upn.sum())}"
    assert n_upn_p == EXPECT["upn_parsed"], f"uPN parse {n_upn_p}"

    # Canonical glomerulus vocabularies: olfactory from the ORNs, thermo/hygro from the TRN/HRNs.
    glom_names = np.array(sorted(orn_g.dropna().unique()), dtype="<U16")
    thermo_glom_names = np.array(sorted(trn_g.dropna().unique()), dtype="<U16")
    assert len(glom_names) == EXPECT["n_olf_glom"], f"olfactory glomeruli {len(glom_names)}"
    assert len(thermo_glom_names) == EXPECT["n_thermo_glom"], f"thermo glomeruli {len(thermo_glom_names)}"
    assert upn_g.dropna().nunique() == EXPECT["upn_gloms"], "uPN glomerulus count"
    total_glom = len(glom_names) + len(thermo_glom_names)
    assert total_glom == 61, f"expected 61 glomeruli total, got {total_glom}"

    olf_pos = {g: i for i, g in enumerate(glom_names.tolist())}
    th_pos = {g: i for i, g in enumerate(thermo_glom_names.tolist())}
    glom_id = np.array([olf_pos.get(g, -1) for g in glomerulus.tolist()], dtype=np.int32)
    thermo_glom_id = np.array([th_pos.get(g, -1) for g in glomerulus.tolist()], dtype=np.int32)

    orn_per_glom = orn_g.dropna().value_counts()
    info = {
        "regex_orn": RE_ORN, "regex_thermo_hygro": RE_TRN, "regex_uniglomerular_alpn": RE_UPN,
        "orn_parsed": n_orn_p, "orn_total": int(is_orn.sum()),
        "orn_parse_rate": round(n_orn_p / int(is_orn.sum()), 4),
        "orn_unparsed_cell_types": {"<null cell_type>": int(is_orn.sum() - n_orn_p)},
        "thermo_hygro_parsed": n_trn_p, "thermo_hygro_total": int(is_trn.sum()),
        "thermo_hygro_parse_rate": round(n_trn_p / int(is_trn.sum()), 4),
        "uniglomerular_alpn_total": int(is_upn.sum()), "uniglomerular_alpn_parsed": n_upn_p,
        "uniglomerular_alpn_parse_rate": round(n_upn_p / int(is_upn.sum()), 4),
        "uniglomerular_alpn_unparsed_cell_types": {
            str(k): int(v) for k, v in ct[is_upn][upn_g.isna().to_numpy()].value_counts().to_dict().items()},
        "n_olfactory_glomeruli": len(glom_names),
        "n_thermo_hygro_glomeruli": len(thermo_glom_names),
        "n_glomeruli_total": total_glom,
        "olfactory_glomeruli": glom_names.tolist(),
        "thermo_hygro_glomeruli": thermo_glom_names.tolist(),
        "orns_per_glomerulus": {str(k): int(v) for k, v in orn_per_glom.sort_index().items()},
        "orns_per_glomerulus_min": int(orn_per_glom.min()),
        "orns_per_glomerulus_min_glom": str(orn_per_glom.idxmin()),
        "orns_per_glomerulus_median": float(orn_per_glom.median()),
        "orns_per_glomerulus_max": int(orn_per_glom.max()),
        "orns_per_glomerulus_max_glom": str(orn_per_glom.idxmax()),
        "upn_glomeruli_outside_olfactory_set": sorted(set(upn_g.dropna()) - set(glom_names.tolist())),
    }
    print(f"[ports] glomeruli: {len(glom_names)} olfactory + {len(thermo_glom_names)} thermo/hygro "
          f"= {total_glom} | ORNs/glom min={orn_per_glom.min()} ({orn_per_glom.idxmin()}) "
          f"median={orn_per_glom.median():.0f} max={orn_per_glom.max()} ({orn_per_glom.idxmax()})",
          flush=True)
    return glomerulus, glom_id, glom_names, thermo_glom_id, thermo_glom_names, info


def block_edge_matrix(M: sp.csr_matrix, block_id: np.ndarray) -> dict:
    """4x4 block connectivity of the real substrate. M is POST x PRE, so M[post_block, pre_block]."""
    Mc = M.tocoo()
    pre_b, post_b = block_id[Mc.col], block_id[Mc.row]
    edges = np.zeros((4, 4), dtype=np.int64)
    syns = np.zeros((4, 4), dtype=np.int64)
    np.add.at(edges, (pre_b, post_b), 1)
    np.add.at(syns, (pre_b, post_b), np.abs(Mc.data).astype(np.int64))
    named = {f"{BLOCK_NAMES[i]}->{BLOCK_NAMES[j]}": int(edges[i, j]) for i in range(4) for j in range(4)}
    assert edges.sum() == M.nnz, "block edge counts do not sum to nnz"
    print("[ports] block edges (pre->post): " + ", ".join(
        f"{k}={v:,}" for k, v in named.items() if v), flush=True)
    return {
        "block_order": list(BLOCK_NAMES),
        "note": "edges[i][j] = number of pre-block-i -> post-block-j edges in the real substrate",
        "edges": edges.tolist(),
        "synapses": syns.tolist(),
        "edges_named": named,
        "synapses_named": {f"{BLOCK_NAMES[i]}->{BLOCK_NAMES[j]}": int(syns[i, j])
                           for i in range(4) for j in range(4)},
        "total_edges": int(edges.sum()),
    }


def labeled_line_strength(M: sp.csr_matrix, cell_class: np.ndarray, glomerulus: np.ndarray,
                          sub: pd.DataFrame) -> dict:
    """For each glomerulus with both ORNs and a uniglomerular PN: what fraction of that PN's
    RECEPTOR fan-in comes from its own glomerulus? Chance = that glomerulus' share of all RNs."""
    is_rn = np.isin(cell_class, RN_CLASSES)
    is_upn = ((cell_class == "ALPN") & (sub["cell_sub_class"] == "uniglomerular").to_numpy())
    rn_glom = np.where(is_rn, glomerulus, "")
    n_rn = int(is_rn.sum())

    per_pn, per_glom = [], {}
    tot_e = tot_own_e = tot_s = tot_own_s = 0
    Mcsr = M.tocsr()
    pn_ids = np.flatnonzero(is_upn)
    for i in pn_ids:
        g = glomerulus[i]
        if not g or not (rn_glom == g).any():
            continue  # PN's glomerulus has no receptor neurons in the substrate
        cols = Mcsr.indices[Mcsr.indptr[i]:Mcsr.indptr[i + 1]]
        vals = np.abs(Mcsr.data[Mcsr.indptr[i]:Mcsr.indptr[i + 1]])
        keep = is_rn[cols]
        cols, vals = cols[keep], vals[keep]
        if len(cols) == 0:
            continue
        own = rn_glom[cols] == g
        e_frac = float(own.sum()) / len(cols)
        s_frac = float(vals[own].sum()) / float(vals.sum())
        chance = float((rn_glom == g).sum()) / n_rn
        per_pn.append((g, e_frac, s_frac, chance))
        per_glom.setdefault(g, []).append((e_frac, s_frac, chance))
        tot_e += len(cols); tot_own_e += int(own.sum())
        tot_s += float(vals.sum()); tot_own_s += float(vals[own].sum())

    glom_e = {g: float(np.mean([x[0] for x in v])) for g, v in per_glom.items()}
    glom_s = {g: float(np.mean([x[1] for x in v])) for g, v in per_glom.items()}
    chance_mean = float(np.mean([x[3] for x in per_pn]))
    out = {
        "definition": ("fraction of a uniglomerular ALPN's receptor-neuron (block RN) fan-in that "
                       "originates in that PN's own glomerulus"),
        "n_pns_measured": len(per_pn),
        "n_glomeruli_measured": len(per_glom),
        "edge_weighted_pooled": round(tot_own_e / tot_e, 4),
        "synapse_weighted_pooled": round(tot_own_s / tot_s, 4),
        "edge_weighted_median_per_glomerulus": round(float(np.median(list(glom_e.values()))), 4),
        "synapse_weighted_median_per_glomerulus": round(float(np.median(list(glom_s.values()))), 4),
        "chance_level_mean": round(chance_mean, 4),
        "enrichment_over_chance_edge_weighted": round((tot_own_e / tot_e) / chance_mean, 1),
        "per_glomerulus_edge_weighted": {g: round(v, 4) for g, v in sorted(glom_e.items())},
        "per_glomerulus_synapse_weighted": {g: round(v, 4) for g, v in sorted(glom_s.items())},
    }
    print(f"[ports] labeled line: {out['edge_weighted_pooled']:.1%} edge-weighted "
          f"(median {out['edge_weighted_median_per_glomerulus']:.1%} per glomerulus), "
          f"{out['synapse_weighted_pooled']:.1%} synapse-weighted, "
          f"chance {chance_mean:.1%} -> {out['enrichment_over_chance_edge_weighted']}x", flush=True)
    assert out["edge_weighted_pooled"] > 0.5, "labeled line collapsed -- glomerulus parse is wrong"
    return out


def build() -> dict:
    root_ids = np.load(SUB / "root_ids.npy").astype(np.int64)
    M = sp.load_npz(SUB / "al_substrate.npz").tocsr()
    N = len(root_ids)
    assert N == EXPECT["N"], f"substrate N {N}, expected {EXPECT['N']}"
    assert M.shape == (N, N), f"substrate shape {M.shape}"
    print(f"[ports] substrate: N={N:,}, edges={M.nnz:,}", flush=True)

    sub = load_annotations(root_ids)
    cell_class, block_id, class_info = assign_classes(sub)
    (glomerulus, glom_id, glom_names,
     thermo_glom_id, thermo_glom_names, glom_info) = assign_glomeruli(sub, cell_class)

    orn_idx = np.flatnonzero(cell_class == "olfactory").astype(np.int32)
    thermo_idx = np.flatnonzero(np.isin(cell_class, ("thermosensory", "hygrosensory"))).astype(np.int32)
    ln_idx = np.flatnonzero(cell_class == "ALLN").astype(np.int32)
    pn_idx = np.flatnonzero(cell_class == "ALPN").astype(np.int32)
    halo_idx = np.flatnonzero(cell_class == "unlabeled").astype(np.int32)

    assert len(orn_idx) == 2279 and len(thermo_idx) == 103
    assert len(ln_idx) == 429 and len(pn_idx) == 683
    assert len(orn_idx) + len(thermo_idx) + len(ln_idx) + len(pn_idx) + len(halo_idx) == N
    assert np.array_equal(np.flatnonzero(block_id == 0), np.sort(np.concatenate([orn_idx, thermo_idx])))
    assert np.array_equal(np.flatnonzero(block_id == 3), halo_idx)

    blocks = block_edge_matrix(M, block_id)
    labeled_line = labeled_line_strength(M, cell_class, glomerulus, sub)

    np.savez_compressed(
        SUB / "ports.npz",
        cell_class=cell_class, glomerulus=glomerulus,
        glom_id=glom_id, glom_names=glom_names,
        thermo_glom_id=thermo_glom_id, thermo_glom_names=thermo_glom_names,
        block_id=block_id,
        orn_idx=orn_idx, thermo_idx=thermo_idx, ln_idx=ln_idx, pn_idx=pn_idx, halo_idx=halo_idx,
        # extra, not part of the required schema: the raw annotation string, so the 258 non-AL
        # halo neurons keep their identity for anyone who wants it.
        cell_class_raw=sub["cell_class"].fillna("").to_numpy().astype("<U24"),
    )

    manifest = {
        "purpose": "biological I/O ports for the al-02 antennal-lobe substrate",
        "annotation_source_url": ANNOTATION_URL,
        "annotation_local_path": str(ANNOTATION_TSV.relative_to(REPO_ROOT)),
        "annotation_sha256": sha256_of(ANNOTATION_TSV),
        "annotation_rows": int(sum(1 for _ in ANNOTATION_TSV.open()) - 1),
        "substrate_npz": "substrate/al_substrate.npz",
        "N": N,
        "edges": int(M.nnz),
        "join_match_rate": 1.0,
        "join_matched": N,
        "cell_class_vocabulary": ["olfactory", "ALLN", "ALPN", "thermosensory", "hygrosensory",
                                  "unlabeled"],
        "block_id_meaning": {"0": "RN (olfactory+thermo+hygro)", "1": "LN (ALLN)",
                             "2": "PN (ALPN)", "3": "halo"},
        "counts": {
            **class_info,
            "n_halo": int(len(halo_idx)),
            "port_sizes": {"orn_idx": int(len(orn_idx)), "thermo_idx": int(len(thermo_idx)),
                           "ln_idx": int(len(ln_idx)), "pn_idx": int(len(pn_idx)),
                           "halo_idx": int(len(halo_idx))},
            "block_sizes": {BLOCK_NAMES[b]: int((block_id == b).sum()) for b in range(4)},
        },
        "halo_note": (
            "The spec expected halo == 1195 (the cell_class NULLs, which we reproduce exactly). That "
            "set does NOT partition the substrate: 3494 AL-class + 1195 null = 4689, leaving 258 "
            "neurons with a non-AL cell_class unassigned. Their super_class breakdown is central/"
            "ascending/sensory/endocrine -- the pass-through population the spec called halo -- so "
            "they are block 3 and halo_idx has 1453 entries. n_cell_class_null records the 1195."),
        "glomeruli": glom_info,
        "block_connectivity": blocks,
        "labeled_line_strength": labeled_line,
    }
    (SUB / "ports_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ports] wrote {SUB/'ports.npz'} and {SUB/'ports_manifest.json'}", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--print-manifest", action="store_true",
                    help="dump the full manifest to stdout after building")
    args = ap.parse_args()
    manifest = build()
    if args.print_manifest:
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
