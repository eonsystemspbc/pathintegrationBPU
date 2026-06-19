#!/usr/bin/env python3
"""Pool-fidelity report: how biological are the sensory/output "ports"?

The BPU trainer injects task input only into the ``sensory`` pool and reads only the
``output`` pool (``src/models.py`` ``index_add``/``index_select``). Those pools are assigned
by an *extra-regional synapse-flow* heuristic (``src/pools.py``): a neuron is ``sensory`` if it
receives most of its input from *outside* the region's ROIs, ``output`` if it sends most of its
output outside. That is a connectivity proxy, **not** a cell-type label.

This script quantifies, per connectome, how well that proxy lines up with the region's *known
input/output cell types* (where type annotations exist). It reads only the prepared
``pool_assignments.csv`` (+ ``neurons.csv`` for type strings) — it never touches training or
controls, so it cannot change any result.

Key invariance (why this is a fidelity report, not a result): the connectome model AND every
control (random / degree_shuffle / weight_shuffle) are wired through the *same* sensory/output
indices. So the connectome-vs-control comparison is invariant to whether these pools are
biologically labeled correctly; this report bears on the *interpretation* of the I/O ("does input
enter at biologically input-side neurons?"), not on the causal connectome>random claim.

Usage:
    .venv/bin/python scripts/connectome/report_pool_fidelity.py \
        --out docs/results/pool_fidelity/README.md
    .venv/bin/python scripts/connectome/report_pool_fidelity.py \
        --connectomes connectomes/cx_polar_bump_seed0 connectomes/flywire_mushroom_body
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

# Per-region known input-side / output-side cell-type families, as regexes on the type/instance
# string. "canonical" = textbook afferent/efferent classes; "broad" additionally counts cell types
# that genuinely project into/out of the region but are integrative rather than primary afferents.
REGION_FAMILIES: dict[str, dict[str, object]] = {
    "cx": {
        "label": "central complex (hemibrain)",
        # canonical CX input families: ellipsoid-body ring (ER), extrinsic ring (ExR), tubercle->bulb
        # (TuBu), noduli inputs (LNO/LCNO/GLNO), superior-posterior-slope (SpsP/IbSpsP).
        "sensory_canonical": r"^(?:ER|ExR|TuBu|LNO|LCNO|GLNO|SpsP|IbSpsP)",
        "sensory_broad": r"^(?:ER|ExR|TuBu|LNO|LCNO|GLNO|SpsP|IbSpsP)",
        # canonical premotor output: PFL2/PFL3 (-> descending), PFR. broad: all CX columnar cells that
        # project OUT of the CX (FS->SMP, FC->CRE, FR).
        "output_canonical": r"^(?:PFL|PFR)",
        "output_broad": r"^(?:PFL|PFR|FS|FC|FR)",
        "internal_expected": r"^(?:EPG|PEG|PEN|EL|FB|vDelta|hDelta|P6|PFN|Delta)",
    },
    "mb": {
        "label": "mushroom body",
        # canonical olfactory input = projection neurons (any *PN*). broad input additionally counts
        # dopaminergic reinforcement neurons (PAM/PPL/DAN): these deliver the reward/punishment (US)
        # signal that the task feeds as the reward/punishment input channels.
        "sensory_canonical": r"PN",
        "sensory_broad": r"PN|PAM|PPL|DAN",
        "output_canonical": r"MBON",
        "output_broad": r"MBON|MBIN",
        "internal_expected": r"KC",  # Kenyon cells (MB-intrinsic)
    },
    "optic": {
        "label": "optic lobe",
        "sensory_canonical": r"^(?:R[1-8]|L[1-5]|Mi|Tm|C[23]|Lawf)",  # photoreceptor/lamina/medulla
        "sensory_broad": r"^(?:R[1-8]|L[1-5]|Mi|Tm|C[23]|Lawf|Dm|Pm|TmY)",
        "output_canonical": r"^(?:T4|T5|LC|LPLC|LT|TmY)",  # motion/projection output
        "output_broad": r"^(?:T4|T5|LC|LPLC|LT|TmY|Y)",
        "internal_expected": r".",
    },
}

# Map a connectome directory name to its region key (which family set applies).
DIR_TO_REGION = {
    "cx_polar_bump_seed0": "cx",
    "cx_structure_polar_frozen": "cx",
    "cx_structure_polar_observed": "cx",
    "flywire_mushroom_body": "mb",
    "hemibrain_mushroom_body_plume": "mb",
    "larva_bpu": "mb",
    "flywire_optic_lobe_bpu": "optic",
}

DEFAULT_CONNECTOMES = [
    "connectomes/cx_polar_bump_seed0",
    "connectomes/flywire_mushroom_body",
    "connectomes/hemibrain_mushroom_body_plume",
    "connectomes/flywire_optic_lobe_bpu",
]


def _name_series(df: pd.DataFrame) -> pd.Series:
    """Best available cell-identity string per row (type, falling back to instance)."""
    type_col = df["type"].fillna("").astype(str) if "type" in df else pd.Series([""] * len(df))
    inst_col = df["instance"].fillna("").astype(str) if "instance" in df else pd.Series([""] * len(df))
    combined = (type_col + " " + inst_col).str.strip()
    return combined


def _pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.0f}%" if d else "—"


def analyze_one(conn_dir: Path) -> dict[str, object]:
    pools = pd.read_csv(conn_dir / "pool_assignments.csv")
    # Prefer type/instance already in the pool file; if absent/empty, merge from neurons.csv.
    have_types = "type" in pools and pools["type"].fillna("").astype(str).str.strip().ne("").any()
    if not have_types and (conn_dir / "neurons.csv").exists():
        neurons = pd.read_csv(conn_dir / "neurons.csv", usecols=lambda c: c in {"bodyId", "type", "instance"})
        pools = pools.merge(neurons, on="bodyId", how="left", suffixes=("", "_n"))
        for col in ("type", "instance"):
            if f"{col}_n" in pools:
                pools[col] = pools[col].where(pools[col].notna() & (pools[col].astype(str).str.strip() != ""), pools[f"{col}_n"])
    pools["__name"] = _name_series(pools)

    region = DIR_TO_REGION.get(conn_dir.name)
    fams = REGION_FAMILIES.get(region, {})

    n_total = len(pools)
    counts = {p: int((pools["pool"] == p).sum()) for p in ("sensory", "internal", "output")}
    typed = pools["__name"].str.strip().ne("")
    type_cov = int(typed.sum())

    out: dict[str, object] = {
        "dir": conn_dir.name,
        "region_label": fams.get("label", region or conn_dir.name),
        "N": n_total,
        "counts": counts,
        "type_coverage": type_cov,
        "type_coverage_pct": _pct(type_cov, n_total),
        "pools": {},
    }

    for pool, can_key, broad_key in (
        ("sensory", "sensory_canonical", "sensory_broad"),
        ("output", "output_canonical", "output_broad"),
        ("internal", "internal_expected", "internal_expected"),
    ):
        sub = pools[pools["pool"] == pool]
        sub_typed = sub[sub["__name"].str.strip().ne("")]
        rec: dict[str, object] = {
            "N": len(sub),
            "typed": len(sub_typed),
            "typed_pct": _pct(len(sub_typed), len(sub)),
        }
        if len(sub_typed) and can_key in fams:
            can = sub_typed["__name"].str.contains(fams[can_key], regex=True, case=True)
            broad = sub_typed["__name"].str.contains(fams[broad_key], regex=True, case=True)
            # precision is computed over TYPED neurons only (untyped cannot be judged)
            rec["match_canonical"] = int(can.sum())
            rec["match_canonical_pct_of_typed"] = _pct(int(can.sum()), len(sub_typed))
            rec["match_broad"] = int(broad.sum())
            rec["match_broad_pct_of_typed"] = _pct(int(broad.sum()), len(sub_typed))
        # top type prefixes (for transparency)
        pref = sub_typed["__name"].str.extract(r"^([A-Za-z]+)")[0].fillna("(other)")
        rec["top_types"] = pref.value_counts().head(8).to_dict()
        out["pools"][pool] = rec
    # assignment-reason breakdown (how the label was decided)
    if "assignment_reason" in pools:
        out["reason"] = (
            pools.groupby(["pool", "assignment_reason"]).size().to_dict()
        )
    return out


def render_markdown(results: list[dict[str, object]]) -> str:
    lines: list[str] = []
    lines.append("# Pool fidelity — how biological are the sensory/output ports?\n")
    lines.append(
        "Generated by `scripts/connectome/report_pool_fidelity.py`. The BPU trainer injects task "
        "input only into the **sensory** pool and reads only the **output** pool. Those pools are "
        "assigned by an *extra-regional synapse-flow* heuristic (`src/pools.py`), i.e. *receives most "
        "of its input / sends most of its output outside the region's ROIs* — a connectivity proxy, "
        "not a cell-type label. This table reports how well that proxy matches the region's known "
        "input/output cell types, where type annotations exist.\n"
    )
    lines.append(
        "> **Why this is a fidelity report, not a result.** The connectome model and every control "
        "(`random` / `degree_shuffle` / `weight_shuffle`) are wired through the *same* sensory/output "
        "indices, so the connectome-vs-control comparison is invariant to how biologically these pools "
        "are labeled. These numbers bear on the *interpretation* of the I/O (does input enter at "
        "biologically input-side neurons?), not on the connectome>random claim.\n"
    )
    lines.append(
        "Match % is computed over **typed neurons only** (untyped neurons cannot be judged). "
        "*canonical* = textbook afferent/efferent classes; *broad* = additionally counts cell types "
        "that genuinely project into/out of the region but are integrative/modulatory rather than "
        "primary afferents.\n"
    )
    lines.append("## Takeaways\n")
    lines.append(
        "- **Central complex (typed):** the `output` pool is almost entirely genuine CX "
        "*out-projecting* cell types (PFL/PFR premotor + FS/FC/FR columnar), with the canonical "
        "premotor steering families (PFL/PFR) a minority of those; the `sensory` pool is a mix of "
        "canonical ring/noduli input families (ER/ExR/LNO/SpsP) and fan-body columnar neurons that "
        "receive extra-CX input. So the ports land on biologically input-side / output-side cells, "
        "though `sensory` is broader than 'primary afferents'.\n"
        "- **Mushroom body (hemibrain, typed):** the ROI-flow heuristic recovers MB cell-class "
        "structure without any cell-type input — the `internal` pool is dominated by Kenyon cells, "
        "the `output` pool is majority MBON, and the `sensory` pool is dopaminergic (PAM) "
        "reinforcement neurons plus projection neurons (the US + CS input streams). This is the "
        "type-verified evidence that `sensory≈input / internal≈KC / output≈MBON`.\n"
        "- **FlyWire MB / optic lobe (used for the main results):** the export carries no cell-type "
        "labels, so the pools cannot be type-verified there; the hemibrain MB above is the validation "
        "by analogy, and pool *proportions* are consistent with it.\n"
        "- **Bottom line:** 'sensory'/'output' are extra-regional-flow-biased pools, not perfect "
        "afferent/efferent rosters, but where types exist they line up with the region's real "
        "input/output cell classes. And because controls share the identical pools, none of this "
        "changes the connectome>random comparison.\n"
    )
    for r in results:
        lines.append(f"\n## {r['region_label']}  — `{r['dir']}`\n")
        c = r["counts"]
        lines.append(
            f"N = {r['N']} · sensory {c['sensory']} / internal {c['internal']} / output {c['output']} · "
            f"type coverage {r['type_coverage']}/{r['N']} ({r['type_coverage_pct']})\n"
        )
        if r["type_coverage"] == 0:
            lines.append(
                "**Cell types unavailable in this export** (`type`/`instance` columns empty; only "
                "`predictedNt` present). The sensory/internal/output pools are therefore pure ROI-flow "
                "proxies. By connectivity logic they are *expected* to place input-projecting cells "
                "(e.g. PNs) in `sensory`, region-intrinsic cells (e.g. KCs) in `internal`, and "
                "out-projecting cells (e.g. MBONs) in `output`, and the pool sizes are consistent with "
                "that — but the correspondence is **not type-verified here**. See the typed sibling "
                "connectome for the validated version.\n"
            )
        lines.append("| pool | N | typed | canonical match (of typed) | broad match (of typed) | top types |")
        lines.append("|---|---|---|---|---|---|")
        for pool in ("sensory", "output", "internal"):
            p = r["pools"][pool]
            can = p.get("match_canonical_pct_of_typed", "—")
            broad = p.get("match_broad_pct_of_typed", "—")
            top = ", ".join(f"{k}×{v}" for k, v in list(p["top_types"].items())[:5]) or "—"
            lines.append(
                f"| {pool} | {p['N']} | {p['typed']} ({p['typed_pct']}) | {can} | {broad} | {top} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--connectomes", nargs="*", default=DEFAULT_CONNECTOMES,
                    help="connectome directories containing pool_assignments.csv (+ neurons.csv)")
    ap.add_argument("--out", default=None, help="markdown output path (also printed to stdout)")
    args = ap.parse_args()

    results = []
    for c in args.connectomes:
        d = Path(c) if Path(c).is_absolute() else REPO / c
        if not (d / "pool_assignments.csv").exists():
            print(f"skip {c}: no pool_assignments.csv")
            continue
        results.append(analyze_one(d))

    md = render_markdown(results)
    print(md)
    if args.out:
        out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
