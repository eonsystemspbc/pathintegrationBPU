from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CX_ROI_LABELS, OutputPaths


SENSORY_FAMILIES = ("ring", "ExR", "LNO", "LCNO", "GLNO", "SpsP")
OUTPUT_FAMILIES = ("PFL", "PFL2", "PFL3", "PFR")


class PoolAssignmentError(RuntimeError):
    """Raised when neuron pool assignment is incomplete or inconsistent."""


def _as_string(row: pd.Series) -> str:
    parts = []
    for col in ("type", "instance"):
        if col in row and pd.notna(row[col]):
            parts.append(str(row[col]))
    return " ".join(parts)


def _family_bias(name: str) -> str | None:
    lowered = name.lower()
    for token in SENSORY_FAMILIES:
        if token.lower() in lowered:
            return "sensory"
    for token in OUTPUT_FAMILIES:
        if token.lower() in lowered:
            return "output"
    return None


def _normalize_roi_counts(roi_counts: pd.DataFrame) -> pd.DataFrame:
    if roi_counts.empty:
        return pd.DataFrame(columns=["bodyId", "roi", "pre", "post"])
    lower_map = {col.lower(): col for col in roi_counts.columns}
    rename = {}
    for canonical in ("bodyId", "roi", "pre", "post"):
        candidates = [canonical, canonical.lower()]
        if canonical == "bodyId":
            candidates.extend(["bodyid", "body_id"])
        for candidate in candidates:
            if candidate in roi_counts.columns:
                rename[candidate] = canonical
                break
            if candidate.lower() in lower_map:
                rename[lower_map[candidate.lower()]] = canonical
                break
    out = roi_counts.rename(columns=rename).copy()
    missing = {"bodyId", "roi"}.difference(out.columns)
    if missing:
        raise PoolAssignmentError(
            f"roi_counts.csv is missing required columns: {sorted(missing)}"
        )
    for col in ("pre", "post"):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["bodyId"] = out["bodyId"].astype("int64")
    out["roi"] = out["roi"].astype(str)
    return out[["bodyId", "roi", "pre", "post"]]


def assign_pools(
    neurons: pd.DataFrame,
    roi_counts: pd.DataFrame,
    primary_rois: tuple[str, ...] = CX_ROI_LABELS,
) -> pd.DataFrame:
    if "bodyId" not in neurons.columns:
        raise PoolAssignmentError("neurons.csv must include bodyId.")
    neurons = neurons.drop_duplicates("bodyId").copy()
    neurons["bodyId"] = neurons["bodyId"].astype("int64")
    neurons = neurons.sort_values("bodyId").reset_index(drop=True)
    if "pre" not in neurons.columns:
        neurons["pre"] = 0.0
    if "post" not in neurons.columns:
        neurons["post"] = 0.0
    neurons["pre"] = pd.to_numeric(neurons["pre"], errors="coerce").fillna(0.0)
    neurons["post"] = pd.to_numeric(neurons["post"], errors="coerce").fillna(0.0)

    counts = _normalize_roi_counts(roi_counts)
    roi_set = set(primary_rois)
    cx_counts = (
        counts[counts["roi"].isin(roi_set)]
        .groupby("bodyId")[["pre", "post"]]
        .sum()
        .rename(columns={"pre": "pre_in_cx_primary", "post": "post_in_cx_primary"})
    )
    merged = neurons.merge(cx_counts, how="left", left_on="bodyId", right_index=True)
    merged["pre_in_cx_primary"] = merged["pre_in_cx_primary"].fillna(0.0)
    merged["post_in_cx_primary"] = merged["post_in_cx_primary"].fillna(0.0)

    merged["outside_input_frac"] = (
        (merged["post"] - merged["post_in_cx_primary"])
        / np.maximum(merged["post"].to_numpy(dtype=float), 1.0)
    ).clip(0.0, 1.0)
    merged["outside_output_frac"] = (
        (merged["pre"] - merged["pre_in_cx_primary"])
        / np.maximum(merged["pre"].to_numpy(dtype=float), 1.0)
    ).clip(0.0, 1.0)

    rows: list[dict[str, object]] = []
    for index, row in merged.reset_index(drop=True).iterrows():
        input_frac = float(row["outside_input_frac"])
        output_frac = float(row["outside_output_frac"])
        diff = input_frac - output_frac
        name = _as_string(row)
        reason = "threshold"
        if abs(diff) < 0.10:
            pool = _family_bias(name) or "internal"
            reason = "type_family_tie_breaker" if pool != "internal" else "near_tie_internal_default"
        elif diff >= 0.10 and input_frac >= 0.20:
            pool = "sensory"
        elif -diff >= 0.10 and output_frac >= 0.20:
            pool = "output"
        else:
            pool = "internal"
            reason = "below_fraction_threshold"
        rows.append(
            {
                "bodyId": int(row["bodyId"]),
                "index": int(index),
                "type": row.get("type", ""),
                "instance": row.get("instance", ""),
                "pool": pool,
                "is_sensory": pool == "sensory",
                "is_internal": pool == "internal",
                "is_output": pool == "output",
                "outside_input_frac": input_frac,
                "outside_output_frac": output_frac,
                "post_in_cx_primary": float(row["post_in_cx_primary"]),
                "pre_in_cx_primary": float(row["pre_in_cx_primary"]),
                "assignment_reason": reason,
            }
        )
    assignments = pd.DataFrame(rows).sort_values("index")
    validate_pool_assignments(assignments, expected_body_ids=neurons["bodyId"].tolist())
    return assignments


def validate_pool_assignments(
    assignments: pd.DataFrame, expected_body_ids: list[int] | None = None
) -> None:
    required = {"bodyId", "pool", "is_sensory", "is_internal", "is_output"}
    missing = required.difference(assignments.columns)
    if missing:
        raise PoolAssignmentError(f"pool assignment missing columns: {sorted(missing)}")
    if assignments["bodyId"].duplicated().any():
        duplicated = assignments.loc[assignments["bodyId"].duplicated(), "bodyId"].tolist()
        raise PoolAssignmentError(f"duplicate pool assignments: {duplicated[:5]}")
    valid_pools = {"sensory", "internal", "output"}
    bad = set(assignments["pool"]).difference(valid_pools)
    if bad:
        raise PoolAssignmentError(f"invalid pool labels: {sorted(bad)}")
    membership = assignments[["is_sensory", "is_internal", "is_output"]].astype(bool)
    counts = membership.sum(axis=1)
    if not (counts == 1).all():
        bad_rows = assignments.loc[counts != 1, "bodyId"].tolist()
        raise PoolAssignmentError(
            f"neurons must belong to exactly one pool; failed bodyIds: {bad_rows[:10]}"
        )
    if expected_body_ids is not None:
        expected = set(int(x) for x in expected_body_ids)
        observed = set(assignments["bodyId"].astype("int64"))
        if expected != observed:
            missing = sorted(expected.difference(observed))[:10]
            extra = sorted(observed.difference(expected))[:10]
            raise PoolAssignmentError(
                f"pool assignments do not match neurons; missing={missing}, extra={extra}"
            )


def write_pool_assignments(paths: OutputPaths, primary_rois: tuple[str, ...]) -> pd.DataFrame:
    neurons = pd.read_csv(paths.neurons_csv)
    roi_counts = pd.read_csv(paths.roi_counts_csv)
    assignments = assign_pools(neurons, roi_counts, primary_rois=primary_rois)
    Path(paths.pool_assignments_csv).parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(paths.pool_assignments_csv, index=False)
    return assignments
