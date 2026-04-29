from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import CX_ROI_LABELS, HEMIBRAIN_DATASET, NEUPRINT_SERVER, OutputPaths


class NeuprintAcquisitionError(RuntimeError):
    """Raised when neuPrint export cannot be completed."""


def _load_neuprint_symbols() -> dict[str, Any]:
    try:
        from neuprint import Client, NeuronCriteria, fetch_adjacencies, fetch_neurons
    except ImportError as exc:
        raise NeuprintAcquisitionError(
            "neuprint-python is required for --mode download. Install the isolated "
            "requirements.txt in this experiment directory."
        ) from exc
    return {
        "Client": Client,
        "NeuronCriteria": NeuronCriteria,
        "fetch_adjacencies": fetch_adjacencies,
        "fetch_neurons": fetch_neurons,
    }


def create_client(
    server: str = NEUPRINT_SERVER,
    dataset: str = HEMIBRAIN_DATASET,
) -> Any:
    symbols = _load_neuprint_symbols()
    if not os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS") and not os.environ.get(
        "NEUPRINT_TOKEN"
    ):
        raise NeuprintAcquisitionError(
            "Set NEUPRINT_APPLICATION_CREDENTIALS before running --mode download."
        )
    return symbols["Client"](server, dataset=dataset)


def _flatten_roi_tree(tree: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(tree, str):
        names.add(_clean_roi_name(tree))
    elif isinstance(tree, dict):
        for key, value in tree.items():
            names.add(_clean_roi_name(str(key)))
            names.update(_flatten_roi_tree(value))
    elif isinstance(tree, (list, tuple, set)):
        for item in tree:
            names.update(_flatten_roi_tree(item))
    return names


def _clean_roi_name(name: str) -> str:
    return name.strip().removesuffix("*")


def fetch_roi_hierarchy(client: Any) -> Any:
    if hasattr(client, "fetch_roi_hierarchy"):
        for kwargs in (
            {"include_subprimary": True, "mark_primary": False, "format": "dict"},
            {"include_subprimary": True, "mark_primary": False},
            {"include_subprimary": True},
            {},
        ):
            try:
                return client.fetch_roi_hierarchy(**kwargs)
            except TypeError:
                continue
    try:
        from neuprint import fetch_roi_hierarchy as fetch_hierarchy
    except ImportError as exc:
        raise NeuprintAcquisitionError("Could not access neuPrint ROI hierarchy API.") from exc
    try:
        return fetch_hierarchy(
            include_subprimary=True,
            mark_primary=False,
            format="dict",
            client=client,
        )
    except TypeError:
        return fetch_hierarchy(client=client)


def resolve_cx_primary_rois(
    hierarchy: Any, requested: Iterable[str] = CX_ROI_LABELS
) -> tuple[str, ...]:
    all_names = _flatten_roi_tree(hierarchy)
    lowered = {name.lower(): name for name in all_names}
    resolved: list[str] = []
    missing: list[str] = []
    for label in requested:
        clean_label = _clean_roi_name(label)
        if clean_label in all_names:
            resolved.append(clean_label)
        elif clean_label.lower() in lowered:
            resolved.append(lowered[clean_label.lower()])
        else:
            missing.append(label)
    if missing:
        raise NeuprintAcquisitionError(
            f"Could not resolve primary CX ROI names from hierarchy: {missing}"
        )
    return tuple(resolved)


def _normalize_connections(connections: pd.DataFrame) -> pd.DataFrame:
    if connections.empty:
        return pd.DataFrame(columns=["bodyId_pre", "bodyId_post", "weight"])
    rename = {}
    candidates = {
        "bodyId_pre": ("bodyId_pre", "pre_bodyId", "pre", "bodyId_x"),
        "bodyId_post": ("bodyId_post", "post_bodyId", "post", "bodyId_y"),
        "weight": ("weight", "syn_count", "synapse_count", "count"),
    }
    for out_col, names in candidates.items():
        for name in names:
            if name in connections.columns:
                rename[name] = out_col
                break
    normalized = connections.rename(columns=rename).copy()
    required = {"bodyId_pre", "bodyId_post", "weight"}
    missing = required.difference(normalized.columns)
    if missing:
        raise NeuprintAcquisitionError(
            f"neuPrint adjacency export is missing required columns: {sorted(missing)}"
        )
    normalized["bodyId_pre"] = normalized["bodyId_pre"].astype("int64")
    normalized["bodyId_post"] = normalized["bodyId_post"].astype("int64")
    normalized["weight"] = pd.to_numeric(normalized["weight"], errors="coerce").fillna(0.0)
    aggregated = (
        normalized.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"]
        .sum()
        .sort_values(["bodyId_pre", "bodyId_post"])
    )
    return aggregated


def _call_fetch_neurons(client: Any, rois: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = _load_neuprint_symbols()
    criteria = symbols["NeuronCriteria"](rois=list(rois), roi_req="any")
    try:
        neurons, roi_counts = symbols["fetch_neurons"](criteria, client=client)
    except TypeError:
        neurons, roi_counts = symbols["fetch_neurons"](criteria)
    if "bodyId" not in neurons.columns:
        raise NeuprintAcquisitionError("neuPrint neuron export did not contain bodyId.")
    return neurons.copy(), roi_counts.copy()


def _call_fetch_adjacencies(client: Any, body_ids: list[int]) -> pd.DataFrame:
    symbols = _load_neuprint_symbols()
    criteria = symbols["NeuronCriteria"](bodyId=body_ids)
    try:
        result = symbols["fetch_adjacencies"](criteria, criteria, client=client)
    except TypeError:
        result = symbols["fetch_adjacencies"](criteria, criteria)
    if isinstance(result, tuple):
        connections = result[-1]
    else:
        connections = result
    return pd.DataFrame(connections).copy()


def download_exports(paths: OutputPaths) -> dict[str, Any]:
    client = create_client()
    hierarchy = fetch_roi_hierarchy(client)
    primary_rois = resolve_cx_primary_rois(hierarchy)
    neurons, roi_counts = _call_fetch_neurons(client, primary_rois)
    neurons = neurons.drop_duplicates("bodyId").sort_values("bodyId")
    body_ids = neurons["bodyId"].astype("int64").tolist()
    connections = _call_fetch_adjacencies(client, body_ids)
    connections = _normalize_connections(connections)
    body_set = set(body_ids)
    connections = connections[
        connections["bodyId_pre"].isin(body_set) & connections["bodyId_post"].isin(body_set)
    ].copy()

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    neurons.to_csv(paths.neurons_csv, index=False)
    roi_counts.to_csv(paths.roi_counts_csv, index=False)
    connections.to_csv(paths.connections_csv, index=False)

    roi_dump = paths.output_dir / "roi_hierarchy.json"
    with roi_dump.open("w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=2, sort_keys=True)

    return {
        "primary_rois": primary_rois,
        "neuron_count": int(len(neurons)),
        "edge_count": int(len(connections)),
        "roi_hierarchy_path": str(roi_dump),
    }


def require_raw_exports(paths: OutputPaths) -> None:
    missing = [
        path
        for path in (paths.neurons_csv, paths.roi_counts_csv, paths.connections_csv)
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing raw neuPrint export(s): "
            + ", ".join(str(path) for path in missing)
            + ". Run --mode download first, or provide cached exports in --output-dir."
        )
