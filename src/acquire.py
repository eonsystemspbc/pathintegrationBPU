from __future__ import annotations

import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import CX_ROI_LABELS, HEMIBRAIN_DATASET, NEUPRINT_SERVER, OutputPaths


class NeuprintAcquisitionError(RuntimeError):
    """Raised when neuPrint export cannot be completed."""


FLYWIRE_ZENODO_RECORD = "10676866"
FLYWIRE_FILE_TEMPLATE = "https://zenodo.org/records/{record}/files/{filename}?download=1"
FLYWIRE_CONNECTION_COLUMNS = {
    "pre_pt_root_id",
    "post_pt_root_id",
    "neuropil",
    "syn_count",
    "gaba_avg",
    "ach_avg",
    "glut_avg",
    "oct_avg",
    "ser_avg",
    "da_avg",
}


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


def _load_pyarrow_feather() -> Any:
    try:
        import pyarrow.feather as feather
    except ImportError as exc:
        raise NeuprintAcquisitionError(
            "pyarrow is required for FlyWire feather files. Install the experiment "
            "requirements.txt, then rerun --mode download --connectome flywire_whole."
        ) from exc
    return feather


def _flywire_filename(kind: str, release: str) -> str:
    return f"{kind}_{release}.feather" if kind != "proofread_root_ids" else f"{kind}_{release}.npy"


def _download_url(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    tmp_path.replace(path)


def _ensure_flywire_file(download_dir: Path, filename: str) -> Path:
    path = download_dir / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    url = FLYWIRE_FILE_TEMPLATE.format(record=FLYWIRE_ZENODO_RECORD, filename=filename)
    print(f"Downloading FlyWire file {filename} from Zenodo...")
    _download_url(url, path)
    return path


def _read_flywire_connections(path: Path) -> pd.DataFrame:
    feather = _load_pyarrow_feather()
    table = feather.read_table(path)
    available = set(table.column_names)
    missing = {"pre_pt_root_id", "post_pt_root_id", "syn_count"}.difference(available)
    if missing:
        raise NeuprintAcquisitionError(
            f"FlyWire proofread connections are missing columns: {sorted(missing)}"
        )
    columns = [col for col in table.column_names if col in FLYWIRE_CONNECTION_COLUMNS]
    return table.select(columns).to_pandas()


def _flywire_transmitter_labels(connections: pd.DataFrame) -> pd.Series:
    nt_columns = {
        "ach_avg": "ACh",
        "gaba_avg": "GABA",
        "glut_avg": "Glu",
        "oct_avg": "Oct",
        "ser_avg": "5HT",
        "da_avg": "DA",
    }
    present = [col for col in nt_columns if col in connections.columns]
    if not present:
        return pd.Series(dtype=object)
    weights = connections["syn_count"].astype(float)
    scores: dict[str, pd.Series] = {}
    for col in present:
        weighted = weights * pd.to_numeric(connections[col], errors="coerce").fillna(0.0)
        scores[nt_columns[col]] = weighted.groupby(connections["pre_pt_root_id"]).sum()
    score_df = pd.DataFrame(scores).fillna(0.0)
    totals = score_df.sum(axis=1)
    winners = score_df.idxmax(axis=1)
    confidence = score_df.max(axis=1) / totals.replace(0.0, np.nan)
    labels = winners.where((confidence >= 0.70) & winners.isin(["ACh", "GABA", "Glu"]), "")
    labels.name = "predictedNt"
    return labels


def _write_flywire_neurons(
    paths: OutputPaths,
    root_ids_path: Path,
    connections: pd.DataFrame,
) -> pd.DataFrame:
    root_ids = np.load(root_ids_path).astype(np.int64)
    body_ids = pd.Index(root_ids, name="bodyId").drop_duplicates().sort_values()
    pre_totals = (
        connections.groupby("pre_pt_root_id")["syn_count"].sum().rename("pre").astype(float)
    )
    post_totals = (
        connections.groupby("post_pt_root_id")["syn_count"].sum().rename("post").astype(float)
    )
    labels = _flywire_transmitter_labels(connections)
    neurons = pd.DataFrame({"bodyId": body_ids.to_numpy(dtype=np.int64)})
    neurons = neurons.merge(pre_totals, how="left", left_on="bodyId", right_index=True)
    neurons = neurons.merge(post_totals, how="left", left_on="bodyId", right_index=True)
    neurons = neurons.merge(labels, how="left", left_on="bodyId", right_index=True)
    neurons["type"] = ""
    neurons["instance"] = ""
    neurons["pre"] = neurons["pre"].fillna(0.0)
    neurons["post"] = neurons["post"].fillna(0.0)
    neurons["predictedNt"] = neurons["predictedNt"].fillna("")
    neurons = neurons[["bodyId", "type", "instance", "pre", "post", "predictedNt"]]
    neurons.to_csv(paths.neurons_csv, index=False)
    return neurons


def _write_flywire_roi_counts(paths: OutputPaths, connections: pd.DataFrame) -> pd.DataFrame:
    if "neuropil" not in connections.columns:
        counts = pd.DataFrame(
            {
                "bodyId": pd.concat(
                    [connections["pre_pt_root_id"], connections["post_pt_root_id"]]
                )
                .drop_duplicates()
                .astype("int64"),
                "roi": "whole_brain",
                "pre": 0.0,
                "post": 0.0,
            }
        )
        counts.to_csv(paths.roi_counts_csv, index=False)
        return counts
    pre = (
        connections.groupby(["pre_pt_root_id", "neuropil"])["syn_count"]
        .sum()
        .rename("pre")
        .reset_index()
        .rename(columns={"pre_pt_root_id": "bodyId", "neuropil": "roi"})
    )
    post = (
        connections.groupby(["post_pt_root_id", "neuropil"])["syn_count"]
        .sum()
        .rename("post")
        .reset_index()
        .rename(columns={"post_pt_root_id": "bodyId", "neuropil": "roi"})
    )
    counts = pre.merge(post, how="outer", on=["bodyId", "roi"])
    counts["pre"] = counts["pre"].fillna(0.0)
    counts["post"] = counts["post"].fillna(0.0)
    counts["bodyId"] = counts["bodyId"].astype("int64")
    counts.to_csv(paths.roi_counts_csv, index=False)
    return counts


def _write_flywire_connections(paths: OutputPaths, connections: pd.DataFrame) -> pd.DataFrame:
    out = (
        connections.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False)["syn_count"]
        .sum()
        .rename(
            columns={
                "pre_pt_root_id": "bodyId_pre",
                "post_pt_root_id": "bodyId_post",
                "syn_count": "weight",
            }
        )
    )
    out["bodyId_pre"] = out["bodyId_pre"].astype("int64")
    out["bodyId_post"] = out["bodyId_post"].astype("int64")
    out["weight"] = out["weight"].astype(np.float32)
    out.to_csv(paths.connections_csv, index=False)
    return out


def download_flywire_exports(
    paths: OutputPaths,
    release: str = "783",
    download_dir: Path | None = None,
) -> dict[str, Any]:
    download_dir = (
        Path(download_dir)
        if download_dir is not None
        else paths.cache_dir / f"flywire_release_{release}"
    )
    download_dir.mkdir(parents=True, exist_ok=True)
    root_ids_path = _ensure_flywire_file(
        download_dir, _flywire_filename("proofread_root_ids", release)
    )
    connections_path = _ensure_flywire_file(
        download_dir, _flywire_filename("proofread_connections", release)
    )
    connections = _read_flywire_connections(connections_path)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    neurons = _write_flywire_neurons(paths, root_ids_path, connections)
    _write_flywire_roi_counts(paths, connections)
    aggregated = _write_flywire_connections(paths, connections)
    source_metadata = {
        "connectome": "flywire_whole",
        "release": release,
        "zenodo_record": FLYWIRE_ZENODO_RECORD,
        "root_ids_path": str(root_ids_path),
        "proofread_connections_path": str(connections_path),
        "source_rows": int(len(connections)),
        "aggregated_edge_count": int(len(aggregated)),
    }
    with (paths.output_dir / "flywire_sources.json").open("w", encoding="utf-8") as f:
        json.dump(source_metadata, f, indent=2, sort_keys=True)
    return {
        "primary_rois": ("whole_brain",),
        "neuron_count": int(len(neurons)),
        "edge_count": int(len(aggregated)),
        "flywire_sources_path": str(paths.output_dir / "flywire_sources.json"),
        "download_dir": str(download_dir),
    }


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
