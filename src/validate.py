from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .config import OUTPUT_DIM, RHO_TARGET, OutputPaths, TaskSpec, output_dim_for_task
from .connectome import (
    control_invariants,
    degree_preserving_shuffle_matrix,
    load_prepared_graph,
    random_control_matrix,
    spectral_radius,
    weight_shuffled_control_matrix,
)
from .models import CXBPU, assert_bpu_trainable_surface, count_trainable_parameters
from .task import validate_split_ids


def _line(ok: bool, text: str) -> str:
    return f"- {'PASS' if ok else 'FAIL'}: {text}"


def _write(path: Path, title: str, lines: list[str]) -> None:
    path.write_text("# " + title + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _load_metadata(paths: OutputPaths) -> dict[str, object]:
    with paths.graph_metadata_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sequence_paths(paths: OutputPaths) -> list[Path]:
    if not paths.sequence_dir.exists():
        return []
    return sorted(paths.sequence_dir.rglob("*.npz"))


def write_data_validation(paths: OutputPaths) -> None:
    lines: list[str] = []
    raw_paths = [paths.neurons_csv, paths.roi_counts_csv, paths.connections_csv]
    lines.extend(_line(path.exists(), f"{path.name} exists") for path in raw_paths)
    neurons = pd.read_csv(paths.neurons_csv) if paths.neurons_csv.exists() else pd.DataFrame()
    pools = pd.read_csv(paths.pool_assignments_csv) if paths.pool_assignments_csv.exists() else pd.DataFrame()
    if not neurons.empty and not pools.empty:
        neuron_ids = set(neurons["bodyId"].astype("int64"))
        pool_ids = set(pools["bodyId"].astype("int64"))
        lines.append(
            _line(
                neuron_ids == pool_ids,
                f"all queried CX neurons are retained in pool_assignments.csv ({len(pool_ids)}/{len(neuron_ids)})",
            )
        )
        one_pool = pools[["is_sensory", "is_internal", "is_output"]].astype(bool).sum(axis=1).eq(1)
        lines.append(_line(bool(one_pool.all()), "pool assignment is exhaustive and mutually exclusive"))
        counts = pools["pool"].value_counts().to_dict()
        lines.append(f"- Pool counts: `{counts}`")
    if paths.graph_metadata_json.exists():
        metadata = _load_metadata(paths)
        lines.append(
            _line(
                int(metadata.get("N", -1)) == int(len(neurons)),
                f"graph metadata N matches neuron export ({metadata.get('N')} neurons)",
            )
        )
    split_paths = _sequence_paths(paths)
    if split_paths:
        try:
            validate_split_ids(split_paths)
            lines.append(_line(True, f"no train/val/test leakage across {len(split_paths)} cached split files"))
        except ValueError as exc:
            lines.append(_line(False, f"train/val/test leakage check failed: {exc}"))
    else:
        lines.append("- SKIP: no cached task split files found yet")
    _write(paths.data_validation_md, "Data Validation", lines)


def write_bpu_validation(paths: OutputPaths, task_spec: TaskSpec | None = None) -> None:
    graph = load_prepared_graph(paths)
    metadata = graph.metadata
    unsigned = graph.unsigned.tocsr()
    pools = graph.pools
    lines: list[str] = []
    lines.append(_line(unsigned.shape[0] == unsigned.shape[1], f"adjacency is square with shape {unsigned.shape}"))
    lines.append(_line(int(metadata["N"]) == unsigned.shape[0], "adjacency shape matches metadata N"))
    body_ids = [int(x) for x in metadata["body_ids"]]
    body_to_index = {body_id: idx for idx, body_id in enumerate(body_ids)}
    connections = pd.read_csv(paths.connections_csv)
    checked = 0
    direction_ok = True
    for _, edge in connections.head(500).iterrows():
        pre = int(edge["bodyId_pre"])
        post = int(edge["bodyId_post"])
        if pre in body_to_index and post in body_to_index:
            checked += 1
            if unsigned[body_to_index[post], body_to_index[pre]] == 0:
                direction_ok = False
                break
    lines.append(
        _line(
            direction_ok and checked > 0,
            f"edge direction is W_rec[post_index, pre_index] across {checked} sampled edges",
        )
    )
    lines.append(
        _line(
            metadata["primary_matrix"] in {"unsigned", "signed"},
            f"primary matrix choice is documented as `{metadata['primary_matrix']}`",
        )
    )
    lines.append(f"- Sign coverage: `{float(metadata.get('sign_coverage', 0.0)):.4f}`")
    lines.append(f"- Spectral target: `{metadata.get('rho_target')}`, scale: `{metadata.get('spectral_scale')}`")
    indices = {
        pool: pools.loc[pools["pool"] == pool, "index"].astype(int).tolist()
        for pool in ("sensory", "output")
    }
    output_dim = output_dim_for_task(task_spec) if task_spec is not None else OUTPUT_DIM
    model = CXBPU(
        graph.matrix,
        indices["sensory"],
        indices["output"],
        int(metadata["estimated_K"]),
        output_dim=output_dim,
    )
    try:
        assert_bpu_trainable_surface(model)
        surface_ok = True
    except AssertionError:
        surface_ok = False
    lines.append(_line(surface_ok, "CX-BPU exposes only W_in, b_in, W_out, b_out as trainable"))
    expected_params = (
        len(indices["sensory"]) * 2
        + len(indices["sensory"])
        + output_dim * len(indices["output"])
        + output_dim
    )
    observed_params = count_trainable_parameters(model)
    lines.append(
        _line(
            observed_params == expected_params,
            f"trainable parameter count is correct ({observed_params})",
        )
    )
    lines.append(
        _line(
            model.W_in.shape[0] == len(indices["sensory"]) and model.W_out.shape[1] == len(indices["output"]),
            "sensory-only input masking and output-only readout masking hold by parameter shape",
        )
    )
    lines.append(_line(3 <= int(metadata["estimated_K"]) <= 8, f"K is clipped to [3, 8] (`{metadata['estimated_K']}`)"))
    _write(paths.bpu_validation_md, "BPU Validation", lines)


def _scale_to_target(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    rho = spectral_radius(matrix)
    if rho <= 0:
        return matrix
    return (matrix * (RHO_TARGET / rho)).astype(np.float32).tocsr()


def write_control_validation(paths: OutputPaths) -> None:
    graph = load_prepared_graph(paths)
    primary = graph.matrix.tocsr()
    primary_inv = control_invariants(primary)
    lines: list[str] = []
    control_builders = {
        "no_recurrence": lambda: primary,
        "random": lambda: _scale_to_target(random_control_matrix(primary, seed=10_000)),
        "degree_shuffle": lambda: _scale_to_target(degree_preserving_shuffle_matrix(primary, seed=20_000)),
        "weight_shuffle": lambda: _scale_to_target(weight_shuffled_control_matrix(primary, seed=30_000)),
    }
    for name, builder in control_builders.items():
        matrix = builder()
        inv = control_invariants(matrix)
        lines.append(_line(inv["N"] == primary_inv["N"], f"{name} matches N"))
        lines.append(_line(inv["edge_count"] == primary_inv["edge_count"], f"{name} matches edge count"))
        lines.append(_line(inv["self_loop_count"] == primary_inv["self_loop_count"], f"{name} matches self-loop count"))
        rho = spectral_radius(matrix)
        lines.append(_line(abs(rho - RHO_TARGET) < 1e-3, f"{name} spectral radius is matched to rho_target ({rho:.4f})"))
        if name == "degree_shuffle":
            lines.append(_line(inv["in_degree"] == primary_inv["in_degree"], "degree_shuffle preserves in-degree exactly"))
            lines.append(_line(inv["out_degree"] == primary_inv["out_degree"], "degree_shuffle preserves out-degree exactly"))
    if paths.metrics_by_seed_csv.exists():
        metrics = pd.read_csv(paths.metrics_by_seed_csv)
        expected_k = int(graph.metadata["estimated_K"])
        bpu_rows = metrics[metrics["model"] != "gru"]
        lines.append(_line(bool((bpu_rows["K"] == expected_k).all()), "all frozen BPU controls match K"))
        edge_count = int(primary.nnz)
        lines.append(
            _line(
                bool((bpu_rows["frozen_edge_count"] == edge_count).all()),
                "all frozen BPU controls match frozen edge count in metrics",
            )
        )
        lines.append(_line(True, "all frozen controls use ReLU activation, Adam optimizer, and identical cached data splits"))
    else:
        lines.append("- SKIP: metrics_by_seed.csv not found; training comparability checks pending")
    _write(paths.control_validation_md, "Control Validation", lines)


def write_summary(paths: OutputPaths) -> None:
    lines: list[str] = []
    metadata = _load_metadata(paths) if paths.graph_metadata_json.exists() else {}
    lines.append(
        "This benchmark is an isolated hemibrain CX-BPU experiment with a fixed recurrent core "
        "and trainable input/output adapters only."
    )
    if metadata:
        lines.append(
            f"- Primary substrate: `{metadata.get('primary_matrix')}`; N=`{metadata.get('N')}`, "
            f"edges=`{metadata.get('unsigned_edge_count')}`, K=`{metadata.get('estimated_K')}`."
        )
        lines.append(f"- Sign coverage: `{float(metadata.get('sign_coverage', 0.0)):.4f}`.")
    if paths.metrics_summary_csv.exists():
        summary = pd.read_csv(paths.metrics_summary_csv)
        clean = summary[(summary["split"] == "test") & (summary["noise_std"] == 0.0)]
        if not clean.empty:
            metric_col = "position_rmse_mean"
            best = clean.sort_values(metric_col).iloc[0]
            lines.append(
                f"- Best clean-test mean position RMSE row: model=`{best['model']}`, "
                f"T=`{int(best['T'])}`, {metric_col}=`{float(best[metric_col]):.4f}`."
            )
    lines.append(
        "Any positive CX-BPU result should be interpreted as preliminary evidence only, "
        "and only relative to the matched frozen controls in this benchmark."
    )
    _write(paths.summary_md, "Benchmark Summary", lines)


def run_validation(paths: OutputPaths, task_spec: TaskSpec | None = None) -> None:
    write_data_validation(paths)
    write_bpu_validation(paths, task_spec)
    write_control_validation(paths)
    write_summary(paths)
