from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

from src.channels import (
    CONNECTOME_FLYWIRE_OPTIC_LOBE,
    TASK_MB_ASSOCIATIVE_LEARNING,
    TASK_OPTIC_FLOW,
    build_adapter_mapping,
    task_channel_spec,
)
from src.config import CONNECTOME_HEMIBRAIN_CX, CONNECTOME_HEMIBRAIN_MUSHROOM_BODY, TASK_CX_POLAR_BUMP
from src.selector import select_connectome, selection_to_markdown


def _load_script(name: str, filename: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_selector_matches_cx_path_mb_assoc_and_optic_flow() -> None:
    assert select_connectome(TASK_CX_POLAR_BUMP).selected.connectome == CONNECTOME_HEMIBRAIN_CX
    assert (
        select_connectome(TASK_MB_ASSOCIATIVE_LEARNING).selected.connectome
        == CONNECTOME_HEMIBRAIN_MUSHROOM_BODY
    )
    assert select_connectome(TASK_OPTIC_FLOW).selected.connectome == CONNECTOME_FLYWIRE_OPTIC_LOBE


def test_adapter_mapping_records_channel_to_region_ports() -> None:
    spec = task_channel_spec(TASK_CX_POLAR_BUMP)
    mapping = build_adapter_mapping(TASK_CX_POLAR_BUMP, CONNECTOME_HEMIBRAIN_CX)

    assert spec.input_dim == mapping.input_dim
    assert spec.output_dim == mapping.output_dim
    assert mapping.mapping_score > 0.5
    assert set(mapping.input_channel_to_port) == {"forward_velocity", "turn_velocity"}
    assert "heading_bump" in mapping.output_channel_to_port


def test_selection_serializes_markdown_ranking() -> None:
    result = select_connectome(TASK_MB_ASSOCIATIVE_LEARNING)
    text = selection_to_markdown(result)
    assert "Candidate Ranking" in text
    assert "hemibrain_mushroom_body" in text
    assert result.to_dict()["mapping"]["input_dim"] > 0


def test_low_power_proxy_estimates_sparse_advantage(tmp_path: Path) -> None:
    low_power = _load_script("low_power_proxy", "run_low_power_proxy_benchmark.py")
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    matrix = sparse.random(20, 20, density=0.05, format="csr", random_state=1, dtype=np.float32)
    sparse.save_npz(graph_dir / "adjacency_unsigned.npz", matrix)
    (graph_dir / "graph_metadata.json").write_text(
        '{"connectome": "toy", "N": 20, "unsigned_edge_count": 20}',
        encoding="utf-8",
    )

    rows = low_power.footprint_rows(graph_dir, (32,))

    assert rows[0].dense_recurrent_ops_per_step == 400
    assert rows[0].sparse_recurrent_ops_per_step == matrix.nnz
    assert rows[0].ops_reduction > 1.0
    assert rows[0].memory_reduction > 1.0


def test_patent_plan_writes_command_artifacts(tmp_path: Path) -> None:
    planner = _load_script("patent_plan", "plan_patent_experiments.py")
    commands = planner.build_plan(
        output_root=tmp_path / "outputs",
        seeds=(0, 1),
        epochs=2,
        device="cpu",
    )
    planner.write_plan(tmp_path / "plan", commands)

    plan_md = (tmp_path / "plan" / "patent_experiment_plan.md").read_text(encoding="utf-8")
    assert "selector_cx_polar_bump" in plan_md
    assert "cross_region_match_mismatch_5seed" in plan_md
    assert (tmp_path / "plan" / "run_patent_experiments.sh").exists()
