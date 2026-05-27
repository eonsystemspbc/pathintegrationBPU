from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from src.config import build_paths
from src.connectome import prepare_connectome
from test_integration_smoke import write_mock_exports


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_cross_region_transfer.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("cross_region", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepared_mock_graph(root: Path) -> Path:
    paths = build_paths(root)
    write_mock_exports(root)
    prepare_connectome(paths)
    return root


def test_cross_region_runner_smoke(tmp_path: Path) -> None:
    cross = _load_module()
    cx_dir = _prepared_mock_graph(tmp_path / "cx_graph")
    mb_dir = _prepared_mock_graph(tmp_path / "mb_graph")
    out_dir = tmp_path / "cross"

    code = cross.main(
        [
            "--mode",
            "all",
            "--pairs",
            "cross",
            "--cx-dir",
            str(cx_dir),
            "--mb-dir",
            str(mb_dir),
            "--output-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--seeds",
            "0",
            "--epochs",
            "1",
            "--assoc-batch-size",
            "2",
            "--path-batch-size",
            "2",
            "--num-workers",
            "0",
            "--log-every-seconds",
            "0",
            "--assoc-train-batches",
            "1",
            "--assoc-val-batches",
            "1",
            "--assoc-test-batches",
            "1",
            "--assoc-num-odors",
            "6",
            "--assoc-odor-dim",
            "6",
            "--assoc-odors-per-episode",
            "3",
            "--assoc-reversal-count",
            "1",
            "--assoc-odor-sparsity",
            "0.5",
            "--assoc-max-neurons",
            "6",
            "--path-train-count",
            "4",
            "--path-val-count",
            "2",
            "--path-test-count",
            "2",
            "--path-train-T",
            "5",
            "--path-test-T",
            "5",
            "--path-noise-stds",
            "0.0",
            "--path-recurrent-runtime",
            "sparse",
            "--path-train-recurrent",
            "observed",
            "--heading-bins",
            "8",
        ]
    )

    assert code == 0
    raw = pd.read_csv(out_dir / "cross_region_metrics_by_seed.csv")
    assert set(raw["condition"]) == {"assoc_cx_seeded", "path_mb_seeded"}
    success = pd.read_csv(out_dir / "cross_region_success_by_seed.csv")
    assert set(success["condition"]) == {"assoc_cx_seeded", "path_mb_seeded"}
    assert (out_dir / "cross_region_summary.csv").exists()
    assert (out_dir / "cross_region_task_success.png").exists()
    assert (out_dir / "cross_region_report.md").exists()
