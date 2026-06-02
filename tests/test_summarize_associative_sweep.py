from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_associative_sweep.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("summarize_assoc_sweep", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarizer_discovers_child_metrics_and_writes_report(tmp_path: Path) -> None:
    summarize = _load_module()
    output_dir = tmp_path / "sweep"
    hemibrain = output_dir / "jobs" / "hemibrain_fast_memory_seed0"
    random = output_dir / "jobs" / "random_sparse_fast_memory_seed0"
    hemibrain.mkdir(parents=True)
    random.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "model": "hemibrain_fast_memory",
                "seed": 0,
                "runtime": "sparse_fast_memory",
                "N": 12,
                "trainable_params": 100,
                "test_query_accuracy": 0.70,
                "test_initial_query_accuracy": 0.75,
                "test_reversal_query_accuracy": 0.65,
            }
        ]
    ).to_csv(hemibrain / "metrics_by_seed.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "random_sparse_fast_memory",
                "seed": 0,
                "runtime": "sparse_fast_memory",
                "N": 12,
                "trainable_params": 100,
                "test_query_accuracy": 0.60,
                "test_initial_query_accuracy": 0.64,
                "test_reversal_query_accuracy": 0.56,
            }
        ]
    ).to_csv(random / "metrics_by_seed.csv", index=False)

    code = summarize.write_summary(output_dir, rewrite_metrics=True)

    leaderboard = pd.read_csv(output_dir / "leaderboard.csv")
    assert code == 0
    assert leaderboard.loc[0, "model"] == "hemibrain_fast_memory"
    assert round(float(leaderboard.loc[0, "delta_vs_random_sparse_fast_memory"]), 6) == 0.10
    assert (output_dir / "metrics_by_seed.csv").exists()
    assert (output_dir / "metrics_summary.csv").exists()
    assert (output_dir / "sweep_report.md").exists()
