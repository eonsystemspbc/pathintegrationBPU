from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_ccnlab_learning_curve.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_ccnlab_learning", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plot_ccnlab_learning_curve_writes_summary_and_paired_tables(tmp_path: Path) -> None:
    plotter = _load_module()
    rows = []
    for model, offset in (
        ("connectome_kalman_filter", 0.2),
        ("random_sparse_kalman_filter", 0.1),
        ("degree_preserving_kalman_filter", 0.15),
        ("weight_shuffle_kalman_filter", 0.12),
    ):
        for seed in (0, 1):
            for trial in (1, 2, 3):
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "experiment": "Acquisition",
                        "group": "paired",
                        "phase": "train",
                        "trial": trial,
                        "trial_in_phase": trial,
                        "has_cs": 1,
                        "learning_response_mean": offset + 0.1 * trial + 0.01 * seed,
                    }
                )
    pd.DataFrame(rows).to_csv(tmp_path / "ccnlab_trial_history.csv", index=False)

    rc = plotter.main(
        [
            str(tmp_path),
            "--learner",
            "kalman",
            "--max-trials",
            "3",
            "--early-trials",
            "2",
        ]
    )

    assert rc == 0
    assert (tmp_path / "ccnlab_learning_curve.png").exists()
    summary = pd.read_csv(tmp_path / "ccnlab_learning_curve_summary.csv")
    paired = pd.read_csv(tmp_path / "ccnlab_learning_curve_paired_comparisons.csv")
    assert set(summary["model"]) == {
        "connectome_kalman_filter",
        "random_sparse_kalman_filter",
        "degree_preserving_kalman_filter",
        "weight_shuffle_kalman_filter",
    }
    assert set(paired["baseline_model"]) == {
        "random_sparse_kalman_filter",
        "degree_preserving_kalman_filter",
        "weight_shuffle_kalman_filter",
    }
