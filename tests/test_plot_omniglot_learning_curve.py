from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_omniglot_learning_curve.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_omniglot_learning", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_history(output_dir: Path) -> None:
    rows = []
    values = {
        "hemibrain_conv_fast_memory": [0.91, 0.94, 0.96, 0.965],
        "random_sparse_conv_fast_memory": [0.90, 0.93, 0.95, 0.952],
        "weight_shuffle_conv_fast_memory": [0.89, 0.92, 0.94],
        "conv_protonet": [0.90, 0.925, 0.935, 0.94],
    }
    for model, curve in values.items():
        for seed in (0, 1):
            for epoch, value in enumerate(curve, start=1):
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "epoch": epoch,
                        "train_loss": 1.0 - value,
                        "val_loss": 1.0 - value,
                        "val_query_accuracy": value + 0.001 * seed,
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "loss_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "hemibrain_conv_fast_memory",
                "test_query_accuracy_mean": 0.961,
            },
            {
                "model": "random_sparse_conv_fast_memory",
                "test_query_accuracy_mean": 0.960,
            },
        ]
    ).to_csv(output_dir / "leaderboard.csv", index=False)


def test_learning_curve_plot_writes_figure_and_tables(tmp_path: Path) -> None:
    plotter = _load_module()
    _write_history(tmp_path)

    code = plotter.main(
        [
            str(tmp_path),
            "--label",
            "hemibrain_conv_fast_memory=FlyWire MB seeded",
        ]
    )

    summary = pd.read_csv(tmp_path / "learning_curve_summary.csv")
    paired = pd.read_csv(tmp_path / "learning_curve_paired_comparisons.csv")
    by_seed = pd.read_csv(tmp_path / "learning_curve_by_seed.csv")

    assert code == 0
    assert (tmp_path / "omniglot_learning_curve.png").exists()
    assert (tmp_path / "omniglot_learning_curve.png").stat().st_size > 0
    assert set(summary["model"]) == {
        "hemibrain_conv_fast_memory",
        "random_sparse_conv_fast_memory",
        "weight_shuffle_conv_fast_memory",
        "conv_protonet",
    }
    assert summary["common_epochs"].unique().tolist() == [3]
    seeded = summary.loc[summary["model"] == "hemibrain_conv_fast_memory"].iloc[0]
    assert seeded["common_mean"] == pytest.approx((0.9105 + 0.9405 + 0.9605) / 3)
    assert not paired.empty
    assert set(by_seed["seed"]) == {0, 1}
