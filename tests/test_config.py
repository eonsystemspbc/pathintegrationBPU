from __future__ import annotations

import pytest

from src.config import STRUCTURE_COMPARISON_MODELS, parse_args, output_dim_for_task


def test_structure_comparison_selects_matched_control_models() -> None:
    cfg = parse_args(["--mode", "train", "--comparison", "structure"])
    assert cfg.train.models == STRUCTURE_COMPARISON_MODELS
    assert cfg.train.log_every_seconds == 60.0


def test_explicit_models_override_comparison_preset() -> None:
    cfg = parse_args(
        [
            "--mode",
            "train",
            "--comparison",
            "structure",
            "--models",
            "cx_bpu",
            "random",
        ]
    )
    assert cfg.train.models == ("cx_bpu", "random")


def test_log_interval_can_be_overridden() -> None:
    cfg = parse_args(["--mode", "train", "--log-every-seconds", "15"])
    assert cfg.train.log_every_seconds == 15.0


def test_cx_polar_bump_task_sets_output_dimension() -> None:
    cfg = parse_args(
        [
            "--mode",
            "train",
            "--task",
            "cx_polar_bump",
            "--heading-bins",
            "16",
            "--home-distance-scale",
            "30",
            "--bump-kappa",
            "6",
        ]
    )
    assert cfg.task.kind == "cx_polar_bump"
    assert cfg.task.home_distance_scale == 30.0
    assert cfg.task.bump_kappa == 6.0
    assert output_dim_for_task(cfg.task) == 19


def test_invalid_cx_polar_bump_knobs_fail_fast() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--task", "cx_polar_bump", "--heading-bins", "3"])
