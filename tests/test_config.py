from __future__ import annotations

import pytest

from src.config import (
    CX_LANDMARK_INPUT_DIM,
    WHOLE_BRAIN_COMPARISON_MODELS,
    STRUCTURE_COMPARISON_MODELS,
    input_dim_for_task,
    parse_args,
    output_dim_for_task,
)


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


def test_flywire_whole_selects_scalable_default_models() -> None:
    cfg = parse_args(["--mode", "train", "--connectome", "flywire_whole"])
    assert cfg.connectome == "flywire_whole"
    assert cfg.train.models == WHOLE_BRAIN_COMPARISON_MODELS
    assert cfg.train.models[0] == "connectome_bpu"


def test_whole_brain_comparison_selects_scalable_models() -> None:
    cfg = parse_args(["--mode", "train", "--comparison", "whole_brain"])
    assert cfg.train.models == WHOLE_BRAIN_COMPARISON_MODELS


def test_log_interval_can_be_overridden() -> None:
    cfg = parse_args(["--mode", "train", "--log-every-seconds", "15"])
    assert cfg.train.log_every_seconds == 15.0


def test_recurrent_training_mode_can_be_overridden() -> None:
    cfg = parse_args(["--mode", "train", "--train-recurrent", "observed"])
    assert cfg.train.train_recurrent == "observed"


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


def test_cx_landmark_bump_task_sets_task_knobs_and_dimensions() -> None:
    cfg = parse_args(
        [
            "--mode",
            "train",
            "--task",
            "cx_landmark_bump",
            "--heading-bins",
            "16",
            "--landmark-visible-prob",
            "0.25",
            "--landmark-noise-std",
            "0.07",
            "--passive-displacement-prob",
            "0.12",
            "--passive-displacement-scale",
            "1.4",
        ]
    )
    assert cfg.task.kind == "cx_landmark_bump"
    assert cfg.task.landmark_visible_prob == 0.25
    assert cfg.task.landmark_noise_std == 0.07
    assert cfg.task.passive_displacement_prob == 0.12
    assert cfg.task.passive_displacement_scale == 1.4
    assert output_dim_for_task(cfg.task) == 19
    assert input_dim_for_task(cfg.task) == CX_LANDMARK_INPUT_DIM


def test_invalid_cx_polar_bump_knobs_fail_fast() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--task", "cx_polar_bump", "--heading-bins", "3"])
