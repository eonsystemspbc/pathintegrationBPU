from __future__ import annotations

from src.config import STRUCTURE_COMPARISON_MODELS, parse_args


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
