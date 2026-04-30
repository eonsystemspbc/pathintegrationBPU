from __future__ import annotations

from src.config import STRUCTURE_COMPARISON_MODELS, parse_args


def test_structure_comparison_selects_matched_control_models() -> None:
    cfg = parse_args(["--mode", "train", "--comparison", "structure"])
    assert cfg.train.models == STRUCTURE_COMPARISON_MODELS


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

