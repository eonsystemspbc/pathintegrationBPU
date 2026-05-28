from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.config import TASK_CX_LANDMARK_BUMP, TASK_CX_POLAR_BUMP, TaskSpec, input_dim_for_task
from src.task import build_targets, ensure_splits, load_split


def test_cx_polar_bump_targets_have_expected_geometry() -> None:
    spec = replace(TaskSpec(), kind=TASK_CX_POLAR_BUMP, heading_bins=8)
    controls = np.array(
        [
            [1.0, 0.0],
            [1.0, np.pi / 2.0],
            [0.5, 0.0],
        ],
        dtype=np.float32,
    )
    targets = build_targets(controls, spec)
    bump = targets[:, : spec.heading_bins]
    bearing = targets[:, spec.heading_bins : spec.heading_bins + 2]
    distance = targets[:, spec.heading_bins + 2]

    assert targets.shape == (3, spec.heading_bins + 3)
    assert np.all(bump >= 0.0)
    assert np.all(bump <= 1.0)
    assert np.allclose(np.linalg.norm(bearing, axis=1), 1.0, atol=1e-6)
    assert np.all(distance >= 0.0)


def test_cx_polar_bump_splits_use_separate_cache_and_target_dim(tmp_path) -> None:
    spec = replace(
        TaskSpec(),
        kind=TASK_CX_POLAR_BUMP,
        heading_bins=8,
        train_count=3,
        val_count=2,
        test_count=2,
        train_T=5,
        test_T=(5,),
        noise_stds=(),
    )
    splits = ensure_splits(tmp_path, spec)
    assert splits
    assert all("cx_polar_bump_bins8" in split.path.parts for split in splits)
    train = load_split(next(split.path for split in splits if split.name == "train"))
    assert train["targets"].shape == (3, 5, spec.heading_bins + 3)
    assert str(train["task_kind"]) == TASK_CX_POLAR_BUMP


def test_cx_landmark_bump_splits_include_cue_channels_and_metadata(tmp_path) -> None:
    spec = replace(
        TaskSpec(),
        kind=TASK_CX_LANDMARK_BUMP,
        heading_bins=8,
        train_count=4,
        val_count=2,
        test_count=2,
        train_T=12,
        test_T=(12,),
        noise_stds=(0.10,),
        landmark_visible_prob=0.3,
        landmark_noise_std=0.02,
        passive_displacement_prob=0.2,
        passive_displacement_scale=0.5,
    )
    splits = ensure_splits(tmp_path, spec)
    assert splits
    assert all("cx_landmark_bump_bins8_vis0.30_jump0.20" in split.path.parts for split in splits)
    train = load_split(next(split.path for split in splits if split.name == "train"))
    assert train["inputs"].shape == (4, 12, input_dim_for_task(spec))
    assert train["targets"].shape == (4, 12, spec.heading_bins + 3)
    assert str(train["task_kind"]) == TASK_CX_LANDMARK_BUMP
    cue_mask = train["inputs"][:, :, 5]
    assert np.all((cue_mask == 0.0) | (cue_mask == 1.0))
    assert np.any(cue_mask == 1.0)
    assert float(train["landmark_visible_prob"]) == pytest.approx(spec.landmark_visible_prob)
