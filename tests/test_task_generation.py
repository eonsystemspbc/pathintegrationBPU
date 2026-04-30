from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.config import TASK_CX_POLAR_BUMP, TaskSpec
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
