from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_tartanair_optic_flow_benchmark.py"
)


def _load_module():
    script_dir = str(SCRIPT_PATH.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("tartanair_optic_flow", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tiny_tartanair_tree(root: Path) -> Path:
    traj = root / "ArchVizTinyHouseDay" / "Data_easy" / "P000"
    flow_dir = traj / "flow_lcam_front"
    flow_dir.mkdir(parents=True)
    poses = []
    for i in range(14):
        yaw = 0.01 * i
        quat = Rotation.from_euler("z", yaw).as_quat()
        poses.append([0.08 * i, 0.02 * i, 0.0, *quat])
    np.savetxt(traj / "pose_lcam_front.txt", np.asarray(poses), fmt="%.8f")
    for i in range(13):
        flow = np.zeros((12, 16, 2), dtype=np.float32)
        flow[..., 0] = 1.0 + 0.1 * i
        flow[..., 1] = -0.25
        mask = np.ones((12, 16), dtype=bool)
        np.savez_compressed(
            flow_dir / f"{i:06d}_{i + 1:06d}.npz",
            flow_fwd=flow,
            covisible_mask_fwd=mask,
        )
    return root


def test_relative_target_uses_pose_motion() -> None:
    bench = _load_module()
    poses = np.zeros((2, 7), dtype=np.float64)
    poses[0, 3:] = Rotation.identity().as_quat()
    poses[1, :3] = [1.0, 0.5, 0.0]
    poses[1, 3:] = Rotation.from_euler("z", 0.2).as_quat()
    target = bench.relative_target(
        poses,
        0,
        1,
        pose_convention="camera_to_world",
        yaw_scale=0.1,
        translation_scale=0.5,
    )
    np.testing.assert_allclose(target, np.array([2.0, 2.0, 1.0], dtype=np.float32), atol=1e-5)


def test_tartanair_windows_and_batches_have_expected_shapes(tmp_path: Path) -> None:
    bench = _load_module()
    root = _write_tiny_tartanair_tree(tmp_path)
    spec = bench.TartanAirSpec(
        data_root=root,
        envs=("ArchVizTinyHouseDay",),
        difficulties=("easy",),
        camera_name="lcam_front",
        sequence_len=4,
        hex_rings=2,
        acceptance_pixel_sigma=1.0,
        acceptance_samples=3,
        flow_clip=16.0,
        target_yaw_scale=0.1,
        target_translation_scale=0.5,
        pose_convention="camera_to_world",
        frame_stride=1,
        sample_stride=2,
        max_windows=0,
        split_seed=7,
        split_by_trajectory=False,
        train_fraction=0.6,
        val_fraction=0.2,
    )
    windows = bench.enumerate_tartanair_windows(spec)
    split = bench.split_windows(windows, spec)
    batch = bench.build_batch(
        split.train,
        batch_size=3,
        rng=np.random.default_rng(0),
        spec=spec,
        target_mean=split.target_mean,
        target_std=split.target_std,
        cache=bench.FlowCache(max_items=4),
    )
    assert len(windows) == 5
    assert batch.inputs.shape == (3, 4, spec.input_dim)
    assert batch.targets.shape == (3, 4, spec.output_dim)
    assert np.isfinite(batch.inputs).all()
    assert np.isfinite(batch.targets).all()
