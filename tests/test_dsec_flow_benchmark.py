from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import sparse


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_dsec_flow_benchmark.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("dsec_flow", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_matrix(n: int = 8) -> sparse.csr_matrix:
    rows = np.array([0, 1, 2, 3, 4, 5, 6, 7, 3, 6], dtype=np.int64)
    cols = np.array([1, 2, 3, 4, 5, 6, 7, 0, 0, 2], dtype=np.int64)
    data = np.linspace(0.1, 0.8, rows.size, dtype=np.float32)
    return sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _write_tiny_dsec(root: Path) -> Path:
    seq = root / "train" / "tiny_00"
    events_dir = seq / "events_left"
    flow_dir = seq / "optical_flow_forward_event"
    events_dir.mkdir(parents=True)
    flow_dir.mkdir(parents=True)
    height, width = 48, 64
    with h5py.File(events_dir / "events.h5", "w") as handle:
        group = handle.create_group("events")
        t = np.arange(0, 210_000, 1000, dtype=np.int64)
        x = (np.arange(t.size) * 3 % width).astype(np.int16)
        y = (np.arange(t.size) * 5 % height).astype(np.int16)
        p = (np.arange(t.size) % 2).astype(np.uint8)
        group.create_dataset("t", data=t)
        group.create_dataset("x", data=x)
        group.create_dataset("y", data=y)
        group.create_dataset("p", data=p)
        handle.create_dataset("t_offset", data=np.array(1_000_000, dtype=np.int64))
        ms_to_idx = np.arange(0, 220, dtype=np.int64)
        ms_to_idx = np.minimum(ms_to_idx, t.size - 1)
        handle.create_dataset("ms_to_idx", data=ms_to_idx)
    rectify = np.zeros((height, width, 2), dtype=np.float32)
    yy, xx = np.indices((height, width))
    rectify[..., 0] = xx
    rectify[..., 1] = yy
    with h5py.File(events_dir / "rectify_map.h5", "w") as handle:
        handle.create_dataset("rectify_map", data=rectify)
    bench = _load_module()
    for idx in range(4):
        flow = np.zeros((height, width, 2), dtype=np.float32)
        flow[..., 0] = 0.25 + 0.05 * idx
        flow[..., 1] = -0.15
        valid = np.ones((height, width), dtype=bool)
        bench.write_dsec_flow_png(flow_dir / f"{idx:06d}.png", flow, valid)
    with (seq / "optical_flow_forward_timestamps.txt").open("w", encoding="utf-8") as handle:
        for idx in range(4):
            handle.write(f"{1_000_000 + idx * 20_000} {1_100_000 + idx * 20_000}\n")
    return root


def test_dsec_flow_png_roundtrip(tmp_path: Path) -> None:
    bench = _load_module()
    flow = np.zeros((8, 9, 2), dtype=np.float32)
    flow[..., 0] = 1.25
    flow[..., 1] = -2.5
    valid = np.zeros((8, 9), dtype=bool)
    valid[2:5, 3:7] = True
    path = tmp_path / "flow.png"

    bench.write_dsec_flow_png(path, flow, valid)
    loaded_flow, loaded_valid = bench.read_dsec_flow_png(path)

    np.testing.assert_allclose(loaded_flow, flow, atol=1 / 128)
    np.testing.assert_array_equal(loaded_valid, valid)


def test_dsec_dataset_discovers_samples_and_bins_events(tmp_path: Path) -> None:
    bench = _load_module()
    root = _write_tiny_dsec(tmp_path)
    samples = bench.discover_labeled_samples(root)
    spec = bench.DSECDataSpec(
        data_root=root,
        event_bins=4,
        temporal_groups=2,
        crop_height=32,
        crop_width=40,
        sensor_height=48,
        sensor_width=64,
        augment=False,
    )
    dataset = bench.DSECFlowDataset(samples, spec, train=False)
    events, flow, valid = dataset[0]

    assert len(samples) == 4
    assert events.shape == (8, 32, 40)
    assert flow.shape == (2, 32, 40)
    assert valid.shape == (1, 32, 40)
    assert torch.isfinite(events).all()
    assert events.sum() > 0


def test_connectome_variants_are_matched(tmp_path: Path) -> None:
    bench = _load_module()
    matrix_path = tmp_path / "matrix.npz"
    sparse.save_npz(matrix_path, _toy_matrix())

    seeded = bench.connectome_variant(matrix_path, "connectome_seeded", 8, 0)
    shuffled = bench.connectome_variant(matrix_path, "connectome_weight_shuffle", 8, 0)
    random = bench.connectome_variant(matrix_path, "random_init", 8, 0)

    assert seeded.shape == shuffled.shape == random.shape
    assert seeded.nnz == shuffled.nnz == random.nnz
    np.testing.assert_array_equal(seeded.tocoo().row, shuffled.tocoo().row)
    np.testing.assert_array_equal(seeded.tocoo().col, shuffled.tocoo().col)
    assert not np.allclose(seeded.data, shuffled.data)


def test_dsec_connectome_flow_model_forward() -> None:
    bench = _load_module()
    model = bench.DSECConnectomeFlowNet(
        matrix=_toy_matrix(),
        event_bins=4,
        temporal_groups=2,
        hidden_dim=16,
        ssm_blocks=1,
        corr_radius=1,
        connectome_steps=1,
    )
    events = torch.rand(2, 8, 32, 40)
    flow = torch.zeros(2, 2, 32, 40)
    valid = torch.ones(2, 1, 32, 40)

    preds = model(events, iters=2)
    loss, metrics = bench.sequence_flow_loss(preds, flow, valid, gamma=0.8, max_flow=400.0)
    loss.backward()

    assert len(preds) == 2
    assert preds[-1].shape == (2, 2, 32, 40)
    assert np.isfinite(metrics["epe"])
    assert model.connectome.edge_values.grad is not None
