from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_omniglot_associative_benchmark.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("omniglot_assoc", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_matrix(n: int = 10) -> sparse.coo_matrix:
    rows = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 7], dtype=np.int64)
    cols = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2], dtype=np.int64)
    data = np.linspace(0.05, 0.6, rows.size, dtype=np.float32)
    keep = (rows < n) & (cols < n)
    rows = rows[keep]
    cols = cols[keep]
    data = data[keep]
    return sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)


def test_episodic_batch_encodes_support_query_and_reversal() -> None:
    omni = _load_module()
    train_bank, _, _ = omni.synthetic_feature_banks(
        feature_dim=12,
        samples_per_class=8,
        train_classes=12,
        val_classes=8,
        test_classes=8,
        seed=5,
        class_noise_std=0.01,
    )
    spec = omni.EpisodeSpec(
        way=4,
        shot=1,
        queries_per_class=2,
        reversal_count=2,
        feature_dim=train_bank.feature_dim,
        feature_noise_std=0.0,
    )
    batch = omni.generate_episode_batch(train_bank, spec, batch_size=3, rng=np.random.default_rng(7))

    assert batch.inputs.shape == (3, spec.timesteps, spec.input_dim)
    assert batch.targets.shape == (3, spec.timesteps)
    assert np.all(batch.support_mask.sum(axis=1) == spec.way * spec.shot + spec.reversal_count * spec.shot)
    assert np.all(batch.initial_query_mask.sum(axis=1) == spec.way * spec.queries_per_class)
    assert np.all(batch.reversal_query_mask.sum(axis=1) == spec.way * spec.queries_per_class)
    assert np.all(batch.query_mask == batch.initial_query_mask + batch.reversal_query_mask)

    label_slice = slice(spec.feature_dim, spec.feature_dim + spec.way)
    support_rows = batch.support_mask.astype(bool)
    query_rows = batch.query_mask.astype(bool)
    assert np.all(batch.inputs[support_rows, label_slice].sum(axis=1) == 1.0)
    assert np.all(batch.inputs[query_rows, label_slice].sum(axis=1) == 0.0)


def test_nearest_support_baseline_solves_easy_synthetic_episode() -> None:
    omni = _load_module()
    _, _, test_bank = omni.synthetic_feature_banks(
        feature_dim=16,
        samples_per_class=12,
        train_classes=10,
        val_classes=10,
        test_classes=10,
        seed=9,
        class_noise_std=0.001,
    )
    spec = omni.EpisodeSpec(
        way=5,
        shot=1,
        queries_per_class=2,
        reversal_count=0,
        feature_dim=test_bank.feature_dim,
        feature_noise_std=0.0,
    )
    metrics = omni.evaluate_nearest_support(
        test_bank,
        spec,
        batch_size=8,
        batches=3,
        seed=10,
        temperature=0.1,
    )
    assert metrics["query_accuracy"] > 0.95
    assert np.isnan(metrics["reversal_query_accuracy"])


def test_fast_memory_query_key_ignores_label_channels() -> None:
    import torch

    omni = _load_module()
    feature_dim = 3
    output_dim = 2
    input_dim = feature_dim + output_dim + 2
    model = omni.MatrixFastMemoryRNN(
        recurrent=_toy_matrix(n=8),
        input_dim=input_dim,
        output_dim=output_dim,
        feature_dim=feature_dim,
        runtime="dense",
        state_clip=5.0,
        memory_decay=1.0,
        memory_temperature=0.5,
        encoder_steps=2,
        seed=11,
    )
    support_col = feature_dim + output_dim
    query_col = support_col + 1
    label_slice = slice(feature_dim, feature_dim + output_dim)
    inputs = torch.zeros(1, 2, input_dim)
    inputs[:, :, :feature_dim] = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    inputs[:, 0, label_slice] = torch.tensor([[1.0, 0.0]])
    inputs[:, 0, support_col] = 1.0
    inputs[:, 1, query_col] = 1.0

    changed_query_labels = inputs.clone()
    changed_query_labels[:, 1, label_slice] = torch.tensor([[0.0, 1.0]])

    with torch.no_grad():
        logits = model(inputs)
        changed_logits = model(changed_query_labels)

    assert torch.allclose(logits[:, 1, :], changed_logits[:, 1, :], atol=1e-6)


def test_fast_memory_prototypes_are_count_normalized() -> None:
    import torch

    omni = _load_module()
    feature_dim = 3
    output_dim = 2
    input_dim = feature_dim + output_dim + 2
    model = omni.MatrixFastMemoryRNN(
        recurrent=_toy_matrix(n=8),
        input_dim=input_dim,
        output_dim=output_dim,
        feature_dim=feature_dim,
        runtime="dense",
        state_clip=5.0,
        memory_decay=1.0,
        memory_temperature=0.5,
        encoder_steps=2,
        seed=11,
    )
    support_col = feature_dim + output_dim
    query_col = support_col + 1
    label_slice = slice(feature_dim, feature_dim + output_dim)

    one_support = torch.zeros(1, 2, input_dim)
    one_support[:, :, :feature_dim] = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    one_support[:, 0, label_slice] = torch.tensor([[1.0, 0.0]])
    one_support[:, 0, support_col] = 1.0
    one_support[:, 1, query_col] = 1.0

    two_supports = torch.zeros(1, 3, input_dim)
    two_supports[:, :, :feature_dim] = torch.tensor(
        [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    )
    two_supports[:, 0, label_slice] = torch.tensor([[1.0, 0.0]])
    two_supports[:, 1, label_slice] = torch.tensor([[1.0, 0.0]])
    two_supports[:, 0, support_col] = 1.0
    two_supports[:, 1, support_col] = 1.0
    two_supports[:, 2, query_col] = 1.0

    with torch.no_grad():
        one_logits = model(one_support)[:, 1, :]
        two_logits = model(two_supports)[:, 2, :]

    assert torch.allclose(one_logits, two_logits, atol=1e-6)


def test_fast_memory_can_freeze_recurrent_weights() -> None:
    omni = _load_module()
    feature_dim = 4
    output_dim = 3
    input_dim = feature_dim + output_dim + 2
    trainable = omni.MatrixFastMemoryRNN(
        recurrent=_toy_matrix(n=8),
        input_dim=input_dim,
        output_dim=output_dim,
        feature_dim=feature_dim,
        runtime="sparse",
        state_clip=5.0,
        memory_decay=0.9,
        memory_temperature=0.5,
        encoder_steps=1,
        seed=11,
        freeze_recurrent=False,
    )
    frozen = omni.MatrixFastMemoryRNN(
        recurrent=_toy_matrix(n=8),
        input_dim=input_dim,
        output_dim=output_dim,
        feature_dim=feature_dim,
        runtime="sparse",
        state_clip=5.0,
        memory_decay=0.9,
        memory_temperature=0.5,
        encoder_steps=1,
        seed=11,
        freeze_recurrent=True,
    )

    assert trainable.W_rec_values.requires_grad
    assert not frozen.W_rec_values.requires_grad
    assert frozen.trainable_parameter_count() < trainable.trainable_parameter_count()


def test_conv_protonet_forward_uses_raw_pixel_features() -> None:
    import torch

    omni = _load_module()
    image_size = 16
    way = 3
    model = omni.ConvProtoNetClassifier(
        feature_dim=image_size * image_size,
        output_dim=way,
        image_size=image_size,
        channels=4,
        embedding_dim=8,
        temperature=0.5,
        memory_decay=0.9,
        seed=13,
    )
    inputs = torch.zeros(2, 4, image_size * image_size + way + 2)
    inputs[:, :, : image_size * image_size] = torch.rand(2, 4, image_size * image_size)
    inputs[:, 0, image_size * image_size] = 1.0
    inputs[:, 0, image_size * image_size + way] = 1.0

    logits = model(inputs)

    assert logits.shape == (2, 4, way)


def test_conv_fast_memory_forward_routes_raw_pixels_through_connectome() -> None:
    import torch

    omni = _load_module()
    image_size = 16
    raw_feature_dim = image_size * image_size
    way = 3
    model = omni.ConvMatrixFastMemoryRNN(
        recurrent=_toy_matrix(n=10),
        input_dim=raw_feature_dim + way + 2,
        output_dim=way,
        raw_feature_dim=raw_feature_dim,
        image_size=image_size,
        conv_channels=4,
        conv_embedding_dim=8,
        runtime="dense",
        state_clip=5.0,
        memory_decay=0.9,
        memory_temperature=0.5,
        encoder_steps=1,
        seed=17,
    )
    inputs = torch.zeros(2, 4, raw_feature_dim + way + 2)
    inputs[:, :, :raw_feature_dim] = torch.rand(2, 4, raw_feature_dim)
    inputs[:, 0, raw_feature_dim] = 1.0
    inputs[:, 0, raw_feature_dim + way] = 1.0

    logits = model(inputs)

    assert logits.shape == (2, 4, way)
    assert model.recurrent_parameter_count() == 100


def test_conv_fast_memory_residual_matches_conv_protonet_when_core_disabled() -> None:
    import torch

    omni = _load_module()
    image_size = 16
    raw_feature_dim = image_size * image_size
    way = 3
    seed = 23
    protonet = omni.ConvProtoNetClassifier(
        feature_dim=raw_feature_dim,
        output_dim=way,
        image_size=image_size,
        channels=4,
        embedding_dim=8,
        temperature=0.5,
        memory_decay=0.9,
        seed=seed,
    )
    hybrid = omni.ConvMatrixFastMemoryRNN(
        recurrent=_toy_matrix(n=10),
        input_dim=raw_feature_dim + way + 2,
        output_dim=way,
        raw_feature_dim=raw_feature_dim,
        image_size=image_size,
        conv_channels=4,
        conv_embedding_dim=8,
        runtime="dense",
        state_clip=5.0,
        memory_decay=0.9,
        memory_temperature=0.5,
        encoder_steps=1,
        seed=seed,
        protonet_residual_weight=1.0,
        protonet_temperature=0.5,
        protonet_memory_decay=0.9,
        connectome_logit_weight=0.0,
    )
    protonet.eval()
    hybrid.eval()
    generator = torch.Generator().manual_seed(123)
    inputs = torch.zeros(2, 4, raw_feature_dim + way + 2)
    inputs[:, :, :raw_feature_dim] = torch.rand(
        2,
        4,
        raw_feature_dim,
        generator=generator,
    )
    inputs[:, 0, raw_feature_dim] = 1.0
    inputs[:, 0, raw_feature_dim + way] = 1.0
    inputs[:, 1, raw_feature_dim + 1] = 1.0
    inputs[:, 1, raw_feature_dim + way] = 1.0
    inputs[:, 2:, raw_feature_dim + way + 1] = 1.0

    with torch.no_grad():
        protonet_logits = protonet(inputs)
        hybrid_logits = hybrid(inputs)

    assert torch.allclose(protonet_logits, hybrid_logits, atol=1e-6)


def test_omniglot_associative_smoke_run_writes_metrics_and_report(tmp_path: Path) -> None:
    omni = _load_module()
    matrix_path = tmp_path / "adjacency_unsigned.npz"
    out = tmp_path / "omniglot_assoc"
    sparse.save_npz(matrix_path, _toy_matrix().tocsr())

    code = omni.main(
        [
            "--dataset",
            "synthetic",
            "--matrix",
            str(matrix_path),
            "--output-dir",
            str(out),
            "--device",
            "cpu",
            "--models",
            "hemibrain_seeded",
            "hemibrain_fast_memory",
            "mlp_protonet",
            "gru",
            "nearest_support",
            "--seeds",
            "0",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--train-batches",
            "1",
            "--val-batches",
            "1",
            "--test-batches",
            "1",
            "--way",
            "3",
            "--shot",
            "1",
            "--queries-per-class",
            "1",
            "--synthetic-feature-dim",
            "6",
            "--synthetic-samples-per-class",
            "6",
            "--synthetic-train-classes",
            "8",
            "--synthetic-val-classes",
            "8",
            "--synthetic-test-classes",
            "8",
            "--gru-hidden",
            "8",
            "--log-every-seconds",
            "0",
        ]
    )

    assert code == 0
    metrics = pd.read_csv(out / "metrics_by_seed.csv")
    assert sorted(metrics["model"].tolist()) == [
        "gru",
        "hemibrain_fast_memory",
        "hemibrain_seeded",
        "mlp_protonet",
        "nearest_support",
    ]
    assert (out / "metrics_summary.csv").exists()
    assert (out / "loss_history.csv").exists()
    assert (out / "omniglot_associative_report.md").exists()
    assert (out / "omniglot_associative_accuracy.png").exists()
    assert (out / "run_config.json").exists()
