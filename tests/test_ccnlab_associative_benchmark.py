from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_ccnlab_associative_benchmark.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("ccnlab_assoc", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_connectome_rpe_model_learns_cue_reward_association() -> None:
    bench = _load_module()
    recurrent = sparse.eye(12, dtype=np.float32, format="coo")
    encoder = bench.ConnectomeCueEncoder(
        recurrent=recurrent,
        cue_dim=1,
        time_basis_dim=2,
        feature_dim=6,
        encoder_steps=1,
        recurrent_gain=0.5,
        input_scale=1.0,
        hidden_scale=1.0,
        raw_input_scale=1.0,
        state_clip=5.0,
        seed=11,
    )
    model = bench.ConnectomeRPEConditioningModel(
        encoder=encoder,
        alpha=0.5,
        alpha_bias=0.0,
        trace_decay=0.8,
        weight_decay=0.0,
        response_clip=5.0,
        nonnegative_response=True,
    )

    initial = model.act([1.0], [], 0.0, 0)
    model.reset()
    for _ in range(40):
        model.act([1.0], [], 1.0, 0)
    learned = model.act([1.0], [], 0.0, 0)

    assert initial == pytest.approx(0.0)
    assert learned > 0.2


def test_feature_rescorla_wagner_learns_cue_reward_association() -> None:
    bench = _load_module()
    recurrent = sparse.eye(12, dtype=np.float32, format="coo")
    encoder = bench.ConnectomeCueEncoder(
        recurrent=recurrent,
        cue_dim=1,
        time_basis_dim=0,
        feature_dim=6,
        encoder_steps=1,
        recurrent_gain=0.5,
        input_scale=1.0,
        hidden_scale=1.0,
        raw_input_scale=1.0,
        state_clip=5.0,
        seed=11,
    )
    model = bench.FeatureRescorlaWagner(encoder=encoder, alpha=0.5)

    initial = model.act([1.0], [], 0.0, 0)
    model.reset()
    for _ in range(20):
        model.act([1.0], [], 1.0, 0)
    learned = model.act([1.0], [], 0.0, 0)

    assert initial == pytest.approx(0.0)
    assert learned > 0.9


def test_connectome_factories_preserve_matched_control_sizes() -> None:
    bench = _load_module()

    class FakeExperiment:
        cs_space = ("A", "B")
        ctx_space = ("K1",)

    args = bench.parse_args(
        [
            "--models",
            "hemibrain_seeded",
            "random_sparse",
            "weight_shuffle",
            "--feature-dim",
            "5",
            "--time-basis-dim",
            "3",
            "--encoder-steps",
            "1",
        ]
    )
    base = sparse.coo_matrix(
        (
            np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            (
                np.asarray([0, 1, 2, 3, 4]),
                np.asarray([1, 2, 3, 4, 0]),
            ),
        ),
        shape=(6, 6),
        dtype=np.float32,
    )

    rows = []
    for model_name in ("hemibrain_seeded", "random_sparse", "weight_shuffle"):
        factory, metadata = bench.make_connectome_factory(
            model_name=model_name,
            exp=FakeExperiment(),
            base_matrix=base,
            seed=0,
            args=args,
        )
        model = factory("g", 0)
        rows.append(metadata)
        assert metadata.N == 6
        assert metadata.init_nonzero_edges == base.nnz
        assert metadata.feature_dim == 5
        assert metadata.encoded_dim == 2 + 1 + 3 + 5
        assert model.act([1.0, 0.0], [1.0], 0.0, 0) == pytest.approx(0.0)

    assert {row.trainable_params for row in rows} == {rows[0].trainable_params}
    assert {row.recurrent_params for row in rows} == {base.nnz}


def test_graph_feature_factories_match_learning_rule_parameterization() -> None:
    bench = _load_module()

    class FakeExperiment:
        cs_space = ("A", "B")
        ctx_space = ("K1",)
        stimuli = {"g": [[None, None, None]]}

    args = bench.parse_args(
        [
            "--models",
            "connectome_rescorla_wagner",
            "random_sparse_rescorla_wagner",
            "weight_shuffle_rescorla_wagner",
            "connectome_kalman_filter",
            "random_sparse_kalman_filter",
            "weight_shuffle_kalman_filter",
            "connectome_temporal_difference",
            "random_sparse_temporal_difference",
            "weight_shuffle_temporal_difference",
            "--feature-learner-dim",
            "4",
            "--encoder-steps",
            "1",
        ]
    )
    base = sparse.coo_matrix(
        (
            np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            (
                np.asarray([0, 1, 2, 3, 4]),
                np.asarray([1, 2, 3, 4, 0]),
            ),
        ),
        shape=(6, 6),
        dtype=np.float32,
    )

    groups = {
        "rw": (
            "connectome_rescorla_wagner",
            "random_sparse_rescorla_wagner",
            "weight_shuffle_rescorla_wagner",
        ),
        "kalman": (
            "connectome_kalman_filter",
            "random_sparse_kalman_filter",
            "weight_shuffle_kalman_filter",
        ),
        "td": (
            "connectome_temporal_difference",
            "random_sparse_temporal_difference",
            "weight_shuffle_temporal_difference",
        ),
    }

    for names in groups.values():
        rows = []
        for model_name in names:
            factory, metadata = bench.make_graph_feature_factory(
                model_name=model_name,
                exp=FakeExperiment(),
                base_matrix=base,
                seed=0,
                args=args,
            )
            model = factory("g", 0)
            rows.append(metadata)
            assert metadata.N == 6
            assert metadata.init_nonzero_edges == base.nnz
            assert metadata.feature_dim == 4
            assert metadata.input_dim == 3
            assert model.act([1.0, 0.0], [1.0], 0.0, 0) == pytest.approx(0.0)
        assert {row.trainable_params for row in rows} == {rows[0].trainable_params}
        assert {row.recurrent_params for row in rows} == {base.nnz}

    rw_metadata = bench.make_graph_feature_factory(
        "connectome_rescorla_wagner", FakeExperiment(), base, 0, args
    )[1]
    kalman_metadata = bench.make_graph_feature_factory(
        "connectome_kalman_filter", FakeExperiment(), base, 0, args
    )[1]
    td_metadata = bench.make_graph_feature_factory(
        "connectome_temporal_difference", FakeExperiment(), base, 0, args
    )[1]
    assert rw_metadata.encoded_dim == 7
    assert rw_metadata.trainable_params == 7
    assert kalman_metadata.encoded_dim == 7
    assert kalman_metadata.trainable_params == 7 + 7 * 7
    assert td_metadata.encoded_dim == 7 * 3
    assert td_metadata.trainable_params == 7 * 3


def test_multi_gpu_command_supports_ccnlab(tmp_path: Path) -> None:
    sweep_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_multi_gpu_associative_sweep.py"
    )
    spec = importlib.util.spec_from_file_location("multi_gpu_assoc_ccnlab", sweep_path)
    assert spec is not None
    assert spec.loader is not None
    sweep = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sweep
    spec.loader.exec_module(sweep)

    args = sweep.parse_args(
        [
            "--benchmark",
            "ccnlab",
            "--output-dir",
            str(tmp_path / "sweep"),
            "--models",
            "hemibrain_seeded",
            "--seeds",
            "0",
            "--gpus",
            "0",
            "--python",
            "python",
            "--",
            "--ccnlab-root",
            "/tmp/ccnlab",
            "--subjects",
            "1",
        ]
    )
    jobs = sweep.build_jobs(args)
    command = sweep.command_for_job(jobs[0], args)

    assert command[:2] == [
        "python",
        str(sweep.ROOT / "scripts" / "run_ccnlab_associative_benchmark.py"),
    ]
    assert "--ccnlab-root" in command
    assert command[-6:] == [
        "--device",
        "cuda",
        "--models",
        "hemibrain_seeded",
        "--seeds",
        "0",
    ]
