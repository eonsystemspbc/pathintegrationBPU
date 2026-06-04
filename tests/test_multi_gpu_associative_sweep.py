from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_multi_gpu_associative_sweep.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("multi_gpu_assoc", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_jobs_and_child_command(tmp_path: Path) -> None:
    sweep = _load_module()
    args = sweep.parse_args(
        [
            "--benchmark",
            "meta_album",
            "--output-dir",
            str(tmp_path / "sweep"),
            "--models",
            "hemibrain_seeded",
            "random_sparse",
            "--seeds",
            "0",
            "1",
            "--gpus",
            "0",
            "1",
            "--python",
            "python",
            "--",
            "--dataset",
            "synthetic",
            "--epochs",
            "1",
        ]
    )

    jobs = sweep.build_jobs(args)
    command = sweep.command_for_job(jobs[0], args)

    assert len(jobs) == 4
    assert jobs[0].output_dir == tmp_path / "sweep" / "jobs" / "hemibrain_seeded_seed0"
    assert command[:2] == [
        "python",
        str(sweep.ROOT / "scripts" / "run_meta_album_associative_benchmark.py"),
    ]
    assert command[-8:] == [
        "--output-dir",
        str(jobs[0].output_dir),
        "--device",
        "cuda",
        "--models",
        "hemibrain_seeded",
        "--seeds",
        "0",
    ]


def test_dry_run_writes_sweep_log(tmp_path: Path) -> None:
    sweep = _load_module()
    output_dir = tmp_path / "dry_run"

    code = sweep.main(
        [
            "--benchmark",
            "omniglot",
            "--output-dir",
            str(output_dir),
            "--models",
            "hemibrain_seeded",
            "--seeds",
            "0",
            "--gpus",
            "0",
            "--python",
            "python",
            "--dry-run",
            "--",
            "--dataset",
            "synthetic",
            "--epochs",
            "1",
        ]
    )

    text = (output_dir / "sweep.log").read_text(encoding="utf-8")
    assert code == 0
    assert "sweep-start" in text
    assert "dry-run gpu=0 model=hemibrain_seeded seed=0" in text
    assert "run_omniglot_associative_benchmark.py" in text


def test_merge_job_outputs_writes_combined_metrics_and_summary(tmp_path: Path) -> None:
    sweep = _load_module()
    output_dir = tmp_path / "sweep"
    job0 = output_dir / "jobs" / "hemibrain_seeded_seed0"
    job1 = output_dir / "jobs" / "hemibrain_seeded_seed1"
    job0.mkdir(parents=True)
    job1.mkdir(parents=True)
    rows = [
        {
            "model": "hemibrain_seeded",
            "seed": 0,
            "runtime": "sparse",
            "N": 12,
            "timesteps": 8,
            "init_nonzero_edges": 40,
            "recurrent_params": 40,
            "trainable_params": 200,
            "best_val_loss": 0.3,
            "test_loss": 0.35,
            "test_query_accuracy": 0.8,
        },
        {
            "model": "hemibrain_seeded",
            "seed": 1,
            "runtime": "sparse",
            "N": 12,
            "timesteps": 8,
            "init_nonzero_edges": 40,
            "recurrent_params": 40,
            "trainable_params": 200,
            "best_val_loss": 0.2,
            "test_loss": 0.30,
            "test_query_accuracy": 0.9,
        },
    ]
    pd.DataFrame([rows[0]]).to_csv(job0 / "metrics_by_seed.csv", index=False)
    pd.DataFrame([rows[1]]).to_csv(job1 / "metrics_by_seed.csv", index=False)
    pd.DataFrame(
        [
            {"model": "hemibrain_seeded", "seed": 0, "epoch": 1, "train_loss": 0.4},
            {"model": "hemibrain_seeded", "seed": 1, "epoch": 1, "train_loss": 0.3},
        ]
    ).to_csv(job0 / "loss_history.csv", index=False)

    records = [
        {
            "index": 0,
            "benchmark": "meta_album",
            "gpu": "0",
            "output_dir": str(job0),
            "return_code": 0,
        },
        {
            "index": 1,
            "benchmark": "meta_album",
            "gpu": "1",
            "output_dir": str(job1),
            "return_code": 0,
        },
    ]

    metrics, history, summary = sweep.merge_job_outputs(output_dir, records)

    assert metrics.shape[0] == 2
    assert history.shape[0] == 2
    assert summary.loc[0, "test_query_accuracy_mean"] == pytest.approx(0.85)
    assert (output_dir / "metrics_by_seed.csv").exists()
    assert (output_dir / "loss_history.csv").exists()
    assert (output_dir / "metrics_summary.csv").exists()
    leaderboard = pd.read_csv(output_dir / "leaderboard.csv")
    assert leaderboard.loc[0, "rank"] == 1
    assert leaderboard.loc[0, "test_query_accuracy_mean"] == pytest.approx(0.85)


def test_merge_job_outputs_skips_empty_loss_history(tmp_path: Path) -> None:
    sweep = _load_module()
    output_dir = tmp_path / "sweep"
    job_dir = output_dir / "jobs" / "nearest_support_seed0"
    job_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "model": "nearest_support",
                "seed": 0,
                "runtime": "none",
                "N": 0,
                "timesteps": 8,
                "init_nonzero_edges": 0,
                "recurrent_params": 0,
                "trainable_params": 0,
                "best_val_loss": 0.2,
                "test_loss": 0.25,
                "test_query_accuracy": 0.9,
            }
        ]
    ).to_csv(job_dir / "metrics_by_seed.csv", index=False)
    (job_dir / "loss_history.csv").write_text("", encoding="utf-8")

    records = [
        {
            "index": 0,
            "benchmark": "omniglot",
            "gpu": "0",
            "output_dir": str(job_dir),
            "return_code": 0,
        }
    ]

    metrics, history, summary = sweep.merge_job_outputs(output_dir, records)

    assert metrics.shape[0] == 1
    assert history.empty
    assert summary.loc[0, "test_query_accuracy_mean"] == 0.9
    assert (output_dir / "metrics_by_seed.csv").exists()
    assert not (output_dir / "loss_history.csv").exists()
    assert (output_dir / "metrics_summary.csv").exists()
    assert (output_dir / "leaderboard.csv").exists()


def test_rmse_leaderboard_sorts_lower_and_positive_delta_is_better(tmp_path: Path) -> None:
    sweep = _load_module()
    output_dir = tmp_path / "optic_sweep"
    seeded = output_dir / "jobs" / "optic_lobe_seeded_seed0"
    random = output_dir / "jobs" / "random_sparse_seed0"
    seeded.mkdir(parents=True)
    random.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "model": "optic_lobe_seeded",
                "seed": 0,
                "N": 10,
                "trainable_params": 40,
                "test_overall_rmse": 0.30,
                "test_yaw_rmse": 0.20,
                "test_translation_rmse": 0.35,
            }
        ]
    ).to_csv(seeded / "metrics_by_seed.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "random_sparse",
                "seed": 0,
                "N": 10,
                "trainable_params": 40,
                "test_overall_rmse": 0.45,
                "test_yaw_rmse": 0.30,
                "test_translation_rmse": 0.50,
            }
        ]
    ).to_csv(random / "metrics_by_seed.csv", index=False)
    records = [
        {"index": 0, "benchmark": "optic_flow", "gpu": "0", "output_dir": str(seeded), "return_code": 0},
        {"index": 1, "benchmark": "optic_flow", "gpu": "1", "output_dir": str(random), "return_code": 0},
    ]

    sweep.merge_job_outputs(output_dir, records)

    leaderboard = pd.read_csv(output_dir / "leaderboard.csv")
    paired = pd.read_csv(output_dir / "paired_comparisons.csv")
    assert leaderboard.loc[0, "model"] == "optic_lobe_seeded"
    assert round(float(leaderboard.loc[0, "delta_vs_random_sparse"]), 6) == 0.15
    paired_row = paired[
        (paired["model"] == "optic_lobe_seeded")
        & (paired["baseline_model"] == "random_sparse")
        & (paired["metric"] == "test_overall_rmse")
    ].iloc[0]
    assert round(float(paired_row["mean_delta"]), 6) == 0.15


def test_matched_topology_comparisons_are_same_architecture_only(tmp_path: Path) -> None:
    sweep = _load_module()
    output_dir = tmp_path / "ccnlab_sweep"
    models = [
        "connectome_kalman_filter",
        "random_sparse_kalman_filter",
        "degree_preserving_kalman_filter",
        "weight_shuffle_kalman_filter",
        "random_sparse_rescorla_wagner",
    ]
    records = []
    values = {
        "connectome_kalman_filter": [0.70, 0.72],
        "random_sparse_kalman_filter": [0.68, 0.69],
        "degree_preserving_kalman_filter": [0.67, 0.68],
        "weight_shuffle_kalman_filter": [0.66, 0.67],
        "random_sparse_rescorla_wagner": [0.75, 0.76],
    }
    index = 0
    for model in models:
        for seed, value in enumerate(values[model]):
            job_dir = output_dir / "jobs" / f"{model}_seed{seed}"
            job_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "model": model,
                        "seed": seed,
                        "N": 10 if "kalman" in model else 0,
                        "feature_dim": 8 if "kalman" in model else 2,
                        "trainable_params": 72,
                        "test_ccnlab_score": value,
                    }
                ]
            ).to_csv(job_dir / "metrics_by_seed.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "model": model,
                        "seed": seed,
                        "experiment": "Acquisition",
                        "group": "paired",
                        "phase": "train",
                        "trial": 1,
                        "trial_in_phase": 1,
                        "has_cs": 1,
                        "learning_response_mean": value,
                    }
                ]
            ).to_csv(job_dir / "ccnlab_trial_history.csv", index=False)
            records.append(
                {
                    "index": index,
                    "benchmark": "ccnlab",
                    "gpu": "0",
                    "output_dir": str(job_dir),
                    "return_code": 0,
                }
            )
            index += 1

    sweep.merge_job_outputs(output_dir, records)
    sweep.write_sweep_report(
        output_dir,
        records,
        pd.read_csv(output_dir / "metrics_by_seed.csv"),
        pd.DataFrame(),
        pd.read_csv(output_dir / "metrics_summary.csv"),
    )

    paired = pd.read_csv(output_dir / "paired_comparisons.csv")
    matched = pd.read_csv(output_dir / "matched_topology_comparisons.csv")
    trial_history = pd.read_csv(output_dir / "ccnlab_trial_history.csv")
    assert set(trial_history["model"]) == set(models)

    kalman_paired = paired[
        (paired["model"] == "connectome_kalman_filter")
        & (paired["metric"] == "test_ccnlab_score")
    ]
    assert set(kalman_paired["baseline_model"]) >= {
        "random_sparse_kalman_filter",
        "degree_preserving_kalman_filter",
        "weight_shuffle_kalman_filter",
        "random_sparse_rescorla_wagner",
    }
    kalman_matched = matched[
        (matched["model"] == "connectome_kalman_filter")
        & (matched["metric"] == "test_ccnlab_score")
    ]
    assert set(kalman_matched["baseline_model"]) == {
        "random_sparse_kalman_filter",
        "degree_preserving_kalman_filter",
        "weight_shuffle_kalman_filter",
    }
    assert set(kalman_matched["comparison_type"]) == {"matched_topology_control"}
    assert (output_dir / "sweep_report.md").read_text(encoding="utf-8").count(
        "random_sparse_rescorla_wagner"
    ) == 1
