from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


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
    assert summary.loc[0, "test_query_accuracy_mean"] == 0.85
    assert (output_dir / "metrics_by_seed.csv").exists()
    assert (output_dir / "loss_history.csv").exists()
    assert (output_dir / "metrics_summary.csv").exists()
    leaderboard = pd.read_csv(output_dir / "leaderboard.csv")
    assert leaderboard.loc[0, "rank"] == 1
    assert leaderboard.loc[0, "test_query_accuracy_mean"] == 0.85


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
