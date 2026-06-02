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
