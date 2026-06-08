#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BENCHMARK_SCRIPTS = {
    "ccnlab": ROOT / "scripts" / "run_ccnlab_associative_benchmark.py",
    "dsec_flow": ROOT / "scripts" / "run_dsec_flow_benchmark.py",
    "meta_album": ROOT / "scripts" / "run_meta_album_associative_benchmark.py",
    "omniglot": ROOT / "scripts" / "run_omniglot_associative_benchmark.py",
    "optic_flow": ROOT / "scripts" / "run_optic_flow_benchmark.py",
}


def _pandas():
    import pandas as pd

    return pd


def _write_artifact_manifest(
    output_dir: Path,
    config: dict[str, object],
) -> None:
    from src.run_manifest import write_artifact_manifest

    write_artifact_manifest(
        output_dir,
        config=config,
        extra={"stage": "multi_gpu_associative_sweep"},
    )


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def _tail_lines(path: Path, line_count: int) -> list[str]:
    if line_count <= 0 or not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-line_count:]


class SweepLogger:
    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / "sweep.log"
        self.handle = self.path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"{_timestamp()} {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


@dataclass(frozen=True)
class SweepJob:
    index: int
    benchmark: str
    model: str
    seed: int
    output_dir: Path
    log_path: Path


@dataclass
class RunningJob:
    job: SweepJob
    gpu: str
    process: subprocess.Popen[bytes]
    log_handle: object
    started_at: float


def _safe_token(value: object) -> str:
    raw = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)


def detect_cuda_devices() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() not in {"", "-1"}:
        return [item.strip() for item in visible.split(",") if item.strip()]
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return []
    devices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return devices


def runner_args(args: argparse.Namespace) -> list[str]:
    values = list(args.runner_args)
    if values and values[0] == "--":
        values = values[1:]
    return values


def build_jobs(args: argparse.Namespace) -> list[SweepJob]:
    jobs_dir = args.output_dir / "jobs"
    jobs: list[SweepJob] = []
    for model in args.models:
        for seed in args.seeds:
            job_name = f"{_safe_token(model)}_seed{int(seed)}"
            output_dir = jobs_dir / job_name
            jobs.append(
                SweepJob(
                    index=len(jobs),
                    benchmark=args.benchmark,
                    model=str(model),
                    seed=int(seed),
                    output_dir=output_dir,
                    log_path=output_dir / "run.log",
                )
            )
    return jobs


def command_for_job(job: SweepJob, args: argparse.Namespace) -> list[str]:
    script = BENCHMARK_SCRIPTS[job.benchmark]
    return [
        args.python,
        str(script),
        *runner_args(args),
        "--output-dir",
        str(job.output_dir),
        "--device",
        args.child_device,
        "--models",
        job.model,
        "--seeds",
        str(job.seed),
    ]


def launch_job(job: SweepJob, gpu: str, args: argparse.Namespace) -> RunningJob:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    command = command_for_job(job, args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    log_handle = job.log_path.open("w", encoding="utf-8")
    log_handle.write(f"started_at={_timestamp()}\n")
    log_handle.write(f"gpu={gpu}\n")
    log_handle.write(f"cuda_visible_devices={env['CUDA_VISIBLE_DEVICES']}\n")
    log_handle.write(f"command={shlex.join(command)}\n\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return RunningJob(
        job=job,
        gpu=str(gpu),
        process=process,
        log_handle=log_handle,
        started_at=time.time(),
    )


def _record_for_finished_job(
    entry: RunningJob,
    return_code: int,
    status: str,
) -> dict[str, object]:
    return {
        "index": entry.job.index,
        "benchmark": entry.job.benchmark,
        "model": entry.job.model,
        "seed": entry.job.seed,
        "gpu": entry.gpu,
        "output_dir": str(entry.job.output_dir),
        "log_path": str(entry.job.log_path),
        "return_code": int(return_code),
        "elapsed_seconds": float(time.time() - entry.started_at),
        "status": status,
    }


def _log_failure_tail(
    logger: SweepLogger,
    entry: RunningJob,
    line_count: int,
) -> None:
    tail = _tail_lines(entry.job.log_path, line_count)
    if not tail:
        return
    logger.log(
        "job-log-tail "
        f"index={entry.job.index} model={entry.job.model} seed={entry.job.seed} "
        f"path={entry.job.log_path} lines={len(tail)}"
    )
    for line in tail:
        logger.log(f"job-log index={entry.job.index} {line}")


def _terminate_running(
    running: list[RunningJob],
    logger: SweepLogger,
) -> list[dict[str, object]]:
    for entry in running:
        if entry.process.poll() is None:
            logger.log(
                "job-terminate "
                f"index={entry.job.index} model={entry.job.model} seed={entry.job.seed} "
                f"gpu={entry.gpu}"
            )
            entry.process.terminate()
    deadline = time.time() + 10.0
    records: list[dict[str, object]] = []
    for entry in running:
        while entry.process.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if entry.process.poll() is None:
            entry.process.kill()
        return_code = entry.process.wait()
        entry.log_handle.close()
        records.append(_record_for_finished_job(entry, return_code, "terminated"))
    return records


def _log_status(
    logger: SweepLogger,
    running: list[RunningJob],
    pending: list[SweepJob],
    records: list[dict[str, object]],
) -> None:
    running_bits = [
        (
            f"{entry.job.index}:{entry.job.model}/seed{entry.job.seed}"
            f"/gpu{entry.gpu}/elapsed={_format_duration(time.time() - entry.started_at)}"
        )
        for entry in running
    ]
    logger.log(
        "sweep-status "
        f"done={len(records)} running={len(running)} pending={len(pending)} "
        f"active=[{';'.join(running_bits)}]"
    )


def run_sweep(
    args: argparse.Namespace,
    jobs: list[SweepJob],
    logger: SweepLogger,
) -> list[dict[str, object]]:
    pending = list(jobs)
    running: list[RunningJob] = []
    records: list[dict[str, object]] = []
    available_gpus = [str(gpu) for gpu in args.gpus]
    if not available_gpus:
        raise RuntimeError("No GPUs were provided or detected.")
    max_parallel = min(int(args.max_parallel_jobs), len(available_gpus))
    failed = False
    last_status = time.time()

    while pending or running:
        while pending and len(running) < max_parallel and available_gpus and not failed:
            gpu = available_gpus.pop(0)
            job = pending.pop(0)
            logger.log(
                "job-start "
                f"index={job.index} model={job.model} seed={job.seed} gpu={gpu} "
                f"output_dir={job.output_dir} log={job.log_path}"
            )
            running.append(launch_job(job, gpu, args))

        if not running:
            break

        time.sleep(args.poll_seconds)
        now = time.time()
        if args.status_seconds > 0 and now - last_status >= args.status_seconds:
            _log_status(logger, running, pending, records)
            last_status = now
        for entry in list(running):
            return_code = entry.process.poll()
            if return_code is None:
                continue
            entry.log_handle.close()
            running.remove(entry)
            available_gpus.append(entry.gpu)
            elapsed = time.time() - entry.started_at
            status = "ok" if return_code == 0 else "failed"
            logger.log(
                "job-done "
                f"index={entry.job.index} model={entry.job.model} seed={entry.job.seed} "
                f"gpu={entry.gpu} status={status} return_code={return_code} "
                f"elapsed={_format_duration(elapsed)} log={entry.job.log_path}"
            )
            records.append(_record_for_finished_job(entry, int(return_code), status))
            if return_code != 0:
                _log_failure_tail(logger, entry, args.tail_lines_on_failure)
            if return_code != 0 and not args.keep_going:
                failed = True
                records.extend(_terminate_running(running, logger))
                running.clear()
                pending.clear()
                break
    return records


def write_job_table(output_dir: Path, records: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "benchmark",
        "model",
        "seed",
        "gpu",
        "status",
        "return_code",
        "elapsed_seconds",
        "output_dir",
        "log_path",
    ]
    with (output_dir / "sweep_jobs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def _read_job_csv(
    records: list[dict[str, object]],
    filename: str,
) -> object:
    return _read_job_csv_any(records, (filename,))


def _read_job_csv_any(
    records: list[dict[str, object]],
    filenames: tuple[str, ...],
) -> object:
    pd = _pandas()
    frames = []
    for record in records:
        if int(record.get("return_code", 1)) != 0:
            continue
        path = next(
            (
                Path(str(record["output_dir"])) / filename
                for filename in filenames
                if (Path(str(record["output_dir"])) / filename).exists()
            ),
            Path(str(record["output_dir"])) / filenames[0],
        )
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty and not list(frame.columns):
            continue
        frame.insert(0, "job_index", int(record["index"]))
        frame.insert(1, "benchmark", str(record["benchmark"]))
        frame.insert(2, "gpu", str(record["gpu"]))
        frame.insert(3, "job_output_dir", str(record["output_dir"]))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _summary_from_metrics(metrics: object) -> object:
    pd = _pandas()
    if metrics.empty:
        return pd.DataFrame()
    mean_std_columns = [
        "test_ccnlab_score",
        "test_ccnlab_correlation",
        "test_ccnlab_ratio",
        "test_ccnlab_finite_score_count",
        "best_val_loss",
        "test_loss",
        "test_query_accuracy",
        "test_initial_query_accuracy",
        "test_reversal_query_accuracy",
        "test_initial_probe_accuracy",
        "test_reversal_probe_accuracy",
        "test_overall_rmse",
        "test_yaw_rmse",
        "test_forward_rmse",
        "test_lateral_rmse",
        "test_translation_rmse",
        "test_yaw_r2",
        "test_forward_r2",
        "test_lateral_r2",
        "best_val_epe",
        "final_val_epe",
        "final_val_1pe",
        "final_val_2pe",
        "final_val_3pe",
        "final_val_ae",
    ]
    first_columns = [
        "runtime",
        "init_nonzero_edges",
        "trainable_params",
        "recurrent_params",
        "N",
        "connectome_N",
        "connectome_edges",
        "input_dim",
        "feature_dim",
        "encoded_dim",
        "subjects",
        "test_ccnlab_experiment_count",
        "timesteps",
        "freeze_recurrent",
        "recurrent_prior_l2",
    ]
    aggregations: dict[str, tuple[str, str]] = {}
    for column in mean_std_columns:
        if column in metrics:
            aggregations[f"{column}_mean"] = (column, "mean")
            aggregations[f"{column}_std"] = (column, "std")
    for column in first_columns:
        if column in metrics:
            aggregations[column] = (column, "first")
    return metrics.groupby("model", as_index=False).agg(**aggregations)


def _primary_metric(summary: object) -> str | None:
    preferred = [
        "test_ccnlab_score_mean",
        "best_val_epe_mean",
        "final_val_epe_mean",
        "test_query_accuracy_mean",
        "test_overall_rmse_mean",
        "test_reversal_query_accuracy_mean",
        "test_initial_query_accuracy_mean",
        "test_initial_probe_accuracy_mean",
        "test_reversal_probe_accuracy_mean",
    ]
    for column in preferred:
        if column in summary:
            return column
    for column in summary.columns:
        if column.endswith("_accuracy_mean"):
            return str(column)
    return None


def _higher_is_better(metric: str | None) -> bool:
    if metric is None:
        return True
    lower_is_better_tokens = ("loss", "rmse", "mae", "error", "epe", "ae")
    return not any(token in metric for token in lower_is_better_tokens)


def _metric_delta(values: object, baseline: float, metric: str) -> object:
    if _higher_is_better(metric):
        return values - baseline
    return baseline - values


MATCHED_TOPOLOGY_CONTROLS = {
    "hemibrain_seeded": ("random_sparse", "degree_preserving_random", "weight_shuffle"),
    "connectome_seeded": ("random_sparse", "degree_preserving_random", "weight_shuffle"),
    "hemibrain_conv_fast_memory": (
        "random_sparse_conv_fast_memory",
        "weight_shuffle_conv_fast_memory",
    ),
    "connectome_conv_fast_memory": (
        "random_sparse_conv_fast_memory",
        "weight_shuffle_conv_fast_memory",
    ),
    "hemibrain_fast_memory": ("random_sparse_fast_memory", "weight_shuffle_fast_memory"),
    "connectome_fast_memory": ("random_sparse_fast_memory", "weight_shuffle_fast_memory"),
    "connectome_rescorla_wagner": (
        "random_sparse_rescorla_wagner",
        "degree_preserving_rescorla_wagner",
        "weight_shuffle_rescorla_wagner",
    ),
    "connectome_kalman_filter": (
        "random_sparse_kalman_filter",
        "degree_preserving_kalman_filter",
        "weight_shuffle_kalman_filter",
    ),
    "connectome_temporal_difference": (
        "random_sparse_temporal_difference",
        "degree_preserving_temporal_difference",
        "weight_shuffle_temporal_difference",
    ),
    "optic_lobe_seeded": ("random_sparse", "weight_shuffle"),
    "cx_bpu": ("random", "weight_shuffle"),
}


def _matched_controls_for_model(model: str) -> tuple[str, ...]:
    return tuple(MATCHED_TOPOLOGY_CONTROLS.get(str(model), ()))


def _comparison_type(model: str, baseline: str) -> str:
    if baseline in _matched_controls_for_model(model):
        return "matched_topology_control"
    return "reference_baseline"


def _leaderboard_from_summary(summary: object) -> object:
    pd = _pandas()
    if summary.empty:
        return pd.DataFrame()
    leaderboard = summary.copy()
    metric = _primary_metric(leaderboard)
    if metric is not None:
        leaderboard = leaderboard.sort_values(
            metric,
            ascending=not _higher_is_better(metric),
            na_position="last",
        )
    else:
        leaderboard = leaderboard.sort_values("model")
    leaderboard = leaderboard.reset_index(drop=True)
    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
    if metric is not None:
        for baseline in (
            "nearest_support",
            "random_sparse_conv_fast_memory",
            "weight_shuffle_conv_fast_memory",
            "random_sparse_fast_memory",
            "weight_shuffle_fast_memory",
            "random_sparse_rescorla_wagner",
            "degree_preserving_rescorla_wagner",
            "weight_shuffle_rescorla_wagner",
            "random_sparse_kalman_filter",
            "degree_preserving_kalman_filter",
            "weight_shuffle_kalman_filter",
            "random_sparse_temporal_difference",
            "degree_preserving_temporal_difference",
            "weight_shuffle_temporal_difference",
            "random_sparse",
            "degree_preserving_random",
            "weight_shuffle",
            "random_weight_topology",
            "shuffled_topology",
        ):
            matches = leaderboard.loc[leaderboard["model"] == baseline, metric]
            if not matches.empty:
                leaderboard[f"delta_vs_{baseline}"] = _metric_delta(
                    leaderboard[metric],
                    float(matches.iloc[0]),
                    metric,
                )
    return leaderboard


def _paired_comparisons_from_metrics(metrics: object) -> object:
    pd = _pandas()
    if metrics.empty or "seed" not in metrics or "model" not in metrics:
        return pd.DataFrame()
    metric_columns = [
        "test_ccnlab_score",
        "test_ccnlab_correlation",
        "test_ccnlab_ratio",
        "test_query_accuracy",
        "test_reversal_query_accuracy",
        "test_initial_query_accuracy",
        "test_initial_probe_accuracy",
        "test_reversal_probe_accuracy",
        "test_overall_rmse",
        "test_yaw_rmse",
        "test_translation_rmse",
        "test_loss",
    ]
    metric_columns = [column for column in metric_columns if column in metrics]
    baseline_models = [
        "random_sparse_conv_fast_memory",
        "weight_shuffle_conv_fast_memory",
        "random_sparse_fast_memory",
        "weight_shuffle_fast_memory",
        "random_sparse_rescorla_wagner",
        "degree_preserving_rescorla_wagner",
        "weight_shuffle_rescorla_wagner",
        "random_sparse_kalman_filter",
        "degree_preserving_kalman_filter",
        "weight_shuffle_kalman_filter",
        "random_sparse_temporal_difference",
        "degree_preserving_temporal_difference",
        "weight_shuffle_temporal_difference",
        "nearest_support",
        "random_sparse",
        "degree_preserving_random",
        "weight_shuffle",
        "random_weight_topology",
        "shuffled_topology",
    ]
    rows: list[dict[str, float | int | str]] = []
    for metric in metric_columns:
        pivot = metrics.pivot_table(index="seed", columns="model", values=metric, aggfunc="mean")
        for model in pivot.columns:
            model_baselines = list(dict.fromkeys([*_matched_controls_for_model(str(model)), *baseline_models]))
            for baseline in model_baselines:
                if model == baseline or baseline not in pivot.columns:
                    continue
                delta = _metric_delta(pivot[model], pivot[baseline], metric).dropna()
                if delta.empty:
                    continue
                count = int(delta.shape[0])
                mean_delta = float(delta.mean())
                std_delta = float(delta.std(ddof=1)) if count > 1 else float("nan")
                se_delta = std_delta / math.sqrt(count) if count > 1 else float("nan")
                ci95 = 1.96 * se_delta if count > 1 else float("nan")
                rows.append(
                    {
                        "model": str(model),
                        "baseline_model": baseline,
                        "comparison_type": _comparison_type(str(model), baseline),
                        "metric": metric,
                        "paired_seed_count": count,
                        "mean_delta": mean_delta,
                        "std_delta": std_delta,
                        "se_delta": se_delta,
                        "ci95_low": mean_delta - ci95 if count > 1 else float("nan"),
                        "ci95_high": mean_delta + ci95 if count > 1 else float("nan"),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _matched_topology_comparisons(paired: object) -> object:
    pd = _pandas()
    if paired.empty or "comparison_type" not in paired:
        return pd.DataFrame()
    return paired.loc[paired["comparison_type"] == "matched_topology_control"].copy()


def _format_metric(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "nan"
    return f"{number:.4f}"


def _markdown_table(frame: object, columns: list[str]) -> str:
    available = [column for column in columns if column in frame]
    if frame.empty or not available:
        return "_No rows._"
    lines = [
        "| " + " | ".join(available) + " |",
        "| " + " | ".join("---" for _ in available) + " |",
    ]
    for _, row in frame[available].iterrows():
        values = [_format_metric(row[column]) for column in available]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_sweep_report(
    output_dir: Path,
    records: list[dict[str, object]],
    metrics: object,
    history: object,
    summary: object,
) -> object:
    leaderboard = _leaderboard_from_summary(summary)
    if not leaderboard.empty:
        leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    paired = _paired_comparisons_from_metrics(metrics)
    if not paired.empty:
        paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
        matched_paired = _matched_topology_comparisons(paired)
        if not matched_paired.empty:
            matched_paired.to_csv(output_dir / "matched_topology_comparisons.csv", index=False)
    failed = [record for record in records if int(record.get("return_code", 1)) != 0]
    metric = _primary_metric(summary) if not summary.empty else None
    table_columns = [
        "rank",
        "model",
        "test_ccnlab_score_mean",
        "test_ccnlab_score_std",
        "test_ccnlab_correlation_mean",
        "test_ccnlab_ratio_mean",
        "test_query_accuracy_mean",
        "test_query_accuracy_std",
        "test_overall_rmse_mean",
        "test_overall_rmse_std",
        "test_yaw_rmse_mean",
        "test_translation_rmse_mean",
        "test_yaw_r2_mean",
        "test_initial_query_accuracy_mean",
        "test_reversal_query_accuracy_mean",
        "delta_vs_random_sparse_conv_fast_memory",
        "delta_vs_weight_shuffle_conv_fast_memory",
        "delta_vs_random_sparse_fast_memory",
        "delta_vs_weight_shuffle_fast_memory",
        "delta_vs_nearest_support",
        "delta_vs_degree_preserving_random",
        "delta_vs_degree_preserving_rescorla_wagner",
        "delta_vs_degree_preserving_kalman_filter",
        "delta_vs_degree_preserving_temporal_difference",
        "delta_vs_random_weight_topology",
        "delta_vs_shuffled_topology",
        "delta_vs_random_sparse",
        "delta_vs_weight_shuffle",
        "N",
        "feature_dim",
        "trainable_params",
        "freeze_recurrent",
        "recurrent_prior_l2",
    ]
    paired_metric = metric.removesuffix("_mean") if metric is not None else ""
    paired_table = paired
    if not paired.empty:
        paired_table = paired_table.loc[paired_table["metric"] == paired_metric]
        if "comparison_type" in paired_table:
            matched = paired_table.loc[
                paired_table["comparison_type"] == "matched_topology_control"
            ]
            if not matched.empty:
                paired_table = matched
    lines = [
        "# Associative Sweep Report",
        "",
        f"Output directory: `{output_dir}`",
        f"Jobs: `{len(records)}`; failed: `{len(failed)}`; metric rows: `{len(metrics)}`; history rows: `{len(history)}`.",
        f"Primary ranking metric: `{metric or 'none'}`.",
        "",
        "## Leaderboard",
        "",
        _markdown_table(leaderboard, table_columns),
        "",
        "## Paired Comparisons",
        "",
        _markdown_table(
            paired_table,
            [
                "model",
                "baseline_model",
                "comparison_type",
                "metric",
                "paired_seed_count",
                "mean_delta",
                "ci95_low",
                "ci95_high",
            ],
        ),
        "",
        "## Interpretation",
        "",
        (
            "A useful connectome signal is the seeded connectome model beating "
            "same-family random-sparse, degree-preserving, and weight-shuffled "
            "controls across several seeds. Benchmark-specific non-connectomic "
            "baselines should be treated as task-fit references rather than "
            "topology controls."
        ),
        "",
    ]
    if failed:
        lines.extend(
            [
                "## Failed Jobs",
                "",
                _markdown_table(
                    _pandas().DataFrame(failed),
                    ["index", "benchmark", "model", "seed", "gpu", "return_code", "log_path"],
                ),
                "",
            ]
        )
    (output_dir / "sweep_report.md").write_text("\n".join(lines), encoding="utf-8")
    return leaderboard


def log_leaderboard(logger: SweepLogger, leaderboard: object, topn: int = 8) -> None:
    if leaderboard.empty:
        return
    metric = _primary_metric(leaderboard)
    for _, row in leaderboard.head(topn).iterrows():
        model_name = str(row["model"])
        parts = [f"rank={int(row['rank'])}", f"model={model_name}"]
        if metric is not None and metric in row:
            parts.append(f"{metric}={_format_metric(row[metric])}")
        matched_delta_columns = [
            f"delta_vs_{baseline}" for baseline in _matched_controls_for_model(model_name)
        ]
        reference_delta_columns = (
            "delta_vs_random_sparse",
            "delta_vs_weight_shuffle",
            "delta_vs_random_sparse_conv_fast_memory",
            "delta_vs_weight_shuffle_conv_fast_memory",
            "delta_vs_random_sparse_fast_memory",
            "delta_vs_weight_shuffle_fast_memory",
            "delta_vs_random_sparse_rescorla_wagner",
            "delta_vs_degree_preserving_rescorla_wagner",
            "delta_vs_weight_shuffle_rescorla_wagner",
            "delta_vs_random_sparse_kalman_filter",
            "delta_vs_degree_preserving_kalman_filter",
            "delta_vs_weight_shuffle_kalman_filter",
            "delta_vs_random_sparse_temporal_difference",
            "delta_vs_degree_preserving_temporal_difference",
            "delta_vs_weight_shuffle_temporal_difference",
            "delta_vs_nearest_support",
        )
        for column in dict.fromkeys([*matched_delta_columns, *reference_delta_columns]):
            if column in row:
                parts.append(f"{column}={_format_metric(row[column])}")
        logger.log("leaderboard " + " ".join(parts))


def merge_job_outputs(
    output_dir: Path,
    records: list[dict[str, object]],
) -> tuple[object, object, object]:
    metrics = _read_job_csv_any(records, ("metrics_by_seed.csv", "dsec_metrics_by_seed.csv"))
    history = _read_job_csv(records, "loss_history.csv")
    experiment_scores = _read_job_csv(records, "experiment_scores.csv")
    ccnlab_timestep_history = _read_job_csv(records, "ccnlab_timestep_history.csv")
    ccnlab_trial_history = _read_job_csv(records, "ccnlab_trial_history.csv")
    summary = _summary_from_metrics(metrics)
    leaderboard = _leaderboard_from_summary(summary)
    paired = _paired_comparisons_from_metrics(metrics)
    matched_paired = _matched_topology_comparisons(paired)
    if not metrics.empty:
        metrics.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    if not history.empty:
        history.to_csv(output_dir / "loss_history.csv", index=False)
    if not experiment_scores.empty:
        experiment_scores.to_csv(output_dir / "experiment_scores.csv", index=False)
    if not ccnlab_timestep_history.empty:
        ccnlab_timestep_history.to_csv(
            output_dir / "ccnlab_timestep_history.csv",
            index=False,
        )
    if not ccnlab_trial_history.empty:
        ccnlab_trial_history.to_csv(output_dir / "ccnlab_trial_history.csv", index=False)
    if not summary.empty:
        summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    if not leaderboard.empty:
        leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    if not paired.empty:
        paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    if not matched_paired.empty:
        matched_paired.to_csv(output_dir / "matched_topology_comparisons.csv", index=False)
    return metrics, history, summary


def write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    records: list[dict[str, object]],
) -> None:
    config = {
        "benchmark": args.benchmark,
        "models": args.models,
        "seeds": args.seeds,
        "gpus": args.gpus,
        "max_parallel_jobs": args.max_parallel_jobs,
        "child_device": args.child_device,
        "runner_args": runner_args(args),
        "status_seconds": args.status_seconds,
        "tail_lines_on_failure": args.tail_lines_on_failure,
        "job_count": len(records),
        "failed_job_count": sum(1 for record in records if int(record.get("return_code", 1)) != 0),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_artifact_manifest(output_dir, config)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch independent benchmark model/seed jobs across multiple GPUs "
            "and merge their metrics."
        )
    )
    parser.add_argument("--benchmark", choices=tuple(BENCHMARK_SCRIPTS), default="meta_album")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--gpus",
        nargs="*",
        default=None,
        help="GPU IDs to use. Defaults to CUDA_VISIBLE_DEVICES or nvidia-smi discovery.",
    )
    parser.add_argument("--max-parallel-jobs", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--child-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--status-seconds",
        type=float,
        default=30.0,
        help="Emit sweep-level running job status at this interval. Use 0 to disable.",
    )
    parser.add_argument(
        "--tail-lines-on-failure",
        type=int,
        default=80,
        help="Print this many lines from a failed child run.log into sweep.log.",
    )
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print child commands without launching jobs.",
    )
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the selected benchmark after `--`.",
    )
    args = parser.parse_args(argv)
    if args.max_parallel_jobs <= 0:
        detected = args.gpus if args.gpus is not None else detect_cuda_devices()
        args.max_parallel_jobs = len(detected)
    if args.gpus is None:
        args.gpus = detect_cuda_devices()
    args.gpus = [str(gpu) for gpu in args.gpus]
    if args.child_device == "cuda" and not args.gpus:
        parser.error("No GPUs detected. Pass --gpus explicitly or use --child-device cpu.")
    if args.max_parallel_jobs < 1:
        parser.error("--max-parallel-jobs must be positive")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.status_seconds < 0:
        parser.error("--status-seconds must be nonnegative")
    if args.tail_lines_on_failure < 0:
        parser.error("--tail-lines-on-failure must be nonnegative")
    if not runner_args(args):
        parser.error("Pass benchmark-specific arguments after `--`.")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    logger = SweepLogger(args.output_dir)
    try:
        logger.log(
            "sweep-start "
            f"benchmark={args.benchmark} jobs={len(jobs)} gpus={','.join(args.gpus)} "
            f"max_parallel_jobs={args.max_parallel_jobs} output_dir={args.output_dir} "
            f"sweep_log={logger.path}"
        )
        if args.dry_run:
            for job in jobs:
                gpu = args.gpus[job.index % max(len(args.gpus), 1)] if args.gpus else "cpu"
                logger.log(
                    f"dry-run gpu={gpu} model={job.model} seed={job.seed} "
                    f"command={shlex.join(command_for_job(job, args))}"
                )
            return 0

        records = run_sweep(args, jobs, logger)
        write_job_table(args.output_dir, records)
        metrics, history, summary = merge_job_outputs(args.output_dir, records)
        write_run_config(args.output_dir, args, records)
        leaderboard = write_sweep_report(args.output_dir, records, metrics, history, summary)
        log_leaderboard(logger, leaderboard)
        failed = [record for record in records if int(record.get("return_code", 1)) != 0]
        logger.log(
            "sweep-complete "
            f"jobs={len(records)} failed={len(failed)} "
            f"metrics_rows={len(metrics)} summary_rows={len(summary)} "
            f"output_dir={args.output_dir}"
        )
        return 1 if failed else 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
