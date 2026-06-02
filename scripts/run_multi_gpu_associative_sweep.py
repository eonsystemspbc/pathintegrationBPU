#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
    "meta_album": ROOT / "scripts" / "run_meta_album_associative_benchmark.py",
    "omniglot": ROOT / "scripts" / "run_omniglot_associative_benchmark.py",
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
    pd = _pandas()
    frames = []
    for record in records:
        if int(record.get("return_code", 1)) != 0:
            continue
        path = Path(str(record["output_dir"])) / filename
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
        "best_val_loss",
        "test_loss",
        "test_query_accuracy",
        "test_initial_query_accuracy",
        "test_reversal_query_accuracy",
        "test_initial_probe_accuracy",
        "test_reversal_probe_accuracy",
    ]
    first_columns = [
        "runtime",
        "init_nonzero_edges",
        "trainable_params",
        "recurrent_params",
        "N",
        "timesteps",
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


def merge_job_outputs(
    output_dir: Path,
    records: list[dict[str, object]],
) -> tuple[object, object, object]:
    metrics = _read_job_csv(records, "metrics_by_seed.csv")
    history = _read_job_csv(records, "loss_history.csv")
    summary = _summary_from_metrics(metrics)
    if not metrics.empty:
        metrics.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    if not history.empty:
        history.to_csv(output_dir / "loss_history.csv", index=False)
    if not summary.empty:
        summary.to_csv(output_dir / "metrics_summary.csv", index=False)
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
            "Launch independent associative benchmark model/seed jobs across multiple GPUs "
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
        metrics, _, summary = merge_job_outputs(args.output_dir, records)
        write_run_config(args.output_dir, args, records)
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
