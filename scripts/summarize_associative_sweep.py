#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_multi_gpu_associative_sweep as sweep  # noqa: E402


def _read_csv_if_present(path: Path):
    pd = sweep._pandas()
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _discover_job_metrics(output_dir: Path, filename: str):
    pd = sweep._pandas()
    frames = []
    for path in sorted((output_dir / "jobs").glob(f"*/{filename}")):
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty and not list(frame.columns):
            continue
        frame.insert(0, "job_output_dir", str(path.parent))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_sweep_tables(output_dir: Path):
    metrics = _read_csv_if_present(output_dir / "metrics_by_seed.csv")
    if metrics.empty:
        metrics = _discover_job_metrics(output_dir, "metrics_by_seed.csv")
    history = _read_csv_if_present(output_dir / "loss_history.csv")
    if history.empty:
        history = _discover_job_metrics(output_dir, "loss_history.csv")
    records = _read_csv_if_present(output_dir / "sweep_jobs.csv")
    return metrics, history, records


def write_summary(output_dir: Path, rewrite_metrics: bool) -> int:
    pd = sweep._pandas()
    metrics, history, records_frame = load_sweep_tables(output_dir)
    if metrics.empty:
        print(f"No metrics found under {output_dir}", file=sys.stderr)
        return 1
    summary = sweep._summary_from_metrics(metrics)
    leaderboard = sweep._leaderboard_from_summary(summary)
    if rewrite_metrics:
        metrics.to_csv(output_dir / "metrics_by_seed.csv", index=False)
        if not history.empty:
            history.to_csv(output_dir / "loss_history.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    records = records_frame.to_dict("records") if not records_frame.empty else []
    sweep.write_sweep_report(output_dir, records, metrics, history, summary)

    columns = [
        "rank",
        "model",
        "test_query_accuracy_mean",
        "test_overall_rmse_mean",
        "test_yaw_rmse_mean",
        "test_translation_rmse_mean",
        "test_initial_query_accuracy_mean",
        "test_reversal_query_accuracy_mean",
        "delta_vs_random_sparse_conv_fast_memory",
        "delta_vs_weight_shuffle_conv_fast_memory",
        "delta_vs_random_sparse_fast_memory",
        "delta_vs_weight_shuffle_fast_memory",
        "delta_vs_nearest_support",
        "delta_vs_random_weight_topology",
        "delta_vs_shuffled_topology",
        "delta_vs_random_sparse",
        "N",
        "trainable_params",
    ]
    available = [column for column in columns if column in leaderboard]
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(leaderboard[available].to_string(index=False))
    print(f"\nwrote {output_dir / 'leaderboard.csv'}")
    if (output_dir / "paired_comparisons.csv").exists():
        print(f"wrote {output_dir / 'paired_comparisons.csv'}")
    print(f"wrote {output_dir / 'sweep_report.md'}")
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize an associative multi-GPU sweep output directory."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--no-rewrite-metrics",
        action="store_true",
        help="Do not rewrite metrics_by_seed.csv/loss_history.csv when discovered from child jobs.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    return write_summary(args.output_dir, rewrite_metrics=not args.no_rewrite_metrics)


if __name__ == "__main__":
    raise SystemExit(main())
