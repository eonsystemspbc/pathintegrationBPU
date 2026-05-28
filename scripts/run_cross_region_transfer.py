#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DEFAULT_LANDMARK_NOISE_STD,
    DEFAULT_LANDMARK_VISIBLE_PROB,
    DEFAULT_PASSIVE_DISPLACEMENT_PROB,
    DEFAULT_PASSIVE_DISPLACEMENT_SCALE,
    RECURRENT_RUNTIME_CHOICES,
    RECURRENT_TRAIN_CHOICES,
    TASK_CX_POLAR_BUMP,
    TASK_CHOICES,
    TaskSpec,
    TrainConfig,
    build_paths,
)
from src.train import run_training  # noqa: E402


ASSOCIATIVE_TASK = "associative"
PATH_TASK = "path"
SUBSTRATE_CX = "hemibrain_cx"
SUBSTRATE_MB = "hemibrain_mushroom_body"

GRAPH_REQUIRED = (
    "graph_metadata.json",
    "pool_assignments.csv",
    "adjacency_unsigned.npz",
)
GRAPH_OPTIONAL = (
    "adjacency_signed.npz",
    "neurons.csv",
    "roi_counts.csv",
    "connections.csv",
    "data_validation.md",
    "bpu_validation.md",
    "control_validation.md",
)


@dataclass(frozen=True)
class Condition:
    name: str
    task_family: str
    substrate: str
    matched: bool


CONDITIONS = {
    "assoc_cx_seeded": Condition(
        name="assoc_cx_seeded",
        task_family=ASSOCIATIVE_TASK,
        substrate=SUBSTRATE_CX,
        matched=False,
    ),
    "assoc_mb_seeded": Condition(
        name="assoc_mb_seeded",
        task_family=ASSOCIATIVE_TASK,
        substrate=SUBSTRATE_MB,
        matched=True,
    ),
    "path_cx_seeded": Condition(
        name="path_cx_seeded",
        task_family=PATH_TASK,
        substrate=SUBSTRATE_CX,
        matched=True,
    ),
    "path_mb_seeded": Condition(
        name="path_mb_seeded",
        task_family=PATH_TASK,
        substrate=SUBSTRATE_MB,
        matched=False,
    ),
}


def _load_associative_module():
    script_path = ROOT / "scripts" / "run_mb_associative_learning.py"
    spec = importlib.util.spec_from_file_location("mb_assoc_cross_region", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import associative trainer at {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _conditions_for(mode: str, pairs: str) -> list[Condition]:
    if pairs == "cross":
        names = ("assoc_cx_seeded", "path_mb_seeded")
    elif pairs == "matched":
        names = ("assoc_mb_seeded", "path_cx_seeded")
    elif pairs == "all":
        names = (
            "assoc_mb_seeded",
            "assoc_cx_seeded",
            "path_cx_seeded",
            "path_mb_seeded",
        )
    else:
        raise ValueError(f"unknown pair selection: {pairs}")

    selected = [CONDITIONS[name] for name in names]
    if mode == "associative":
        return [condition for condition in selected if condition.task_family == ASSOCIATIVE_TASK]
    if mode == "path":
        return [condition for condition in selected if condition.task_family == PATH_TASK]
    return selected


def _substrate_dir(args: argparse.Namespace, substrate: str) -> Path:
    if substrate == SUBSTRATE_CX:
        return args.cx_dir.resolve()
    if substrate == SUBSTRATE_MB:
        return args.mb_dir.resolve()
    raise ValueError(f"unknown substrate: {substrate}")


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def _copy_graph_artifacts(source_dir: Path, dest_dir: Path) -> None:
    for name in GRAPH_REQUIRED:
        _require_file(source_dir / name, f"prepared graph artifact {name}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in GRAPH_REQUIRED + GRAPH_OPTIONAL:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, dest_dir / name)


def _assoc_epochs(args: argparse.Namespace) -> int:
    return args.assoc_epochs if args.assoc_epochs is not None else args.epochs


def _path_epochs(args: argparse.Namespace) -> int:
    return args.path_epochs if args.path_epochs is not None else args.epochs


def _assoc_batch_size(args: argparse.Namespace) -> int:
    return args.assoc_batch_size if args.assoc_batch_size is not None else args.batch_size


def _path_batch_size(args: argparse.Namespace) -> int:
    return args.path_batch_size if args.path_batch_size is not None else args.batch_size


def _run_associative_condition(
    condition: Condition,
    substrate_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    matrix = substrate_dir / "adjacency_unsigned.npz"
    _require_file(matrix, "associative substrate matrix")
    assoc = _load_associative_module()
    argv = [
        "--matrix",
        str(matrix),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--models",
        args.assoc_model,
        "--recurrent-runtime",
        args.assoc_recurrent_runtime,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--epochs",
        str(_assoc_epochs(args)),
        "--batch-size",
        str(_assoc_batch_size(args)),
        "--train-batches",
        str(args.assoc_train_batches),
        "--val-batches",
        str(args.assoc_val_batches),
        "--test-batches",
        str(args.assoc_test_batches),
        "--lr",
        str(args.assoc_lr),
        "--patience",
        str(args.assoc_patience),
        "--grad-clip",
        str(args.assoc_grad_clip),
        "--state-clip",
        str(args.assoc_state_clip),
        "--log-every-seconds",
        str(args.log_every_seconds),
        "--num-odors",
        str(args.assoc_num_odors),
        "--odor-dim",
        str(args.assoc_odor_dim),
        "--odors-per-episode",
        str(args.assoc_odors_per_episode),
        "--reversal-count",
        str(args.assoc_reversal_count),
        "--reversal-repeats",
        str(args.assoc_reversal_repeats),
        "--odor-sparsity",
        str(args.assoc_odor_sparsity),
        "--odor-noise-std",
        str(args.assoc_odor_noise_std),
        "--data-seed",
        str(args.assoc_data_seed),
        "--init-seed",
        str(args.assoc_init_seed),
        "--val-seed",
        str(args.assoc_val_seed),
        "--test-seed",
        str(args.assoc_test_seed),
    ]
    if args.assoc_max_neurons > 0:
        argv.extend(["--max-neurons", str(args.assoc_max_neurons)])
    print(
        "condition-start "
        f"condition={condition.name} task=associative substrate={condition.substrate} "
        f"matrix={matrix}",
        flush=True,
    )
    code = assoc.main(argv)
    if code != 0:
        raise RuntimeError(f"associative trainer failed for {condition.name} with exit {code}")


def _run_path_condition(
    condition: Condition,
    substrate_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    _copy_graph_artifacts(substrate_dir, output_dir)
    paths = build_paths(output_dir, output_dir)
    train_config = TrainConfig(
        seeds=tuple(args.seeds),
        epochs=_path_epochs(args),
        batch_size=_path_batch_size(args),
        num_workers=args.num_workers,
        lr=args.path_lr,
        patience=args.path_patience,
        grad_clip=args.path_grad_clip,
        include_gru=False,
        device=args.device,
        models=(args.path_model,),
        log_every_seconds=args.log_every_seconds,
        recurrent_runtime=args.path_recurrent_runtime,
        train_recurrent=args.path_train_recurrent,
    )
    task_spec = TaskSpec(
        train_count=args.path_train_count,
        val_count=args.path_val_count,
        test_count=args.path_test_count,
        train_T=args.path_train_T,
        test_T=tuple(args.path_test_T),
        noise_stds=tuple(args.path_noise_stds),
        kind=args.path_task,
        heading_bins=args.heading_bins,
        home_distance_scale=args.home_distance_scale,
        bump_kappa=args.bump_kappa,
        landmark_visible_prob=args.landmark_visible_prob,
        landmark_noise_std=args.landmark_noise_std,
        passive_displacement_prob=args.passive_displacement_prob,
        passive_displacement_scale=args.passive_displacement_scale,
    )
    print(
        "condition-start "
        f"condition={condition.name} task={args.path_task} substrate={condition.substrate} "
        f"graph_dir={substrate_dir}",
        flush=True,
    )
    run_training(paths, train_config, task_spec)


def _read_condition_metrics(condition: Condition, condition_dir: Path) -> pd.DataFrame:
    metrics_path = condition_dir / "metrics_by_seed.csv"
    _require_file(metrics_path, f"metrics for {condition.name}")
    metrics = pd.read_csv(metrics_path)
    metrics.insert(0, "condition", condition.name)
    metrics.insert(1, "task_family", condition.task_family)
    metrics.insert(2, "substrate", condition.substrate)
    metrics.insert(3, "matched_region_task", bool(condition.matched))
    return metrics


def _best_path_rows(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame[
        (frame["task_family"] == PATH_TASK)
        & (frame.get("split", pd.Series(index=frame.index, dtype=object)) == "test")
        & (frame.get("noise_std", pd.Series(index=frame.index, dtype=float)).fillna(0.0) == 0.0)
    ].copy()
    if clean.empty:
        return clean
    rows = []
    for (_, seed), group in clean.groupby(["condition", "seed"], dropna=False):
        rows.append(group.sort_values("T").iloc[-1])
    return pd.DataFrame(rows)


def _success_rows(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    assoc = raw[raw["task_family"] == ASSOCIATIVE_TASK]
    if not assoc.empty:
        for _, row in assoc.iterrows():
            rows.append(
                {
                    "condition": row["condition"],
                    "task_family": ASSOCIATIVE_TASK,
                    "substrate": row["substrate"],
                    "matched_region_task": bool(row["matched_region_task"]),
                    "seed": int(row["seed"]),
                    "success_metric": "test_reversal_probe_accuracy",
                    "success_value": float(row["test_reversal_probe_accuracy"]),
                    "higher_is_better": True,
                    "secondary_metric": "test_loss",
                    "secondary_value": float(row["test_loss"]),
                }
            )
    path = _best_path_rows(raw)
    if not path.empty:
        metric = "home_bearing_angular_error"
        if metric not in path.columns:
            metric = "position_rmse"
        secondary = "home_distance_rmse" if "home_distance_rmse" in path.columns else "best_val_loss"
        for _, row in path.iterrows():
            rows.append(
                {
                    "condition": row["condition"],
                    "task_family": PATH_TASK,
                    "substrate": row["substrate"],
                    "matched_region_task": bool(row["matched_region_task"]),
                    "seed": int(row["seed"]),
                    "success_metric": metric,
                    "success_value": float(row[metric]),
                    "higher_is_better": False,
                    "secondary_metric": secondary,
                    "secondary_value": float(row[secondary]),
                }
            )
    return pd.DataFrame(rows)


def _write_summary(output_dir: Path, raw: pd.DataFrame) -> pd.DataFrame:
    success = _success_rows(raw)
    if success.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            success.groupby(
                [
                    "condition",
                    "task_family",
                    "substrate",
                    "matched_region_task",
                    "success_metric",
                    "higher_is_better",
                    "secondary_metric",
                ],
                as_index=False,
            )
            .agg(
                success_mean=("success_value", "mean"),
                success_std=("success_value", "std"),
                secondary_mean=("secondary_value", "mean"),
                secondary_std=("secondary_value", "std"),
                seeds=("seed", "nunique"),
            )
            .sort_values(["task_family", "matched_region_task", "condition"], ascending=[True, False, True])
        )
    success.to_csv(output_dir / "cross_region_success_by_seed.csv", index=False)
    summary.to_csv(output_dir / "cross_region_summary.csv", index=False)
    return summary


def _plot_cross_region(output_dir: Path, raw: pd.DataFrame) -> None:
    success = _success_rows(raw)
    if success.empty:
        return
    tasks = list(success["task_family"].drop_duplicates())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.2), dpi=180)
    if len(tasks) == 1:
        axes = [axes]
    colors = {True: "#1f77b4", False: "#d62728"}
    for ax, task in zip(axes, tasks):
        task_rows = success[success["task_family"] == task].copy()
        order = task_rows.groupby("condition")["matched_region_task"].max().sort_values(ascending=False).index
        values = [
            task_rows[task_rows["condition"] == condition]["success_value"].to_numpy()
            for condition in order
        ]
        labels = [str(condition).replace("_", "\n") for condition in order]
        means = [float(vals.mean()) for vals in values]
        sems = [
            float(vals.std(ddof=1) / (len(vals) ** 0.5)) if len(vals) > 1 else 0.0
            for vals in values
        ]
        bar_colors = [
            colors[bool(task_rows[task_rows["condition"] == condition]["matched_region_task"].iloc[0])]
            for condition in order
        ]
        ax.bar(range(len(order)), means, yerr=sems, color=bar_colors, alpha=0.85, capsize=4)
        for idx, vals in enumerate(values):
            ax.scatter([idx] * len(vals), vals, color="black", s=18, zorder=3)
        metric = str(task_rows["success_metric"].iloc[0]).replace("_", " ")
        better = "higher is better" if bool(task_rows["higher_is_better"].iloc[0]) else "lower is better"
        title = "Associative reversal" if task == ASSOCIATIVE_TASK else "Angular path integration"
        ax.set_title(title)
        ax.set_ylabel(f"{metric} ({better})")
        ax.set_xticks(range(len(order)), labels)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_region_task_success.png")
    plt.close(fig)


def _write_report(output_dir: Path, args: argparse.Namespace, summary: pd.DataFrame) -> None:
    lines = [
        "# Cross-Region Transfer Benchmark",
        "",
        "This run tests whether a connectome substrate is most useful for a task matched to its biological region.",
        "",
        "Conditions:",
        "",
        "- `assoc_mb_seeded`: mushroom-body substrate on odor-valence associative reversal.",
        "- `assoc_cx_seeded`: central-complex substrate on odor-valence associative reversal.",
        "- `path_cx_seeded`: central-complex substrate on CX-style angular path integration.",
        "- `path_mb_seeded`: mushroom-body substrate on CX-style angular path integration.",
        "",
        "The cross conditions are `assoc_cx_seeded` and `path_mb_seeded`. The matched references are `assoc_mb_seeded` and `path_cx_seeded`.",
        "",
        "Important caveat: this is a region-specificity stress test, not a perfect size-matched null. CX and MB substrates can differ in neuron count, edge count, and pool assignments. Use same-size random and weight-shuffled controls inside each task for stronger claims.",
        "",
        "## Command Configuration",
        "",
        "```json",
        json.dumps(vars(args), indent=2, default=str),
        "```",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("No summary rows were produced.")
    else:
        lines.extend(["```", summary.to_string(index=False), "```"])
    lines.extend(
        [
            "",
            "Primary outputs:",
            "",
            "- `cross_region_metrics_by_seed.csv`: raw metrics with condition metadata.",
            "- `cross_region_success_by_seed.csv`: one task-success row per condition and seed.",
            "- `cross_region_summary.csv`: mean/std task-success summary.",
            "- `cross_region_task_success.png`: matched vs cross-region task-success figure.",
            "",
        ]
    )
    (output_dir / "cross_region_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run cross-region connectome transfer tests: CX on associative learning "
            "and mushroom body on CX-style angular path integration."
        )
    )
    parser.add_argument("--mode", choices=("all", "associative", "path"), default="all")
    parser.add_argument(
        "--pairs",
        choices=("cross", "matched", "all"),
        default="cross",
        help="Run cross-region mismatches, matched references, or both.",
    )
    parser.add_argument("--cx-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--mb-dir",
        type=Path,
        default=Path("outputs/hemibrain_mushroom_body_plume"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cross_region_transfer"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every-seconds", type=float, default=30.0)

    parser.add_argument("--assoc-model", default="hemibrain_seeded")
    parser.add_argument("--assoc-recurrent-runtime", choices=("sparse", "dense"), default="sparse")
    parser.add_argument("--assoc-max-neurons", type=int, default=0)
    parser.add_argument("--assoc-epochs", type=int, default=None)
    parser.add_argument(
        "--assoc-batch-size",
        type=int,
        default=None,
        help="Associative batch size. Defaults to --batch-size.",
    )
    parser.add_argument("--assoc-train-batches", type=int, default=200)
    parser.add_argument("--assoc-val-batches", type=int, default=40)
    parser.add_argument("--assoc-test-batches", type=int, default=80)
    parser.add_argument("--assoc-lr", type=float, default=1e-3)
    parser.add_argument("--assoc-patience", type=int, default=5)
    parser.add_argument("--assoc-grad-clip", type=float, default=1.0)
    parser.add_argument("--assoc-state-clip", type=float, default=5.0)
    parser.add_argument("--assoc-num-odors", type=int, default=64)
    parser.add_argument("--assoc-odor-dim", type=int, default=64)
    parser.add_argument("--assoc-odors-per-episode", type=int, default=6)
    parser.add_argument("--assoc-reversal-count", type=int, default=3)
    parser.add_argument("--assoc-reversal-repeats", type=int, default=1)
    parser.add_argument("--assoc-odor-sparsity", type=float, default=0.20)
    parser.add_argument("--assoc-odor-noise-std", type=float, default=0.03)
    parser.add_argument("--assoc-data-seed", type=int, default=12345)
    parser.add_argument("--assoc-init-seed", type=int, default=7000)
    parser.add_argument("--assoc-val-seed", type=int, default=22000)
    parser.add_argument("--assoc-test-seed", type=int, default=33000)

    parser.add_argument("--path-model", default="connectome_bpu")
    parser.add_argument("--path-task", choices=TASK_CHOICES, default=TASK_CX_POLAR_BUMP)
    parser.add_argument("--path-epochs", type=int, default=None)
    parser.add_argument(
        "--path-batch-size",
        type=int,
        default=None,
        help="Path-integration batch size. Defaults to --batch-size.",
    )
    parser.add_argument("--path-train-count", type=int, default=10_000)
    parser.add_argument("--path-val-count", type=int, default=2_000)
    parser.add_argument("--path-test-count", type=int, default=2_000)
    parser.add_argument("--path-train-T", type=int, default=50)
    parser.add_argument("--path-test-T", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--path-noise-stds", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--path-lr", type=float, default=1e-3)
    parser.add_argument("--path-patience", type=int, default=4)
    parser.add_argument("--path-grad-clip", type=float, default=1.0)
    parser.add_argument("--path-recurrent-runtime", choices=RECURRENT_RUNTIME_CHOICES, default="auto")
    parser.add_argument(
        "--path-train-recurrent",
        choices=RECURRENT_TRAIN_CHOICES,
        default="observed",
        help="Default trains one recurrent parameter per observed connectome edge.",
    )
    parser.add_argument("--heading-bins", type=int, default=32)
    parser.add_argument("--home-distance-scale", type=float, default=25.0)
    parser.add_argument("--bump-kappa", type=float, default=8.0)
    parser.add_argument("--landmark-visible-prob", type=float, default=DEFAULT_LANDMARK_VISIBLE_PROB)
    parser.add_argument("--landmark-noise-std", type=float, default=DEFAULT_LANDMARK_NOISE_STD)
    parser.add_argument("--passive-displacement-prob", type=float, default=DEFAULT_PASSIVE_DISPLACEMENT_PROB)
    parser.add_argument("--passive-displacement-scale", type=float, default=DEFAULT_PASSIVE_DISPLACEMENT_SCALE)
    args = parser.parse_args(argv)
    if args.assoc_odors_per_episode > args.assoc_num_odors:
        parser.error("--assoc-odors-per-episode cannot exceed --assoc-num-odors")
    if args.assoc_reversal_count > args.assoc_odors_per_episode:
        parser.error("--assoc-reversal-count cannot exceed --assoc-odors-per-episode")
    if not (0.0 < args.assoc_odor_sparsity <= 1.0):
        parser.error("--assoc-odor-sparsity must be in (0, 1]")
    if args.heading_bins < 4:
        parser.error("--heading-bins must be at least 4")
    if not (0.0 <= args.landmark_visible_prob <= 1.0):
        parser.error("--landmark-visible-prob must be in [0, 1]")
    if args.landmark_noise_std < 0.0:
        parser.error("--landmark-noise-std must be non-negative")
    if not (0.0 <= args.passive_displacement_prob <= 1.0):
        parser.error("--passive-displacement-prob must be in [0, 1]")
    if args.passive_displacement_scale < 0.0:
        parser.error("--passive-displacement-scale must be non-negative")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.cx_dir = args.cx_dir.resolve()
    args.mb_dir = args.mb_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    conditions = _conditions_for(args.mode, args.pairs)
    print(
        "cross-run-start "
        f"output_dir={args.output_dir} pairs={args.pairs} mode={args.mode} "
        f"conditions={','.join(condition.name for condition in conditions)}",
        flush=True,
    )
    raw_frames: list[pd.DataFrame] = []
    for condition in conditions:
        substrate_dir = _substrate_dir(args, condition.substrate)
        condition_dir = args.output_dir / condition.name
        if condition.task_family == ASSOCIATIVE_TASK:
            _run_associative_condition(condition, substrate_dir, condition_dir, args)
        else:
            _run_path_condition(condition, substrate_dir, condition_dir, args)
        raw_frames.append(_read_condition_metrics(condition, condition_dir))

    raw = pd.concat(raw_frames, ignore_index=True, sort=False) if raw_frames else pd.DataFrame()
    raw.to_csv(args.output_dir / "cross_region_metrics_by_seed.csv", index=False)
    summary = _write_summary(args.output_dir, raw)
    _plot_cross_region(args.output_dir, raw)
    _write_report(args.output_dir, args, summary)
    print(
        "cross-run-complete "
        f"metrics={args.output_dir / 'cross_region_metrics_by_seed.csv'} "
        f"summary={args.output_dir / 'cross_region_summary.csv'} "
        f"figure={args.output_dir / 'cross_region_task_success.png'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
