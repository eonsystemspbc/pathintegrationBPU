#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "hemibrain_cx_bpu_matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "hemibrain_cx_bpu_xdg_cache"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


DEFAULT_MODELS = (
    "hemibrain_conv_fast_memory",
    "random_sparse_conv_fast_memory",
    "weight_shuffle_conv_fast_memory",
    "conv_protonet",
)
DISPLAY_NAMES = {
    "hemibrain_conv_fast_memory": "Connectome seeded",
    "random_sparse_conv_fast_memory": "Random sparse",
    "weight_shuffle_conv_fast_memory": "Weight shuffle",
    "conv_protonet": "Conv ProtoNet",
    "nearest_support": "Nearest support",
}
COLORS = {
    "hemibrain_conv_fast_memory": "#147d7e",
    "random_sparse_conv_fast_memory": "#4d6fb3",
    "weight_shuffle_conv_fast_memory": "#c77d23",
    "conv_protonet": "#5c5c5c",
    "nearest_support": "#8f8f8f",
}


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", np.trapz)
    return float(fn(y, x))


def _display_name(model: str, labels: dict[str, str]) -> str:
    return labels.get(model, DISPLAY_NAMES.get(model, model))


def _parse_labels(items: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--label must be formatted as model=Display Name, got {item!r}")
        model, label = item.split("=", 1)
        labels[model.strip()] = label.strip()
    return labels


def load_history(output_dir: Path, metric: str, models: list[str]) -> pd.DataFrame:
    path = output_dir / "loss_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    history = pd.read_csv(path)
    required = {"model", "seed", "epoch", metric}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    history = history[history["model"].isin(models)].copy()
    if history.empty:
        raise ValueError(f"none of the requested models were found in {path}")
    history["epoch"] = history["epoch"].astype(int)
    history["seed"] = history["seed"].astype(int)
    history[metric] = history[metric].astype(float)
    return history


def common_epoch_count(history: pd.DataFrame, requested: int | None) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError("--common-epochs must be positive")
        return int(requested)
    per_run_max = history.groupby(["model", "seed"])["epoch"].max()
    if per_run_max.empty:
        raise ValueError("history has no model/seed epoch rows")
    return int(per_run_max.min())


def per_seed_learning_summary(
    history: pd.DataFrame,
    metric: str,
    common_epochs: int,
) -> pd.DataFrame:
    horizon = history[history["epoch"] <= common_epochs].copy()
    by_epoch = (
        horizon.groupby(["model", "seed", "epoch"], as_index=False)[metric]
        .mean()
        .sort_values(["model", "seed", "epoch"])
    )
    rows: list[dict[str, float | int | str]] = []
    for (model, seed), group in by_epoch.groupby(["model", "seed"]):
        group = group.sort_values("epoch")
        epochs = group["epoch"].to_numpy(dtype=float)
        values = group[metric].to_numpy(dtype=float)
        row: dict[str, float | int | str] = {
            "model": str(model),
            "seed": int(seed),
            "common_epochs": int(common_epochs),
            "common_mean": float(values.mean()),
            "common_auc": _trapezoid(values, epochs),
            "final_common_val": float(values[-1]),
        }
        for epoch in (1, 3, 5, 10):
            matches = group.loc[group["epoch"] == epoch, metric]
            row[f"epoch{epoch}"] = float(matches.iloc[0]) if not matches.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_learning_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    mean_columns = [
        "epoch1",
        "epoch3",
        "epoch5",
        "epoch10",
        "common_auc",
        "common_mean",
        "final_common_val",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "common_epochs": ("common_epochs", "first"),
        "seed_count": ("seed", "nunique"),
    }
    for column in mean_columns:
        if column in per_seed:
            aggregations[column] = (column, "mean")
            aggregations[f"{column}_std"] = (column, "std")
    return (
        per_seed.groupby("model", as_index=False)
        .agg(**aggregations)
        .sort_values("common_mean", ascending=False)
        .reset_index(drop=True)
    )


def paired_learning_comparisons(
    per_seed: pd.DataFrame,
    baselines: list[str],
) -> pd.DataFrame:
    metrics = ["common_mean", "common_auc", "final_common_val", "epoch1", "epoch3", "epoch5", "epoch10"]
    rows: list[dict[str, float | int | str]] = []
    for metric in metrics:
        if metric not in per_seed:
            continue
        pivot = per_seed.pivot_table(index="seed", columns="model", values=metric, aggfunc="mean")
        for model in pivot.columns:
            for baseline in baselines:
                if model == baseline or baseline not in pivot.columns:
                    continue
                delta = (pivot[model] - pivot[baseline]).dropna()
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
                        "metric": metric,
                        "paired_seed_count": count,
                        "mean_delta": mean_delta,
                        "std_delta": std_delta,
                        "se_delta": se_delta,
                        "ci95_low": mean_delta - ci95 if count > 1 else float("nan"),
                        "ci95_high": mean_delta + ci95 if count > 1 else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def load_test_accuracy(output_dir: Path, models: list[str]) -> pd.DataFrame:
    path = output_dir / "leaderboard.csv"
    if not path.exists():
        return pd.DataFrame()
    leaderboard = pd.read_csv(path)
    if "model" not in leaderboard or "test_query_accuracy_mean" not in leaderboard:
        return pd.DataFrame()
    return leaderboard.loc[
        leaderboard["model"].isin(models),
        ["model", "test_query_accuracy_mean"],
    ].copy()


def plot_learning_curve(
    output_path: Path,
    history: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str,
    common_epochs: int,
    labels: dict[str, str],
    title: str,
    test_accuracy: pd.DataFrame,
) -> None:
    common = history[history["epoch"] <= common_epochs].copy()
    by_epoch = common.groupby(["model", "epoch"], as_index=False).agg(
        mean=(metric, "mean"),
        std=(metric, "std"),
        n=("seed", "nunique"),
    )
    by_epoch["sem"] = by_epoch["std"] / np.sqrt(by_epoch["n"].clip(lower=1))

    ordered_models = list(summary["model"])
    fig, (ax_curve, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.8),
        dpi=180,
        gridspec_kw={"width_ratios": [1.9, 1.0]},
    )

    for model in ordered_models:
        group = by_epoch[by_epoch["model"] == model].sort_values("epoch")
        if group.empty:
            continue
        color = COLORS.get(model, None)
        x = group["epoch"].to_numpy(dtype=float)
        y = group["mean"].to_numpy(dtype=float) * 100.0
        sem = group["sem"].fillna(0.0).to_numpy(dtype=float) * 100.0
        label = _display_name(model, labels)
        ax_curve.plot(x, y, color=color, linewidth=2.2, label=label)
        ax_curve.fill_between(x, y - sem, y + sem, color=color, alpha=0.15, linewidth=0)

    ax_curve.set_title("Validation Learning Curve", fontsize=12)
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Validation query accuracy (%)")
    ax_curve.set_xlim(1, common_epochs)
    ymin = max(0.0, float(by_epoch["mean"].min()) * 100.0 - 0.6)
    ymax = min(100.0, float(by_epoch["mean"].max()) * 100.0 + 0.5)
    ax_curve.set_ylim(ymin, ymax)
    ax_curve.grid(True, alpha=0.25)
    ax_curve.legend(frameon=False, fontsize=8, loc="best")

    bar_frame = summary.copy()
    x = np.arange(len(bar_frame))
    means = bar_frame["common_mean"].to_numpy(dtype=float) * 100.0
    colors = [COLORS.get(model, "#666666") for model in bar_frame["model"]]
    ax_bar.bar(x, means, color=colors, width=0.72)
    ax_bar.set_title(f"Mean Over {common_epochs} Epochs", fontsize=12)
    ax_bar.set_ylabel("Validation accuracy (%)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [_display_name(str(model), labels) for model in bar_frame["model"]],
        rotation=28,
        ha="right",
        fontsize=8,
    )
    ax_bar.set_ylim(max(0.0, float(means.min()) - 0.6), min(100.0, float(means.max()) + 0.5))
    ax_bar.grid(True, axis="y", alpha=0.25)

    if not test_accuracy.empty:
        test_map = dict(zip(test_accuracy["model"], test_accuracy["test_query_accuracy_mean"]))
        for idx, model in enumerate(bar_frame["model"]):
            value = test_map.get(model)
            if value is None:
                continue
            ax_bar.text(
                idx,
                means[idx] + 0.08,
                f"test {float(value) * 100:.2f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    fig.suptitle(title, fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Omniglot associative sweep validation learning curves."
    )
    parser.add_argument("output_dir", type=Path, help="Sweep output directory.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Models to include in the figure.",
    )
    parser.add_argument("--metric", default="val_query_accuracy")
    parser.add_argument(
        "--common-epochs",
        type=int,
        default=None,
        help="Restrict the comparison to this many shared epochs. Defaults to the shortest model/seed run.",
    )
    parser.add_argument(
        "--baseline-models",
        nargs="+",
        default=["random_sparse_conv_fast_memory", "weight_shuffle_conv_fast_memory"],
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Override a display label, formatted as model=Display Name.",
    )
    parser.add_argument(
        "--title",
        default="Omniglot Reversal Learning: Connectome Fast Memory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path. Defaults to <output_dir>/omniglot_learning_curve.png.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    labels = _parse_labels(args.label)
    history = load_history(output_dir, args.metric, args.models)
    common_epochs = common_epoch_count(history, args.common_epochs)
    per_seed = per_seed_learning_summary(history, args.metric, common_epochs)
    summary = aggregate_learning_summary(per_seed)
    paired = paired_learning_comparisons(per_seed, args.baseline_models)
    test_accuracy = load_test_accuracy(output_dir, args.models)

    output_path = args.output or (output_dir / "omniglot_learning_curve.png")
    plot_learning_curve(
        output_path=output_path,
        history=history,
        summary=summary,
        metric=args.metric,
        common_epochs=common_epochs,
        labels=labels,
        title=args.title,
        test_accuracy=test_accuracy,
    )
    per_seed.to_csv(output_dir / "learning_curve_by_seed.csv", index=False)
    summary.to_csv(output_dir / "learning_curve_summary.csv", index=False)
    if not paired.empty:
        paired.to_csv(output_dir / "learning_curve_paired_comparisons.csv", index=False)

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary.round(6).to_string(index=False))
    print(f"\nwrote {output_path}")
    print(f"wrote {output_dir / 'learning_curve_summary.csv'}")
    print(f"wrote {output_dir / 'learning_curve_by_seed.csv'}")
    if not paired.empty:
        print(f"wrote {output_dir / 'learning_curve_paired_comparisons.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
