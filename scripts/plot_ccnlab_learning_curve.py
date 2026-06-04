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


MODEL_ORDER = (
    "connectome_kalman_filter",
    "degree_preserving_kalman_filter",
    "random_sparse_kalman_filter",
    "weight_shuffle_kalman_filter",
    "connectome_rescorla_wagner",
    "degree_preserving_rescorla_wagner",
    "random_sparse_rescorla_wagner",
    "weight_shuffle_rescorla_wagner",
    "connectome_temporal_difference",
    "degree_preserving_temporal_difference",
    "random_sparse_temporal_difference",
    "weight_shuffle_temporal_difference",
    "connectome_seeded",
    "hemibrain_seeded",
    "degree_preserving_random",
    "random_sparse",
    "weight_shuffle",
)
COLORS = {
    "connectome": "#147d7e",
    "degree": "#7d5ab6",
    "random": "#4d6fb3",
    "weight": "#c77d23",
    "baseline": "#707070",
}


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", np.trapz)
    return float(fn(y, x))


def model_color(model: str) -> str:
    if model.startswith(("connectome", "hemibrain")):
        return COLORS["connectome"]
    if model.startswith("degree_preserving"):
        return COLORS["degree"]
    if model.startswith("random_sparse"):
        return COLORS["random"]
    if model.startswith("weight_shuffle"):
        return COLORS["weight"]
    return COLORS["baseline"]


def display_name(model: str) -> str:
    replacements = {
        "connectome": "Connectome",
        "hemibrain": "Connectome",
        "random_sparse": "Random sparse",
        "degree_preserving": "Degree-preserving",
        "weight_shuffle": "Weight shuffle",
        "rescorla_wagner": "RW",
        "kalman_filter": "Kalman",
        "temporal_difference": "TD",
        "seeded": "RPE",
        "random": "Random",
    }
    label = model
    for old, new in replacements.items():
        label = label.replace(old, new)
    return " ".join(part for part in label.split("_") if part)


def learner_matches(model: str, learner: str) -> bool:
    if learner == "all":
        return True
    if learner == "kalman":
        return model.endswith("_kalman_filter")
    if learner == "rw":
        return model.endswith("_rescorla_wagner")
    if learner == "td":
        return model.endswith("_temporal_difference")
    if learner == "rpe":
        return model in {
            "connectome_seeded",
            "hemibrain_seeded",
            "random_sparse",
            "degree_preserving_random",
            "weight_shuffle",
        }
    raise ValueError(f"unknown learner: {learner}")


def is_topology_model(model: str) -> bool:
    return model.startswith(
        ("connectome", "hemibrain", "random_sparse", "degree_preserving", "weight_shuffle")
    )


def matched_controls(model: str) -> tuple[str, ...]:
    if model in {"connectome_seeded", "hemibrain_seeded"}:
        return ("random_sparse", "degree_preserving_random", "weight_shuffle")
    prefixes = ("connectome_", "hemibrain_")
    if not model.startswith(prefixes):
        return ()
    suffix = model.split("_", 1)[1]
    if suffix == "rescorla_wagner":
        return (
            "random_sparse_rescorla_wagner",
            "degree_preserving_rescorla_wagner",
            "weight_shuffle_rescorla_wagner",
        )
    if suffix == "kalman_filter":
        return (
            "random_sparse_kalman_filter",
            "degree_preserving_kalman_filter",
            "weight_shuffle_kalman_filter",
        )
    if suffix == "temporal_difference":
        return (
            "random_sparse_temporal_difference",
            "degree_preserving_temporal_difference",
            "weight_shuffle_temporal_difference",
        )
    return ()


def load_trial_history(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "ccnlab_trial_history.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; rerun the CCNLab sweep with a version that writes "
            "ccnlab_trial_history.csv"
        )
    history = pd.read_csv(path)
    required = {
        "model",
        "seed",
        "experiment",
        "group",
        "phase",
        "trial_in_phase",
        "has_cs",
        "learning_response_mean",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    history = history.copy()
    history["phase"] = history["phase"].fillna("").replace("", "unphased")
    history["seed"] = history["seed"].astype(int)
    history["trial_in_phase"] = history["trial_in_phase"].astype(int)
    history["learning_response_mean"] = history["learning_response_mean"].astype(float)
    return history


def select_history(
    history: pd.DataFrame,
    models: list[str] | None,
    learner: str,
    experiments: list[str],
    phases: list[str],
    groups: list[str],
    include_no_cs: bool,
    max_trials: int,
) -> pd.DataFrame:
    selected = history.copy()
    if models:
        selected = selected[selected["model"].isin(models)].copy()
    else:
        selected = selected[
            selected["model"].map(is_topology_model)
            & selected["model"].map(lambda model: learner_matches(str(model), learner))
        ].copy()
    if experiments:
        selected = selected[selected["experiment"].isin(experiments)].copy()
    if phases:
        selected = selected[selected["phase"].isin(phases)].copy()
    if groups:
        selected = selected[selected["group"].isin(groups)].copy()
    if not include_no_cs:
        selected = selected[selected["has_cs"].astype(int) == 1].copy()
    if max_trials > 0:
        selected = selected[selected["trial_in_phase"] <= max_trials].copy()
    selected = selected[np.isfinite(selected["learning_response_mean"])].copy()
    if selected.empty:
        raise ValueError("no CCNLab trial-history rows remain after filtering")
    return selected


def model_sort_key(model: str) -> tuple[int, str]:
    try:
        return (MODEL_ORDER.index(model), model)
    except ValueError:
        return (len(MODEL_ORDER), model)


def panel_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["experiment"].astype(str)
        + " | "
        + frame["group"].astype(str)
        + " | "
        + frame["phase"].astype(str)
    )


def plot_curves(
    selected: pd.DataFrame,
    output_path: Path,
    title: str,
    max_panels: int,
) -> None:
    selected = selected.copy()
    selected["panel"] = panel_key(selected)
    panels = (
        selected.groupby("panel")["learning_response_mean"]
        .size()
        .sort_values(ascending=False)
        .head(max_panels)
        .index.tolist()
    )
    selected = selected[selected["panel"].isin(panels)].copy()
    n_panels = len(panels)
    ncols = min(3, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 3.2 * nrows),
        dpi=150,
        squeeze=False,
    )
    models = sorted(selected["model"].unique(), key=model_sort_key)
    for ax, panel in zip(axes.ravel(), panels):
        frame = selected[selected["panel"] == panel]
        by_seed = (
            frame.groupby(["model", "seed", "trial_in_phase"], as_index=False)[
                "learning_response_mean"
            ]
            .mean()
            .sort_values(["model", "seed", "trial_in_phase"])
        )
        summary = by_seed.groupby(["model", "trial_in_phase"], as_index=False).agg(
            mean=("learning_response_mean", "mean"),
            std=("learning_response_mean", "std"),
            seed_count=("seed", "nunique"),
        )
        for model in models:
            curve = summary[summary["model"] == model].sort_values("trial_in_phase")
            if curve.empty:
                continue
            x = curve["trial_in_phase"].to_numpy(dtype=float)
            y = curve["mean"].to_numpy(dtype=float)
            std = curve["std"].fillna(0.0).to_numpy(dtype=float)
            count = np.maximum(curve["seed_count"].to_numpy(dtype=float), 1.0)
            se = std / np.sqrt(count)
            ax.plot(x, y, label=display_name(model), color=model_color(model), linewidth=1.8)
            ax.fill_between(x, y - se, y + se, color=model_color(model), alpha=0.14, linewidth=0)
        ax.set_title(panel, fontsize=9)
        ax.set_xlabel("Trial in phase")
        ax.set_ylabel("Mean response")
        ax.grid(True, alpha=0.22)
    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, max(1, len(labels))))
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def per_seed_summary(selected: pd.DataFrame, early_trials: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    selected = selected.copy()
    selected["panel"] = panel_key(selected)
    for (model, seed), group in selected.groupby(["model", "seed"]):
        by_trial = (
            group.groupby("trial_in_phase", as_index=False)["learning_response_mean"]
            .mean()
            .sort_values("trial_in_phase")
        )
        x = by_trial["trial_in_phase"].to_numpy(dtype=float)
        y = by_trial["learning_response_mean"].to_numpy(dtype=float)
        early = by_trial[by_trial["trial_in_phase"] <= early_trials]
        rows.append(
            {
                "model": str(model),
                "seed": int(seed),
                "trial_count": int(len(by_trial)),
                "panel_count": int(group["panel"].nunique()),
                "early_trials": int(early_trials),
                "early_mean": float(early["learning_response_mean"].mean())
                if not early.empty
                else float("nan"),
                "curve_auc": _trapezoid(y, x) if len(by_trial) > 1 else float(y[0]),
                "curve_mean": float(np.mean(y)),
                "final_mean": float(y[-1]),
            }
        )
    return pd.DataFrame(rows)


def aggregate_summary(by_seed: pd.DataFrame) -> pd.DataFrame:
    return (
        by_seed.groupby("model", as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            trial_count=("trial_count", "mean"),
            panel_count=("panel_count", "mean"),
            early_mean=("early_mean", "mean"),
            early_mean_std=("early_mean", "std"),
            curve_auc=("curve_auc", "mean"),
            curve_auc_std=("curve_auc", "std"),
            curve_mean=("curve_mean", "mean"),
            curve_mean_std=("curve_mean", "std"),
            final_mean=("final_mean", "mean"),
            final_mean_std=("final_mean", "std"),
        )
        .sort_values("model", key=lambda column: column.map(model_sort_key))
    )


def paired_comparisons(by_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    metrics = ("early_mean", "curve_auc", "curve_mean", "final_mean")
    pivot = {
        metric: by_seed.pivot_table(index="seed", columns="model", values=metric, aggfunc="mean")
        for metric in metrics
    }
    for model in by_seed["model"].unique():
        controls = matched_controls(str(model))
        if not controls:
            continue
        for baseline in controls:
            for metric, table in pivot.items():
                if model not in table or baseline not in table:
                    continue
                delta = (table[model] - table[baseline]).dropna()
                if delta.empty:
                    continue
                count = int(delta.shape[0])
                std = float(delta.std(ddof=1)) if count > 1 else float("nan")
                se = std / math.sqrt(count) if count > 1 else float("nan")
                ci = 1.96 * se if count > 1 else float("nan")
                rows.append(
                    {
                        "model": str(model),
                        "baseline_model": str(baseline),
                        "metric": metric,
                        "paired_seed_count": count,
                        "mean_delta": float(delta.mean()),
                        "std_delta": std,
                        "se_delta": se,
                        "ci95_low": float(delta.mean() - ci) if count > 1 else float("nan"),
                        "ci95_high": float(delta.mean() + ci) if count > 1 else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CCNLab trial-by-trial response curves from a sweep output."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--learner",
        choices=("all", "kalman", "rw", "td", "rpe"),
        default="kalman",
        help="Default model family to plot when --models is omitted.",
    )
    parser.add_argument("--experiments", nargs="+", default=[])
    parser.add_argument("--phases", nargs="+", default=[])
    parser.add_argument("--groups", nargs="+", default=[])
    parser.add_argument("--include-no-cs", action="store_true")
    parser.add_argument("--max-trials", type=int, default=40)
    parser.add_argument("--early-trials", type=int, default=5)
    parser.add_argument("--max-panels", type=int, default=12)
    parser.add_argument("--title", default="CCNLab Trial-by-Trial Response Curves")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    history = load_trial_history(args.output_dir)
    selected = select_history(
        history,
        models=args.models,
        learner=args.learner,
        experiments=args.experiments,
        phases=args.phases,
        groups=args.groups,
        include_no_cs=args.include_no_cs,
        max_trials=args.max_trials,
    )
    output_path = args.output or (args.output_dir / "ccnlab_learning_curve.png")
    plot_curves(
        selected=selected,
        output_path=output_path,
        title=args.title,
        max_panels=args.max_panels,
    )
    by_seed = per_seed_summary(selected, args.early_trials)
    summary = aggregate_summary(by_seed)
    paired = paired_comparisons(by_seed)
    by_seed.to_csv(args.output_dir / "ccnlab_learning_curve_by_seed.csv", index=False)
    summary.to_csv(args.output_dir / "ccnlab_learning_curve_summary.csv", index=False)
    if not paired.empty:
        paired.to_csv(
            args.output_dir / "ccnlab_learning_curve_paired_comparisons.csv",
            index=False,
        )
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary.to_string(index=False))
    print(f"\nwrote {output_path}")
    print(f"wrote {args.output_dir / 'ccnlab_learning_curve_summary.csv'}")
    print(f"wrote {args.output_dir / 'ccnlab_learning_curve_by_seed.csv'}")
    if not paired.empty:
        print(f"wrote {args.output_dir / 'ccnlab_learning_curve_paired_comparisons.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
