#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _read_first(patterns: list[Path]) -> pd.DataFrame | None:
    for pattern in patterns:
        matches = sorted(pattern.parent.glob(pattern.name))
        if matches:
            return pd.read_csv(matches[-1])
    return None


def _load_run(root: Path, label: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    run_dir = root / label
    args_files = sorted(run_dir.glob("*_args.json"))
    args = {}
    if args_files:
        with args_files[-1].open("r", encoding="utf-8") as f:
            args = json.load(f)
    train = _read_first([run_dir / "*_train.csv"])
    eval_df = _read_first([run_dir / "*_eval.csv"])
    if train is not None:
        train = train.copy()
        train["model"] = label
    if eval_df is not None:
        eval_df = eval_df.copy()
        eval_df["model"] = label
    return train, eval_df, args


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["rnn_param_matched", "mb_bpu"])
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    plot_dir = args.root / "analysis"
    plot_dir.mkdir(parents=True, exist_ok=True)

    trains = []
    evals = []
    run_args: dict[str, dict] = {}
    for model in args.models:
        train, eval_df, model_args = _load_run(args.root, model)
        run_args[model] = model_args
        if train is not None:
            trains.append(train)
        if eval_df is not None:
            evals.append(eval_df)

    summary_rows = []
    if trains:
        train_all = pd.concat(trains, ignore_index=True)
        train_all.to_csv(plot_dir / "training_logs_combined.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        for model, group in train_all.groupby("model"):
            group = group.sort_values("T")
            ax.plot(group["T"], group["median"], label=f"{model} median")
            ax.plot(group["T"], group["mean"], linestyle="--", alpha=0.7, label=f"{model} mean")
            last = group.iloc[-1]
            summary_rows.append(
                {
                    "model": model,
                    "source": "train",
                    "T": int(last["T"]),
                    "score": float(last["median"]),
                    "mean": float(last["mean"]),
                    "median": float(last["median"]),
                    "min": float(last["min"]),
                    "max": float(last["max"]),
                }
            )
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Episode reward")
        ax.set_title("Training reward")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / "training_reward.png")
        plt.close(fig)

    if evals:
        eval_all = pd.concat(evals, ignore_index=True)
        eval_all.to_csv(plot_dir / "eval_logs_combined.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        for model, group in eval_all.groupby("model"):
            group = group.sort_values("T")
            ax.plot(group["T"], group["r_mean"], marker="o", label=model)
            last = group.iloc[-1]
            summary_rows.append(
                {
                    "model": model,
                    "source": "eval",
                    "T": int(last["T"]),
                    "score": float(last["r_mean"]),
                    "mean": float(last["r_mean"]),
                    "median": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                }
            )
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Mean deterministic eval reward")
        ax.set_title("Evaluation reward")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / "eval_reward.png")
        plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary.to_csv(plot_dir / "summary.csv", index=False)
        ranked = summary.sort_values(["source", "score"], ascending=[True, False])
        best_by_source = ranked.groupby("source").head(1)
        best_by_source.to_csv(plot_dir / "best_by_source.csv", index=False)

    with (plot_dir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump(run_args, f, indent=2, sort_keys=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
