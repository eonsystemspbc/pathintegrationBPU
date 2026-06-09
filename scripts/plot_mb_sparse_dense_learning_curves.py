#!/usr/bin/env python3
"""Overlay associative-reversal learning curves for three MB recurrent families.

The figure compares, on the *same* standard odor-valence associative-reversal
task, three recurrent-matrix families and their matched controls:

  * normal  -- the full unpruned hemibrain sparse recurrent matrix
  * pruned  -- the sensory-output short-path-bridge pruned sparse matrix
  * dense   -- a fully-connected trainable recurrent matrix

Each family contributes a connectome-seeded model plus controls:
  sparse families (normal, pruned): random_sparse and weight_shuffle controls;
  dense family: random_dense control.

Color encodes the model identity (connectome / random / weight-shuffle) and
line style encodes the family (normal solid, pruned dashed, dense dotted), so
"sparse vs dense for the connectome and for random" reads directly off the plot.

Sparse families are trained at their tuned lr 1e-3; the dense family has ~100x
more trainable parameters and is trained at its tuned lr 1e-4. That per-family
learning rate is annotated on the figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]

# Model identity -> color (shared across families so sparse/dense line up by hue).
MODEL_COLORS = {
    "connectome": "#1f77b4",  # hemibrain_seeded / hemibrain_dense
    "random": "#ff7f0e",      # random_sparse / random_dense
    "weight_shuffle": "#2ca02c",
}
MODEL_LABELS = {
    "connectome": "Connectome-seeded",
    "random": "Random",
    "weight_shuffle": "Weight shuffle",
}
# Family -> line style.
FAMILY_STYLES = {"normal": "-", "pruned": "--", "dense": ":"}
FAMILY_LABELS = {"normal": "Normal (unpruned sparse)", "pruned": "Pruned sparse", "dense": "Dense"}

# Map each raw model name in a family's logs to a shared model-identity key.
MODEL_IDENTITY = {
    "hemibrain_seeded": "connectome",
    "hemibrain_dense": "connectome",
    "random_sparse": "random",
    "random_dense": "random",
    "weight_shuffle": "weight_shuffle",
}


def load_family(paths: list[Path]) -> pd.DataFrame:
    """Concatenate one or more loss_history.csv files for a single family."""
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def mean_curve(df: pd.DataFrame, model: str, column: str) -> pd.DataFrame:
    sub = df[df["model"] == model]
    if sub.empty:
        raise ValueError(f"model {model!r} absent from history")
    return (
        sub.groupby("epoch", as_index=False)[column]
        .mean()
        .sort_values("epoch")
        .reset_index(drop=True)
    )


def discover_seed_histories(run_dir: Path) -> list[Path]:
    """Return loss_history.csv files, either directly in run_dir or in seed* subdirs."""
    direct = run_dir / "loss_history.csv"
    if direct.exists():
        return [direct]
    seed_files = sorted(run_dir.glob("seed*/loss_history.csv"))
    if not seed_files:
        raise FileNotFoundError(f"no loss_history.csv under {run_dir}")
    return seed_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normal-dir",
        type=Path,
        default=ROOT / "outputs/mb_assoc_normal_regen_2seed_40ep",
        help="Run dir for the unpruned sparse family (loss_history.csv or seed*/ subdirs).",
    )
    parser.add_argument(
        "--pruned-dir",
        type=Path,
        default=ROOT / "outputs/mb_assoc_pruned_regen_2seed_40ep",
    )
    parser.add_argument(
        "--dense-dir",
        type=Path,
        default=ROOT / "outputs/mb_assoc_dense_only_lr1e4_3seed_100ep",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "docs/results/mb_associative_pruned_vs_unpruned_2seed"
        / "mb_sparse_dense_reversal_learning_curves_core_controls.png",
    )
    parser.add_argument("--xmax", type=float, default=100.0, help="Epoch axis upper limit.")
    args = parser.parse_args()

    families = {
        "normal": (load_family(discover_seed_histories(args.normal_dir)),
                   ["hemibrain_seeded", "random_sparse", "weight_shuffle"], "lr 1e-3"),
        "pruned": (load_family(discover_seed_histories(args.pruned_dir)),
                   ["hemibrain_seeded", "random_sparse", "weight_shuffle"], "lr 1e-3"),
        "dense": (load_family(discover_seed_histories(args.dense_dir)),
                  ["hemibrain_dense", "random_dense"], "lr 1e-4"),
    }

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    for family, (df, models, _lr) in families.items():
        style = FAMILY_STYLES[family]
        for model in models:
            ident = MODEL_IDENTITY[model]
            color = MODEL_COLORS[ident]
            acc = mean_curve(df, model, "val_reversal_probe_accuracy")
            loss = mean_curve(df, model, "val_loss")
            ax_acc.plot(acc["epoch"], acc["val_reversal_probe_accuracy"] * 100.0,
                        color=color, linestyle=style, linewidth=1.8)
            ax_loss.plot(loss["epoch"], loss["val_loss"],
                         color=color, linestyle=style, linewidth=1.8)

    ax_acc.set_title("Validation reversal accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(60, 100)
    ax_acc.grid(True, alpha=0.25)

    ax_loss.set_title("Validation loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("BCE loss")
    ax_loss.grid(True, alpha=0.25)

    for ax in (ax_acc, ax_loss):
        ax.set_xlim(0, args.xmax)

    # Two-part legend: family (line style) and model identity (color).
    family_handles = [
        Line2D([0], [0], color="0.25", linestyle=FAMILY_STYLES[f],
               label=f"{FAMILY_LABELS[f]} ({lr})")
        for f, (_d, _m, lr) in families.items()
    ]
    model_handles = [
        Line2D([0], [0], color=MODEL_COLORS[k], linestyle="-", label=MODEL_LABELS[k])
        for k in ("connectome", "random", "weight_shuffle")
    ]
    leg1 = ax_acc.legend(handles=family_handles, title="Family (line style)",
                         frameon=False, fontsize=8, loc="lower right")
    ax_acc.add_artist(leg1)
    ax_acc.legend(handles=model_handles, title="Model (color)",
                  frameon=False, fontsize=8, loc="center right")

    fig.suptitle(
        "Sparse vs dense MB recurrent matrices on odor-valence associative reversal",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
