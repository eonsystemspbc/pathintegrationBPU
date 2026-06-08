#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_mb_associative_learning as mb  # noqa: E402


POOL_SENSORY = "sensory"
POOL_INTERNAL = "internal"
POOL_OUTPUT = "output"
DEFAULT_COMPARE_MODELS = (
    mb.MODEL_HEMIBRAIN,
    mb.MODEL_RANDOM,
    mb.MODEL_DEGREE_PRESERVING,
    mb.MODEL_WEIGHT_SHUFFLE,
)
PRIMARY_METRICS = (
    "test_query_accuracy",
    "test_initial_probe_accuracy",
    "test_reversal_probe_accuracy",
    "test_loss",
    "best_val_loss",
)


@dataclass(frozen=True)
class PruneResult:
    matrix: sparse.coo_matrix
    keep_indices: np.ndarray
    candidate_internal_indices: np.ndarray
    kept_internal_indices: np.ndarray
    metadata: dict[str, object]


def load_pool_assignments(path: Path, n: int) -> pd.DataFrame:
    pools = pd.read_csv(path)
    required = {"index", "pool"}
    missing = required.difference(pools.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    pools = pools.copy()
    pools["index"] = pools["index"].astype(int)
    pools["pool"] = pools["pool"].astype(str)
    pools = pools[(pools["index"] >= 0) & (pools["index"] < int(n))]
    if pools["index"].duplicated().any():
        dupes = pools.loc[pools["index"].duplicated(), "index"].tolist()
        raise ValueError(f"pool assignments contain duplicate indices: {dupes[:10]}")
    if pools.empty:
        raise ValueError(f"no pool assignments overlap matrix size {n}")
    return pools.sort_values("index").reset_index(drop=True)


def pool_indices(pools: pd.DataFrame, n: int, pool_name: str) -> np.ndarray:
    if pool_name == POOL_SENSORY and "is_sensory" in pools.columns:
        mask = pools["is_sensory"].astype(str).str.lower().isin({"true", "1"})
    elif pool_name == POOL_OUTPUT and "is_output" in pools.columns:
        mask = pools["is_output"].astype(str).str.lower().isin({"true", "1"})
    elif pool_name == POOL_INTERNAL and "is_internal" in pools.columns:
        mask = pools["is_internal"].astype(str).str.lower().isin({"true", "1"})
    else:
        mask = pools["pool"] == pool_name
    idx = pools.loc[mask, "index"].to_numpy(dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < int(n))]
    return np.unique(idx)


def _unique_neighbors_by_column(matrix: sparse.csc_matrix, nodes: np.ndarray) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for node in np.asarray(nodes, dtype=np.int64):
        start = int(matrix.indptr[node])
        stop = int(matrix.indptr[node + 1])
        if stop > start:
            chunks.append(matrix.indices[start:stop])
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(chunks).astype(np.int64, copy=False))


def _unique_neighbors_by_row(matrix: sparse.csr_matrix, nodes: np.ndarray) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for node in np.asarray(nodes, dtype=np.int64):
        start = int(matrix.indptr[node])
        stop = int(matrix.indptr[node + 1])
        if stop > start:
            chunks.append(matrix.indices[start:stop])
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(chunks).astype(np.int64, copy=False))


def reachable_within_hops(
    matrix: sparse.coo_matrix,
    starts: np.ndarray,
    max_hops: int,
    *,
    direction: str,
) -> dict[int, int]:
    """Return node -> shortest hop distance using W_rec[post, pre] orientation."""
    if max_hops < 0:
        raise ValueError("max_hops must be nonnegative")
    starts = np.unique(np.asarray(starts, dtype=np.int64))
    reached: dict[int, int] = {int(node): 0 for node in starts}
    frontier = starts
    if direction == "forward":
        lookup = matrix.tocsc()
        neighbors = _unique_neighbors_by_column
    elif direction == "backward":
        lookup = matrix.tocsr()
        neighbors = _unique_neighbors_by_row
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

    for hop in range(1, int(max_hops) + 1):
        if frontier.size == 0:
            break
        next_nodes = neighbors(lookup, frontier)
        if next_nodes.size == 0:
            break
        new_nodes = np.array(
            [int(node) for node in next_nodes if int(node) not in reached],
            dtype=np.int64,
        )
        for node in new_nodes:
            reached[int(node)] = hop
        frontier = new_nodes
    return reached


def rank_internal_bridge_nodes(
    matrix: sparse.coo_matrix,
    sensory: np.ndarray,
    output: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    if candidates.size == 0:
        return candidates
    matrix = matrix.astype(np.float32).tocoo()
    matrix.sum_duplicates()
    n = int(matrix.shape[0])
    rows = matrix.row.astype(np.int64, copy=False)
    cols = matrix.col.astype(np.int64, copy=False)
    abs_weight = np.abs(matrix.data.astype(np.float64, copy=False))
    in_degree = np.bincount(rows, minlength=n)
    out_degree = np.bincount(cols, minlength=n)
    weighted_in = np.bincount(rows, weights=abs_weight, minlength=n)
    weighted_out = np.bincount(cols, weights=abs_weight, minlength=n)

    sensory_mask = np.zeros(n, dtype=bool)
    sensory_mask[np.asarray(sensory, dtype=np.int64)] = True
    output_mask = np.zeros(n, dtype=bool)
    output_mask[np.asarray(output, dtype=np.int64)] = True

    from_sensory = np.bincount(
        rows[sensory_mask[cols]],
        weights=abs_weight[sensory_mask[cols]],
        minlength=n,
    )
    to_output = np.bincount(
        cols[output_mask[rows]],
        weights=abs_weight[output_mask[rows]],
        minlength=n,
    )
    candidates = np.asarray(candidates, dtype=np.int64)
    score = (
        np.log1p(in_degree[candidates] + out_degree[candidates])
        + np.log1p(weighted_in[candidates] + weighted_out[candidates])
        + 2.0 * np.log1p(from_sensory[candidates])
        + 2.0 * np.log1p(to_output[candidates])
    )
    order = np.lexsort((candidates, -score))
    return candidates[order]


def prune_recurrent_matrix(
    matrix: sparse.coo_matrix,
    pools: pd.DataFrame,
    *,
    max_hops: int,
    max_internal_nodes: int,
) -> PruneResult:
    matrix = matrix.astype(np.float32).tocoo()
    matrix.sum_duplicates()
    n = int(matrix.shape[0])
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got {matrix.shape}")
    sensory = pool_indices(pools, n, POOL_SENSORY)
    internal = pool_indices(pools, n, POOL_INTERNAL)
    output = pool_indices(pools, n, POOL_OUTPUT)
    if sensory.size == 0:
        raise ValueError("cannot prune without sensory pool nodes")
    if output.size == 0:
        raise ValueError("cannot prune without output pool nodes")

    forward = reachable_within_hops(matrix, sensory, max_hops, direction="forward")
    backward = reachable_within_hops(matrix, output, max_hops, direction="backward")
    bridge = np.array(
        sorted(set(forward).intersection(backward).intersection(set(internal.tolist()))),
        dtype=np.int64,
    )
    ranked = rank_internal_bridge_nodes(matrix, sensory, output, bridge)
    if max_internal_nodes > 0:
        kept_internal = np.sort(ranked[: int(max_internal_nodes)])
    else:
        kept_internal = np.sort(ranked)
    keep = np.unique(np.concatenate([sensory, kept_internal, output]).astype(np.int64))
    pruned = matrix.tocsr()[keep, :][:, keep].tocoo()
    pruned.sum_duplicates()
    if pruned.nnz == 0:
        raise ValueError(
            "pruned matrix has no edges. Increase --prune-max-hops or "
            "--prune-max-internal-nodes."
        )

    before_counts = pools["pool"].value_counts().sort_index().to_dict()
    kept_pool_rows = pools[pools["index"].isin(set(keep.tolist()))]
    after_counts = kept_pool_rows["pool"].value_counts().sort_index().to_dict()
    metadata: dict[str, object] = {
        "strategy": "sensory_output_short_path_bridge",
        "orientation": "W_rec[post_index, pre_index]",
        "max_hops": int(max_hops),
        "max_internal_nodes": int(max_internal_nodes),
        "original_N": n,
        "original_edges": int(matrix.nnz),
        "pruned_N": int(pruned.shape[0]),
        "pruned_edges": int(pruned.nnz),
        "sensory_count": int(sensory.size),
        "output_count": int(output.size),
        "candidate_internal_count": int(bridge.size),
        "kept_internal_count": int(kept_internal.size),
        "pool_counts_before": {str(k): int(v) for k, v in before_counts.items()},
        "pool_counts_after": {str(k): int(v) for k, v in after_counts.items()},
    }
    return PruneResult(
        matrix=pruned,
        keep_indices=keep,
        candidate_internal_indices=bridge,
        kept_internal_indices=kept_internal,
        metadata=metadata,
    )


def write_matrix_artifacts(
    output_dir: Path,
    matrix: sparse.coo_matrix,
    pools: pd.DataFrame,
    *,
    prefix: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / f"{prefix}_adjacency_unsigned.npz"
    sparse.save_npz(matrix_path, matrix.astype(np.float32).tocoo())
    indices = np.arange(matrix.shape[0], dtype=np.int64)
    nodes = pd.DataFrame({"original_index": indices, "pruned_index": indices})
    pool_subset = pools[pools["index"].isin(set(indices.tolist()))].copy()
    if not pool_subset.empty:
        pool_subset = pool_subset.rename(columns={"index": "original_index"})
        nodes = nodes.merge(pool_subset, how="left", on="original_index")
    nodes.to_csv(output_dir / f"{prefix}_nodes.csv", index=False)
    return matrix_path


def write_pruned_artifacts(output_dir: Path, result: PruneResult, pools: pd.DataFrame) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "pruned_adjacency_unsigned.npz"
    sparse.save_npz(matrix_path, result.matrix.astype(np.float32).tocoo())

    keep_df = pd.DataFrame(
        {
            "original_index": result.keep_indices,
            "pruned_index": np.arange(result.keep_indices.size, dtype=np.int64),
        }
    )
    pool_subset = pools[pools["index"].isin(set(result.keep_indices.tolist()))].copy()
    pool_subset = pool_subset.rename(columns={"index": "original_index"})
    keep_df = keep_df.merge(pool_subset, how="left", on="original_index")
    keep_df.to_csv(output_dir / "pruned_nodes.csv", index=False)

    metadata = dict(result.metadata)
    metadata["kept_original_indices"] = [int(x) for x in result.keep_indices.tolist()]
    (output_dir / "pruning_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = [
        "# Pruned Mushroom Body Recurrent Matrix",
        "",
        "Pruning keeps sensory and output pool nodes, then keeps internal nodes that",
        "are both reachable from sensory nodes and can reach output nodes within the",
        "configured hop budget. If too many internal bridge nodes are found, they are",
        "ranked by degree and direct sensory/output bridge weight.",
        "",
        "```",
        json.dumps(result.metadata, indent=2, sort_keys=True),
        "```",
        "",
    ]
    (output_dir / "pruning_report.md").write_text("\n".join(lines), encoding="utf-8")
    return matrix_path


def _append_option(argv: list[str], name: str, value: object) -> None:
    argv.extend([name, str(value)])


def runner_argv(matrix: Path, output_dir: Path, args: argparse.Namespace) -> list[str]:
    argv: list[str] = [
        "--matrix",
        str(matrix),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--recurrent-runtime",
        args.recurrent_runtime,
        "--max-neurons",
        "0",
        "--models",
    ]
    argv.extend(args.models)
    argv.append("--seeds")
    argv.extend(str(seed) for seed in args.seeds)
    for name in (
        "epochs",
        "batch_size",
        "train_batches",
        "val_batches",
        "test_batches",
        "lr",
        "patience",
        "grad_clip",
        "state_clip",
        "log_every_seconds",
        "num_odors",
        "odor_dim",
        "odors_per_episode",
        "reversal_count",
        "reversal_repeats",
        "odor_sparsity",
        "odor_noise_std",
        "data_seed",
        "init_seed",
        "val_seed",
        "test_seed",
    ):
        _append_option(argv, f"--{name.replace('_', '-')}", getattr(args, name))
    return argv


def run_condition(condition: str, matrix: Path, output_dir: Path, args: argparse.Namespace) -> None:
    condition_dir = output_dir / condition
    print(
        f"condition-start condition={condition} matrix={matrix} output_dir={condition_dir}",
        flush=True,
    )
    code = mb.main(runner_argv(matrix, condition_dir, args))
    if code != 0:
        raise RuntimeError(f"{condition} run failed with exit code {code}")


def summarize_condition_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    seed_frames = []
    for condition in ("unpruned", "pruned"):
        condition_dir = output_dir / condition
        summary = pd.read_csv(condition_dir / "metrics_summary.csv")
        summary.insert(0, "condition", condition)
        frames.append(summary)
        metrics = pd.read_csv(condition_dir / "metrics_by_seed.csv")
        metrics.insert(0, "condition", condition)
        seed_frames.append(metrics)
    combined_summary = pd.concat(frames, ignore_index=True)
    combined_seeds = pd.concat(seed_frames, ignore_index=True)
    combined_summary.to_csv(output_dir / "pruned_vs_unpruned_summary.csv", index=False)
    combined_seeds.to_csv(output_dir / "pruned_vs_unpruned_metrics_by_seed.csv", index=False)
    return combined_summary, combined_seeds


def paired_pruned_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    models = sorted(metrics["model"].dropna().unique().tolist())
    for model in models:
        unpruned = metrics[
            (metrics["condition"] == "unpruned") & (metrics["model"] == model)
        ]
        pruned = metrics[(metrics["condition"] == "pruned") & (metrics["model"] == model)]
        for metric in PRIMARY_METRICS:
            if metric not in metrics.columns:
                continue
            merged = unpruned[["seed", metric]].merge(
                pruned[["seed", metric]],
                on="seed",
                suffixes=("_unpruned", "_pruned"),
            )
            if merged.empty:
                continue
            diffs = (
                merged[f"{metric}_pruned"].astype(float)
                - merged[f"{metric}_unpruned"].astype(float)
            )
            n = int(diffs.shape[0])
            mean = float(diffs.mean())
            std = float(diffs.std(ddof=1)) if n > 1 else float("nan")
            sem = float(std / math.sqrt(n)) if n > 1 else float("nan")
            ci = 1.96 * sem if n > 1 else float("nan")
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "N": n,
                    "mean_delta_pruned_minus_unpruned": mean,
                    "std": std,
                    "sem": sem,
                    "ci95_low": mean - ci if n > 1 else float("nan"),
                    "ci95_high": mean + ci if n > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def write_comparison_report(
    output_dir: Path,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    prune_metadata: dict[str, object],
) -> None:
    lines = [
        "# Pruned vs Unpruned MB Associative RNN Comparison",
        "",
        "This run compares the same `AssociativeRNN` odor-valence reversal task on",
        "the original recurrent matrix and a sensory-output pruned recurrent matrix.",
        "Controls are regenerated separately for each matrix, so random-sparse,",
        "degree-preserving, and weight-shuffled controls remain matched to the graph",
        "size and support being tested.",
        "",
        "## Pruning Metadata",
        "",
        "```json",
        json.dumps(prune_metadata, indent=2, sort_keys=True),
        "```",
        "",
        "## Summary",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "## Paired Pruned - Unpruned Deltas",
        "",
        "```",
        deltas.to_string(index=False) if not deltas.empty else "No paired deltas.",
        "```",
        "",
    ]
    (output_dir / "pruned_vs_unpruned_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a sensory-output pruned mushroom-body recurrent matrix and "
            "compare it with the unpruned matrix on the standard odor-valence "
            "AssociativeRNN task."
        )
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz"),
        help="Prepared mushroom-body adjacency npz.",
    )
    parser.add_argument(
        "--pool-assignments",
        type=Path,
        default=Path("outputs/hemibrain_mushroom_body_plume/pool_assignments.csv"),
        help="Pool assignment CSV aligned to the matrix indices.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mb_associative_pruned_comparison"),
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=mb.MODEL_CHOICES,
        default=list(DEFAULT_COMPARE_MODELS),
    )
    parser.add_argument("--recurrent-runtime", choices=mb.RUNTIME_CHOICES, default="sparse")
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=0,
        help="Use leading N neurons before pruning for smoke tests. 0 keeps the full matrix.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--prune-max-hops", type=int, default=2)
    parser.add_argument(
        "--prune-max-internal-nodes",
        type=int,
        default=1024,
        help="Maximum internal bridge nodes to keep. 0 keeps all bridge nodes.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-batches", type=int, default=200)
    parser.add_argument("--val-batches", type=int, default=40)
    parser.add_argument("--test-batches", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--state-clip", type=float, default=5.0)
    parser.add_argument("--log-every-seconds", type=float, default=30.0)
    parser.add_argument("--num-odors", type=int, default=64)
    parser.add_argument("--odor-dim", type=int, default=64)
    parser.add_argument("--odors-per-episode", type=int, default=6)
    parser.add_argument("--reversal-count", type=int, default=3)
    parser.add_argument("--reversal-repeats", type=int, default=1)
    parser.add_argument("--odor-sparsity", type=float, default=0.20)
    parser.add_argument("--odor-noise-std", type=float, default=0.03)
    parser.add_argument("--data-seed", type=int, default=12345)
    parser.add_argument("--init-seed", type=int, default=7000)
    parser.add_argument("--val-seed", type=int, default=22000)
    parser.add_argument("--test-seed", type=int, default=33000)
    args = parser.parse_args(argv)
    if args.max_neurons < 0:
        parser.error("--max-neurons must be nonnegative")
    if args.prune_max_hops < 0:
        parser.error("--prune-max-hops must be nonnegative")
    if args.prune_max_internal_nodes < 0:
        parser.error("--prune-max-internal-nodes must be nonnegative")
    if args.odors_per_episode > args.num_odors:
        parser.error("--odors-per-episode cannot exceed --num-odors")
    if args.reversal_count > args.odors_per_episode:
        parser.error("--reversal-count cannot exceed --odors-per-episode")
    if not (0.0 < args.odor_sparsity <= 1.0):
        parser.error("--odor-sparsity must be in (0, 1]")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_matrix = mb.load_base_matrix(args.matrix, args.max_neurons)
    pools = load_pool_assignments(args.pool_assignments, base_matrix.shape[0])
    matrix_dir = output_dir / "matrices"
    unpruned_matrix = write_matrix_artifacts(
        matrix_dir,
        base_matrix,
        pools,
        prefix="unpruned",
    )
    prune_result = prune_recurrent_matrix(
        base_matrix,
        pools,
        max_hops=args.prune_max_hops,
        max_internal_nodes=args.prune_max_internal_nodes,
    )
    pruned_matrix = write_pruned_artifacts(matrix_dir, prune_result, pools)
    print(
        "prune-ready "
        f"original_N={prune_result.metadata['original_N']} "
        f"pruned_N={prune_result.metadata['pruned_N']} "
        f"original_edges={prune_result.metadata['original_edges']} "
        f"pruned_edges={prune_result.metadata['pruned_edges']} "
        f"kept_internal={prune_result.metadata['kept_internal_count']}",
        flush=True,
    )

    run_condition("unpruned", unpruned_matrix, output_dir, args)
    run_condition("pruned", pruned_matrix, output_dir, args)
    summary, per_seed = summarize_condition_outputs(output_dir)
    deltas = paired_pruned_deltas(per_seed)
    deltas.to_csv(output_dir / "pruned_vs_unpruned_paired_deltas.csv", index=False)
    write_comparison_report(output_dir, summary, deltas, prune_result.metadata)
    print(
        "comparison-complete "
        f"summary={output_dir / 'pruned_vs_unpruned_summary.csv'} "
        f"deltas={output_dir / 'pruned_vs_unpruned_paired_deltas.csv'} "
        f"report={output_dir / 'pruned_vs_unpruned_report.md'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
