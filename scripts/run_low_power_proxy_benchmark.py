#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class FootprintRow:
    graph_dir: str
    connectome: str
    N: int
    recurrent_edges: int
    density: float
    bitwidth: int
    dense_recurrent_bytes: int
    sparse_csr_bytes: int
    dense_recurrent_ops_per_step: int
    sparse_recurrent_ops_per_step: int
    ops_reduction: float
    memory_reduction: float
    latency_ms_per_sequence_mean: float
    latency_ms_per_sequence_min: float


def _load_metadata(graph_dir: Path) -> dict[str, object]:
    metadata_path = graph_dir / "graph_metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {"connectome": graph_dir.name}


def _load_matrix(graph_dir: Path) -> sparse.csr_matrix | None:
    for name in ("adjacency_unsigned.npz", "adjacency_signed.npz"):
        path = graph_dir / name
        if path.exists():
            return sparse.load_npz(path).tocsr()
    return None


def _latency_stats(graph_dir: Path) -> tuple[float, float]:
    candidates = [
        graph_dir / "metrics_by_seed.csv",
        graph_dir / "metrics_summary.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        metrics = pd.read_csv(path)
        if "latency_ms_per_sequence" not in metrics.columns:
            continue
        values = pd.to_numeric(metrics["latency_ms_per_sequence"], errors="coerce").dropna()
        if len(values):
            return float(values.mean()), float(values.min())
    return float("nan"), float("nan")


def footprint_rows(graph_dir: Path, bitwidths: tuple[int, ...]) -> list[FootprintRow]:
    metadata = _load_metadata(graph_dir)
    matrix = _load_matrix(graph_dir)
    if matrix is not None:
        n = int(matrix.shape[0])
        edge_count = int(matrix.nnz)
    else:
        n = int(metadata.get("N", 0))
        edge_count = int(metadata.get("unsigned_edge_count", metadata.get("edge_count", 0)))
    dense_slots = int(n * n)
    density = float(edge_count / dense_slots) if dense_slots else 0.0
    latency_mean, latency_min = _latency_stats(graph_dir)
    rows = []
    for bitwidth in bitwidths:
        value_bytes = max(int(bitwidth) // 8, 1)
        dense_bytes = dense_slots * value_bytes
        sparse_bytes = edge_count * value_bytes + edge_count * 4 + (n + 1) * 4
        rows.append(
            FootprintRow(
                graph_dir=str(graph_dir),
                connectome=str(metadata.get("connectome", graph_dir.name)),
                N=n,
                recurrent_edges=edge_count,
                density=density,
                bitwidth=int(bitwidth),
                dense_recurrent_bytes=int(dense_bytes),
                sparse_csr_bytes=int(sparse_bytes),
                dense_recurrent_ops_per_step=dense_slots,
                sparse_recurrent_ops_per_step=edge_count,
                ops_reduction=float(dense_slots / edge_count) if edge_count else float("nan"),
                memory_reduction=float(dense_bytes / sparse_bytes) if sparse_bytes else float("nan"),
                latency_ms_per_sequence_mean=latency_mean,
                latency_ms_per_sequence_min=latency_min,
            )
        )
    return rows


def write_report(output_dir: Path, table: pd.DataFrame) -> None:
    lines = [
        "# Low-Power Sparse Deployment Proxy",
        "",
        "This report estimates sparse recurrent inference footprint relative to a dense recurrent matrix with the same neuron count.",
        "It is a proxy: hardware power claims still need direct measurement on the target edge device.",
        "",
        "## Summary",
        "",
        "```",
        table.to_string(index=False) if not table.empty else "No graph directories were processed.",
        "```",
        "",
        "## Interpretation",
        "",
        "- `ops_reduction` is dense recurrent slots divided by observed sparse recurrent edges.",
        "- `memory_reduction` is dense recurrent weight storage divided by approximate CSR storage at the selected bitwidth.",
        "- Latency columns are attached from existing metrics files when available.",
        "",
        "Recommended next hardware run: repeat the same model on CPU, Jetson/Raspberry Pi, and any neuromorphic or sparse-accelerator target, then add measured joules per sequence.",
        "",
    ]
    (output_dir / "low_power_proxy_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate sparse-vs-dense low-power deployment proxies.")
    parser.add_argument("--graph-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/low_power_proxy"))
    parser.add_argument("--bitwidths", nargs="+", type=int, default=[32, 16, 8])
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[FootprintRow] = []
    for graph_dir in args.graph_dir:
        rows.extend(footprint_rows(graph_dir, tuple(args.bitwidths)))
    table = pd.DataFrame([row.__dict__ for row in rows])
    if not table.empty:
        table = table.sort_values(["graph_dir", "bitwidth"], ascending=[True, False])
    table.to_csv(args.output_dir / "low_power_proxy_summary.csv", index=False)
    write_report(args.output_dir, table)
    print(f"wrote low-power proxy summary to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
