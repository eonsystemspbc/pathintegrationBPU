#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.selector import select_connectome, selection_to_markdown  # noqa: E402


def _load_metadata(paths: list[Path]) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            item = json.load(f)
        connectome = str(item.get("connectome") or path.parent.name)
        metadata[connectome] = item
    return metadata


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank task-matched biological connectome priors.")
    parser.add_argument(
        "--task",
        required=True,
        help=(
            "Task name, e.g. cartesian, cx_polar_bump, cx_landmark_bump, "
            "mb_associative_learning, optic_flow, embodied_foraging."
        ),
    )
    parser.add_argument(
        "--available-connectomes",
        nargs="*",
        default=None,
        help="Optional allow-list of connectomes that are prepared/available.",
    )
    parser.add_argument(
        "--graph-metadata",
        nargs="*",
        type=Path,
        default=[],
        help="Optional graph_metadata.json files used to populate K/pool-count evidence.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = select_connectome(
        args.task,
        available_connectomes=args.available_connectomes,
        graph_metadata=_load_metadata(args.graph_metadata),
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    markdown = selection_to_markdown(result)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    if args.json_output is None and args.markdown_output is None:
        print(markdown)
    else:
        print(
            f"selected {result.selected.connectome} for {result.task_name}; "
            f"K={result.expected_k}; controls={','.join(result.matched_controls)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
