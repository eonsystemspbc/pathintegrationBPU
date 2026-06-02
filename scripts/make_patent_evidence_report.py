#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.channels import (  # noqa: E402
    TASK_MB_ASSOCIATIVE_LEARNING,
    TASK_OPTIC_FLOW,
)
from src.config import TASK_CX_LANDMARK_BUMP, TASK_CX_POLAR_BUMP  # noqa: E402
from src.selector import select_connectome, selection_to_markdown  # noqa: E402


TASKS = (
    TASK_CX_POLAR_BUMP,
    TASK_CX_LANDMARK_BUMP,
    TASK_MB_ASSOCIATIVE_LEARNING,
    TASK_OPTIC_FLOW,
)


def _relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _best_rows(path: Path) -> list[str]:
    table = _read_csv(path)
    if table is None or table.empty:
        return []
    metric_candidates = [
        "test_reversal_probe_accuracy_mean",
        "test_overall_rmse_mean",
        "home_bearing_angular_error_mean",
        "position_rmse_mean",
        "mse_mean",
    ]
    metric = next((col for col in metric_candidates if col in table.columns), None)
    if metric is None or "model" not in table.columns:
        return [table.head(5).to_string(index=False)]
    ascending = not metric.endswith("accuracy_mean")
    ordered = table.sort_values(metric, ascending=ascending).head(5)
    return [ordered.to_string(index=False)]


def _collect_result_tables(results_root: Path) -> list[tuple[str, Path, list[str]]]:
    tables: list[tuple[str, Path, list[str]]] = []
    for path in sorted(results_root.rglob("metrics_summary.csv")):
        rows = _best_rows(path)
        if rows:
            tables.append(("metrics_summary", path, rows))
    for path in sorted(results_root.rglob("cross_region_summary.csv")):
        rows = _best_rows(path)
        if rows:
            tables.append(("cross_region_summary", path, rows))
    for path in sorted(results_root.rglob("low_power_proxy_summary.csv")):
        rows = _best_rows(path)
        if rows:
            tables.append(("low_power_proxy_summary", path, rows))
    return tables


def _manifest_summary(results_root: Path) -> list[str]:
    lines = []
    for path in sorted(results_root.rglob("run_manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        git = manifest.get("git", {})
        files = manifest.get("files", {})
        lines.append(
            f"- `{_relative(path, results_root)}`: commit `{git.get('commit')}`, "
            f"files recorded `{len(files)}`"
        )
    return lines


def _missing_checklist(results_root: Path) -> list[str]:
    checks = {
        "selector JSON/Markdown outputs": any(results_root.rglob("selectors/*.json")),
        "CX path multi-seed metrics": any(results_root.rglob("*cx_path*/metrics_summary.csv")),
        "MB associative multi-seed metrics": any(results_root.rglob("*mb_associative*/metrics_summary.csv")),
        "cross-region match/mismatch report": any(results_root.rglob("*cross_region*/cross_region_report.md")),
        "optic-flow benchmark report": any(results_root.rglob("*optic_flow*/optic_flow_report.md")),
        "low-power proxy report": any(results_root.rglob("low_power_proxy_report.md")),
        "run manifests with checksums": any(results_root.rglob("run_manifest.json")),
    }
    return [
        f"- {'DONE' if done else 'TODO'}: {name}"
        for name, done in checks.items()
    ]


def write_report(results_root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Patent Evidence Report",
        "",
        "Working invention summary: a system that selects a biological brain-region connectome as a task-matched sparse recurrent prior, maps task-specific channels into sensory/output ports, trains under controlled curricula, and evaluates sparse deployment evidence for embodied perception/control.",
        "",
        "## Pipeline",
        "",
        "```mermaid",
        "flowchart LR",
        "  T[Task channel spec] --> S[Connectome selector]",
        "  S --> C[Biological sparse recurrent support]",
        "  C --> A[Adapter mapping into sensory/output pools]",
        "  A --> R[Controlled curricula and matched controls]",
        "  R --> E[Metrics, manifests, low-power proxy]",
        "```",
        "",
        "## Selector Decisions",
        "",
    ]
    for task in TASKS:
        result = select_connectome(task)
        lines.extend(
            [
                f"### {task}",
                "",
                f"- Selected: `{result.selected.connectome}`",
                f"- K: `{result.expected_k}`",
                f"- Controls: `{', '.join(result.matched_controls)}`",
                f"- Rationale: {result.decision_reasons[2] if len(result.decision_reasons) > 2 else result.decision_reasons[0]}",
                "",
            ]
        )
    lines.extend(
        [
            "## Evidence Tables",
            "",
        ]
    )
    tables = _collect_result_tables(results_root)
    if not tables:
        lines.append("No metrics tables found yet. Run `scripts/plan_patent_experiments.py` and execute the generated AWS commands.")
        lines.append("")
    for label, path, rendered in tables:
        lines.extend(
            [
                f"### {_relative(path, results_root)}",
                "",
                f"Type: `{label}`",
                "",
                "```",
                *rendered,
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Provenance",
            "",
        ]
    )
    manifest_lines = _manifest_summary(results_root)
    lines.extend(manifest_lines if manifest_lines else ["- TODO: no run_manifest.json files found yet."])
    lines.extend(
        [
            "",
            "## Missing Evidence Checklist",
            "",
            *_missing_checklist(results_root),
            "",
            "## Claim-Support Notes",
            "",
            "- Strongest current support should come from region-matched MB associative learning, optic-flow support controls, and cross-region match/mismatch.",
            "- The low-power claim should be limited to sparse inference proxy evidence until direct hardware energy measurements are added.",
            "- Whole-brain results should be treated as a substrate comparison unless biological sensory/motor port labels are tightened.",
            "",
            "## Appendix: Full Selector Output",
            "",
        ]
    )
    for task in TASKS:
        lines.extend([selection_to_markdown(select_connectome(task)), ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate patent evidence artifacts into one Markdown report.")
    parser.add_argument("--results-root", type=Path, default=Path("outputs/patent_evidence"))
    parser.add_argument("--output", type=Path, default=Path("outputs/patent_evidence/patent_evidence_report.md"))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    write_report(args.results_root, args.output)
    print(f"wrote patent evidence report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
