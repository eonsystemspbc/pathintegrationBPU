#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.channels import TASK_MB_ASSOCIATIVE_LEARNING, TASK_OPTIC_FLOW  # noqa: E402
from src.config import TASK_CX_LANDMARK_BUMP, TASK_CX_POLAR_BUMP  # noqa: E402
from src.selector import select_connectome  # noqa: E402


@dataclass(frozen=True)
class ExperimentCommand:
    name: str
    claim_element: str
    purpose: str
    command: tuple[str, ...]
    expected_artifacts: tuple[str, ...]
    priority: str = "required"

    def shell(self) -> str:
        return " ".join(shlex.quote(part) for part in self.command)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["shell"] = self.shell()
        return data


def _seeds(values: tuple[int, ...]) -> list[str]:
    return [str(seed) for seed in values]


def build_plan(
    output_root: Path,
    seeds: tuple[int, ...],
    epochs: int,
    device: str,
) -> list[ExperimentCommand]:
    evidence = output_root / "patent_evidence"
    graphs = evidence / "graphs"
    results = evidence / "results"
    selectors = evidence / "selectors"
    commands: list[ExperimentCommand] = []
    for task in (TASK_CX_POLAR_BUMP, TASK_MB_ASSOCIATIVE_LEARNING, TASK_OPTIC_FLOW, TASK_CX_LANDMARK_BUMP):
        selected = select_connectome(task)
        commands.append(
            ExperimentCommand(
                name=f"selector_{task}",
                claim_element="task-matched biological connectome selection",
                purpose=f"Record deterministic selector decision for {task}.",
                command=(
                    "python",
                    "scripts/select_connectome.py",
                    "--task",
                    task,
                    "--json-output",
                    str(selectors / f"{task}.json"),
                    "--markdown-output",
                    str(selectors / f"{task}.md"),
                ),
                expected_artifacts=(
                    str(selectors / f"{task}.json"),
                    str(selectors / f"{task}.md"),
                ),
                priority="required",
            )
        )

    commands.extend(
        [
            ExperimentCommand(
                name="download_cx_graph_exports",
                claim_element="biological sparse recurrent prior",
                purpose="Download or reuse raw hemibrain central-complex exports.",
                command=(
                    "python",
                    "run_benchmark.py",
                    "--mode",
                    "download",
                    "--connectome",
                    "hemibrain_cx",
                    "--output-dir",
                    str(graphs / "hemibrain_cx"),
                    "--cache-dir",
                    str(graphs / "hemibrain_cx"),
                ),
                expected_artifacts=(
                    str(graphs / "hemibrain_cx" / "neurons.csv"),
                    str(graphs / "hemibrain_cx" / "connections.csv"),
                ),
            ),
            ExperimentCommand(
                name="prepare_cx_graph",
                claim_element="biological sparse recurrent prior",
                purpose="Prepare the hemibrain central-complex substrate and record graph/pool metadata.",
                command=(
                    "python",
                    "run_benchmark.py",
                    "--mode",
                    "prepare",
                    "--connectome",
                    "hemibrain_cx",
                    "--output-dir",
                    str(graphs / "hemibrain_cx"),
                    "--cache-dir",
                    str(graphs / "hemibrain_cx"),
                ),
                expected_artifacts=(
                    str(graphs / "hemibrain_cx" / "graph_metadata.json"),
                    str(graphs / "hemibrain_cx" / "pool_assignments.csv"),
                    str(graphs / "hemibrain_cx" / "run_manifest.json"),
                ),
            ),
            ExperimentCommand(
                name="download_mb_graph_exports",
                claim_element="biological sparse recurrent prior",
                purpose="Download or reuse raw hemibrain mushroom-body exports.",
                command=(
                    "python",
                    "run_benchmark.py",
                    "--mode",
                    "download",
                    "--connectome",
                    "hemibrain_mushroom_body",
                    "--output-dir",
                    str(graphs / "hemibrain_mb"),
                    "--cache-dir",
                    str(graphs / "hemibrain_mb"),
                ),
                expected_artifacts=(
                    str(graphs / "hemibrain_mb" / "neurons.csv"),
                    str(graphs / "hemibrain_mb" / "connections.csv"),
                ),
            ),
            ExperimentCommand(
                name="prepare_mb_graph",
                claim_element="biological sparse recurrent prior",
                purpose="Prepare the hemibrain mushroom-body substrate and record graph/pool metadata.",
                command=(
                    "python",
                    "run_benchmark.py",
                    "--mode",
                    "prepare",
                    "--connectome",
                    "hemibrain_mushroom_body",
                    "--output-dir",
                    str(graphs / "hemibrain_mb"),
                    "--cache-dir",
                    str(graphs / "hemibrain_mb"),
                ),
                expected_artifacts=(
                    str(graphs / "hemibrain_mb" / "graph_metadata.json"),
                    str(graphs / "hemibrain_mb" / "pool_assignments.csv"),
                    str(graphs / "hemibrain_mb" / "run_manifest.json"),
                ),
            ),
            ExperimentCommand(
                name="cx_path_observed_5seed",
                claim_element="controlled curriculum training on matched connectome",
                purpose="Train the central-complex substrate on path integration with support-preserving recurrent training and matched controls.",
                command=(
                    "python",
                    "run_benchmark.py",
                    "--mode",
                    "all",
                    "--connectome",
                    "hemibrain_cx",
                    "--task",
                    "cx_polar_bump",
                    "--comparison",
                    "structure",
                    "--train-recurrent",
                    "observed",
                    "--recurrent-runtime",
                    "sparse",
                    "--seeds",
                    *_seeds(seeds),
                    "--epochs",
                    str(epochs),
                    "--device",
                    device,
                    "--output-dir",
                    str(results / "cx_path_observed_5seed"),
                    "--cache-dir",
                    str(results / "cx_path_observed_5seed"),
                ),
                expected_artifacts=(
                    str(results / "cx_path_observed_5seed" / "metrics_summary.csv"),
                    str(results / "cx_path_observed_5seed" / "metrics_by_seed.csv"),
                    str(results / "cx_path_observed_5seed" / "run_manifest.json"),
                ),
            ),
            ExperimentCommand(
                name="cx_landmark_observed_5seed",
                claim_element="controlled curriculum and cue-correction training",
                purpose="Stress the central-complex substrate with landmark cues and passive displacement.",
                command=(
                    "python",
                    "run_benchmark.py",
                    "--mode",
                    "all",
                    "--connectome",
                    "hemibrain_cx",
                    "--task",
                    "cx_landmark_bump",
                    "--comparison",
                    "structure",
                    "--train-recurrent",
                    "observed",
                    "--recurrent-runtime",
                    "sparse",
                    "--seeds",
                    *_seeds(seeds),
                    "--epochs",
                    str(epochs),
                    "--device",
                    device,
                    "--output-dir",
                    str(results / "cx_landmark_observed_5seed"),
                    "--cache-dir",
                    str(results / "cx_landmark_observed_5seed"),
                ),
                expected_artifacts=(
                    str(results / "cx_landmark_observed_5seed" / "metrics_summary.csv"),
                    str(results / "cx_landmark_observed_5seed" / "run_manifest.json"),
                ),
                priority="recommended",
            ),
            ExperimentCommand(
                name="mb_associative_observed_5seed",
                claim_element="task-region match against controls",
                purpose="Train mushroom-body substrate on odor-valence associative learning and matched sparse controls.",
                command=(
                    "python",
                    "scripts/run_mb_associative_learning.py",
                    "--matrix",
                    str(graphs / "hemibrain_mb" / "adjacency_unsigned.npz"),
                    "--output-dir",
                    str(results / "mb_associative_observed_5seed"),
                    "--models",
                    "hemibrain_seeded",
                    "weight_shuffle",
                    "random_sparse",
                    "--recurrent-runtime",
                    "sparse",
                    "--seeds",
                    *_seeds(seeds),
                    "--epochs",
                    str(epochs),
                    "--device",
                    device,
                ),
                expected_artifacts=(
                    str(results / "mb_associative_observed_5seed" / "metrics_summary.csv"),
                    str(results / "mb_associative_observed_5seed" / "associative_learning_report.md"),
                ),
            ),
            ExperimentCommand(
                name="cross_region_match_mismatch_5seed",
                claim_element="task-matched selector validation",
                purpose="Run CX and MB on matched and swapped task families to show region identity matters.",
                command=(
                    "python",
                    "scripts/run_cross_region_transfer.py",
                    "--mode",
                    "all",
                    "--pairs",
                    "all",
                    "--cx-dir",
                    str(graphs / "hemibrain_cx"),
                    "--mb-dir",
                    str(graphs / "hemibrain_mb"),
                    "--output-dir",
                    str(results / "cross_region_match_mismatch_5seed"),
                    "--seeds",
                    *_seeds(seeds),
                    "--epochs",
                    str(epochs),
                    "--device",
                    device,
                    "--path-recurrent-runtime",
                    "sparse",
                    "--path-train-recurrent",
                    "observed",
                ),
                expected_artifacts=(
                    str(results / "cross_region_match_mismatch_5seed" / "cross_region_summary.csv"),
                    str(results / "cross_region_match_mismatch_5seed" / "cross_region_report.md"),
                ),
            ),
            ExperimentCommand(
                name="optic_flow_5seed",
                claim_element="embodied perception task with visual connectome prior",
                purpose="Train optic-lobe support on optic-flow ego-motion prediction against support/weight controls.",
                command=(
                    "python",
                    "scripts/run_optic_flow_benchmark.py",
                    "--mode",
                    "all",
                    "--output-dir",
                    str(results / "optic_flow_5seed"),
                    "--cache-dir",
                    str(results / "optic_flow_5seed"),
                    "--models",
                    "optic_lobe_seeded",
                    "random_weight_topology",
                    "shuffled_topology",
                    "random_sparse",
                    "--seeds",
                    *_seeds(seeds),
                    "--epochs",
                    str(epochs),
                    "--device",
                    device,
                ),
                expected_artifacts=(
                    str(results / "optic_flow_5seed" / "metrics_summary.csv"),
                    str(results / "optic_flow_5seed" / "optic_flow_report.md"),
                ),
            ),
            ExperimentCommand(
                name="low_power_proxy",
                claim_element="low-power sparse deployment evidence",
                purpose="Estimate sparse-vs-dense memory/operation footprint and attach measured latency when metrics are present.",
                command=(
                    "python",
                    "scripts/run_low_power_proxy_benchmark.py",
                    "--graph-dir",
                    str(results / "cx_path_observed_5seed"),
                    "--graph-dir",
                    str(results / "optic_flow_5seed"),
                    "--output-dir",
                    str(evidence / "low_power_proxy"),
                ),
                expected_artifacts=(
                    str(evidence / "low_power_proxy" / "low_power_proxy_summary.csv"),
                    str(evidence / "low_power_proxy" / "low_power_proxy_report.md"),
                ),
            ),
            ExperimentCommand(
                name="patent_evidence_report",
                claim_element="reduction-to-practice evidence package",
                purpose="Aggregate selectors, results, controls, manifests, and missing-evidence checklist.",
                command=(
                    "python",
                    "scripts/make_patent_evidence_report.py",
                    "--results-root",
                    str(evidence),
                    "--output",
                    str(evidence / "patent_evidence_report.md"),
                ),
                expected_artifacts=(str(evidence / "patent_evidence_report.md"),),
            ),
        ]
    )
    return commands


def write_plan(output_dir: Path, commands: list[ExperimentCommand]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "patent_experiment_plan.json"
    md_path = output_dir / "patent_experiment_plan.md"
    sh_path = output_dir / "run_patent_experiments.sh"
    json_path.write_text(
        json.dumps([command.to_dict() for command in commands], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_lines = [
        "# Patent Experiment Plan",
        "",
        "Run these from the repository root on the AWS machine. Required commands are ordered before the final report command.",
        "",
        "| step | priority | claim element | purpose |",
        "| ---: | --- | --- | --- |",
    ]
    for idx, command in enumerate(commands, start=1):
        md_lines.append(
            f"| {idx} | {command.priority} | {command.claim_element} | {command.purpose} |"
        )
    md_lines.extend(["", "## Commands", ""])
    for idx, command in enumerate(commands, start=1):
        md_lines.extend(
            [
                f"### {idx}. {command.name}",
                "",
                "```bash",
                command.shell(),
                "```",
                "",
                "Expected artifacts:",
                "",
            ]
        )
        md_lines.extend(f"- `{artifact}`" for artifact in command.expected_artifacts)
        md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    sh_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    sh_lines.extend(command.shell() for command in commands)
    sh_path.write_text("\n".join(sh_lines) + "\n", encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AWS commands for patent evidence runs.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--plan-dir", type=Path, default=Path("outputs/patent_evidence_plan"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    commands = build_plan(
        output_root=args.output_root,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        device=args.device,
    )
    write_plan(args.plan_dir, commands)
    print(f"wrote {len(commands)} commands to {args.plan_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
