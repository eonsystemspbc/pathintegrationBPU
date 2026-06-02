from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from .channels import (
    CONNECTOME_FLYWIRE_OPTIC_LOBE,
    TASK_EMBODIED_FORAGING,
    TASK_MB_ASSOCIATIVE_LEARNING,
    TASK_OPTIC_FLOW,
    AdapterMapping,
    RegionPortSpec,
    build_adapter_mapping,
    region_port_spec,
    task_channel_spec,
    task_name,
)
from .config import (
    CONNECTOME_FLYWIRE_MUSHROOM_BODY,
    CONNECTOME_FLYWIRE_WHOLE,
    CONNECTOME_HEMIBRAIN_CX,
    CONNECTOME_HEMIBRAIN_MUSHROOM_BODY,
    STRUCTURE_COMPARISON_MODELS,
    TASK_CARTESIAN,
    TASK_CX_LANDMARK_BUMP,
    TASK_CX_POLAR_BUMP,
    WHOLE_BRAIN_COMPARISON_MODELS,
    TaskSpec,
)


@dataclass(frozen=True)
class CandidateConnectome:
    connectome: str
    region_name: str
    source: str
    task_tags: tuple[str, ...]
    primary_rois: tuple[str, ...]
    default_k: int
    default_controls: tuple[str, ...]
    train_recurrent_modes: tuple[str, ...]
    evidence_tags: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def port_spec(self) -> RegionPortSpec:
        return region_port_spec(self.connectome)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionScore:
    candidate: CandidateConnectome
    total_score: float
    semantic_score: float
    port_score: float
    evidence_score: float
    availability_score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["candidate"] = self.candidate.to_dict()
        return data


@dataclass(frozen=True)
class SelectionResult:
    task_name: str
    selected: CandidateConnectome
    mapping: AdapterMapping
    expected_k: int
    matched_controls: tuple[str, ...]
    train_recurrent_modes: tuple[str, ...]
    ranking: tuple[SelectionScore, ...]
    decision_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "task_name": self.task_name,
            "selected": self.selected.to_dict(),
            "mapping": self.mapping.to_dict(),
            "expected_k": self.expected_k,
            "matched_controls": list(self.matched_controls),
            "train_recurrent_modes": list(self.train_recurrent_modes),
            "ranking": [score.to_dict() for score in self.ranking],
            "decision_reasons": list(self.decision_reasons),
        }


def default_candidate_library() -> tuple[CandidateConnectome, ...]:
    return (
        CandidateConnectome(
            connectome=CONNECTOME_HEMIBRAIN_CX,
            region_name="hemibrain central complex",
            source="neuPrint hemibrain v1.2.1",
            task_tags=(
                "path_integration",
                "heading",
                "ring_attractor",
                "home_vector",
                "navigation",
                "landmark",
                "cue_correction",
                "self_motion",
                "locomotion",
                "angular_velocity",
                "embodied_control",
            ),
            primary_rois=("EB", "PB", "FB", "NO"),
            default_k=3,
            default_controls=STRUCTURE_COMPARISON_MODELS,
            train_recurrent_modes=("frozen", "observed"),
            evidence_tags=(TASK_CARTESIAN, TASK_CX_POLAR_BUMP, TASK_CX_LANDMARK_BUMP),
            notes=("Best matched to path-integration, heading, and cue-correction tasks.",),
        ),
        CandidateConnectome(
            connectome=CONNECTOME_HEMIBRAIN_MUSHROOM_BODY,
            region_name="hemibrain mushroom body",
            source="neuPrint hemibrain v1.2.1",
            task_tags=(
                "odor",
                "olfaction",
                "sparse_code",
                "reward",
                "punishment",
                "associative_learning",
                "memory",
                "valence",
                "reversal_learning",
            ),
            primary_rois=("MB(R)", "MB(L)"),
            default_k=3,
            default_controls=("hemibrain_seeded", "weight_shuffle", "random_sparse"),
            train_recurrent_modes=("observed",),
            evidence_tags=(TASK_MB_ASSOCIATIVE_LEARNING,),
            notes=("Best matched to odor-valence associative memory and reversal tasks.",),
        ),
        CandidateConnectome(
            connectome=CONNECTOME_FLYWIRE_MUSHROOM_BODY,
            region_name="FlyWire mushroom body",
            source="FlyWire release 783",
            task_tags=(
                "odor",
                "olfaction",
                "sparse_code",
                "associative_learning",
                "memory",
                "valence",
            ),
            primary_rois=(
                "MB_CA_L",
                "MB_CA_R",
                "MB_ML_L",
                "MB_ML_R",
                "MB_PED_L",
                "MB_PED_R",
                "MB_VL_L",
                "MB_VL_R",
            ),
            default_k=3,
            default_controls=("connectome_bpu", "weight_shuffle", "random"),
            train_recurrent_modes=("frozen", "observed"),
            evidence_tags=(TASK_MB_ASSOCIATIVE_LEARNING,),
            notes=("Alternative MB substrate when whole FlyWire labels are desired.",),
        ),
        CandidateConnectome(
            connectome=CONNECTOME_FLYWIRE_OPTIC_LOBE,
            region_name="FlyWire optic lobe",
            source="FlyWire release 783 optic-lobe ROI export",
            task_tags=(
                "vision",
                "retinotopy",
                "motion",
                "optic_flow",
                "ego_motion",
                "embodied_perception",
            ),
            primary_rois=("ME_R", "ME_L", "LO_R", "LO_L", "LOP_R", "LOP_L"),
            default_k=2,
            default_controls=(
                "optic_lobe_seeded",
                "random_weight_topology",
                "shuffled_topology",
                "random_sparse",
            ),
            train_recurrent_modes=("observed",),
            evidence_tags=(TASK_OPTIC_FLOW,),
            notes=("Best matched to optic-flow and visual ego-motion perception.",),
        ),
        CandidateConnectome(
            connectome=CONNECTOME_FLYWIRE_WHOLE,
            region_name="FlyWire whole brain",
            source="FlyWire release 783",
            task_tags=(
                "broad_substrate",
                "embodied_perception",
                "embodied_control",
                "navigation",
                "vision",
                "taste",
                "locomotion",
                "motor_control",
            ),
            primary_rois=("whole_brain",),
            default_k=3,
            default_controls=WHOLE_BRAIN_COMPARISON_MODELS,
            train_recurrent_modes=("frozen",),
            evidence_tags=(TASK_EMBODIED_FORAGING,),
            notes=(
                "Whole-brain pools are heuristic; use as broad substrate or embodied-control fallback.",
            ),
        ),
    )


def _task_tags(task: TaskSpec | str) -> set[str]:
    return set(task_channel_spec(task).semantic_tags)


def _availability(connectome: str, available_connectomes: Iterable[str] | None) -> float:
    if available_connectomes is None:
        return 1.0
    return 1.0 if connectome in set(available_connectomes) else 0.0


def _metadata_for(
    connectome: str,
    graph_metadata: Mapping[str, Mapping[str, object]] | None,
) -> Mapping[str, object] | None:
    if graph_metadata is None:
        return None
    return graph_metadata.get(connectome)


def score_candidate(
    task: TaskSpec | str,
    candidate: CandidateConnectome,
    available_connectomes: Iterable[str] | None = None,
    graph_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> SelectionScore:
    task_tags = _task_tags(task)
    candidate_tags = set(candidate.task_tags)
    overlap = sorted(task_tags.intersection(candidate_tags))
    semantic_score = len(overlap) / max(len(task_tags), 1)
    metadata = _metadata_for(candidate.connectome, graph_metadata)
    pool_counts = None
    if metadata is not None and isinstance(metadata.get("pool_counts"), dict):
        pool_counts = {
            str(key): int(value)
            for key, value in dict(metadata["pool_counts"]).items()
        }
    mapping = build_adapter_mapping(task, candidate.port_spec, pool_counts=pool_counts)
    port_score = mapping.mapping_score
    evidence_score = 1.0 if task_name(task) in set(candidate.evidence_tags) else 0.0
    availability_score = _availability(candidate.connectome, available_connectomes)
    total = (
        0.48 * semantic_score
        + 0.32 * port_score
        + 0.12 * evidence_score
        + 0.08 * availability_score
    )
    if availability_score == 0.0:
        total *= 0.25
    reasons = [
        f"semantic overlap: {', '.join(overlap) if overlap else 'none'}",
        f"adapter mapping score: {port_score:.3f}",
        "prior result exists for this task family" if evidence_score else "no direct prior result recorded for this task family",
    ]
    if metadata is not None:
        reasons.append(
            f"prepared graph metadata available: N={metadata.get('N')}, K={metadata.get('estimated_K')}"
        )
    return SelectionScore(
        candidate=candidate,
        total_score=float(total),
        semantic_score=float(semantic_score),
        port_score=float(port_score),
        evidence_score=float(evidence_score),
        availability_score=float(availability_score),
        reasons=tuple(reasons),
    )


def select_connectome(
    task: TaskSpec | str,
    candidates: Iterable[CandidateConnectome] | None = None,
    available_connectomes: Iterable[str] | None = None,
    graph_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> SelectionResult:
    library = tuple(candidates) if candidates is not None else default_candidate_library()
    ranking = tuple(
        sorted(
            (
                score_candidate(
                    task,
                    candidate,
                    available_connectomes=available_connectomes,
                    graph_metadata=graph_metadata,
                )
                for candidate in library
            ),
            key=lambda score: (
                score.total_score,
                score.semantic_score,
                score.port_score,
                score.candidate.connectome,
            ),
            reverse=True,
        )
    )
    if not ranking:
        raise ValueError("No connectome candidates were provided.")
    selected = ranking[0].candidate
    metadata = _metadata_for(selected.connectome, graph_metadata)
    expected_k = selected.default_k
    pool_counts = None
    if metadata is not None:
        expected_k = int(metadata.get("estimated_K", expected_k))
        if isinstance(metadata.get("pool_counts"), dict):
            pool_counts = {
                str(key): int(value)
                for key, value in dict(metadata["pool_counts"]).items()
            }
    mapping = build_adapter_mapping(task, selected.port_spec, pool_counts=pool_counts)
    decision_reasons = (
        f"selected {selected.connectome} for task {task_name(task)}",
        f"top score {ranking[0].total_score:.3f}; runner-up {ranking[1].total_score:.3f}"
        if len(ranking) > 1
        else f"top score {ranking[0].total_score:.3f}",
        *ranking[0].reasons,
    )
    return SelectionResult(
        task_name=task_name(task),
        selected=selected,
        mapping=mapping,
        expected_k=int(expected_k),
        matched_controls=selected.default_controls,
        train_recurrent_modes=selected.train_recurrent_modes,
        ranking=ranking,
        decision_reasons=tuple(decision_reasons),
    )


def selection_to_markdown(result: SelectionResult) -> str:
    lines = [
        f"# Connectome Selection: {result.task_name}",
        "",
        f"Selected connectome: `{result.selected.connectome}` ({result.selected.region_name})",
        f"Expected recurrent depth K: `{result.expected_k}`",
        f"Matched controls: `{', '.join(result.matched_controls)}`",
        f"Train recurrent modes: `{', '.join(result.train_recurrent_modes)}`",
        "",
        "## Decision Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in result.decision_reasons)
    lines.extend(
        [
            "",
            "## Adapter Mapping",
            "",
            f"- Input dimension: `{result.mapping.input_dim}`",
            f"- Output dimension: `{result.mapping.output_dim}`",
            f"- Sensory pool: `{result.mapping.sensory_pool}`",
            f"- Output pool: `{result.mapping.output_pool}`",
            "",
            "### Input Channels",
            "",
        ]
    )
    lines.extend(
        f"- `{name}` -> {port}"
        for name, port in result.mapping.input_channel_to_port.items()
    )
    lines.extend(["", "### Output Channels", ""])
    lines.extend(
        f"- `{name}` -> {port}"
        for name, port in result.mapping.output_channel_to_port.items()
    )
    lines.extend(["", "## Candidate Ranking", ""])
    lines.append("| rank | connectome | total | semantic | port | evidence | available |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for rank, score in enumerate(result.ranking, start=1):
        lines.append(
            "| "
            f"{rank} | `{score.candidate.connectome}` | "
            f"{score.total_score:.3f} | {score.semantic_score:.3f} | "
            f"{score.port_score:.3f} | {score.evidence_score:.3f} | "
            f"{score.availability_score:.3f} |"
        )
    return "\n".join(lines) + "\n"
