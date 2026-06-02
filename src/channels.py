from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .config import (
    CONNECTOME_FLYWIRE_MUSHROOM_BODY,
    CONNECTOME_FLYWIRE_WHOLE,
    CONNECTOME_HEMIBRAIN_CX,
    CONNECTOME_HEMIBRAIN_MUSHROOM_BODY,
    CX_LANDMARK_INPUT_DIM,
    TASK_CARTESIAN,
    TASK_CX_LANDMARK_BUMP,
    TASK_CX_POLAR_BUMP,
    TaskSpec,
    input_dim_for_task,
    output_dim_for_task,
)


CONNECTOME_FLYWIRE_OPTIC_LOBE = "flywire_optic_lobe"
TASK_MB_ASSOCIATIVE_LEARNING = "mb_associative_learning"
TASK_OPTIC_FLOW = "optic_flow"
TASK_EMBODIED_FORAGING = "embodied_foraging"


@dataclass(frozen=True)
class Channel:
    name: str
    role: str
    size: int
    semantic_tags: tuple[str, ...]
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TaskChannelSpec:
    task_name: str
    input_channels: tuple[Channel, ...]
    output_channels: tuple[Channel, ...]
    sensory_tags: tuple[str, ...]
    target_tags: tuple[str, ...]

    @property
    def input_dim(self) -> int:
        return int(sum(channel.size for channel in self.input_channels))

    @property
    def output_dim(self) -> int:
        return int(sum(channel.size for channel in self.output_channels))

    @property
    def semantic_tags(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.sensory_tags).union(self.target_tags)))

    def to_dict(self) -> dict[str, object]:
        return {
            "task_name": self.task_name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_channels": [channel.to_dict() for channel in self.input_channels],
            "output_channels": [channel.to_dict() for channel in self.output_channels],
            "sensory_tags": list(self.sensory_tags),
            "target_tags": list(self.target_tags),
            "semantic_tags": list(self.semantic_tags),
        }


@dataclass(frozen=True)
class RegionPortSpec:
    connectome: str
    region_name: str
    sensory_ports: tuple[str, ...]
    output_ports: tuple[str, ...]
    supported_tags: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AdapterMapping:
    task_name: str
    connectome: str
    region_name: str
    input_dim: int
    output_dim: int
    sensory_pool: str
    output_pool: str
    input_channel_to_port: dict[str, str]
    output_channel_to_port: dict[str, str]
    mapping_score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def task_name(task: TaskSpec | str) -> str:
    if isinstance(task, TaskSpec):
        return str(task.kind)
    return str(task)


def task_channel_spec(task: TaskSpec | str) -> TaskChannelSpec:
    if isinstance(task, TaskSpec):
        if task.kind == TASK_CARTESIAN:
            return TaskChannelSpec(
                task_name=TASK_CARTESIAN,
                input_channels=(
                    Channel("forward_velocity", "input", 1, ("self_motion", "locomotion")),
                    Channel("turn_velocity", "input", 1, ("angular_velocity", "heading")),
                ),
                output_channels=(
                    Channel("heading_cos", "output", 1, ("heading", "path_integration")),
                    Channel("heading_sin", "output", 1, ("heading", "path_integration")),
                    Channel("home_x", "output", 1, ("home_vector", "navigation")),
                    Channel("home_y", "output", 1, ("home_vector", "navigation")),
                ),
                sensory_tags=("self_motion", "locomotion", "angular_velocity"),
                target_tags=("path_integration", "heading", "home_vector", "navigation"),
            )
        if task.kind == TASK_CX_POLAR_BUMP:
            return TaskChannelSpec(
                task_name=TASK_CX_POLAR_BUMP,
                input_channels=(
                    Channel("forward_velocity", "input", 1, ("self_motion", "locomotion")),
                    Channel("turn_velocity", "input", 1, ("angular_velocity", "heading")),
                ),
                output_channels=(
                    Channel(
                        "heading_bump",
                        "output",
                        int(task.heading_bins),
                        ("heading", "ring_attractor", "path_integration"),
                    ),
                    Channel("home_bearing_cos", "output", 1, ("home_vector", "navigation")),
                    Channel("home_bearing_sin", "output", 1, ("home_vector", "navigation")),
                    Channel("home_distance", "output", 1, ("home_vector", "navigation")),
                ),
                sensory_tags=("self_motion", "locomotion", "angular_velocity"),
                target_tags=(
                    "path_integration",
                    "heading",
                    "ring_attractor",
                    "home_vector",
                    "navigation",
                ),
            )
        if task.kind == TASK_CX_LANDMARK_BUMP:
            return TaskChannelSpec(
                task_name=TASK_CX_LANDMARK_BUMP,
                input_channels=(
                    Channel("forward_velocity", "input", 1, ("self_motion", "locomotion")),
                    Channel("turn_velocity", "input", 1, ("angular_velocity", "heading")),
                    Channel("landmark_visible", "input", 1, ("landmark", "cue")),
                    Channel("landmark_bearing_cos", "input", 1, ("landmark", "cue", "bearing")),
                    Channel("landmark_bearing_sin", "input", 1, ("landmark", "cue", "bearing")),
                    Channel("landmark_distance", "input", 1, ("landmark", "cue", "distance")),
                ),
                output_channels=(
                    Channel(
                        "heading_bump",
                        "output",
                        int(task.heading_bins),
                        ("heading", "ring_attractor", "path_integration"),
                    ),
                    Channel("home_bearing_cos", "output", 1, ("home_vector", "navigation")),
                    Channel("home_bearing_sin", "output", 1, ("home_vector", "navigation")),
                    Channel("home_distance", "output", 1, ("home_vector", "navigation")),
                ),
                sensory_tags=(
                    "self_motion",
                    "locomotion",
                    "angular_velocity",
                    "landmark",
                    "cue",
                    "bearing",
                    "distance",
                ),
                target_tags=(
                    "path_integration",
                    "heading",
                    "ring_attractor",
                    "home_vector",
                    "navigation",
                    "cue_correction",
                ),
            )
    name = task_name(task)
    if name == TASK_CARTESIAN:
        return task_channel_spec(TaskSpec(kind=TASK_CARTESIAN))
    if name == TASK_CX_POLAR_BUMP:
        return task_channel_spec(TaskSpec(kind=TASK_CX_POLAR_BUMP))
    if name == TASK_CX_LANDMARK_BUMP:
        return task_channel_spec(TaskSpec(kind=TASK_CX_LANDMARK_BUMP))
    if name == TASK_MB_ASSOCIATIVE_LEARNING:
        return TaskChannelSpec(
            task_name=name,
            input_channels=(
                Channel("odor_pattern", "input", 64, ("odor", "olfaction", "sparse_code")),
                Channel("reward_feedback", "input", 1, ("reward", "valence")),
                Channel("punishment_feedback", "input", 1, ("punishment", "valence")),
                Channel("query_gate", "input", 1, ("memory_query", "association")),
            ),
            output_channels=(
                Channel("odor_valence", "output", 1, ("valence", "associative_learning")),
            ),
            sensory_tags=("odor", "olfaction", "reward", "punishment", "sparse_code"),
            target_tags=("associative_learning", "memory", "valence", "reversal_learning"),
        )
    if name == TASK_OPTIC_FLOW:
        return TaskChannelSpec(
            task_name=name,
            input_channels=(
                Channel("hex_lattice_luminance", "input", 61, ("vision", "retinotopy", "optic_flow")),
            ),
            output_channels=(
                Channel("yaw_rate", "output", 1, ("ego_motion", "optic_flow")),
                Channel("forward_translation", "output", 1, ("ego_motion", "optic_flow")),
                Channel("lateral_translation", "output", 1, ("ego_motion", "optic_flow")),
            ),
            sensory_tags=("vision", "retinotopy", "motion", "optic_flow"),
            target_tags=("ego_motion", "optic_flow", "embodied_perception"),
        )
    if name == TASK_EMBODIED_FORAGING:
        return TaskChannelSpec(
            task_name=name,
            input_channels=(
                Channel("left_taste", "input", 1, ("taste", "contact_sensation")),
                Channel("right_taste", "input", 1, ("taste", "contact_sensation")),
            ),
            output_channels=(
                Channel("turn_command", "output", 1, ("motor_control", "steering")),
                Channel("forward_speed", "output", 1, ("motor_control", "locomotion")),
            ),
            sensory_tags=("taste", "contact_sensation", "foraging"),
            target_tags=("embodied_control", "motor_control", "steering", "locomotion"),
        )
    raise ValueError(f"Unknown task channel spec: {name}")


def region_port_spec(connectome: str) -> RegionPortSpec:
    if connectome == CONNECTOME_HEMIBRAIN_CX:
        return RegionPortSpec(
            connectome=connectome,
            region_name="hemibrain central complex",
            sensory_ports=("ring/ExR/LNO/LCNO/GLNO/SpsP sensory-biased pool",),
            output_ports=("PFL/PFL2/PFL3/PFR output-biased pool",),
            supported_tags=(
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
            notes=(
                "CX ports are derived from ROI flow and known type-family biases.",
                "Adapter weights inject task channels only into sensory pool neurons and read only output pool neurons in the BPU trainer.",
            ),
        )
    if connectome in {CONNECTOME_HEMIBRAIN_MUSHROOM_BODY, CONNECTOME_FLYWIRE_MUSHROOM_BODY}:
        label = (
            "hemibrain mushroom body"
            if connectome == CONNECTOME_HEMIBRAIN_MUSHROOM_BODY
            else "FlyWire mushroom body"
        )
        return RegionPortSpec(
            connectome=connectome,
            region_name=label,
            sensory_ports=("olfactory/KC input-biased pool",),
            output_ports=("MBON/output-biased pool",),
            supported_tags=(
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
            notes=(
                "MB ports are currently ROI-flow pools; future releases should add explicit PN/KC/MBON labels when available.",
            ),
        )
    if connectome == CONNECTOME_FLYWIRE_OPTIC_LOBE:
        return RegionPortSpec(
            connectome=connectome,
            region_name="FlyWire optic lobe",
            sensory_ports=("retinotopic photoreceptor/medulla input lattice",),
            output_ports=("motion/ego-motion readout pool",),
            supported_tags=(
                "vision",
                "retinotopy",
                "motion",
                "optic_flow",
                "ego_motion",
                "embodied_perception",
            ),
            notes=(
                "Optic-lobe benchmark maps a hex-lattice visual channel into the sparse optic-lobe recurrent support.",
            ),
        )
    if connectome == CONNECTOME_FLYWIRE_WHOLE:
        return RegionPortSpec(
            connectome=connectome,
            region_name="FlyWire whole brain",
            sensory_ports=("degree/input-dominant heuristic sensory pool",),
            output_ports=("degree/output-dominant heuristic output pool",),
            supported_tags=(
                "broad_substrate",
                "embodied_perception",
                "embodied_control",
                "navigation",
                "vision",
                "taste",
                "locomotion",
                "motor_control",
            ),
            notes=(
                "Whole-brain ports are heuristic pools and should be treated as a substrate comparison until biological labels are added.",
            ),
        )
    raise ValueError(f"Unknown connectome port spec: {connectome}")


def _best_port_for_channel(channel: Channel, ports: tuple[str, ...]) -> str:
    if not ports:
        return "unmapped"
    lowered_tags = set(channel.semantic_tags)
    for port in ports:
        port_lower = port.lower()
        if any(tag.replace("_", " ") in port_lower or tag in port_lower for tag in lowered_tags):
            return port
    return ports[0]


def build_adapter_mapping(
    task: TaskSpec | str,
    region: RegionPortSpec | str,
    pool_counts: Mapping[str, int] | None = None,
) -> AdapterMapping:
    spec = task_channel_spec(task)
    ports = region_port_spec(region) if isinstance(region, str) else region
    task_tags = set(spec.semantic_tags)
    region_tags = set(ports.supported_tags)
    overlap = sorted(task_tags.intersection(region_tags))
    sensory_ok = bool(set(spec.sensory_tags).intersection(region_tags))
    target_ok = bool(set(spec.target_tags).intersection(region_tags))
    denom = max(len(task_tags), 1)
    score = 0.75 * (len(overlap) / denom) + 0.125 * float(sensory_ok) + 0.125 * float(target_ok)
    if pool_counts is not None:
        score += 0.05 * float(int(pool_counts.get("sensory", 0)) > 0)
        score += 0.05 * float(int(pool_counts.get("output", 0)) > 0)
    score = float(min(score, 1.0))
    reasons = [
        f"semantic overlap: {', '.join(overlap) if overlap else 'none'}",
        f"sensory tags {'match' if sensory_ok else 'do not match'} region ports",
        f"target tags {'match' if target_ok else 'do not match'} region ports",
    ]
    if pool_counts is not None:
        reasons.append(
            "pool counts: "
            + ", ".join(f"{key}={int(value)}" for key, value in sorted(pool_counts.items()))
        )
    return AdapterMapping(
        task_name=spec.task_name,
        connectome=ports.connectome,
        region_name=ports.region_name,
        input_dim=spec.input_dim,
        output_dim=spec.output_dim,
        sensory_pool="sensory",
        output_pool="output",
        input_channel_to_port={
            channel.name: _best_port_for_channel(channel, ports.sensory_ports)
            for channel in spec.input_channels
        },
        output_channel_to_port={
            channel.name: _best_port_for_channel(channel, ports.output_ports)
            for channel in spec.output_channels
        },
        mapping_score=score,
        reasons=tuple(reasons),
    )


def expected_dims_for_task(task: TaskSpec | str) -> tuple[int, int]:
    if isinstance(task, TaskSpec):
        return input_dim_for_task(task), output_dim_for_task(task)
    spec = task_channel_spec(task)
    if spec.task_name == TASK_CX_LANDMARK_BUMP:
        return CX_LANDMARK_INPUT_DIM, spec.output_dim
    return spec.input_dim, spec.output_dim
