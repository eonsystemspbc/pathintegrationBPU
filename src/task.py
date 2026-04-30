from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DT, TaskSpec


@dataclass(frozen=True)
class SequenceSplit:
    name: str
    T: int
    noise_std: float
    path: Path


def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def _run_turn_controls(T: int, rng: np.random.Generator) -> np.ndarray:
    controls = np.zeros((T, 2), dtype=np.float32)
    t = 0
    mode = "run"
    while t < T:
        if mode == "run":
            duration = int(rng.integers(6, 18))
            v = float(rng.uniform(0.55, 1.15))
            omega_base = float(rng.normal(0.0, 0.025))
            for _ in range(duration):
                if t >= T:
                    break
                controls[t, 0] = max(0.0, v + rng.normal(0.0, 0.04))
                controls[t, 1] = omega_base + rng.normal(0.0, 0.02)
                t += 1
            mode = "turn"
        else:
            duration = int(rng.integers(2, 7))
            sign = float(rng.choice([-1.0, 1.0]))
            omega = sign * float(rng.uniform(0.18, 0.62))
            v = float(rng.uniform(0.05, 0.35))
            for _ in range(duration):
                if t >= T:
                    break
                controls[t, 0] = max(0.0, v + rng.normal(0.0, 0.03))
                controls[t, 1] = omega + rng.normal(0.0, 0.04)
                t += 1
            mode = "run"
    return controls


def integrate_trajectory(controls: np.ndarray) -> np.ndarray:
    T = controls.shape[0]
    targets = np.zeros((T, 4), dtype=np.float32)
    theta = 0.0
    x = 0.0
    y = 0.0
    for t in range(T):
        v = float(controls[t, 0])
        omega = float(controls[t, 1])
        theta = float(wrap_angle(theta + omega * DT))
        x += v * np.cos(theta) * DT
        y += v * np.sin(theta) * DT
        targets[t] = (np.cos(theta), np.sin(theta), x, y)
    return targets


def generate_sequences(
    count: int,
    T: int,
    rng: np.random.Generator,
    split_name: str,
    noise_std: float = 0.0,
) -> dict[str, np.ndarray]:
    clean_inputs = np.zeros((count, T, 2), dtype=np.float32)
    inputs = np.zeros((count, T, 2), dtype=np.float32)
    targets = np.zeros((count, T, 4), dtype=np.float32)
    ids = np.empty((count,), dtype=f"<U{max(16, len(split_name) + 12)}")
    for i in range(count):
        controls = _run_turn_controls(T, rng)
        clean_inputs[i] = controls
        if noise_std > 0:
            noisy = controls + rng.normal(0.0, noise_std, size=controls.shape).astype(np.float32)
            noisy[:, 0] = np.maximum(noisy[:, 0], 0.0)
            inputs[i] = noisy.astype(np.float32)
        else:
            inputs[i] = controls
        targets[i] = integrate_trajectory(controls)
        ids[i] = f"{split_name}-{i:06d}"
    return {
        "inputs": inputs,
        "clean_inputs": clean_inputs,
        "targets": targets,
        "ids": ids,
        "T": np.array(T, dtype=np.int32),
        "noise_std": np.array(noise_std, dtype=np.float32),
        "cache_version": np.array(0, dtype=np.int32),
    }


def with_input_noise(
    base_data: dict[str, np.ndarray],
    rng: np.random.Generator,
    split_name: str,
    noise_std: float,
    cache_version: int,
) -> dict[str, np.ndarray]:
    clean_inputs = base_data["clean_inputs"].astype(np.float32)
    if noise_std > 0:
        inputs = clean_inputs + rng.normal(0.0, noise_std, size=clean_inputs.shape).astype(np.float32)
        inputs[:, :, 0] = np.maximum(inputs[:, :, 0], 0.0)
    else:
        inputs = clean_inputs.copy()
    count, T, _ = clean_inputs.shape
    ids = np.array([f"{split_name}-{i:06d}" for i in range(count)])
    return {
        "inputs": inputs.astype(np.float32),
        "clean_inputs": clean_inputs,
        "targets": base_data["targets"].astype(np.float32),
        "ids": ids,
        "T": np.array(T, dtype=np.int32),
        "noise_std": np.array(noise_std, dtype=np.float32),
        "cache_version": np.array(cache_version, dtype=np.int32),
    }


def split_path(sequence_dir: Path, name: str, T: int, noise_std: float = 0.0) -> Path:
    if name == "test_noise":
        return sequence_dir / f"{name}_T{T}_noise{noise_std:.2f}.npz"
    return sequence_dir / f"{name}_T{T}.npz"


def ensure_splits(sequence_dir: Path, spec: TaskSpec) -> list[SequenceSplit]:
    sequence_dir.mkdir(parents=True, exist_ok=True)
    desired: list[tuple[str, int, int, float]] = [
        ("train", spec.train_T, spec.train_count, 0.0),
        ("val", spec.train_T, spec.val_count, 0.0),
    ]
    desired.extend(("test", T, spec.test_count, 0.0) for T in spec.test_T)
    desired.extend(
        ("test_noise", 200, spec.test_count, float(noise)) for noise in spec.noise_stds
    )
    splits: list[SequenceSplit] = []
    noise_base: dict[str, np.ndarray] | None = None
    for split_index, (name, T, count, noise_std) in enumerate(desired):
        path = split_path(sequence_dir, name, T, noise_std)
        regenerate = not path.exists()
        if path.exists():
            cached = load_split(path)
            cached_inputs = cached["inputs"]
            cached_noise = float(cached.get("noise_std", np.array(-1.0)))
            cached_version = int(cached.get("cache_version", np.array(-1)))
            regenerate = (
                cached_inputs.shape[:2] != (count, T)
                or not np.isclose(cached_noise, noise_std)
                or cached_version != spec.cache_version
            )
        if regenerate:
            seed_seq = np.random.SeedSequence(
                [spec.data_seed, split_index, T, int(round(noise_std * 1000))]
            )
            rng = np.random.default_rng(seed_seq)
            if name == "test_noise":
                if noise_base is None:
                    base_seed = np.random.SeedSequence([spec.data_seed, 90_200, T])
                    base_rng = np.random.default_rng(base_seed)
                    noise_base = generate_sequences(
                        count,
                        T,
                        base_rng,
                        f"{name}_T{T}_base",
                        noise_std=0.0,
                    )
                data = with_input_noise(
                    noise_base,
                    rng,
                    f"{name}_T{T}_n{noise_std:.2f}",
                    noise_std,
                    spec.cache_version,
                )
            else:
                data = generate_sequences(count, T, rng, f"{name}_T{T}_n{noise_std:.2f}", noise_std)
                data["cache_version"] = np.array(spec.cache_version, dtype=np.int32)
            np.savez_compressed(path, **data)
        splits.append(SequenceSplit(name=name, T=T, noise_std=noise_std, path=path))
    validate_split_ids([split.path for split in splits])
    return splits


def load_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def validate_split_ids(paths: list[Path]) -> None:
    seen: set[str] = set()
    overlaps: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        ids = load_split(path)["ids"].astype(str)
        for seq_id in ids:
            if seq_id in seen:
                overlaps.append(seq_id)
            seen.add(seq_id)
    if overlaps:
        raise ValueError(f"Train/val/test sequence IDs overlap: {overlaps[:5]}")
