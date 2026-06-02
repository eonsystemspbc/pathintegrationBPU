from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import OutputPaths


MANIFEST_FILES = (
    "neurons.csv",
    "roi_counts.csv",
    "connections.csv",
    "pool_assignments.csv",
    "graph_metadata.json",
    "adjacency_unsigned.npz",
    "adjacency_signed.npz",
    "metrics_by_seed.csv",
    "metrics_summary.csv",
    "loss_history.csv",
    "data_validation.md",
    "bpu_validation.md",
    "control_validation.md",
    "summary.md",
)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return None
    value = completed.stdout.strip()
    return value or None


def git_provenance() -> dict[str, object]:
    return {
        "commit": _git_value(["rev-parse", "HEAD"]),
        "branch": _git_value(["branch", "--show-current"]),
        "remote_origin": _git_value(["remote", "get-url", "origin"]),
        "status_short": _git_value(["status", "--short"]),
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def output_file_manifest(paths: OutputPaths) -> dict[str, dict[str, object]]:
    manifest: dict[str, dict[str, object]] = {}
    for name in MANIFEST_FILES:
        path = paths.output_dir / name
        if not path.exists():
            continue
        manifest[name] = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
    sequence_files = sorted(paths.sequence_dir.rglob("*.npz")) if paths.sequence_dir.exists() else []
    if sequence_files:
        manifest["sequence_files"] = {
            "path": str(paths.sequence_dir),
            "count": len(sequence_files),
            "sha256": hashlib.sha256(
                "\n".join(file_sha256(path) for path in sequence_files).encode("utf-8")
            ).hexdigest(),
        }
    return manifest


def directory_file_manifest(
    output_dir: Path,
    include_globs: tuple[str, ...] = ("*.csv", "*.json", "*.md", "*.npz", "*.png", "*.log"),
) -> dict[str, dict[str, object]]:
    output_dir = Path(output_dir)
    files: dict[str, dict[str, object]] = {}
    seen: set[Path] = set()
    for pattern in include_globs:
        for path in sorted(output_dir.rglob(pattern)):
            if path.name == "run_manifest.json" or path in seen or not path.is_file():
                continue
            seen.add(path)
            key = str(path.relative_to(output_dir))
            files[key] = {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
    return files


def write_run_manifest(
    paths: OutputPaths,
    config: object | Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    manifest = {
        "git": git_provenance(),
        "output_dir": str(paths.output_dir),
        "cache_dir": str(paths.cache_dir),
        "config": _jsonable(config) if config is not None else None,
        "files": output_file_manifest(paths),
        "extra": _jsonable(dict(extra or {})),
    }
    out = paths.output_dir / "run_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def write_artifact_manifest(
    output_dir: Path,
    config: object | Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "git": git_provenance(),
        "output_dir": str(output_dir),
        "config": _jsonable(config) if config is not None else None,
        "files": directory_file_manifest(output_dir),
        "extra": _jsonable(dict(extra or {})),
    }
    out = output_dir / "run_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
