"""Experiment and run manifest helpers.

``experiment.lock`` identifies input bytes only. ``run_manifest.json`` identifies generated run
artifacts. Keeping those separate avoids circular provenance and keeps cache keys stable.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
import tarfile
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import BaseModel

from diffBloch.config.schema import ExperimentConfig, load_config


class InputLock(BaseModel):
    """Hash and size for one input reference."""

    ref: str
    sha256: str
    bytes: int


class ExperimentLock(BaseModel):
    """``experiment.lock``: exact input identity, never generated outputs."""

    structure: InputLock
    observations: InputLock
    orientations: InputLock | None = None


class ArtifactHash(BaseModel):
    """Hash and media metadata for a generated run artifact."""

    path: str
    sha256: str
    bytes: int
    media_type: str


class RunManifest(BaseModel):
    """``run_manifest.json``: generated artifact hashes and execution identity."""

    experiment_lock_sha256: str
    resolved_config: ArtifactHash
    config_diff: ArtifactHash
    data_used: ArtifactHash
    history: ArtifactHash
    parameter_table: ArtifactHash
    objective_terms: ArtifactHash | None = None
    snapshot: ArtifactHash | None = None
    refined_model: ArtifactHash | None = None
    code_version: str
    environment: dict[str, str]


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 hex digest for ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_lock_for(path: str | Path, *, ref: str) -> InputLock:
    """Build an ``InputLock`` for an experiment input file."""
    resolved = Path(path)
    return InputLock(ref=ref, sha256=sha256_file(resolved), bytes=resolved.stat().st_size)


def artifact_hash_for(path: str | Path, *, root: str | Path) -> ArtifactHash:
    """Build an ``ArtifactHash`` for a generated run artifact."""
    artifact = Path(path)
    root_path = Path(root)
    media_type = mimetypes.guess_type(artifact.name)[0] or "application/octet-stream"
    return ArtifactHash(
        path=artifact.relative_to(root_path).as_posix(),
        sha256=sha256_file(artifact),
        bytes=artifact.stat().st_size,
        media_type=media_type,
    )


def load_experiment(directory: str | Path) -> tuple[ExperimentConfig, ExperimentLock]:
    """Load ``experiment.yaml`` and verify ``experiment.lock`` against input bytes."""
    root = Path(directory)
    cfg = load_config(root / "experiment.yaml")
    lock_path = root / "experiment.lock"
    lock = ExperimentLock.model_validate(yaml.safe_load(lock_path.read_text()))
    _verify_input(root, cfg.inputs.structure, lock.structure)
    _verify_input(root, cfg.inputs.observations, lock.observations)
    if cfg.inputs.orientations is not None or lock.orientations is not None:
        if cfg.inputs.orientations is None or lock.orientations is None:
            raise ValueError("orientations must appear in both experiment.yaml and experiment.lock")
        _verify_input(root, cfg.inputs.orientations, lock.orientations)
    return cfg, lock


def write_run_manifest(path: str | Path, manifest: RunManifest) -> None:
    """Write ``run_manifest.json`` in a stable, human-readable form."""
    Path(path).write_text(manifest.model_dump_json(indent=2) + "\n")


def select_reference_rotations(
    rotations: list[dict[str, object]], selector: str
) -> list[dict[str, object]]:
    """Select reference rotations by ``all``, ``first:N``, or comma-separated ``rotation_idx``."""
    selector = selector.strip()
    if not selector:
        raise ValueError("rotation selector must not be empty")
    if selector == "all":
        return rotations
    if selector.startswith("first:"):
        count = int(selector.split(":", 1)[1])
        if count < 1:
            raise ValueError("first:N requires N >= 1")
        return rotations[:count]
    requested = {int(value.strip()) for value in selector.split(",") if value.strip()}
    if not requested:
        raise ValueError("rotation selector must not be empty")
    selected = [
        rotation for rotation in rotations if cast(int, rotation["rotation_idx"]) in requested
    ]
    missing = requested - {cast(int, rotation["rotation_idx"]) for rotation in selected}
    if missing:
        raise ValueError(f"unknown rotation_idx values: {sorted(missing)}")
    return selected


def pack_run(
    run_directory: str | Path,
    *,
    format: Literal["zip", "tar", "bagit", "ro-crate"] = "zip",
) -> Path:
    """Export a canonical run directory for transfer/archive/publication.

    The run directory is the working format. Archives are export artifacts.
    """
    run_dir = Path(run_directory)
    if not (run_dir / "run_manifest.json").is_file():
        raise FileNotFoundError(f"{run_dir}/run_manifest.json")
    if format == "zip":
        return _zip_tree(run_dir, _export_path(run_dir, ".zip"))
    if format == "tar":
        return _tar_tree(run_dir, _export_path(run_dir, ".tar"))
    if format == "bagit":
        return _pack_bagit(run_dir)
    if format == "ro-crate":
        return _pack_ro_crate(run_dir)
    raise ValueError(f"unsupported run package format: {format}")


def _verify_input(root: Path, ref: str, lock: InputLock) -> None:
    if ref != lock.ref:
        raise ValueError(f"lock ref mismatch for {ref!r}: {lock.ref!r}")
    path = root / ref
    actual = input_lock_for(path, ref=ref)
    if actual.sha256 != lock.sha256 or actual.bytes != lock.bytes:
        raise ValueError(f"input drift detected for {ref}")


def _export_path(run_dir: Path, suffix: str) -> Path:
    return run_dir.parent / f"{run_dir.name}{suffix}"


def _zip_tree(source: Path, output: Path) -> Path:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in _iter_files(source):
            archive.write(path, path.relative_to(source.parent))
    return output


def _tar_tree(source: Path, output: Path) -> Path:
    with tarfile.open(output, "w") as archive:
        archive.add(source, arcname=source.name)
    return output


def _pack_bagit(run_dir: Path) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        bag = Path(tmp) / f"{run_dir.name}.bag"
        data = bag / "data"
        shutil.copytree(run_dir, data)
        (bag / "bagit.txt").write_text("BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n")
        manifest = "\n".join(
            f"{sha256_file(path)}  {path.relative_to(bag).as_posix()}" for path in _iter_files(data)
        )
        (bag / "manifest-sha256.txt").write_text(manifest + "\n")
        return _zip_tree(bag, _export_path(run_dir, ".bagit.zip"))


def _pack_ro_crate(run_dir: Path) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        crate = Path(tmp) / f"{run_dir.name}.ro-crate"
        shutil.copytree(run_dir, crate)
        metadata = {
            "@context": "https://w3id.org/ro/crate/1.1/context",
            "@graph": [
                {"@id": "ro-crate-metadata.json", "@type": "CreativeWork", "about": {"@id": "./"}},
                {"@id": "./", "@type": "Dataset", "name": run_dir.name},
            ],
        }
        (crate / "ro-crate-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return _zip_tree(crate, _export_path(run_dir, ".ro-crate.zip"))


def _iter_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())
