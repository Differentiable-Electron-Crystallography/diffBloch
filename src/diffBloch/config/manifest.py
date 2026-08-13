"""Experiment lock and checkpoint-lock helpers.

``experiment.lock`` identifies input bytes only; the preprocess and refinement locks identify
generated artifacts and execution identity. Keeping those separate avoids circular provenance and
keeps cache keys stable.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import subprocess
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from diffBloch import __version__
from diffBloch.config.schema import ExperimentConfig, load_config

type CellParameters = tuple[float, float, float, float, float, float]


class InputLock(BaseModel):
    """Hash and size for one input reference."""

    ref: str
    sha256: str
    bytes: int


class ExperimentLock(BaseModel):
    """``experiment.lock``: exact input identity, never generated outputs.

    ``experimental_data`` is one lock for a single dataset or a list in ``inputs.exp_data`` order
    for a pooled (``inputs.multi_dataset``) experiment.
    """

    structure: InputLock
    experimental_data: InputLock | list[InputLock]


class ArtifactHash(BaseModel):
    """Hash and media metadata for a generated run artifact."""

    path: str
    sha256: str
    bytes: int
    media_type: str


class RecipeStep(BaseModel):
    """One step's identity in a preprocess recipe: its name + serialized params (or ``None``).

    Mirrors :class:`~diffBloch.preprocess.pipeline.StepRecord` as plain, comparable data -- the lock
    stores the recipe as a readable list of these, decoupled from the preprocess step vocabulary.
    """

    name: str
    params: dict[str, Any] | None = None


class PreprocessLock(BaseModel):
    """``plan.<stem>.lock``: binds one dataset's ``Plan`` checkpoint to everything that determined it.

    A checkpoint is safe to reuse only when the current run matches on every axis -- the structure
    and *this dataset's* input bytes, the authoritative PETS cell shared by the experiment, the
    dataset-scoped config projection (:func:`dataset_config_digest`), this dataset's file-local
    ignored rotations, the software version, and the composed recipe -- AND the ``.npz`` verifies
    against ``plan``. The identity is deliberately **per dataset**: nothing here hashes the whole
    ``experiment.lock`` or the full ``inputs.exp_data`` list, so adding, removing, or reordering
    *other* datasets in a pooled experiment never restales this one's checkpoint unless the first
    dataset's authoritative cell changes. The recipe axis distinguishes checkpoints built from the
    same inputs and config by different step sequences; ``code_version`` is the
    software-implementation axis the recipe (step shape + params) cannot capture. The full
    ``code_version`` string (``__version__+g<sha>[.dirty]``) is recorded here as a build stamp, but
    the reuse gate compares only its release ``__version__`` (see :func:`preprocess_lock_status`),
    so the checkpoint survives commits within a release. Identity only: hashes + a readable recipe,
    never payload.
    """

    structure: InputLock
    experimental_data: InputLock  # this dataset's file only, ref = its inputs.exp_data entry
    # The experiment-wide PETS cell every dataset's shared grid/refinement metric was built from,
    # recorded on every per-dataset lock because dataset 0's cell shapes all pooled checkpoints.
    authoritative_cell: CellParameters
    # File-local (position within this dataset's own PETS file), sorted. Lives here rather than in
    # the config digest: the config's ignore_orientations indexes the pooled rotation space, so a
    # given file's checkpoint identity is only the slice that lands on it.
    ignored_rotations: tuple[int, ...]
    config_digest: str
    code_version: str
    recipe: list[RecipeStep]
    plan: ArtifactHash


class RefinementLock(BaseModel):
    """``refinement.lock``: binds refined-structure outputs to everything that produced them.

    The refinement-stage counterpart to :class:`PreprocessLock`. Refinement runs on top of
    already-settled per-dataset ``Plan``\\ s, so everything that determines *those* -- inputs,
    sample, blochwave, preprocess config, recipe -- is already pinned by ``plan_lock_sha256s`` (the
    hashes of the exact ``plan.<stem>.lock`` files this run refined from, in ``inputs.exp_data``
    order). What this lock adds is what refinement itself contributes on top: the
    refinement-determining config (:func:`refinement_config_digest`, which includes the train/val
    ``split`` -- the split partitions rotations at refinement time and no longer shapes the
    checkpointed plans) and the code version that ran it, plus hashes of the refined outputs.
    Verifiable independently of whether the plan locks are still present or match --
    ``plan_lock_sha256s`` is a recorded fact about that run, not a live re-check.
    """

    plan_lock_sha256s: list[str]
    refinement_config_digest: str
    code_version: str
    refined_structure: ArtifactHash
    refined_parameters: ArtifactHash


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
    """Load ``experiment.yaml`` and verify ``experiment.lock`` (in ``reproducibility/``) against
    input bytes."""
    root = Path(directory)
    cfg = load_config(root / "experiment.yaml")
    lock_path = root / "reproducibility" / "experiment.lock"
    lock = ExperimentLock.model_validate(yaml.safe_load(lock_path.read_text()))
    _verify_input(root, cfg.inputs.structure, lock.structure)
    _verify_experimental_data(root, cfg.inputs.exp_data, lock.experimental_data)
    return cfg, lock


def dataset_config_digest(config: ExperimentConfig, *, exp_data: str) -> str:
    """SHA256 of the config that determines *one dataset's* settled ``Plan`` -- its lock identity.

    Keyed on the resolved :class:`ExperimentConfig` (not the ``experiment.yaml`` bytes): stable
    under comment/whitespace/field-order edits, sensitive to any validated-value change in scope.
    ``sort_keys`` makes it order-independent.

    Scope is an **explicit projection** onto exactly what determines the settled per-dataset
    ``Plan`` -- so a committed checkpoint is restaled only by a change that could alter it:

    - ``inputs`` -- rewritten to this dataset's view: ``exp_data`` is the single ``exp_data`` ref
      the checkpoint belongs to (never the full list -- other datasets joining or leaving the pool
      cannot alter this one's plan), and ``multi_dataset`` is dropped for the same reason (a
      dataset's settled plan is independent of whether it is pooled);
    - ``sample``, ``blochwave`` -- shape the grid and beams; ``blochwave.ignore_orientations``
      is dropped here because it indexes the *pooled* rotation space -- the translated file-local
      slice lives explicitly in :attr:`PreprocessLock.ignored_rotations` instead, so an ignore edit
      restales exactly the datasets it lands on;
    - ``preprocess`` -- shapes and configures the fitting steps, but ``orientation``/``thickness``
      only when the matching ``optimize_orientation``/``optimize_thickness`` flag enables that step
      (the step's own params already ride in the recipe axis whenever it actually runs -- see
      :func:`~diffBloch.preprocess.pipeline.as_step` -- so including them here unconditionally would
      restale a checkpoint over config that provably never touched it), and always excluding
      ``thickness.plot`` (reporting-only, never touches the Plan even when ``thickness`` is in
      scope);
    - ``loss_metrics`` -- the residual ``optimize_orientation``/``optimize_thickness`` search
      minimises (:meth:`~diffBloch.config.schema.LossMetricsConfig.to_scores`).

    Everything else is excluded because it cannot change the Plan: ``name`` (a label), and all of
    ``refinement`` -- including ``split``, which partitions rotations at refinement time and no
    longer shapes the checkpointed plan (it rides in :func:`refinement_config_digest`). This is the
    config axis of the per-dataset preprocess lock only, not a whole-config identity.
    """
    dump = config.model_dump(mode="json")
    blochwave = {k: v for k, v in dump["blochwave"].items() if k != "ignore_orientations"}
    blochwave["solver"] = {"preprocess": dump["blochwave"]["solver"]["preprocess"]}
    preprocess = dict(dump["preprocess"])
    if not preprocess["optimize_orientation"]:
        preprocess.pop("orientation", None)
    if not preprocess["optimize_thickness"]:
        preprocess.pop("thickness", None)
    elif "thickness" in preprocess:
        # thickness.plot only selects whether a PNG gets written; it never touches the Plan.
        preprocess["thickness"] = {k: v for k, v in preprocess["thickness"].items() if k != "plot"}
    dataset_identity = {
        "inputs": {
            "structure": dump["inputs"]["structure"],
            "exp_data": exp_data,
            "load_hydrogens": dump["inputs"]["load_hydrogens"],
        },
        # Resolve the optional per-dataset mapping to this Plan's effective seed. This keeps a
        # change for dataset B from invalidating dataset A's checkpoint.
        "sample": {"thicknesses": list(config.sample.seed_thicknesses_for(exp_data))},
        "blochwave": blochwave,
        "preprocess": preprocess,
        "loss_metrics": dump["loss_metrics"],
    }
    canonical = json.dumps(dataset_identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def refinement_config_digest(config: ExperimentConfig) -> str:
    """SHA256 of the *refinement-determining* config -- the refinement lock's config identity.

    The complement of :func:`dataset_config_digest`: everything that function excludes from the
    per-dataset checkpoint's identity under ``refinement`` (``optimizer`` / ``steps`` /
    ``trainable`` / ``thickness_nn`` / ``split``) is exactly what determines the gradient-refined
    result on top of already-settled ``Plan``\\ s, so this hashes the whole ``refinement`` section.
    ``split`` belongs here (not in the preprocess digest) because the train/validation partition is
    applied when the pooled plan is handed to refinement -- it never shapes a checkpointed
    per-dataset plan. ``loss_metrics`` is a top-level ``ExperimentConfig`` field (not under
    ``refinement``, so this dump never sees it): it determines the preprocess search, so it belongs
    solely to :func:`dataset_config_digest`.
    """
    dump = config.model_dump(mode="json")
    identity = {
        "refinement": dump["refinement"],
        "solver": dump["blochwave"]["solver"]["refinement"],
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def code_version() -> str:
    """The software-version identity of the compute (checkpoint validity + run-manifest stamp).

    Returns :data:`diffBloch.__version__`, best-effort suffixed with the git short-SHA and a
    ``.dirty`` marker when running inside a checkout. Falls back to the bare version in an installed
    wheel (no git / no repo). This full string is the *stamp* recorded in the run manifest and the
    checkpoint lock (it says exactly which build produced an artifact). The checkpoint *reuse gate*,
    however, keys only on the release ``__version__`` (see :func:`_release`), so a committed
    checkpoint stays reusable across commits within a release -- the SHA/``.dirty`` detail is
    recorded but does not invalidate. The trade-off is a weaker guard: a physics change without a
    version bump reuses; release discipline plus ``--refresh`` (regenerate) is the escape hatch.
    """
    root = str(Path(__file__).parent)
    try:
        sha = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if sha.returncode != 0:
            return __version__
        status = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        suffix = sha.stdout.strip() + (".dirty" if status.stdout.strip() else "")
        return f"{__version__}+g{suffix}"
    except (OSError, subprocess.SubprocessError):
        return __version__


def write_preprocess_lock(path: str | Path, lock: PreprocessLock) -> None:
    """Write ``plan.lock`` in a stable, human-readable form (beside ``experiment.lock``)."""
    Path(path).write_text(lock.model_dump_json(indent=2) + "\n")


def read_preprocess_lock(path: str | Path) -> PreprocessLock:
    """Read a ``plan.lock`` written by :func:`write_preprocess_lock`."""
    return PreprocessLock.model_validate_json(Path(path).read_text())


def write_refinement_lock(path: str | Path, lock: RefinementLock) -> None:
    """Write ``refinement.lock`` in a stable, human-readable form (beside ``refined_structure.cif``)."""
    Path(path).write_text(lock.model_dump_json(indent=2) + "\n")


def read_refinement_lock(path: str | Path) -> RefinementLock:
    """Read a ``refinement.lock`` written by :func:`write_refinement_lock`."""
    return RefinementLock.model_validate_json(Path(path).read_text())


PreprocessLockStatus = Literal["reuse", "resume", "stale"]


def _release(code_version: str) -> str:
    """The release-version prefix of a :func:`code_version` string (drops the ``+g<sha>`` suffix).

    ``code_version()`` is ``__version__`` optionally suffixed with ``+g<sha>[.dirty]``; the reuse
    gate keys on the release ``__version__`` alone (``split("+g")[0]``), so a checkpoint stays
    reusable across commits within a release rather than being invalidated by every SHA change.
    """
    return code_version.split("+g", 1)[0]


def preprocess_lock_status(
    lock: PreprocessLock,
    *,
    structure: InputLock,
    experimental_data: InputLock,
    authoritative_cell: CellParameters,
    ignored_rotations: tuple[int, ...],
    config_digest: str,
    code_version: str,
    recipe: list[RecipeStep],
    plan_path: str | Path,
    root: str | Path,
) -> PreprocessLockStatus:
    """How the checkpoint ``lock`` relates to the current run's ``recipe`` -- the resume verdict.

    ``"stale"`` unless the non-recipe axes all match (the structure and this dataset's input bytes,
    the authoritative PETS cell, the file-local ignored rotations, the dataset config digest, and
    the *release* portion of the software version -- :func:`_release`, so a differing git SHA within
    the same release still matches) AND the ``.npz`` verifies against the lock's
    :class:`ArtifactHash` (a tampered/missing checkpoint is stale). Input identity compares
    ``sha256``/``bytes`` only, never ``ref``: renaming a dataset file without changing its bytes
    moves its checkpoint (new stem) but a lock whose recorded ref differs while the bytes match is
    still the same measurement. Given those hold:

    - ``"reuse"`` when the recipe is identical -- the snapshot is exactly this run's output.
    - ``"resume"`` when the lock's recipe is a *proper prefix* of ``recipe`` -- the run appends
      steps, so resume from the snapshot and run only the suffix (append-only / tail resume).
    - ``"stale"`` otherwise (a middle step differs, or the lock's recipe is longer).

    The caller must refuse recipes containing an opaque step *before* reaching here (those can never
    be safely reused); this function assumes a clean, comparable recipe.
    """
    if (
        (lock.structure.sha256, lock.structure.bytes) != (structure.sha256, structure.bytes)
        or (lock.experimental_data.sha256, lock.experimental_data.bytes)
        != (experimental_data.sha256, experimental_data.bytes)
        or lock.authoritative_cell != authoritative_cell
        or lock.ignored_rotations != ignored_rotations
        or lock.config_digest != config_digest
        or _release(lock.code_version) != _release(code_version)
    ):
        return "stale"
    artifact = Path(plan_path)
    if not artifact.exists():
        return "stale"
    current = artifact_hash_for(artifact, root=root)
    if current.sha256 != lock.plan.sha256 or current.bytes != lock.plan.bytes:
        return "stale"
    if lock.recipe == recipe:
        return "reuse"
    k = len(lock.recipe)
    if k < len(recipe) and lock.recipe == recipe[:k]:
        return "resume"
    return "stale"


def _verify_input(root: Path, ref: str, lock: InputLock) -> None:
    if ref != lock.ref:
        raise ValueError(f"lock ref mismatch for {ref!r}: {lock.ref!r}")
    path = root / ref
    actual = input_lock_for(path, ref=ref)
    if actual.sha256 != lock.sha256 or actual.bytes != lock.bytes:
        raise ValueError(f"input drift detected for {ref}")


def _verify_experimental_data(
    root: Path, ref: str | list[str], lock: InputLock | list[InputLock]
) -> None:
    """Verify either one experimental file or every file of a pooled experiment."""
    if isinstance(ref, list) or isinstance(lock, list):
        if not isinstance(ref, list) or not isinstance(lock, list):
            raise ValueError(
                "experiment.lock experimental_data shape does not match inputs.exp_data "
                "(one is a list, the other is not)"
            )
        if len(ref) != len(lock):
            raise ValueError(
                f"experiment.lock has {len(lock)} experimental_data entries, "
                f"inputs.exp_data has {len(ref)}"
            )
        # The length guard above already ensures equal counts (with a friendlier message);
        # strict=True is repo lint policy (B905), not a second check.
        for one_ref, one_lock in zip(ref, lock, strict=True):
            _verify_input(root, one_ref, one_lock)
        return
    _verify_input(root, ref, lock)
