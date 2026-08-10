"""Experiment and run manifest helpers.

``experiment.lock`` identifies input bytes only. ``run_manifest.json`` identifies generated run
artifacts. Keeping those separate avoids circular provenance and keeps cache keys stable.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from diffBloch import __version__
from diffBloch.config.schema import ExperimentConfig, load_config


class InputLock(BaseModel):
    """Hash and size for one input reference."""

    ref: str
    sha256: str
    bytes: int


class ExperimentLock(BaseModel):
    """``experiment.lock``: exact input identity, never generated outputs.

    ``experimental_data`` is a single :class:`InputLock` for the ordinary single-dataset experiment,
    or a list of them (one per file, in ``inputs.exp_data`` order) when ``inputs.multi_dataset`` is
    true.
    """

    structure: InputLock
    experimental_data: InputLock | list[InputLock]


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


class RecipeStep(BaseModel):
    """One step's identity in a preprocess recipe: its name + serialized params (or ``None``).

    Mirrors :class:`~diffBloch.preprocess.pipeline.StepRecord` as plain, comparable data -- the lock
    stores the recipe as a readable list of these, decoupled from the preprocess step vocabulary.
    """

    name: str
    params: dict[str, Any] | None = None


class PreprocessLock(BaseModel):
    """``plan.lock``: binds a serialized ``Plan`` checkpoint to everything that determined it.

    A checkpoint is safe to reuse only when the current run matches on all four axes -- the input
    bytes, the resolved config, the software version, and the composed recipe -- AND the ``.npz``
    verifies against ``plan``. The recipe axis distinguishes checkpoints built from the same inputs
    and config by different step sequences;
    ``code_version`` is the software-implementation axis the recipe (step shape + params) cannot
    capture. The full ``code_version`` string (``__version__+g<sha>[.dirty]``) is recorded here as a
    build stamp, but the reuse gate compares only its release ``__version__`` (see
    :func:`preprocess_lock_status`), so the checkpoint survives commits within a release. Identity
    only: hashes + a readable recipe, never payload.
    """

    experiment_lock_sha256: str
    config_digest: str
    code_version: str
    recipe: list[RecipeStep]
    plan: ArtifactHash


class RefinementLock(BaseModel):
    """``refinement.lock``: binds refined-structure outputs to everything that produced them.

    The refinement-stage counterpart to :class:`PreprocessLock`. Refinement runs on top of an
    already-settled ``Plan``, so everything that determines *that* -- inputs, sample, blochwave,
    preprocess config, recipe -- is already pinned by ``plan_lock_sha256`` (the hash of the exact
    ``plan.lock`` this run refined from, which itself chains back to ``experiment.lock``). What this
    lock adds is what refinement itself contributes on top: the refinement-determining config
    (:func:`refinement_config_digest`) and the code version that ran it, plus hashes of the refined
    outputs. Verifiable independently of whether ``plan.lock`` is still present or matches --
    ``plan_lock_sha256`` is a recorded fact about that run, not a live re-check.
    """

    plan_lock_sha256: str
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


def load_experiment(
    directory: str | Path, *, regenerate_lock: bool = False
) -> tuple[ExperimentConfig, ExperimentLock]:
    """Load ``experiment.yaml`` and verify ``experiment.lock`` (in ``reproducibility/``) against
    input bytes.

    ``regenerate_lock`` (default ``False``) rewrites ``experiment.lock`` from the *current*
    ``inputs.structure``/``inputs.exp_data`` bytes instead of verifying against whatever is already
    there -- for a caller that just edited an input on disk (e.g. patched a structure CIF) and wants
    the lock to catch up rather than raise ``"input drift detected"``. Wired to the app's
    ``--refresh`` flag, which already means "recompute from these inputs, ignore what was cached
    before" for the preprocess checkpoint -- this extends that same meaning to the input lock.
    """
    root = Path(directory)
    cfg = load_config(root / "experiment.yaml")
    lock_path = root / "reproducibility" / "experiment.lock"
    if regenerate_lock:
        lock = ExperimentLock(
            structure=input_lock_for(root / cfg.inputs.structure, ref=cfg.inputs.structure),
            experimental_data=_experimental_data_lock(root, cfg.inputs.exp_data),
        )
        lock_path.write_text(lock.model_dump_json(indent=2) + "\n")
        return cfg, lock
    lock = ExperimentLock.model_validate(yaml.safe_load(lock_path.read_text()))
    _verify_input(root, cfg.inputs.structure, lock.structure)
    _verify_experimental_data(root, cfg.inputs.exp_data, lock.experimental_data)
    return cfg, lock


def _experimental_data_lock(root: Path, ref: str | list[str]) -> InputLock | list[InputLock]:
    """Build the ``experimental_data`` lock value for either the single- or combined-dataset shape."""
    if isinstance(ref, list):
        return [input_lock_for(root / path, ref=path) for path in ref]
    return input_lock_for(root / ref, ref=ref)


def write_run_manifest(path: str | Path, manifest: RunManifest) -> None:
    """Write ``run_manifest.json`` in a stable, human-readable form."""
    Path(path).write_text(manifest.model_dump_json(indent=2) + "\n")


def config_digest(config: ExperimentConfig) -> str:
    """SHA256 of the *preprocess-determining* config -- the checkpoint's config identity.

    Keyed on the resolved :class:`ExperimentConfig` (not the ``experiment.yaml`` bytes): stable
    under comment/whitespace/field-order edits, sensitive to any validated-value change in scope.
    ``sort_keys`` makes it order-independent.

    Scope is an **explicit projection** onto exactly what determines the settled preprocess
    ``Plan`` -- so a committed preprocess checkpoint is restaled only by a change that could alter
    it:

    - ``inputs``, ``sample``, ``blochwave`` -- shape the grid and beams;
    - ``preprocess`` -- shapes and configures the fitting steps, but ``orientation``/``thickness``
      only when the matching ``optimize_orientation``/``optimize_thickness`` flag enables that step
      (the step's own params already ride in the recipe axis whenever it actually runs -- see
      :func:`~diffBloch.preprocess.pipeline.as_step` -- so including them here unconditionally would
      restale a checkpoint over config that provably never touched it), and always excluding
      ``thickness.plot`` (reporting-only, never touches the Plan even when ``thickness`` is in
      scope);
    - ``loss_metrics`` -- the residual ``optimize_orientation``/``optimize_thickness`` search
      minimises (:meth:`~diffBloch.config.schema.LossMetricsConfig.to_scores`), so it determines
      the settled Plan exactly like ``refinement.split`` below;
    - ``refinement.split`` -- orders :attr:`PlanSplit.combined`, the checkpointed plan.

    Everything else is excluded because it cannot change the Plan: ``name`` (a label),
    ``solver.inference`` (terminal scoring only), and the rest of ``refinement``
    (``optimizer`` / ``steps`` / ``trainable`` -- refinement-stage execution). This
    is the config axis of the preprocess lock only, not a whole-config identity.
    """
    dump = config.model_dump(mode="json")
    blochwave = {
        **dump["blochwave"],
        "solver": {"refine": dump["blochwave"]["solver"]["refine"]},
    }
    preprocess = dict(dump["preprocess"])
    if not preprocess["optimize_orientation"]:
        preprocess.pop("orientation", None)
    if not preprocess["optimize_thickness"]:
        preprocess.pop("thickness", None)
    elif "thickness" in preprocess:
        # thickness.plot only selects whether a PNG gets written; it never touches the Plan.
        preprocess["thickness"] = {k: v for k, v in preprocess["thickness"].items() if k != "plot"}
    preprocess_identity = {
        "inputs": dump["inputs"],
        "sample": dump["sample"],
        "blochwave": blochwave,
        "preprocess": preprocess,
        "loss_metrics": dump["loss_metrics"],
        "refinement": {"split": dump["refinement"]["split"]},
    }
    canonical = json.dumps(preprocess_identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def refinement_config_digest(config: ExperimentConfig) -> str:
    """SHA256 of the *refinement-determining* config -- the refinement lock's config identity.

    The complement of :func:`config_digest`: everything that function excludes from the preprocess
    checkpoint's identity (``optimizer`` / ``steps`` / ``trainable`` / ``thickness_nn``, all under
    ``refinement``) is exactly what determines the gradient-refined result on top of an
    already-settled ``Plan``, so this hashes precisely that complement. ``refinement.split`` is
    omitted here too -- it shapes the ``Plan`` itself and is already covered by ``config_digest``
    (and therefore by the ``plan.lock`` a refinement run is built from). ``loss_metrics`` is a
    top-level ``ExperimentConfig`` field (not under ``refinement``, so this dump never sees it) for
    exactly the same reason: it now determines the preprocess search too, so it belongs solely to
    ``config_digest``.
    """
    dump = config.model_dump(mode="json")
    refinement = {k: v for k, v in dump["refinement"].items() if k != "split"}
    canonical = json.dumps(refinement, sort_keys=True, separators=(",", ":"))
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
    experiment_lock_sha256: str,
    config_digest: str,
    code_version: str,
    recipe: list[RecipeStep],
    plan_path: str | Path,
    root: str | Path,
) -> PreprocessLockStatus:
    """How the checkpoint ``lock`` relates to the current run's ``recipe`` -- the resume verdict.

    ``"stale"`` unless the non-recipe axes all match (inputs, config, and the *release* portion of
    the software version -- :func:`_release`, so a differing git SHA within the same release still
    matches) AND the ``.npz`` verifies against the lock's :class:`ArtifactHash` (a tampered/missing
    checkpoint is stale). Given those hold:

    - ``"reuse"`` when the recipe is identical -- the snapshot is exactly this run's output.
    - ``"resume"`` when the lock's recipe is a *proper prefix* of ``recipe`` -- the run appends
      steps, so resume from the snapshot and run only the suffix (append-only / tail resume).
    - ``"stale"`` otherwise (a middle step differs, or the lock's recipe is longer).

    The caller must refuse recipes containing an opaque step *before* reaching here (those can never
    be safely reused); this function assumes a clean, comparable recipe.
    """
    if (
        lock.experiment_lock_sha256 != experiment_lock_sha256
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


def pack_run(
    run_directory: str | Path,
    *,
    package_format: Literal["zip", "tar", "bagit", "ro-crate"] = "zip",
) -> Path:
    """Export a canonical run directory for transfer/archive/publication.

    The run directory is the working format. Archives are export artifacts.
    """
    run_dir = Path(run_directory)
    if not (run_dir / "run_manifest.json").is_file():
        raise FileNotFoundError(f"{run_dir}/run_manifest.json")
    if package_format == "zip":
        return _zip_tree(run_dir, _export_path(run_dir, ".zip"))
    if package_format == "tar":
        return _tar_tree(run_dir, _export_path(run_dir, ".tar"))
    if package_format == "bagit":
        return _pack_bagit(run_dir)
    if package_format == "ro-crate":
        return _pack_ro_crate(run_dir)
    raise ValueError(f"unsupported run package format: {package_format}")


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
    """Dispatch :func:`_verify_input` over either the single-dataset or combined-dataset shape."""
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
        for one_ref, one_lock in zip(ref, lock, strict=True):
            _verify_input(root, one_ref, one_lock)
        return
    _verify_input(root, ref, lock)


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
