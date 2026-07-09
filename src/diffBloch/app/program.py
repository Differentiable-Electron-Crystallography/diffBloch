"""The default experiment runner the ``run infer`` CLI exposes.

:func:`run_experiment` encodes the faithful default recipe as one ordered ``Plan -> Plan`` pipeline:
plan-shaping (``select_beams`` -> ``integrate_rocking_curve`` -> ``mosaicity``) followed by
parameter fitting (``fit_orientation`` -> ``fit_thickness``), then ``run_inference`` evaluates it --
so a caller with an experiment directory gets the faithful result in one call. It is a
*convenience*, not the only path: every step is ordinary public API, so a Python user who wants a
different composition composes their own ``pipeline([...])`` with ``from_experiment`` +
``run_inference`` directly. The CLI stays thin by delegating here and holds no science.

**Checkpoint / resume.** The preprocess (the coupled fit especially) is the run's expensive phase,
so its settled ``Plan`` is checkpointed to the experiment dir (``plan.npz`` + ``plan.lock``) and
reused when still valid. The lock binds the checkpoint to four axes -- input bytes, resolved config,
software version, and the *recipe* (the Plan's own provenance) -- so it is reused only when all
match. The match is a longest-prefix over the recipe: an identical recipe is a full **reuse**; a
recipe that *appends* steps **resumes** from the snapshot and runs only the suffix (the append-only
increment). Any run that computes -- fresh, stale-recompute, ``refresh``, or resume -- regenerates
``plan.npz`` + ``plan.lock``, so the lock always describes the on-disk checkpoint. Diagnostics about
load/resume go through stdlib ``logging`` (not the domain-observation ``logger``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from diffBloch.config import (
    ExperimentConfig,
    PreprocessLock,
    RecipeStep,
    artifact_hash_for,
    code_version,
    config_digest,
    load_experiment,
    preprocess_lock_status,
    read_preprocess_lock,
    sha256_file,
    write_preprocess_lock,
)
from diffBloch.io import read_observations, read_structure
from diffBloch.observability import NULL_LOGGER, Logger
from diffBloch.preprocess import (
    OPAQUE,
    Plan,
    PlanStep,
    build_orientation_plans,
    fit_orientation,
    fit_thickness,
    from_experiment,
    integrate_rocking_curve,
    mosaicity,
    pipeline,
    read_plan,
    run_inference,
    select_beams,
    step_records,
    write_plan,
)
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.inference import InferenceResult
from diffBloch.specs import ScoredSelection, TiltSegmentUnion, TrialCoupling

__all__ = ["run_experiment"]

_log = logging.getLogger(__name__)

_PLAN_NPZ = "plan.npz"
_PLAN_LOCK = "plan.lock"


def run_experiment(
    experiment_dir: str | Path,
    *,
    logger: Logger = NULL_LOGGER,
    checkpoint: bool = True,
    refresh: bool = False,
) -> InferenceResult:
    """Load, preprocess, and score every rotation of the experiment at ``experiment_dir``.

    Loads ``experiment.yaml`` (verifying the input lock), reads the structure + observations, builds
    the geometry via ``from_experiment``, runs the faithful integrated recipe, and evaluates the
    forward model over all rotations with ``run_inference`` -- emitting per-rotation observations to
    ``logger`` (the null default discards them). Returns the
    :class:`~diffBloch.preprocess.inference.InferenceResult` (per-rotation ``R_obs`` + aggregate).

    The recipe is the private's faithful pipeline, **per-trial beam coupling included** (the fit
    re-derives the SOLVE union + SCORED set at every trial orientation -- the private's exact
    objective). This is the opinionated default; a caller wanting a different composition (e.g. the
    cheaper tilt-independent fit) composes their own ``pipeline([...])`` with the public steps.

    ``checkpoint`` (default ``True``) reuses/resumes a valid ``plan.npz`` in the experiment dir and
    writes a fresh one after computing; ``refresh`` forces a full recompute (ignoring any snapshot)
    while still regenerating the checkpoint. ``checkpoint=False`` neither reads nor writes.
    """
    root = Path(experiment_dir)
    cfg, _lock = load_experiment(root)
    structure = read_structure(root / cfg.inputs.structure)
    observations = read_observations(root / cfg.inputs.observations)
    setup = from_experiment(structure, observations, cfg)
    refinement = setup.refinement
    steps = _recipe_steps(cfg, refinement, logger)
    prepared = _prepare(
        setup.plans.combined,
        steps,
        root=root,
        cfg=cfg,
        checkpoint=checkpoint,
        refresh=refresh,
    )
    return run_inference(prepared, refinement, method=cfg.solver.inference, logger=logger)


def _recipe_steps(
    cfg: ExperimentConfig, refinement: RefinementSetup, logger: Logger
) -> list[PlanStep]:
    """The faithful default recipe as an inspectable step list (its provenance keys the lock).

    The orientation fit runs under the private's per-trial coupling (:func:`_trial_coupling`) -- the
    faithful objective. The tilt-independent fit is not offered here; compose it directly if needed.
    """
    return [
        select_beams(cfg.numerics.to_beam_selection()),
        build_orientation_plans(),
        integrate_rocking_curve(cfg.numerics.to_rocking_curve()),
        mosaicity(cfg.numerics.mosaicity),
        fit_orientation(
            refinement,
            cfg.preprocess.orientation.to_search(),
            method=cfg.solver.refine,
            coupling=_trial_coupling(cfg),
            logger=logger,  # per-rotation fit progress (the run's long phase)
        ),
        fit_thickness(refinement, cfg.preprocess.thickness.to_grid(), method=cfg.solver.refine),
    ]


def _trial_coupling(cfg: ExperimentConfig) -> TrialCoupling:
    """Assemble the per-trial coupling from config, with the faithful private policy defaults.

    Coupling is the faithful objective; its policy carries no config block (mirroring
    ``converge_numerics``), so the SOLVE policy uses :class:`~diffBloch.specs.TiltSegmentUnion`'s
    faithful defaults. The SCORED set reuses the *same* Klar window as ``select_beams`` (so the two
    filters cannot disagree) and the config's ``g_max_refine`` as the scoring-resolution cap.
    """
    return TrialCoupling(
        policy=TiltSegmentUnion(),
        scored=ScoredSelection(
            klar=cfg.numerics.to_beam_selection(), g_max=cfg.numerics.g_max_refine
        ),
    )


def _prepare(
    base: Plan,
    steps: list[PlanStep],
    *,
    root: Path,
    cfg: ExperimentConfig,
    checkpoint: bool,
    refresh: bool,
) -> Plan:
    """Run the preprocess ``steps`` on ``base``, reusing/resuming a valid checkpoint if present."""
    records = step_records(steps)
    # A recipe with an unrecorded (opaque) step cannot be safely identified -> never checkpoint it.
    can_checkpoint = checkpoint and OPAQUE not in records
    recipe = [RecipeStep(name=r.name, params=r.params) for r in records]
    npz, lock_path = root / _PLAN_NPZ, root / _PLAN_LOCK

    if can_checkpoint and not refresh and lock_path.exists() and npz.exists():
        lock = read_preprocess_lock(lock_path)
        status = preprocess_lock_status(
            lock,
            experiment_lock_sha256=sha256_file(root / "experiment.lock"),
            config_digest=config_digest(cfg),
            code_version=code_version(),
            recipe=recipe,
            plan_path=npz,
            root=root,
        )
        if status == "reuse":
            _log.info("loaded preprocess checkpoint (full reuse) from %s", npz)
            return read_plan(npz)
        if status == "resume":
            snapshot = read_plan(npz)
            k = len(lock.recipe)
            _log.info(
                "resumed preprocess checkpoint after %d step(s); running %s",
                k,
                [r.name for r in records[k:]],
            )
            result = pipeline(steps[k:])(snapshot)
            _write_checkpoint(result, recipe, root=root, cfg=cfg, npz=npz, lock_path=lock_path)
            return result

    result = pipeline(steps)(base)
    if can_checkpoint:
        _write_checkpoint(result, recipe, root=root, cfg=cfg, npz=npz, lock_path=lock_path)
    return result


def _write_checkpoint(
    plan: Plan,
    recipe: list[RecipeStep],
    *,
    root: Path,
    cfg: ExperimentConfig,
    npz: Path,
    lock_path: Path,
) -> None:
    """Write ``plan.npz`` + regenerate ``plan.lock`` so the lock always describes the npz."""
    write_plan(plan, npz)
    lock = PreprocessLock(
        experiment_lock_sha256=sha256_file(root / "experiment.lock"),
        config_digest=config_digest(cfg),
        code_version=code_version(),
        recipe=recipe,
        plan=artifact_hash_for(npz, root=root),
    )
    write_preprocess_lock(lock_path, lock)
    _log.info("wrote preprocess checkpoint %s + %s", npz.name, lock_path.name)
