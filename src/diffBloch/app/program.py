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
from diffBloch.core.solver import FloatFormat
from diffBloch.io import read_observations, read_structure
from diffBloch.observability import NULL_LOGGER, Logger
from diffBloch.params import Device
from diffBloch.preprocess import (
    OPAQUE,
    Plan,
    PlanStep,
    build_orientation_plans,
    fit_orientation,
    fit_thickness,
    fork,
    from_experiment,
    integrate_rocking_curve,
    mosaicity,
    pipeline,
    read_plan,
    resolve_recipe,
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

# Above this unit-cell volume the coupled orientation search runs its coarse pass in fp32 with the
# gather integrity checks skipped (the large-cell fast path); at or below it the search stays the
# exact fp64 path, byte-identical to the pre-fork recipe. The eigensolve is O(N^3) in the beam count
# and N grows with cell volume, so the fp32 ~1.75x throughput win only pays off for large cells; a
# small cell (quartz ~113 A^3) fits in seconds at fp64 and gains nothing from the coarser search.
# A deliberately wide heuristic gap separates the known regimes -- quartz ~113 vs LTA/zeolites
# ~1861 A^3 -- so classification is unambiguous; it is not a sharp physical boundary. Kept a code
# constant, NOT a config field: a config field would enter config_digest and restale the committed
# quartz checkpoint. It only selects which precision the *search* uses, and the fp64 terminal always
# re-scores the found orientation, so the threshold never affects the reported score's fidelity.
# It is NOT free of accuracy consequence, though: the coarse fp32 search can converge to a different
# -- possibly worse -- basin than fp64 would on the bumpy coupled landscape, and the large branch
# has no fp64 oracle in its own regime (LTA fp64 is ~hours CPU, CUDA-deferred), so that basin parity
# is unverified until the A100 fp32-vs-fp64 check. A search-robustness trade, not a scoring one.
_LARGE_CELL_THRESHOLD_A3 = 1000.0


def run_experiment(
    experiment_dir: str | Path,
    *,
    logger: Logger = NULL_LOGGER,
    checkpoint: bool = True,
    refresh: bool = False,
    device: Device | None = None,
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

    ``device`` (default ``None`` = CPU) runs the forward solve on the given accelerator (e.g.
    ``"cuda"``) for both the coupled preprocess fits and the terminal ``run_inference``. The
    preprocess *geometry* (beam selection, coupling unions) stays CPU-side numpy; only the
    eigensolve -- the O(N^3) cost -- moves to the device (the fits move the seed params, so ``fgb``
    and every per-trial score co-locate there). Device is execution-only: it does not enter the
    checkpoint lock, so a committed CPU checkpoint is still reused when a run moves to GPU.
    """
    root = Path(experiment_dir)
    cfg, _lock = load_experiment(root)
    structure = read_structure(root / cfg.inputs.structure)
    observations = read_observations(root / cfg.inputs.observations)
    setup = from_experiment(structure, observations, cfg)
    refinement = setup.refinement
    steps = _recipe_steps(cfg, refinement, logger, device=device)
    prepared = _prepare(
        setup.plans.combined,
        steps,
        root=root,
        cfg=cfg,
        checkpoint=checkpoint,
        refresh=refresh,
    )
    return run_inference(
        prepared, refinement, method=cfg.solver.inference, device=device, logger=logger
    )


def _recipe_steps(
    cfg: ExperimentConfig,
    refinement: RefinementSetup,
    logger: Logger,
    *,
    device: Device | None = None,
) -> list[PlanStep]:
    """The faithful default recipe as an inspectable step list (its provenance keys the lock).

    The orientation fit runs under the private's per-trial coupling (:func:`_trial_coupling`) -- the
    faithful objective. The tilt-independent fit is not offered here; compose it directly if needed.

    The fit tail is a :func:`~diffBloch.preprocess.fork` on unit-cell volume: a **large cell**
    (> ``_LARGE_CELL_THRESHOLD_A3``) takes the coarse fp32 search with the gather integrity checks
    skipped (``validate=False``, made sound by ``fit_orientation``'s coupled coverage guard); a
    **small cell** takes the exact fp64 path, byte-identical to the pre-fork recipe. The predicate
    reads only the pipeline-invariant grid, so :func:`~diffBloch.preprocess.resolve_recipe` compiles
    the fork to a flat branch before the lock sees it (the branch is fixed per experiment). fp32 /
    ``validate`` are execution-only and stay out of the step identity, so the resolved small-cell
    branch keys identically to today -- the committed quartz checkpoint is untouched.

    ``device`` is threaded to the two fits (execution-only -- it does not alter the recipe
    identity), so the coupled eigensolve runs on the same accelerator as the terminal.
    """
    search = cfg.preprocess.orientation.to_search()
    thickness_grid = cfg.preprocess.thickness.to_grid()
    coupling = _trial_coupling(cfg)

    def orientation_fit(*, precision: FloatFormat, validate: bool) -> PlanStep:
        return fit_orientation(
            refinement,
            search,
            method=cfg.solver.refine,
            coupling=coupling,
            precision=precision,
            validate=validate,
            device=device,
            logger=logger,  # per-rotation fit progress (the run's long phase)
        )

    def thickness_fit(*, precision: FloatFormat) -> PlanStep:
        return fit_thickness(
            refinement, thickness_grid, method=cfg.solver.refine, precision=precision, device=device
        )

    return [
        select_beams(cfg.numerics.to_beam_selection()),
        build_orientation_plans(),
        integrate_rocking_curve(cfg.numerics.to_rocking_curve()),
        mosaicity(cfg.numerics.mosaicity),
        fork(
            lambda grid: grid.cell_volume > _LARGE_CELL_THRESHOLD_A3,
            when_true=[  # large cell: coarse fp32 search, integrity checks skipped
                orientation_fit(precision="fp32", validate=False),
                thickness_fit(precision="fp32"),
            ],
            when_false=[  # small cell: exact fp64, byte-identical to the pre-fork recipe
                orientation_fit(precision="fp64", validate=True),
                thickness_fit(precision="fp64"),
            ],
        ),
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
    # Compile any `fork` away against the base grid (invariant across every step), so the recipe the
    # lock keys on is a flat, fork-free step list -- the fork is Applicative by construction, so its
    # branch is fixed here, before running (see design/decisions/combinators-and-recipe-identity).
    steps = list(resolve_recipe(steps, base.grid))
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
