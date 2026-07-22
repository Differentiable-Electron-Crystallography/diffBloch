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
from diffBloch.engine import (
    ModelRefinementResult,
    build_refinement_model,
    build_refinement_problem,
    run_refinement_model,
)
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
    report_coupling,
    resolve_recipe,
    run_inference,
    select_beams,
    step_records,
    write_plan,
)
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.inference import InferenceResult
from diffBloch.preprocess.scoring import build_engine
from diffBloch.specs import ScoredSelection, TrialCoupling

__all__ = ["preprocess_experiment", "refine_experiment", "run_experiment"]

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


def preprocess_experiment(
    experiment_dir: str | Path,
    *,
    logger: Logger = NULL_LOGGER,
    checkpoint: bool = True,
    refresh: bool = False,
    device: Device | None = None,
    workers: int = 1,
    max_batch: int | None = None,
) -> Plan:
    """Load and preprocess the experiment at ``experiment_dir``, returning the settled ``Plan``.

    The preprocess half of :func:`run_experiment` with the terminal scoring stripped off: it loads
    ``experiment.yaml`` (verifying the input lock), reads the structure + observations, builds the
    geometry via ``from_experiment``, and runs the faithful integrated recipe -- returning the
    settled coupled :class:`~diffBloch.preprocess.plan.Plan` (fitted orientations, tilt-segment
    couplings, pinned scored sets). This is the entry point for callers who want *only* the
    calibrated Plan (to checkpoint it, or to drive their own downstream refinement) without paying
    for the terminal inference pass.

    The recipe is the private's faithful pipeline, **per-trial beam coupling included** (the fit
    re-derives the SOLVE union + SCORED set at every trial orientation -- the private's exact
    objective). Its coupling policy, orientation-search bounds, thickness grid, and whether
    hydrogens are loaded all come from config (``preprocess.coupling`` / ``preprocess.orientation``
    / ``preprocess.thickness`` / ``inputs.load_hydrogens``). A caller wanting a different
    composition (e.g. the cheaper tilt-independent fit) composes their own ``pipeline([...])`` with
    the public steps.

    ``checkpoint`` (default ``True``) reuses/resumes a valid ``plan.npz`` in the experiment dir and
    writes a fresh one after computing; ``refresh`` forces a full recompute (ignoring any snapshot)
    while still regenerating the checkpoint. ``checkpoint=False`` neither reads nor writes.

    ``device`` (default ``None`` = CPU) runs the forward solve on the given accelerator (e.g.
    ``"cuda"``) for the coupled preprocess fits. The preprocess *geometry* (beam selection, coupling
    unions) stays CPU-side numpy; only the eigensolve -- the O(N^3) cost -- moves to the device (the
    fits move the seed params, so ``fgb`` and every per-trial score co-locate there). Device is
    execution-only: it does not enter the checkpoint lock, so a committed CPU checkpoint is still
    reused when a run moves to GPU.

    ``workers`` (default 1, sequential) fans the per-rotation orientation search over a thread pool
    (rotations are independent). On a GPU run the per-trial cost is host-bound around a small
    eigensolve, so overlapping rotations across cores is the main wall-clock lever (measured best
    ~4 on the A100; gains flatten past that as the solves serialise on one CUDA stream). Like
    ``device`` it is execution-only -- the results are identical to a sequential run and it does not
    enter the checkpoint lock. **Cap host threads to 1 when using it** --
    ``OMP_NUM_THREADS``/``MKL_NUM_THREADS``/``TORCH_NUM_THREADS`` (or ``torch.set_num_threads(1)``);
    in a pod torch/BLAS size their pools from the *node* core count, not the cgroup limit, so an
    uncapped run oversubscribes the cores the workers need (measured: capping alone cut the LTA
    per-trial ~1.4 s -> ~0.2 s, before any parallelism).

    ``max_batch`` (default ``None``) caps the ``matrix_exp`` propagator block for the coupled fits.
    ``None`` lets each solve derive a memory-safe block from its beam count; raise it to fill a
    larger accelerator (e.g. ``1024`` on an 80 GB A100). Execution-only (memory, bit-for-bit to
    machine precision), out of the checkpoint lock like ``device``/``workers``. See
    :func:`~diffBloch.engine.build_engine`.
    """
    root = Path(experiment_dir)
    cfg, _lock = load_experiment(root)
    _refinement, prepared = _preprocess(
        root,
        cfg,
        logger=logger,
        checkpoint=checkpoint,
        refresh=refresh,
        device=device,
        workers=workers,
        max_batch=max_batch,
    )
    return prepared


def run_experiment(
    experiment_dir: str | Path,
    *,
    logger: Logger = NULL_LOGGER,
    checkpoint: bool = True,
    refresh: bool = False,
    device: Device | None = None,
    workers: int = 1,
    max_batch: int | None = None,
) -> InferenceResult:
    """Load, preprocess, and score every rotation of the experiment at ``experiment_dir``.

    :func:`preprocess_experiment` followed by the terminal forward model: it settles the coupled
    ``Plan`` (see that function for the recipe, ``checkpoint``/``refresh``, and
    ``device``/``workers`` semantics -- all shared), then evaluates every rotation with
    ``run_inference`` -- emitting
    per-rotation observations to ``logger`` (the null default discards them). Returns the
    :class:`~diffBloch.preprocess.inference.InferenceResult` (per-rotation ``R_obs`` + aggregate).
    ``device`` also runs the terminal eigensolve on the accelerator.
    """
    root = Path(experiment_dir)
    cfg, _lock = load_experiment(root)
    refinement, prepared = _preprocess(
        root,
        cfg,
        logger=logger,
        checkpoint=checkpoint,
        refresh=refresh,
        device=device,
        workers=workers,
        max_batch=max_batch,
    )
    return run_inference(
        prepared,
        refinement,
        method=cfg.solver.inference,
        device=device,
        max_batch=max_batch,
        logger=logger,
    )


def refine_experiment(
    experiment_dir: str | Path,
    *,
    logger: Logger = NULL_LOGGER,
    checkpoint: bool = True,
    refresh: bool = False,
    device: Device | None = None,
    workers: int = 1,
    max_batch: int | None = None,
) -> ModelRefinementResult:
    """Settle the coupled ``Plan`` and gradient-refine the structure against the observed data.

    :func:`preprocess_experiment` for the geometry (checkpoint reuse for free -- see it for the
    recipe and ``checkpoint``/``refresh``/``device``/``workers`` semantics), then run the
    **default** single-stage refinement on that settled ``Plan``. This is the boring config-knobs
    path: the data term (:meth:`~diffBloch.config.schema.ObjectiveConfig.to_loss`), the trainable
    selection
    (:meth:`~diffBloch.config.schema.TrainableConfig.to_spec`), and the optimizer/step budget all
    come from ``experiment.yaml``. It composes no hard constraints or penalties -- scientific
    composition (hydrogen riding, freeze-H, penalties, multi-stage) is a Python/API concern, built
    with :func:`~diffBloch.engine.build_refinement_model`,
    :func:`~diffBloch.engine.build_refinement_problem`, and
    :func:`~diffBloch.engine.with_hydrogen_riding`, then run via ``run_refinement_model``. The
    :class:`~diffBloch.engine.RefinementProblem` here is pure optimization-definition data; the
    imperative loop lives in ``run_refinement_model``. Returns the
    :class:`~diffBloch.engine.ModelRefinementResult` (per-step losses + best snapshot); the refined
    structure is not persisted (deferred).

    ``device`` places the refinement solve on the accelerator: the seed params move there and the
    forward co-locates onto them (as in the preprocess fits).
    """
    root = Path(experiment_dir)
    cfg, _lock = load_experiment(root)
    refinement, prepared = _preprocess(
        root,
        cfg,
        logger=logger,
        checkpoint=checkpoint,
        refresh=refresh,
        device=device,
        workers=workers,
        max_batch=max_batch,
    )
    engine = build_engine(
        prepared,
        refinement,
        loss=cfg.refinement.objective.to_loss(),
        method=cfg.solver.refine,
        max_batch=max_batch,
    )
    initial = refinement.params if device is None else refinement.params.to(device)
    model = build_refinement_model(initial=initial)
    problem = build_refinement_problem()
    return run_refinement_model(
        engine,
        model,
        problem,
        trainable=cfg.refinement.trainable.to_spec(),
        steps=cfg.refinement.steps,
        optimizer=cfg.refinement.optimizer.name,
        lr=cfg.refinement.optimizer.lr,
        logger=logger,
    )


def _preprocess(
    root: Path,
    cfg: ExperimentConfig,
    *,
    logger: Logger,
    checkpoint: bool,
    refresh: bool,
    device: Device | None,
    workers: int,
    max_batch: int | None,
) -> tuple[RefinementSetup, Plan]:
    """Shared spine of the two public entry points: read inputs, run the recipe, settle the Plan.

    Returns the structure ``RefinementSetup`` (which the terminal ``run_inference`` needs)
    alongside the settled ``Plan``, so :func:`preprocess_experiment` can drop the setup and
    :func:`run_experiment` can score with it -- neither loads the inputs twice. Hydrogen sites are
    loaded per ``inputs.load_hydrogens``.
    """
    structure = read_structure(
        root / cfg.inputs.structure, load_hydrogens=cfg.inputs.load_hydrogens
    )
    observations = read_observations(root / cfg.inputs.observations)
    setup = from_experiment(structure, observations, cfg)
    steps = _recipe_steps(
        cfg, setup.refinement, logger, device=device, workers=workers, max_batch=max_batch
    )
    prepared = _prepare(
        setup.plans.combined,
        steps,
        root=root,
        cfg=cfg,
        checkpoint=checkpoint,
        refresh=refresh,
        logger=logger,
    )
    # Report the settled plan's coupling geometry at the consumer boundary: fires on every run
    # (including a checkpoint-reuse refine, which ran no pipeline steps), so the coupling the loop
    # is about to consume is logged before the first step.
    report_coupling(logger)(prepared)
    return setup.refinement, prepared


def _recipe_steps(
    cfg: ExperimentConfig,
    refinement: RefinementSetup,
    logger: Logger,
    *,
    device: Device | None = None,
    workers: int = 1,
    max_batch: int | None = None,
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

    Both ``device`` and ``workers`` are execution-only -- neither alters the recipe identity.
    ``device`` is threaded to *both* fits (so the coupled eigensolve runs on the same accelerator as
    the terminal); ``workers`` fans only the orientation search (the run's long phase) over threads.
    ``max_batch`` (also execution-only) is threaded to both fits: it caps the ``matrix_exp``
    propagator block so a wide coupled segment x the thickness grid can't materialize the whole
    propagator at once (the adaptive-union fit_thickness OOM); ``None`` picks a memory-safe block.
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
            max_batch=max_batch,
            workers=workers,
            logger=logger,  # per-rotation fit progress (the run's long phase)
        )

    def thickness_fit(*, precision: FloatFormat) -> PlanStep:
        return fit_thickness(
            refinement,
            thickness_grid,
            method=cfg.solver.refine,
            precision=precision,
            device=device,
            max_batch=max_batch,
            logger=logger,  # per-rotation thickness-fit progress (the memory-heavy tail phase)
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
    """Assemble the per-trial coupling from config; raise if the coupled fit has no policy.

    Coupling is the faithful objective and carries no default (it determines the physics), so a
    config that runs the coupled orientation fit must declare ``preprocess.coupling``. The SCORED
    set reuses the *same* Klar window as ``select_beams`` (so the two filters cannot disagree) and
    the config's ``g_max_refine`` as the scoring-resolution cap.
    """
    if cfg.preprocess.coupling is None:
        raise ValueError(
            "the coupled orientation fit needs a coupling policy, but preprocess.coupling is "
            "unset; add a preprocess.coupling block (n_splits, g_max, sg_max) to config"
        )
    return TrialCoupling(
        policy=cfg.preprocess.coupling.to_policy(),
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
    logger: Logger = NULL_LOGGER,
) -> Plan:
    """Run the preprocess ``steps`` on ``base``, reusing/resuming a valid checkpoint if present.

    ``logger`` streams a per-step plan summary as the recipe runs (see :func:`pipeline`); it fires
    only when steps actually execute (a fresh or resumed run, not a full-reuse load).
    """
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
            result = pipeline(steps[k:], logger=logger)(snapshot)
            _write_checkpoint(result, recipe, root=root, cfg=cfg, npz=npz, lock_path=lock_path)
            return result

    result = pipeline(steps, logger=logger)(base)
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
