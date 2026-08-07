"""The default experiment runner the ``run infer`` CLI exposes.

:func:`run_experiment` encodes the default recipe as one ordered ``Plan -> Plan`` pipeline:
plan-shaping (``build_orientation_plans``, selecting coupled SOLVE beams from ``g_max``/``sg_max``
and building rocking geometry plus its reduction) followed by
the config-enabled parameter fitting stages (orientation, then thickness), then ``run_inference``
evaluates it -- so a caller with an experiment directory gets a full result in one call. It is a
*convenience*, not the only path: every step is ordinary public API, so a Python user who wants a
different composition composes their own ``pipeline([...])`` with ``from_experiment`` +
``run_inference`` directly. The CLI stays thin by delegating here and holds no science.

**Checkpoint / resume.** The preprocess (the coupled fit especially) is the run's expensive phase,
so its settled ``Plan`` is checkpointed to ``<experiment_dir>/reproducibility/`` (``plan.npz`` +
``plan.lock``, see :func:`_reproducibility_dir`) and reused when still valid. The lock binds the
checkpoint to four axes -- input bytes, resolved config,
software version, and the *recipe* (the Plan's own provenance) -- so it is reused only when all
match. The match is a longest-prefix over the recipe: an identical recipe is a full **reuse**; a
recipe that *appends* steps **resumes** from the snapshot and runs only the suffix (the append-only
increment). Any run that computes -- fresh, stale-recompute, ``refresh``, or resume -- regenerates
``plan.npz`` + ``plan.lock``, so the lock always describes the on-disk checkpoint. Diagnostics about
load/resume go through stdlib ``logging`` (not the domain-observation ``logger``).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import gemmi
import numpy as np
import torch

from diffBloch.config import (
    ExperimentConfig,
    PreprocessLock,
    RecipeStep,
    RefinementLock,
    artifact_hash_for,
    code_version,
    config_digest,
    load_experiment,
    preprocess_lock_status,
    read_preprocess_lock,
    refinement_config_digest,
    sha256_file,
    write_preprocess_lock,
    write_refinement_lock,
)
from diffBloch.engine import (
    ApparentThicknessNN,
    ModelRefinementResult,
    RefinementEngine,
    ThicknessBounds,
    build_refinement_model,
    build_refinement_problem,
    run_refinement_model,
)
from diffBloch.io import read_experimental_data, read_structure
from diffBloch.observability import (
    NULL_LOGGER,
    DeviceSelected,
    Logger,
    MultiLogger,
    RefinedRotationMetrics,
    RefinementOutputsWritten,
)
from diffBloch.params import Device, constrain
from diffBloch.preprocess import (
    OPAQUE,
    ConvergenceTest,
    ConvergenceTolerance,
    Plan,
    PlanStep,
    build_orientation_plans,
    fork,
    from_experiment,
    import_orientations,
    optimize_orientation,
    optimize_thickness,
    pipeline,
    read_plan,
    resolve_recipe,
    run_inference,
    select_beams,
    step_records,
    write_plan,
)
from diffBloch.preprocess.driver import ConvergenceState, run_convergence
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.inference import InferenceResult
from diffBloch.preprocess.scoring import build_engine
from diffBloch.specs import IntegrationGeometry, ScoredHklSelection, TrialCoupling

__all__ = [
    "converge_experiment",
    "preprocess_experiment",
    "refine_experiment",
    "run_experiment",
]

_log = logging.getLogger(__name__)

_PLAN_NPZ = "plan.npz"
_PLAN_LOCK = "plan.lock"
_REFINEMENT_LOCK = "refinement.lock"
_REPRODUCIBILITY_DIRNAME = "reproducibility"


def _reproducibility_dir(root: Path) -> Path:
    """The ``root/reproducibility`` subdirectory for generated bookkeeping, created on first use.

    Keeps ``root`` itself down to the handful of files someone actually reads (``experiment.yaml``,
    the input structure/experimental-data files, ``refined_structure.cif``,
    ``refinement_report.txt``, the thickness-NN shape plot) -- everything else, including
    ``experiment.lock`` (input-identity verification, not a generated output, but bookkeeping all
    the same), lives here instead: preprocess/refinement checkpoints and their locks, and the raw
    parameter/component ``.npz`` snapshots.
    """
    directory = root / _REPRODUCIBILITY_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _select_device(device: Device | None, *, logger: Logger = NULL_LOGGER) -> Device:
    """Resolve the execution device, falling back from CUDA to CPU when this host lacks CUDA."""
    requested = "cpu" if device is None else str(device)
    cuda_available = torch.cuda.is_available()
    selected: Device = "cpu" if device is None else device
    if torch.device(requested).type == "cuda" and not cuda_available:
        selected = "cpu"
    selected_name = str(selected)
    logger.report(
        DeviceSelected(
            requested=requested,
            selected=selected_name,
            cuda_available=cuda_available,
        )
    )
    return selected


# Above this unit-cell volume the coupled orientation search runs its coarse pass with the
# gather integrity checks skipped (the large-cell fast path); at or below it the search stays the
# exact, fully-validated path. The eigensolve is O(N^3) in the beam count
# and N grows with cell volume, so skipping the O(N^2) integrity checks only pays off for large
# cells; a small cell (quartz ~113 A^3) fits in seconds and gains nothing from skipping them.
# A deliberately wide heuristic gap separates the known regimes -- quartz ~113 vs zeolites
# ~1861 A^3 -- so classification is unambiguous; it is not a sharp physical boundary. Kept a code
# constant, NOT a config field: a config field would enter config_digest and restale the committed
# quartz checkpoint. It only selects whether the *search* validates its gathers, and the terminal
# always re-scores the found orientation, so the threshold never affects the reported score's
# fidelity.
_LARGE_CELL_THRESHOLD_A3 = 1000.0


def converge_experiment(
    experiment_dir: str | Path,
    *,
    logger: Logger = NULL_LOGGER,
    device: Device = "cuda",
    n_orientations: int = 1,
) -> ConvergenceState:
    """Run the standard numerical-convergence test for an experiment.

    Starting ``g_max`` prefers the PETS file's own ``dstarmax`` (its processing-resolution cutoff,
    :attr:`~diffBloch.io.record.ExperimentalRecord.dstar_max`) when present; a PETS version that
    doesn't record it (or a file where the tag is otherwise absent) falls back to the experiment's
    configured ``blochwave.g_max``, the previous manual-only behaviour. ``sg_max`` and rocking-curve
    tilt steps still start from the configured simulation settings. The sweep then follows the
    defaults owned by :class:`ConvergenceTest` and :class:`ConvergenceTolerance`, returning the
    smallest settled values found.
    """
    root = Path(experiment_dir)
    device = _select_device(device, logger=logger)
    cfg, _lock = load_experiment(root)

    structure = read_structure(
        root / cfg.inputs.structure, load_hydrogens=cfg.inputs.load_hydrogens
    )
    experimental_data = read_experimental_data(root / cfg.inputs.exp_data)
    setup = from_experiment(structure, experimental_data, cfg)
    refinement = replace(setup.refinement, params=setup.refinement.params.to(device))
    combined = setup.plans.combined
    if n_orientations < 1:
        raise ValueError("n_orientations must be >= 1")
    if n_orientations > len(combined.orientations):
        raise ValueError(
            f"n_orientations={n_orientations} exceeds the experiment's "
            f"{len(combined.orientations)} orientations"
        )
    selected = replace(
        combined,
        orientations=combined.orientations[:n_orientations],
    )
    rocking = cfg.blochwave.to_rocking_curve(setup.integration)
    simulation = cfg.blochwave.to_policy()
    starting_g_max = (
        experimental_data.dstar_max if experimental_data.dstar_max is not None else simulation.g_max
    )
    plan = pipeline(
        [
            select_beams(cfg.blochwave.to_beam_selection(setup.integration)),
            build_orientation_plans(),
        ]
    )(selected)

    _plan, settled = run_convergence(
        plan,
        ConvergenceState(
            g_max=starting_g_max,
            sg_max=simulation.sg_max,
            tilt_steps=rocking.sampling,
        ),
        ConvergenceTest(),
        rocking,
        simulation,
        refinement,
        ConvergenceTolerance(),
        method=cfg.blochwave.solver.refine,
        logger=logger,
    )
    return settled


def preprocess_experiment(
    experiment_dir: str | Path,
    *,
    logger: Logger = NULL_LOGGER,
    checkpoint: bool = True,
    refresh: bool = False,
    device: Device | None = "cuda",
    workers: int = 1,
    max_batch: int | None = None,
    orientations_csv: str | Path | None = None,
    plot_thickness: bool = False,
    plot_thickness_dir: str | Path | None = None,
) -> Plan:
    """Load and preprocess the experiment at ``experiment_dir``, returning the settled ``Plan``.

    The preprocess half of :func:`run_experiment` with the terminal scoring stripped off: it loads
    ``experiment.yaml`` (verifying the input lock), reads the structure + experimental data, builds the
    geometry via ``from_experiment``, and runs the default integrated recipe -- returning the
    settled coupled :class:`~diffBloch.preprocess.plan.Plan` (fitted orientations, tilt-segment
    couplings, pinned scored sets). This is the entry point for callers who want *only* the
    calibrated Plan (to checkpoint it, or to drive their own downstream refinement) without paying
    for the terminal inference pass.

    The recipe includes **per-trial beam coupling** (the fit
    re-derives the SOLVE union + SCORED set at every trial orientation). Its coupling policy,
    orientation-search bounds, thickness grid, and whether
    hydrogens are loaded all come from config (``blochwave`` / ``preprocess.orientation``
    / ``preprocess.thickness`` / ``inputs.load_hydrogens``). A caller wanting a different
    composition (e.g. the cheaper tilt-independent fit) composes their own ``pipeline([...])`` with
    the public steps.

    ``checkpoint`` (default ``True``) reuses/resumes a valid ``plan.npz`` in the experiment dir and
    writes a fresh one after computing; ``refresh`` forces a full recompute (ignoring any snapshot)
    while still regenerating the checkpoint. ``checkpoint=False`` neither reads nor writes.

    ``device`` (default ``"cuda"``) runs the forward solve on the selected accelerator for the
    coupled preprocess fits. If CUDA is requested on a host without CUDA, the app falls back to CPU
    and emits a device-selection event. The preprocess *geometry* (beam selection, coupling unions)
    stays CPU-side numpy; only the eigensolve -- the O(N^3) cost -- moves to the device (the fits
    move the seed params, so ``fgb`` and every per-trial score co-locate there). Device is
    execution-only: it does not enter the checkpoint lock, so a committed CPU checkpoint is still
    reused when a run moves to GPU.

    ``workers`` (default 1, sequential) fans the per-rotation orientation search over a thread pool
    (rotations are independent). On a GPU run the per-trial cost is host-bound around a small
    eigensolve, so overlapping rotations across cores is the main wall-clock lever (a small worker
    count is usually the sweet spot; gains flatten as the solves serialise on one GPU stream). Like
    ``device`` it is execution-only -- the results are identical to a sequential run and it does not
    enter the checkpoint lock. **Cap host threads to 1 when using it** --
    ``OMP_NUM_THREADS``/``MKL_NUM_THREADS``/``TORCH_NUM_THREADS`` (or ``torch.set_num_threads(1)``);
    in a pod torch/BLAS size their pools from the *node* core count, not the cgroup limit, so an
    uncapped run oversubscribes the cores the workers need (capping alone can dominate the speedup,
    before any parallelism).

    ``max_batch`` (default ``None``) caps the ``matrix_exp`` propagator block for the coupled fits.
    ``None`` lets each solve derive a memory-safe block from its beam count; raise it to fill a
    larger accelerator's memory budget. Execution-only (memory, bit-for-bit to
    machine precision), out of the checkpoint lock like ``device``/``workers``. See
    :func:`~diffBloch.engine.build_engine`.

    ``orientations_csv`` (default ``None``), when given, prepends
    :func:`~diffBloch.preprocess.import_orientations` to the recipe: every candidate's orientation
    is overwritten from the CSV (a ``Rotation Index`` / ``Orientation Matrix`` file -- this repo's
    own orientation-search output format) before ``build_orientation_plans`` runs, and
    ``preprocess.optimize_orientation`` then controls whether the search still refines from that
    seed or is skipped so the imported orientations are used as-is. Unlike ``device``/``workers``
    this **is** part of the recipe (the checkpoint lock sees it), so switching the CSV path (or
    adding/removing it) correctly restales any existing checkpoint.

    ``plot_thickness`` (default ``False``) ORs with ``cfg.preprocess.thickness.plot`` -- either
    turns on one wR2-vs-thickness PNG per rotation from ``optimize_thickness``'s grid search, saved
    under ``plot_thickness_dir`` (default ``<inputs.structure's directory>/thickness_optim``).
    Execution-only like ``device``/``workers``, out of the checkpoint lock.
    """
    root = Path(experiment_dir)
    device = _select_device(device, logger=logger)
    cfg, _lock = load_experiment(root)
    _refinement, _integration, prepared, _validation_rotation_indices = _preprocess(
        root,
        cfg,
        logger=logger,
        checkpoint=checkpoint,
        refresh=refresh,
        device=device,
        workers=workers,
        max_batch=max_batch,
        orientations_csv=orientations_csv,
        plot_thickness=plot_thickness,
        plot_thickness_dir=plot_thickness_dir,
    )
    return prepared


def run_experiment(
    experiment_dir: str | Path,
    *,
    logger: Logger = NULL_LOGGER,
    checkpoint: bool = True,
    refresh: bool = False,
    device: Device | None = "cuda",
    workers: int = 1,
    max_batch: int | None = None,
    orientations_csv: str | Path | None = None,
    plot_thickness: bool = False,
    plot_thickness_dir: str | Path | None = None,
) -> InferenceResult:
    """Load, preprocess, and score every rotation of the experiment at ``experiment_dir``.

    :func:`preprocess_experiment` followed by the terminal forward model: it settles the coupled
    ``Plan`` (see that function for the recipe, ``checkpoint``/``refresh``,
    ``device``/``workers``, ``orientations_csv``, and ``plot_thickness``/``plot_thickness_dir``
    semantics -- all shared), then evaluates every rotation with
    ``run_inference`` -- emitting
    per-rotation observations to ``logger`` (the null default discards them). Returns the
    :class:`~diffBloch.preprocess.inference.InferenceResult` (per-rotation ``R_obs`` + aggregate).
    ``device`` also runs the terminal eigensolve on the accelerator.
    """
    root = Path(experiment_dir)
    device = _select_device(device, logger=logger)
    cfg, _lock = load_experiment(root)
    refinement, _integration, prepared, _validation_rotation_indices = _preprocess(
        root,
        cfg,
        logger=logger,
        checkpoint=checkpoint,
        refresh=refresh,
        device=device,
        workers=workers,
        max_batch=max_batch,
        orientations_csv=orientations_csv,
        plot_thickness=plot_thickness,
        plot_thickness_dir=plot_thickness_dir,
    )
    return run_inference(
        prepared,
        refinement,
        method=cfg.blochwave.solver.inference,
        device=device,
        max_batch=max_batch,
        logger=logger,
        absorption=cfg.blochwave.to_absorption(),
    )


def refine_experiment(
    experiment_dir: str | Path,
    *,
    logger: Logger = NULL_LOGGER,
    checkpoint: bool = True,
    refresh: bool = False,
    device: Device | None = "cuda",
    workers: int = 1,
    max_batch: int | None = None,
    verbose: bool = False,
    profile: bool = False,
    checkpoint_activations: bool = True,
    orientations_csv: str | Path | None = None,
    plot_thickness: bool = False,
    plot_thickness_dir: str | Path | None = None,
) -> ModelRefinementResult:
    """Settle the coupled ``Plan`` and gradient-refine the structure against the observed data.

    :func:`preprocess_experiment` for the geometry (checkpoint reuse for free -- see it for the
    recipe and ``checkpoint``/``refresh``/``device``/``workers``/``orientations_csv``/
    ``plot_thickness``/``plot_thickness_dir`` semantics), then run the
    **default** single-stage refinement on that settled ``Plan``. This is the boring config-knobs
    path: the residual (:meth:`~diffBloch.config.schema.LossMetricsConfig.to_loss`), the trainable
    selection (:meth:`~diffBloch.config.schema.TrainableConfig.to_spec`), and
    the optimizer/step budget all come from ``experiment.yaml``. It composes no hard constraints or
    penalties -- scientific
    composition (hydrogen riding, freeze-H, penalties, multi-stage) is a Python/API concern, built
    with :func:`~diffBloch.engine.build_refinement_model`,
    :func:`~diffBloch.engine.build_refinement_problem`, and
    :func:`~diffBloch.engine.with_hydrogen_riding`, then run via ``run_refinement_model``. The
    :class:`~diffBloch.engine.RefinementProblem` here is pure optimization-definition data; the
    imperative loop lives in ``run_refinement_model``. Returns the
    :class:`~diffBloch.engine.ModelRefinementResult` (per-step losses + best snapshot);
    :func:`_write_refinement_outputs` persists the best structure/parameters/summary to
    ``experiment_dir`` alongside a ``refinement.lock`` binding them to the settled ``Plan``
    (``plan.lock``), the refinement-determining config, and the code version that produced them --
    the refinement-stage counterpart to the preprocess checkpoint's own lock.

    ``device`` places the refinement solve on the accelerator: the seed params move there and the
    forward co-locates onto them (as in the preprocess fits).

    ``verbose`` ("verbose refinement") reports one per-rotation wR2/R_obs/diffraction-loss line per
    step in addition to the epoch mean; see :func:`~diffBloch.engine.run_refinement_model`.
    Execution-only, like ``logger``.

    ``profile`` logs per-phase wall time (structure factors, each rotation's solve, backward,
    optimizer step) via stdlib diagnostics logging; see :func:`~diffBloch.engine.run_refinement_model`
    and :func:`~diffBloch.preprocess.scoring.build_engine`. Execution-only and off by default.

    ``checkpoint_activations`` (default ``True``) trades peak memory for one extra forward
    recompute per solve on backward; disabling it removes that recompute at the cost of retaining
    solve intermediates until backward. Execution-only -- gradients are unaffected. See
    :class:`~diffBloch.engine.RefinementEngine`.
    """
    root = Path(experiment_dir)
    device = _select_device(device, logger=logger)
    cfg, _lock = load_experiment(root)
    refinement, integration, prepared, validation_rotation_indices = _preprocess(
        root,
        cfg,
        logger=logger,
        checkpoint=checkpoint,
        refresh=refresh,
        device=device,
        workers=workers,
        max_batch=max_batch,
        orientations_csv=orientations_csv,
        plot_thickness=plot_thickness,
        plot_thickness_dir=plot_thickness_dir,
    )
    # `engine` covers every rotation (train + validation) -- reporting always scores the whole
    # experiment, e.g. the thickness-NN shape table below evaluates the trained curve at
    # validation angles it never saw, which is the point. Only the *training* engine, built
    # separately, excludes validation_rotation_indices from the gradient objective.
    engine = build_engine(
        prepared,
        refinement,
        loss=cfg.loss_metrics.to_loss(),
        method=cfg.blochwave.solver.refine,
        max_batch=max_batch,
        absorption=cfg.blochwave.to_absorption(),
        profile=profile,
        checkpoint_activations=checkpoint_activations,
    )
    train_engine = engine
    selection_engine = None
    if validation_rotation_indices:
        train_only = tuple(
            op
            for op in prepared.orientations
            if op.pattern.rotation_index not in validation_rotation_indices
        )
        validation_only = tuple(
            op
            for op in prepared.orientations
            if op.pattern.rotation_index in validation_rotation_indices
        )
        train_engine = build_engine(
            replace(prepared, orientations=train_only),
            refinement,
            loss=cfg.loss_metrics.to_loss(),
            method=cfg.blochwave.solver.refine,
            max_batch=max_batch,
            absorption=cfg.blochwave.to_absorption(),
            profile=profile,
            checkpoint_activations=checkpoint_activations,
        )
        selection_engine = build_engine(
            replace(prepared, orientations=validation_only),
            refinement,
            loss=cfg.loss_metrics.to_loss(),
            method=cfg.blochwave.solver.refine,
            max_batch=max_batch,
            absorption=cfg.blochwave.to_absorption(),
            profile=profile,
            checkpoint_activations=checkpoint_activations,
        )
    logger.report(cfg.to_declaration(integration))
    initial = refinement.params if device is None else refinement.params.to(device)
    thickness_spec = cfg.refinement.thickness_nn.to_spec()
    thickness_nn: ApparentThicknessNN | None = None
    raw_alphas: np.ndarray | None = None
    if thickness_spec.enabled:
        experimental_data = read_experimental_data(root / cfg.inputs.exp_data)
        raw_alphas = experimental_data.alphas
        thickness_nn = ApparentThicknessNN(
            bounds=ThicknessBounds(
                thickness_spec.min_thickness,
                thickness_spec.max_thickness,
            ),
            normalized_alphas=_normalized_pets_alphas(experimental_data.alphas),
            form=thickness_spec.form,
            sample_thickness=thickness_spec.sample_thickness,
            num_samples=thickness_spec.num_samples,
            init_seed=thickness_spec.init_seed,
        )
        model = build_refinement_model(
            initial=initial,
            components=(thickness_nn,),
            component_params={
                thickness_nn.key: thickness_nn.initial_params(
                    dtype=initial.asu_positions.dtype,
                    device=initial.asu_positions.device,
                )
            },
        )
    else:
        model = build_refinement_model(initial=initial)
    problem = build_refinement_problem()
    result = run_refinement_model(
        train_engine,
        model,
        problem,
        trainable=cfg.refinement.trainable.to_spec(),
        steps=cfg.refinement.steps,
        optimizer=cfg.refinement.optimizer.name,
        lr=cfg.refinement.optimizer.lr,
        logger=logger,
        verbose=verbose,
        profile=profile,
        selection_engine=selection_engine,
    )
    result = _write_refinement_outputs(root, cfg, refinement, result)
    # The settled-result events, then the terminal one: SummaryLogger writes the file on the latter.
    _report_refinement_outcome(
        logger,
        engine,
        result,
        validation_rotation_indices=validation_rotation_indices,
        thickness_nn=thickness_nn,
        raw_alphas=raw_alphas,
    )
    return result


def _report_refinement_outcome(
    logger: Logger,
    engine: RefinementEngine,
    result: ModelRefinementResult,
    *,
    validation_rotation_indices: frozenset[int],
    thickness_nn: ApparentThicknessNN | None,
    raw_alphas: np.ndarray | None,
) -> None:
    """Emit the settled-result events the refinement loop itself cannot produce.

    ``run_refinement_model`` only ever sees the *training* engine, so the final per-rotation scores
    (which cover held-out rotations too) and the trained thickness curve have to be emitted here,
    where the reporting engine and the split are both in scope. :class:`RefinementOutputsWritten`
    goes last and is the run's terminal event -- a sink that must write exactly once, after
    everything else, acts on it.
    """
    for row in engine.per_rotation_metrics(result.best_model):
        logger.report(
            RefinedRotationMetrics(
                rotation_index=row.rotation_index,
                wr2=row.wr2,
                r_obs=row.r_obs,
                n_matched=row.n_matched,
                is_validation=row.rotation_index in validation_rotation_indices,
            )
        )
    if thickness_nn is not None and raw_alphas is not None:
        logger.report(
            thickness_nn.profile(
                result.best_model.component_params[thickness_nn.key],
                engine.orientations,
                raw_alphas,
            )
        )
    logger.report(
        RefinementOutputsWritten(
            structure=result.artifacts["refined_structure"], artifacts=result.artifacts
        )
    )


def _normalized_pets_alphas(alphas: np.ndarray) -> tuple[float, ...]:
    """Legacy MinMaxScaler ``[-1, 1]`` normalization for the PETS alpha coordinate."""
    values = np.asarray(alphas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("PETS alphas must be a finite non-empty 1-D array")
    minimum = float(values.min())
    span = float(values.max() - minimum)
    if span == 0.0:
        return tuple(-1.0 for _ in values)
    normalized = -1.0 + 2.0 * (values - minimum) / span
    return tuple(float(value) for value in normalized)


def _write_refinement_outputs(
    root: Path,
    cfg: ExperimentConfig,
    refinement: RefinementSetup,
    result: ModelRefinementResult,
) -> ModelRefinementResult:
    """Persist the best structure and raw parameter/component snapshots.

    ``refined_structure.cif`` lands at ``root`` (the human-facing output); the raw ``.npz``
    snapshots and the checkpoint/refinement locks go under :func:`_reproducibility_dir` --
    bookkeeping nobody reads directly. See :class:`~diffBloch.app.loggers.summary.SummaryLogger`
    for the actual human-readable summary (``refinement_report.txt``), which supersedes the
    machine-readable ``refinement_summary.json`` this function used to also write.
    """
    reproducibility_dir = _reproducibility_dir(root)
    structure_path = (root / "refined_structure.cif").resolve()
    params_path = (reproducibility_dir / "refined_parameters.npz").resolve()
    components_path = (reproducibility_dir / "refined_components.npz").resolve()
    state = constrain(result.best_params, refinement.spec)
    source_path = root / cfg.inputs.structure
    document = gemmi.cif.read_file(str(source_path))
    block = document.sole_block()
    structure = read_structure(source_path, load_hydrogens=cfg.inputs.load_hydrogens)
    positions = state.positions.detach().cpu().numpy()
    occupancies = state.occupancies.detach().cpu().numpy()
    uij_star = state.uij_star.detach().cpu().numpy()
    reciprocal_basis = refinement.spec.reciprocal_basis
    assert reciprocal_basis is not None
    reciprocal = reciprocal_basis.detach().cpu().numpy()
    reciprocal_lengths = np.linalg.norm(reciprocal, axis=1)
    reciprocal_metric = reciprocal @ reciprocal.T
    atom_loop = block.find_loop("_atom_site_label").get_loop()
    tags = list(atom_loop.tags)
    label_column = tags.index("_atom_site_label")
    atom_rows = {
        atom_loop.values[row * atom_loop.width() + label_column]: row
        for row in range(atom_loop.length())
    }
    for index, label in enumerate(structure.labels):
        row = atom_rows[label]
        updates = {
            "_atom_site_fract_x": positions[index, 0],
            "_atom_site_fract_y": positions[index, 1],
            "_atom_site_fract_z": positions[index, 2],
            "_atom_site_occupancy": occupancies[index],
        }
        if structure.adp.kind[index] == "Uiso":
            updates["_atom_site_U_iso_or_equiv"] = np.sum(
                uij_star[index] * reciprocal_metric
            ) / np.sum(reciprocal_metric * reciprocal_metric)
        for tag, value in updates.items():
            if tag in tags:
                column = tags.index(tag)
                atom_loop[row, column] = f"{float(value):.10g}"

    aniso_column = block.find_loop("_atom_site_aniso_label")
    if aniso_column:
        aniso_loop = aniso_column.get_loop()
        aniso_tags = list(aniso_loop.tags)
        label_column = aniso_tags.index("_atom_site_aniso_label")
        aniso_rows = {
            aniso_loop.values[row * aniso_loop.width() + label_column]: row
            for row in range(aniso_loop.length())
        }
        components = {
            "_atom_site_aniso_U_11": (0, 0),
            "_atom_site_aniso_U_22": (1, 1),
            "_atom_site_aniso_U_33": (2, 2),
            "_atom_site_aniso_U_12": (0, 1),
            "_atom_site_aniso_U_13": (0, 2),
            "_atom_site_aniso_U_23": (1, 2),
        }
        scale = reciprocal_lengths[:, None] * reciprocal_lengths[None, :]
        for index, label in enumerate(structure.labels):
            if structure.adp.kind[index] != "Uani" or label not in aniso_rows:
                continue
            row = aniso_rows[label]
            uij_cif = uij_star[index] / scale
            for tag, (i, j) in components.items():
                if tag in aniso_tags:
                    column = aniso_tags.index(tag)
                    aniso_loop[row, column] = f"{float(uij_cif[i, j]):.10g}"
    document.write_file(str(structure_path))

    params = result.best_params
    empty = np.empty((0,), dtype=np.float64)
    np.savez_compressed(
        str(params_path),
        asu_positions=params.asu_positions.detach().cpu().numpy(),
        uij_raw=empty if params.uij_raw is None else params.uij_raw.detach().cpu().numpy(),
        u_iso_raw=(empty if params.u_iso_raw is None else params.u_iso_raw.detach().cpu().numpy()),
        occupancy_raw=(
            empty if params.occupancy_raw is None else params.occupancy_raw.detach().cpu().numpy()
        ),
    )
    component_arrays = {
        f"{component_key}__{parameter_name}": parameter.detach().cpu().numpy()
        for component_key, parameters in result.best_model.component_params.items()
        for parameter_name, parameter in parameters.items()
    }
    if component_arrays:
        np.savez_compressed(str(components_path), **component_arrays)  # type: ignore[arg-type]

    artifacts: dict[str, str] = {
        "refined_structure": str(structure_path),
        "refined_parameters": str(params_path),
        "plan": str((reproducibility_dir / _PLAN_NPZ).resolve()),
        "plan_lock": str((reproducibility_dir / _PLAN_LOCK).resolve()),
    }
    if component_arrays:
        artifacts["refined_components"] = str(components_path)
    plan_lock_path = reproducibility_dir / _PLAN_LOCK
    if plan_lock_path.exists():
        lock_path = (reproducibility_dir / _REFINEMENT_LOCK).resolve()
        write_refinement_lock(
            lock_path,
            RefinementLock(
                plan_lock_sha256=sha256_file(plan_lock_path),
                refinement_config_digest=refinement_config_digest(cfg),
                code_version=code_version(),
                refined_structure=artifact_hash_for(structure_path, root=root.resolve()),
                refined_parameters=artifact_hash_for(params_path, root=root.resolve()),
            ),
        )
        artifacts["refinement_lock"] = str(lock_path)
    return replace(result, artifacts=artifacts)


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
    orientations_csv: str | Path | None = None,
    plot_thickness: bool = False,
    plot_thickness_dir: str | Path | None = None,
) -> tuple[RefinementSetup, IntegrationGeometry, Plan, frozenset[int]]:
    """Shared spine of the two public entry points: read inputs, run the recipe, settle the Plan.

    Returns the structure ``RefinementSetup`` (which the terminal ``run_inference`` needs), the
    ``IntegrationGeometry`` (the PETS-derived integration semiangle, for reporting), the settled
    ``Plan`` (still over *every* rotation -- the train/validation split never restricts
    preprocessing, only :func:`refine_experiment`'s gradient stage), and the ``rotation_index`` set
    ``cfg.refinement.split`` holds out as validation (empty when ``train_test`` is off), so
    :func:`preprocess_experiment` can drop the setup and :func:`run_experiment` can score with it --
    neither loads the inputs twice. Hydrogen sites are loaded per ``inputs.load_hydrogens``.

    ``orientations_csv`` (the API/CLI argument) overrides ``cfg.preprocess.orientations_csv`` when
    given; the config value (resolved relative to ``root``, like ``inputs.structure``) is the
    default so the import is reproducible from ``experiment.yaml`` alone, not just a CLI one-off.

    ``plot_thickness`` (API/CLI) ORs with ``cfg.preprocess.thickness.plot`` -- either can turn
    plotting on. ``plot_thickness_dir`` overrides the default output directory,
    ``<inputs.structure's directory>/thickness_optim``, when given. Both are execution-only (they
    only decide whether/where a PNG gets written, never the fitted ``Plan``) -- see
    :func:`~diffBloch.config.manifest.config_digest`.
    """
    structure = read_structure(
        root / cfg.inputs.structure, load_hydrogens=cfg.inputs.load_hydrogens
    )
    experimental_data = read_experimental_data(root / cfg.inputs.exp_data)
    setup = from_experiment(structure, experimental_data, cfg)
    validation_rotation_indices = frozenset(
        op.pattern.rotation_index for op in setup.plans.validation.orientations
    )
    effective_orientations_csv = (
        orientations_csv
        if orientations_csv is not None
        else (root / cfg.preprocess.orientations_csv if cfg.preprocess.orientations_csv else None)
    )
    if plot_thickness or cfg.preprocess.thickness.plot:
        from diffBloch.app.loggers.plotting import ThicknessPlotLogger

        effective_plot_dir = (
            Path(plot_thickness_dir)
            if plot_thickness_dir is not None
            else (root / cfg.inputs.structure).parent / "thickness_optim"
        )
        logger = MultiLogger((logger, ThicknessPlotLogger(effective_plot_dir)))
    steps = _recipe_steps(
        cfg,
        setup.refinement,
        setup.integration,
        logger,
        device=device,
        workers=workers,
        max_batch=max_batch,
        orientations_csv=effective_orientations_csv,
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
    return setup.refinement, setup.integration, prepared, validation_rotation_indices


def _recipe_steps(
    cfg: ExperimentConfig,
    refinement: RefinementSetup,
    integration: IntegrationGeometry,
    logger: Logger,
    *,
    device: Device | None = None,
    workers: int = 1,
    max_batch: int | None = None,
    orientations_csv: str | Path | None = None,
) -> list[PlanStep]:
    """The default recipe as an inspectable step list (its provenance keys the lock).

    ``preprocess.optimize_orientation`` and ``preprocess.optimize_thickness`` select which fitting
    stages join the fixed recipe order. When enabled, the orientation fit runs under per-trial
    coupling (:func:`_trial_coupling`). The tilt-independent fit is not offered here; compose it
    directly if needed.

    ``orientations_csv``, when given, prepends :func:`~diffBloch.preprocess.import_orientations`
    (every candidate's orientation is overwritten from the CSV before ``build_orientation_plans``
    runs). ``optimize_orientation`` still controls whether the search step follows it -- ``true``
    refines from the imported seed, ``false`` uses the imported orientations as-is.

    The orientation fit is a :func:`~diffBloch.preprocess.fork` on unit-cell volume: a **large
    cell** (> ``_LARGE_CELL_THRESHOLD_A3``) skips the per-trial gather integrity checks
    (``validate=False``, made sound by ``optimize_orientation``'s coupled coverage guard); a
    **small cell** takes the exact, fully-validated path. The predicate
    reads only the pipeline-invariant grid, so :func:`~diffBloch.preprocess.resolve_recipe` compiles
    the fork to a flat branch before the lock sees it (the branch is fixed per experiment).
    ``validate`` is execution-only and stays out of the step identity, so the resolved branch keys
    the same either way -- the committed quartz checkpoint is untouched. The thickness fit has no
    such split and always runs the same path regardless of cell size.

    Both ``device`` and ``workers`` are execution-only -- neither alters the recipe identity.
    ``device`` is threaded to *both* fits (so the coupled eigensolve runs on the same accelerator as
    the terminal); ``workers`` fans both the independent initial rotation-plan builds and the
    orientation searches over threads.
    ``max_batch`` (also execution-only) is threaded to both fits: it caps the ``matrix_exp``
    propagator block so a wide coupled segment x the thickness grid can't materialize the whole
    propagator at once; ``None`` picks a memory-safe block.
    """
    search = cfg.preprocess.orientation.to_search()
    thickness_grid = cfg.preprocess.thickness.to_grid()
    coupling = _trial_coupling(cfg, integration)

    def orientation_fit(*, validate: bool) -> PlanStep:
        return optimize_orientation(
            refinement,
            search,
            method=cfg.blochwave.solver.refine,
            coupling=coupling,
            validate=validate,
            device=device,
            max_batch=max_batch,
            workers=workers,
            logger=logger,  # per-rotation fit progress (the run's long phase)
            absorption=cfg.blochwave.to_absorption(),
            scores=cfg.loss_metrics.to_scores(),
            residual=cfg.loss_metrics.residual,
        )

    def thickness_fit() -> PlanStep:
        return optimize_thickness(
            refinement,
            thickness_grid,
            method=cfg.blochwave.solver.refine,
            device=device,
            max_batch=max_batch,
            logger=logger,  # per-rotation thickness-fit progress (the memory-heavy tail phase)
            absorption=cfg.blochwave.to_absorption(),
            scores=cfg.loss_metrics.to_scores(),
            residual=cfg.loss_metrics.residual,
        )

    steps: list[PlanStep] = []
    if orientations_csv is not None:
        steps.append(import_orientations(orientations_csv))
    steps.append(
        build_orientation_plans(
            cfg.blochwave.to_rocking_curve(integration),
            cfg.blochwave.mosaicity,
            coupling=cfg.blochwave.to_policy(),
            scoring_selection=cfg.blochwave.to_beam_selection(integration),
            workers=workers,
        )
    )

    def orientation_step() -> PlanStep:
        return fork(
            lambda grid: grid.cell_volume > _LARGE_CELL_THRESHOLD_A3,
            when_true=[orientation_fit(validate=False)],
            when_false=[orientation_fit(validate=True)],
        )

    fitting_steps: list[PlanStep] = []
    if cfg.preprocess.stage_order == "thickness_first":
        if cfg.preprocess.optimize_thickness:
            fitting_steps.append(thickness_fit())
        if cfg.preprocess.optimize_orientation:
            fitting_steps.append(orientation_step())
    else:
        if cfg.preprocess.optimize_orientation:
            fitting_steps.append(orientation_step())
        if cfg.preprocess.optimize_thickness:
            fitting_steps.append(thickness_fit())
    steps.extend(fitting_steps)
    return steps


def _trial_coupling(cfg: ExperimentConfig, integration: IntegrationGeometry) -> TrialCoupling:
    """Assemble the per-trial beam-union policy from the top-level Bloch-wave config.

    The SCORED set reuses the same Klar window as ``select_beams`` and the config's solve cutoff
    (``g_max``) as the scoring-resolution cap.
    """
    return TrialCoupling(
        policy=cfg.blochwave.to_policy(),
        scored=ScoredHklSelection(
            klar=cfg.blochwave.to_beam_selection(integration),
            g_max=cfg.blochwave.g_max,
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
    # lock keys on is a flat, fork-free step list -- the fork's shape is static by construction, so
    # its branch is fixed here, before running.
    steps = list(resolve_recipe(steps, base.structure_factor_grid))
    records = step_records(steps)
    # A recipe with an unrecorded (opaque) step cannot be safely identified -> never checkpoint it.
    can_checkpoint = checkpoint and OPAQUE not in records
    recipe = [RecipeStep(name=r.name, params=r.params) for r in records]
    reproducibility_dir = _reproducibility_dir(root)
    npz, lock_path = reproducibility_dir / _PLAN_NPZ, reproducibility_dir / _PLAN_LOCK

    if can_checkpoint and not refresh and lock_path.exists() and npz.exists():
        lock = read_preprocess_lock(lock_path)
        status = preprocess_lock_status(
            lock,
            experiment_lock_sha256=sha256_file(root / "reproducibility" / "experiment.lock"),
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
        experiment_lock_sha256=sha256_file(root / "reproducibility" / "experiment.lock"),
        config_digest=config_digest(cfg),
        code_version=code_version(),
        recipe=recipe,
        plan=artifact_hash_for(npz, root=root),
    )
    write_preprocess_lock(lock_path, lock)
    _log.info("wrote preprocess checkpoint %s + %s", npz.name, lock_path.name)
