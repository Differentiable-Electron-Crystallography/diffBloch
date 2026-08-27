"""``optimize_orientation``: per-rotation crystal-orientation refinement.

A ``Plan -> Plan`` step that sharpens each orientation by minimising the scaling-optimised wR2 of
the full dynamical simulation against the observed pattern -- the objective exposed by
:meth:`~diffBloch.engine.RefinementEngine.score_orientation`. A local
``scipy.optimize.minimize(method="Nelder-Mead")`` simplex search over the three
goniometer-correction angles directly, seeded from a fixed initial simplex of edge length
``search.step_size`` around ``(alpha, beta, omega) = (0, 0, 0)`` (see :func:`_refine_one`).

The captured ``refinement`` is read-only context the step never mutates; the simulation inside is
deterministic and depends only on its inputs, so it is ordinary computation, not a side effect.

Without ``coupling=``, the SCORED reflection set is held fixed at each orientation's seed
selection across the search (:meth:`~diffBloch.engine.plan.OrientationPlan.with_orientation` only
recomputes the orientation-dependent bases). The rocking-curve tilt set carried by the ``Plan`` is
threaded through every trial unchanged, so each candidate is scored under the *same* integration as
the seed -- the optimize/eval consistency invariant. Ordering ``integrate_rocking_curve`` before this
step therefore couples the search to the integrated model; with rocking off the tilt set is a single
identity, identical to a static search.

**Coupling (opt-in, ``coupling=TrialCoupling(...)``)** re-derives the excitation-selected SOLVE
beams *and* re-selects the SCORED set at every trial's own orientation (``coupling.scored``'s Klar
rsg/dsg window + resolution cap, reapplied to that trial's fresh lab-frame geometry) -- mirroring
the reference implementation's per-trial ``filter_hkls``, rather than pinning scoring to the seed's
selection. A trial's matched-reflection count can therefore differ from the seed's and from other
trials'; the default wR2 formula itself has no reflection-count term
(:func:`diffBloch.core.losses.w_rbragg`: plain ``sum/sum``), so the search can in principle prefer
a trial that improves partly by matching a different, easier subset --
``search.penalize_fewer_reflections`` (:class:`~diffBloch.specs.NelderMeadSearch`, off by default)
guards against exactly this by dividing the comparison score by the matched count
(:func:`_comparable_score`); the reported ``score`` stays the plain metric regardless. The seed is
rebuilt through the same builder, and the last accepted trial is already the
coupled-at-optimized-orientation plan -- no separate ``couple_beams`` step is needed. Atomic ``F_gb``
values are cached lazily by support-grid row: each trial computes only previously unseen beam
differences, while every segment's structure matrix remains a cheap gather-index into that cache.

With ``coupling=None`` (the default) each trial is ``current.with_orientation(...)``, defined on
both the tilt-independent :class:`OrientationPlan` and the
:class:`~diffBloch.engine.plan.CoupledOrientationPlan`, so an already-segmented plan is optimized
under its frozen union.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import minimize
from torch import Tensor

from diffBloch.core.crystal import orientation_basis
from diffBloch.core.dynamical import (
    StructureFactorGather,
    build_structure_factor_gather,
    grid_source_indices,
)
from diffBloch.core.reciprocal import g_vectors
from diffBloch.core.scattering import ScatteringFactorModel
from diffBloch.core.solver import SolverMethod
from diffBloch.engine import RefinementEngine, ScoresFn, wr2_scores
from diffBloch.engine.plan import CoupledOrientationPlan, OrientationPlanLike, StructureFactorGrid
from diffBloch.observability import (
    NULL_LOGGER,
    Logger,
    OrientationOptimizationStarted,
    OrientationOptimizationSummary,
    OrientationOptimized,
)
from diffBloch.params import Device
from diffBloch.preprocess.coupling import build_coupling_segments
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.orientation import goniometer_rotation
from diffBloch.preprocess.pipeline import PlanStep, as_step
from diffBloch.preprocess.plan import (
    Plan,
    dataset_of,
    require_built_plans,
    unique_hkl_count,
)
from diffBloch.preprocess.scoring import active_structure_factor_indices, build_engine
from diffBloch.preprocess.steps.beams import klar_beam_mask
from diffBloch.specs import (
    NO_ABSORPTION,
    Absorption,
    NelderMeadSearch,
    TrialCoupling,
    assert_grid_covers_coupling,
)

__all__ = ["optimize_orientation"]


def optimize_orientation(
    refinement: RefinementSetup,
    search: NelderMeadSearch,
    *,
    method: SolverMethod = "matrix_exp",
    coupling: TrialCoupling | None = None,
    validate: bool = True,
    workers: int = 1,
    device: Device | None = None,
    max_batch: int | None = None,
    logger: Logger = NULL_LOGGER,
    absorption: Absorption = NO_ABSORPTION,
    scattering_factors: ScatteringFactorModel = "lobato2026",
    scores: ScoresFn = wr2_scores,
    residual: str = "wr2",
) -> PlanStep:
    """Return a ``Plan -> Plan`` step refining each orientation by orientation search.

    ``residual`` (default ``"wr2"``) is the display name for ``scores`` -- pass
    ``cfg.loss_metrics.residual`` alongside ``scores=cfg.loss_metrics.to_scores()`` so
    :class:`~diffBloch.observability.OrientationOptimized` reports the score under its real name.

    ``scores`` (default :func:`~diffBloch.engine.wr2_scores`) is the per-thickness metric
    :meth:`~diffBloch.engine.forward.RefinementEngine.score_orientation` searches -- pass
    ``cfg.loss_metrics.to_scores()`` to search the same residual the gradient refinement stage
    minimises (:func:`~diffBloch.config.schema.LossMetricsConfig.to_scores`). Execution-only like
    ``method``: it changes what the search optimizes for, not the recipe's own identity (the
    resolved ``ExperimentConfig.loss_metrics`` already rides in :func:`~diffBloch.config.manifest.dataset_config_digest`).

    ``refinement`` (constraint spec, ASU expansion, atomic numbers, seeded params) is captured
    read-only and rejoined to the geometry ``Plan`` via :func:`build_engine`; the
    orientation-invariant ``F_gb`` is computed once and reused across every orientation and trial.
    ``search`` is a pre-validated :class:`~diffBloch.specs.NelderMeadSearch` (invalid bounds are
    unrepresentable, so this function never re-validates). ``method`` configures the engine's
    solver (``score_orientation`` scores with ``scores``, a scaling-optimised wR2 by default).

    ``coupling`` (default ``None``) opts the optimization into per-trial re-coupling: a
    :class:`~diffBloch.specs.TrialCoupling` re-derives the solve union and re-selects the scored set
    at every trial orientation (see the module docstring for the non-stationary-objective nuance).
    ``None`` keeps the tilt-independent search (one fixed beam set across the search).

    ``validate`` (default ``True``) forwards to the per-trial coupled gather rebuild
    (:func:`~diffBloch.core.dynamical.build_structure_factor_gather`). ``False`` skips its O(N^2)
    integrity checks -- the dominant per-trial cost over a large coupled union -- for the large-cell
    fast path. It is sound **only** because the coupled coverage guard (above) proves the grid
    spans the beam-difference support; without that guarantee a skipped check would let a gather
    silently read zeros. Inert unless ``coupling`` is set (the tilt-independent path rebuilds no
    gather in the search), and, like the coverage guard, it does not enter the recipe identity: the
    checks are pure, so ``False`` yields identical gather indices when coverage holds.

    ``device`` (default ``None`` = CPU) places the search's forward solve on the given accelerator:
    the seed params are moved there and ``engine.fgb`` is computed on-device, so every per-trial
    ``score_orientation`` co-locates onto the param-derived ``fgb.device`` at the use site (the CPU
    trial rebuilds are cheap numpy; only their tensors reach the device). Kept out of the
    recipe identity like ``workers``/``logger`` -- but unlike those it is not bit-exact: the
    solve shifts ~1e-11 cross-device, and because the greedy search accepts on a threshold, that
    shift can flip a near-tie into a full-radius orientation difference (a well-conditioned
    optimization stays; a knife-edge one legitimately diverges). Safe regardless: reproducibility is anchored at
    the checkpoint boundary, so a committed CPU checkpoint is reused (not recomputed) on GPU, cannot
    restale -- only a fresh GPU-computed checkpoint would differ from a CPU one.

    ``workers`` (default 1, sequential) fans the per-rotation searches over a thread pool.
    Rotations are independent, the engine and ``F_gb`` are read-only shared context, results keep
    plan order, and each
    rotation's gather cache is thread-local -- so the results are identical to a sequential run.
    Threads (not processes) suffice because torch's CPU linalg releases the GIL.

    ``max_batch`` (default ``None``) caps the ``matrix_exp`` propagator block; ``None`` lets each
    solve derive a memory-safe block from its beam count. Execution-only and matches the unbounded
    solve to machine precision (memory only), like ``device`` -- raise it to fill a larger GPU. See
    :func:`build_engine`.

    ``logger`` receives an :class:`~diffBloch.observability.OrientationOptimized` per rotation as its
    search completes (the optimization is the run's long phase, so this is the progress stream); the
    default :data:`NULL_LOGGER` discards them. With ``workers > 1`` events arrive in completion order.

    The greedy search restarts at the same radius on every accepting (improving) pass, so the
    radius schedule alone does not bound the pass count. Mirroring :func:`iterate_until`,
    ``search.max_iterations`` caps the total passes *per orientation* and a ``RuntimeError`` is
    raised if it is reached -- silent non-convergence is never returned.

    The cap is a runaway guard: the search terminates by construction for a
    non-degenerate objective (monotone wR2 descent + the radius floor), so the cap only guards
    pathological ridge-walking on (near-)degenerate landscapes. Its default of ``2000`` is
    **calibrated on the quartz anchor under the integrated recipe** (slowest legitimate search: 1288
    passes across 99 rotations, so 2000 has headroom); raise it via config if a dataset with
    shallower minima trips it.
    """

    if workers < 1:
        raise ValueError("workers must be >= 1")

    def run(plan: Plan) -> Plan:
        # Coverage guard for the coupled path: the grid sphere must span the beam-difference
        # support so a per-trial gather cannot address a reflection outside it -- the precondition
        # for running those gathers validate=False (their silent-zero gap has no runtime backstop).
        # O(1), orientation-independent, always on -- fails at setup, not deep in the search.
        if coupling is not None:
            assert_grid_covers_coupling(coupling.policy, plan.structure_factor_grid.g_max)
        engine = build_engine(
            plan,
            refinement,
            method=method,
            max_batch=max_batch,
            absorption=absorption,
            scattering_factors=scattering_factors,
            scores=scores,
        )
        params = refinement.params if device is None else refinement.params.to(device)
        fgb = engine.fgb(params)

        def refine(op: OrientationPlanLike) -> _FitResult:
            trial_fgb: Tensor | Callable[[OrientationPlanLike], Tensor] = fgb
            if coupling is not None:
                cached = fgb.clone()
                known = torch.zeros(cached.shape[0], dtype=torch.bool, device=cached.device)
                assert engine.active_structure_factor_indices is not None
                known[engine.active_structure_factor_indices.to(device=known.device)] = True

                def lazy_fgb(candidate: OrientationPlanLike) -> Tensor:
                    indices = active_structure_factor_indices(
                        (candidate,), plan.structure_factor_grid.gpts
                    ).to(device=cached.device)
                    missing = indices[~known.index_select(0, indices)]
                    if missing.numel():
                        with torch.no_grad():
                            values = engine.structure_factor_values(params, missing)
                            cached.index_copy_(0, missing, values)
                            known[missing] = True
                    return cached

                trial_fgb = lazy_fgb
            return _refine_one(
                engine,
                trial_fgb,
                plan,
                op,
                search=search,
                coupling=coupling,
                validate=validate,
            )

        built = require_built_plans(plan)
        # This step runs once per dataset, before a multi-dataset pool renumbers anything, so its
        # rotation_index is file-local and cannot disambiguate a pooled report on its own. The ref
        # rides on each rotation's own pattern (stamped at setup_datasets), so it is read off the
        # plan rather than passed in -- a recipe step takes no dataset argument.
        dataset = dataset_of(built)
        logger.report(OrientationOptimizationStarted(total_rotations=len(built), dataset=dataset))
        results_by_index: dict[int, _FitResult] = {}
        cap = search.max_iterations

        def report(index: int, result: _FitResult) -> None:
            results_by_index[index] = result
            pattern_index = result.plan.alignment.pattern_index
            n_matched = int(pattern_index.shape[0])
            logger.report(
                OrientationOptimized(
                    rotation_index=result.plan.pattern.rotation_index,
                    score=result.score,
                    seed_score=result.seed_score,
                    alpha=result.alpha,
                    beta=result.beta,
                    omega=result.omega,
                    residual=residual,
                    n_matched_hkl=n_matched,
                    n_trials=result.n_trials,
                    n_passes=result.n_passes,
                    pass_cap=cap,
                    dataset=dataset,
                )
            )

        if workers > 1:
            pool = ThreadPoolExecutor(max_workers=workers)
            try:
                futures = {pool.submit(refine, op): index for index, op in enumerate(built)}
                for future in as_completed(futures):  # emit progress as searches finish
                    index = futures[future]
                    report(index, future.result())
            finally:
                # An early abort surfaces as a logger.report() raising mid-loop. Cancel the
                # not-yet-started searches and don't block on the in-flight ones, so the abort
                # actually saves the remaining budget instead of draining every queued rotation
                # first (the whole point of stopping early). The `with`-block's default
                # shutdown(wait=True) would run them all before the exception surfaced. Rotations
                # already executing cannot be interrupted, so up to `workers` still finish; the
                # queued remainder is dropped. On normal completion nothing is pending -- a no-op.
                pool.shutdown(wait=False, cancel_futures=True)
        else:
            for index, op in enumerate(built):
                report(index, refine(op))
        ordered_results = tuple(results_by_index[i] for i in range(len(built)))
        ordered = tuple(result.plan for result in ordered_results)

        def strong_matched_hkl(op: OrientationPlanLike) -> Tensor:
            pattern_index = op.alignment.pattern_index
            strong = op.pattern.intensities[pattern_index] > 3.0 * op.pattern.sigmas[pattern_index]
            return op.alignment.hkl[strong]

        logger.report(
            OrientationOptimizationSummary(
                n_orientations=len(ordered_results),
                mean_score=sum(result.score for result in ordered_results) / len(ordered_results),
                residual=residual,
                unique_matched_hkl=unique_hkl_count(op.alignment.hkl for op in ordered),
                unique_strong_hkl=unique_hkl_count(strong_matched_hkl(op) for op in ordered),
                unique_observed_hkl=unique_hkl_count(op.pattern.hkl for op in ordered),
                total_trials=sum(result.n_trials for result in ordered_results),
                max_passes=max(result.n_passes for result in ordered_results),
            )
        )
        return replace(plan, orientations=ordered)

    # search rides in the config digest too, but coupling is a composition-site kwarg (not config),
    # so it MUST be in the recipe identity; workers/logger/device are execution-only (device shifts
    # output only to solver tolerance -- see the docstring -- so it stays out of the identity).
    return as_step(
        "optimize_orientation",
        {
            "search": search,
            "coupling": coupling,
            "absorption": absorption,
            "scattering_factors": scattering_factors,
        },
        run,
    )


def _comparable_score(score: float, plan: OrientationPlanLike, search: NelderMeadSearch) -> float:
    """``score``, or ``score`` normalised by matched-reflection count when the search opts in.

    A trial's SCORED set can vary in size (:func:`_coupled_trial` re-selects it per trial, and the
    residual metrics have no term correcting for that), so comparing the raw ``score`` lets a trial
    win by drifting to geometry that matches a smaller, easier subset rather than by optimizing
    better. ``search.penalize_fewer_reflections`` (default ``False``, matching the reference
    implementation's own disabled-by-default normalisation) opts into dividing by the matched
    count: a trial with fewer matched reflections then needs a proportionally lower ``score`` to be
    accepted. ``0`` matched reflections scores ``inf`` (never accepted; ``score`` itself is
    typically ``nan`` there already). With the flag off this is the identity on ``score``.
    """
    if not search.penalize_fewer_reflections:
        return score
    n_matched = int(plan.alignment.pattern_index.shape[0])
    return score / n_matched if n_matched > 0 else float("inf")


class _FitResult(NamedTuple):
    """One rotation's finished search: the fitted plan plus everything :class:`OrientationOptimized`
    reports about it."""

    plan: OrientationPlanLike
    score: float
    n_trials: int
    n_passes: int
    alpha: float
    beta: float
    omega: float
    seed_score: float


def _refine_one(
    engine: RefinementEngine,
    fgb: Tensor | Callable[[OrientationPlanLike], Tensor],
    plan: Plan,
    op: OrientationPlanLike,
    *,
    search: NelderMeadSearch,
    coupling: TrialCoupling | None,
    validate: bool = True,
) -> _FitResult:
    """Local Nelder-Mead search over the goniometer correction ``(alpha, beta, omega)``.

    Every trial composes directly off the fixed seed orientation: ``seed_orientation @
    goniometer_rotation(alpha, beta, omega)`` -- the search explores the whole ``step_size``
    neighbourhood at once rather than annealing a step down. ``scipy.optimize.minimize`` runs
    ``method="Nelder-Mead"`` from a fixed initial simplex of edge length ``search.step_size``
    around ``(alpha, beta, omega) = (0, 0, 0)``, exactly mirroring the reference implementation
    this port is checked against. ``n_passes`` is scipy's reported iteration count (``result.nit``).

    ``seed_score`` is the same metric evaluated once more at the unsearched seed orientation
    (``alpha = beta = omega = 0``) -- one extra forward solve on top of the search's own trials, paid
    so the report can state what the search actually bought (``seed_score - score``) instead of only
    the post-search value.
    """
    grid = plan.structure_factor_grid
    n_trials = 0
    # One rotation's search revisits the same excitation unions across many nearby trials, so the
    # beam set across this rotation's trials.
    gather_cache: dict[bytes, StructureFactorGather] = {}
    seed_orientation = np.asarray(op.orientation, dtype=np.float64)

    def build_trial(orientation: NDArray[np.float64]) -> OrientationPlanLike:
        if coupling is None:
            return op.with_orientation(grid, orientation)
        return _coupled_trial(grid, op, orientation, coupling, gather_cache, validate=validate)

    seed_trial = build_trial(seed_orientation)
    n_trials += 1
    seed_trial_fgb = fgb(seed_trial) if callable(fgb) else fgb
    seed_score = float(engine.score_orientation(seed_trial, seed_trial_fgb))

    def objective(params: NDArray[np.float64]) -> float:
        nonlocal n_trials
        alpha, beta, omega = params
        orientation = seed_orientation @ goniometer_rotation(alpha, beta, omega)
        trial = build_trial(orientation)
        n_trials += 1
        trial_fgb = fgb(trial) if callable(fgb) else fgb
        raw = float(engine.score_orientation(trial, trial_fgb))
        # scipy minimises this directly, so the (opt-in) fewer-reflections guard lives here: a
        # trial cannot win merely by drifting to geometry that matches a smaller, easier subset.
        return _comparable_score(raw, trial, search)

    step = search.step_size
    initial_simplex = np.array(
        [
            [0.0, 0.0, 0.0],
            [step, 0.0, 0.0],
            [0.0, step, 0.0],
            [0.0, 0.0, step],
        ]
    )
    result = minimize(
        objective,
        x0=np.zeros(3),
        method="Nelder-Mead",
        options={
            "initial_simplex": initial_simplex,
            "maxiter": search.max_iterations,
            "xatol": search.x_tolerance,
            "fatol": search.f_tolerance,
        },
    )
    alpha, beta, omega = result.x
    best_orientation = seed_orientation @ goniometer_rotation(alpha, beta, omega)
    current = build_trial(best_orientation)
    n_trials += 1
    current_fgb = fgb(current) if callable(fgb) else fgb
    # result.fun is the comparable (penalized) score minimised above; report the plain score
    # instead (self.scores, under whichever residual ExperimentConfig.loss_metrics configures).
    score = float(engine.score_orientation(current, current_fgb))
    return _FitResult(
        plan=current,
        score=score,
        n_trials=n_trials,
        n_passes=int(result.nit),
        alpha=float(alpha),
        beta=float(beta),
        omega=float(omega),
        seed_score=seed_score,
    )


def _coupled_trial(
    grid: StructureFactorGrid,
    op: OrientationPlanLike,
    orientation: NDArray[np.float64],
    coupling: TrialCoupling,
    gather_cache: dict[bytes, StructureFactorGather] | None = None,
    *,
    validate: bool = True,
) -> CoupledOrientationPlan:
    """Re-couple the solve union and re-select the scored set, both at ``orientation`` (one trial).

    One objective evaluation: (1) ``build_coupling_segments`` re-derives the per-tilt-segment
    excitation union at ``orientation`` (the SOLVE set); (2) ``coupling.scored`` (the Klar
    rsg/dsg window + resolution cap) is re-applied to that fresh union's own lab-frame geometry at
    this orientation, giving the SCORED set its own fresh selection every trial -- mirroring the
    reference implementation's per-trial ``filter_hkls`` -- rather than pinning it to the seed's.
    The observed ``pattern``, ``thickness``, and ``tilt_reduction`` are carried from ``op``
    unchanged. The atomic ``F_gb`` is untouched either way; ``gather_cache`` (keyed by a segment's
    beam-set bytes) reuses the orientation-free per-segment F-gathers across a search's trials --
    identical beam set, identical gather -- collapsing the per-trial rebuild cost.

    ``validate`` (default ``True``) is forwarded to each cache-miss
    :func:`~diffBloch.core.dynamical.build_structure_factor_gather`; ``False`` skips its O(N^2)
    integrity checks on the hot path (safe under the caller's coverage guard). It reaches only that
    build -- the ``CoupledOrientationPlan.build`` below always receives the precomputed
    ``gathers``, so its own ``validate`` never triggers a rebuild here.
    """
    cell = np.asarray(grid.cell, dtype=np.float64)
    tilts = np.asarray(op.tilts, dtype=np.float64)
    segments = build_coupling_segments(
        coupling.policy,
        np.asarray(grid.structure_factor_hkl, dtype=np.int64),
        cell=cell,
        orientation=orientation,
        tilts=tilts,
        energy=op.energy,
        u0=op.u0,
    )
    union_hkl = np.unique(np.concatenate([segment.union_hkl for segment in segments]), axis=0)
    basis = orientation_basis(cell, orientation)
    g = g_vectors(union_hkl, basis)
    keep = klar_beam_mask(
        g,
        energy=op.energy,
        u0=op.u0,
        rsg=coupling.scored.klar.rsg,
        dsg=coupling.scored.klar.dsg,
        semiangle=coupling.scored.klar.integration.semiangle,
        geometry=coupling.scored.klar.integration.geometry,
    )
    keep &= np.linalg.norm(g, axis=1) <= coupling.scored.g_max
    keep |= (union_hkl == 0).all(axis=1)  # 000 anchors psi0; retained when present
    scored_hkl = union_hkl[keep]
    gathers = None
    if gather_cache is not None:
        structure_factor_hkl = np.asarray(grid.structure_factor_hkl)
        # The gather's grid-side ravel is identical for every segment (same
        # structure_factor_hkl/gpts), so build
        # it once here and reuse across this trial's segment builds -- re-raveling the support grid
        # per segment was the residual per-trial cost after the |g|<cap pre-filter. Lazy: only paid
        # when a segment actually misses the cache (an unchanged union rebuilds nothing).
        source: NDArray[np.int64] | None = None
        gathers = []
        for segment in segments:
            key = np.ascontiguousarray(segment.union_hkl).tobytes()
            gather = gather_cache.get(key)
            if gather is None:
                if source is None:
                    source = grid_source_indices(structure_factor_hkl, grid.gpts)
                gather = build_structure_factor_gather(
                    structure_factor_hkl,
                    segment.union_hkl,
                    grid.gpts,
                    validate=validate,
                    structure_factor_indices=source,
                )
                gather_cache[key] = gather
            gathers.append(gather)
    return CoupledOrientationPlan.build(
        grid,
        [(segment.union_hkl, segment.covered_tilt_indices) for segment in segments],
        op.pattern,
        energy=op.energy,
        thickness=op.thickness,
        u0=op.u0,
        orientation=orientation,
        tilts=tilts,
        tilt_reduction=op.tilt_reduction,
        scored_hkl=scored_hkl,
        gathers=gathers,
    )
