"""``fit_orientation``: per-rotation crystal-orientation refinement (Palatinus hexagonal simplex).

A ``Plan -> Plan`` step that sharpens each orientation by minimising the scaling-optimised wR2 of
the full dynamical simulation against the observed pattern -- the objective exposed by
:meth:`~diffBloch.engine.RefinementEngine.score_orientation`. Per rotation it runs the modified
simplex of Palatinus et al. (*Acta Cryst.* A69, 171-188, 2013): a hexagonal search of ``n_steps``
azimuths at a shrinking tilt radius, greedily accepting the first tilt that lowers wR2 and halving
the radius when none does, until the radius falls below ``min_search_angle``.

The trial orientation is ``orientation @ hexagonal_tilt(azimuth, radius)`` -- a right-multiplied
true rotation, so the non-orthonormal ``U`` measured-cell correction is preserved (no
re-orthonormalisation). The captured ``refinement`` is read-only context
the step never mutates; the simulation inside is
deterministic and depends only on its inputs, so it is ordinary computation, not a side effect.

The SCORED reflection set is held fixed at each orientation's seed selection across the search.
With ``coupling=`` the additional SOLVE beams are re-derived per trial, while every scored
reflection is retained in each segment's solve basis. The rocking-curve tilt set carried by the
``Plan`` is
threaded through every trial unchanged, so each candidate is scored under the *same* integration as
the seed -- the fit/eval consistency invariant. Ordering ``integrate_rocking_curve`` before this
step therefore couples the fit to the integrated model; with rocking off the tilt set is a single
identity, identical to a static fit.

**Coupling (opt-in, ``coupling=TrialCoupling(...)``)** re-derives the excitation-selected SOLVE
beams at every trial while pinning the SCORED set to the seed alignment. Pinning is required for a
valid greedy comparison: otherwise a trial can lower wR2 merely by deleting reflections and reach
the degenerate one-reflection ``wr2=0`` minimum. The seed is rebuilt through the same builder, and
the last accepted trial is already the coupled-at-fitted-orientation plan -- no separate
``couple_beams`` step is needed. It is affordable because the atomic ``F_gb`` is
computed once and every segment's structure-factor matrix is a cheap gather-index into it, not a
re-derivation.

With ``coupling=None`` (the default) each trial is ``current.with_orientation(...)``, defined on
both the tilt-independent :class:`OrientationPlan` and the
:class:`~diffBloch.engine.plan.CoupledOrientationPlan`, so an already-segmented plan is fit under
its frozen union.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from diffBloch.core.dynamical import (
    StructureFactorGather,
    build_structure_factor_gather,
    grid_source_indices,
)
from diffBloch.core.solver import FloatFormat, SolverMethod
from diffBloch.engine import RefinementEngine
from diffBloch.engine.plan import CoupledOrientationPlan, OrientationPlanLike, StructureFactorGrid
from diffBloch.observability import (
    NULL_LOGGER,
    Logger,
    OrientationFitSummary,
    OrientationFitted,
)
from diffBloch.params import Device
from diffBloch.preprocess.coupling import build_coupling_segments
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.orientation import hexagonal_tilt
from diffBloch.preprocess.pipeline import PlanStep, as_step
from diffBloch.preprocess.plan import Plan, require_built_plans
from diffBloch.preprocess.scoring import build_engine
from diffBloch.specs import (
    NO_ABSORPTION,
    Absorption,
    HexagonalSearch,
    TrialCoupling,
    assert_grid_covers_coupling,
)

__all__ = ["fit_orientation"]


def fit_orientation(
    refinement: RefinementSetup,
    search: HexagonalSearch,
    *,
    method: SolverMethod = "matrix_exp",
    precision: FloatFormat = "fp64",
    coupling: TrialCoupling | None = None,
    validate: bool = True,
    workers: int = 1,
    device: Device | None = None,
    max_batch: int | None = None,
    logger: Logger = NULL_LOGGER,
    absorption: Absorption = NO_ABSORPTION,
) -> PlanStep:
    """Return a ``Plan -> Plan`` step refining each orientation by Palatinus hexagonal search.

    ``refinement`` (constraint spec, ASU expansion, atomic numbers, seeded params) is captured
    read-only and rejoined to the geometry ``Plan`` via :func:`build_engine`; the
    orientation-invariant ``F_gb`` is computed once and reused across every orientation and trial.
    ``search`` is a pre-validated :class:`~diffBloch.specs.HexagonalSearch` (invalid bounds are
    unrepresentable, so this function never re-validates): ``max_search_angle`` /
    ``min_search_angle`` (degrees) bound the shrinking tilt radius, and ``n_steps`` is the number of
    hexagonal azimuths (6 -> 0, 60, ..., 300 deg). ``method`` configures the
    engine's solver (``score_orientation`` uses a scaling-optimised wR2 internally).

    ``coupling`` (default ``None``) opts the fit into per-trial re-coupling: a
    :class:`~diffBloch.specs.TrialCoupling` re-derives the solve union and re-selects the scored set
    at every trial orientation (see the module docstring for the non-stationary-objective nuance).
    ``None`` keeps the tilt-independent fit (one fixed beam set across the search).

    ``precision`` (default ``"fp64"``) configures the scoring engine's numeric field. ``"fp32"``
    (complex64) roughly halves the per-trial O(N^3) eigensolve for large cells at the cost of a
    coarser, basin-sensitive search -- acceptable here because the fit is re-scored by the
    fp64 terminal; it must never be used for a terminal ``run_inference`` / ``refine``.

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
    trial rebuilds are cheap numpy; only their tensors reach the device). This is the fp32 search's
    device wiring (pair it with ``precision="fp32"`` for the large-cell fork). Kept out of the
    recipe identity like ``workers``/``logger`` -- but unlike those it is not bit-exact: the
    solve shifts ~1e-11 cross-device, and because the greedy search accepts on a threshold, that
    shift can flip a near-tie into a full-radius orientation difference (a well-conditioned fit
    stays; a knife-edge one legitimately diverges). Safe regardless: reproducibility is anchored at
    the checkpoint boundary, so a committed CPU checkpoint is reused (not recomputed) on GPU, cannot
    restale -- only a fresh GPU-computed checkpoint would differ from a CPU one.

    ``workers`` (default 1, sequential) fans the per-rotation searches over a thread pool, the
    private's ``ThreadPoolExecutor(num_workers=8)`` pattern. Rotations are independent, the engine
    and ``F_gb`` are read-only shared context, results keep plan order, and each
    rotation's gather cache is thread-local -- so the results are identical to a sequential run.
    Threads (not processes) suffice because torch's CPU linalg releases the GIL.

    ``max_batch`` (default ``None``) caps the ``matrix_exp`` propagator block; ``None`` lets each
    solve derive a memory-safe block from its beam count. Execution-only and matches the unbounded
    solve to machine precision (memory only), like ``device`` -- raise it to fill a larger GPU. See
    :func:`build_engine`.

    ``logger`` receives an :class:`~diffBloch.observability.OrientationFitted` per rotation as its
    search completes (the fit is the run's long phase, so this is the progress stream); the default
    :data:`NULL_LOGGER` discards them. With ``workers > 1`` events arrive in completion order.

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
            precision=precision,
            max_batch=max_batch,
            absorption=absorption,
        )
        params = refinement.params if device is None else refinement.params.to(device)
        fgb = engine.fgb(params)

        def refine(op: OrientationPlanLike) -> tuple[OrientationPlanLike, float, float, int, int]:
            return _refine_one(
                engine,
                fgb,
                plan,
                op,
                search=search,
                coupling=coupling,
                validate=validate,
            )

        built = require_built_plans(plan)
        results_by_index: dict[
            int, tuple[OrientationPlanLike, float, float, int, int]
        ] = {}
        cap = search.max_iterations

        def report(
            index: int,
            result: tuple[OrientationPlanLike, float, float, int, int],
        ) -> None:
            fitted, wr2, _r_obs, n_trials, n_passes = result
            results_by_index[index] = result
            pattern_index = fitted.alignment.pattern_index
            n_matched = int(pattern_index.shape[0])
            logger.report(
                OrientationFitted(
                    rotation_index=fitted.pattern.rotation_index,
                    wr2=wr2,
                    n_matched_hkl=n_matched,
                    n_trials=n_trials,
                    n_passes=n_passes,
                    pass_cap=cap,
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
        ordered = tuple(result[0] for result in ordered_results)
        fitted_events = [
            (
                result,
                int(result[0].alignment.pattern_index.shape[0]),
                int(
                    (
                        result[0].pattern.intensities[result[0].alignment.pattern_index]
                        > 3.0 * result[0].pattern.sigmas[result[0].alignment.pattern_index]
                    ).sum()
                ),
                int(result[0].pattern.hkl.shape[0]),
            )
            for result in ordered_results
        ]
        logger.report(
            OrientationFitSummary(
                n_orientations=len(fitted_events),
                mean_wr2=sum(item[0][1] for item in fitted_events) / len(fitted_events),
                mean_r_obs=sum(item[0][2] for item in fitted_events) / len(fitted_events),
                total_matched_hkl=sum(item[1] for item in fitted_events),
                total_strong_hkl=sum(item[2] for item in fitted_events),
                total_weak_hkl=sum(item[1] - item[2] for item in fitted_events),
                total_observed_hkl=sum(item[3] for item in fitted_events),
                total_trials=sum(item[0][3] for item in fitted_events),
                max_passes=max(item[0][4] for item in fitted_events),
            )
        )
        return replace(plan, orientations=ordered)

    # search rides in the config digest too, but coupling is a composition-site kwarg (not config),
    # so it MUST be in the recipe identity; workers/logger/device are execution-only (device shifts
    # output only to solver tolerance -- see the docstring -- so it stays out of the identity).
    return as_step(
        "optimize_orientation",
        {"search": search, "coupling": coupling, "absorption": absorption},
        run,
    )


def _refine_one(
    engine: RefinementEngine,
    fgb: Tensor,
    plan: Plan,
    op: OrientationPlanLike,
    *,
    search: HexagonalSearch,
    coupling: TrialCoupling | None,
    validate: bool = True,
) -> tuple[OrientationPlanLike, float, float, int, int]:
    """Palatinus hexagonal search over one orientation.

    Returns ``(fitted, wr2, n_trials, n_passes)``: the best-scoring plan, its final
    scaling-optimised wR2, the number of trial orientations scored, and the number of
    hexagonal-ring sweeps taken (the quantity ``search.max_iterations`` caps -- surfaced so a
    search's cost and its headroom under the cap are observable, e.g. for recalibrating the cap on
    a new compound).

    With ``coupling=None`` each trial is ``current.with_orientation(grid, o)`` -- it rebuilds only
    the orientation-dependent bases and reuses the F-gather, so the beam set, rocking-curve tilts,
    and reduction are held fixed across the search. With a
    :class:`~diffBloch.specs.TrialCoupling`, the seed *and* every trial are rebuilt through
    :func:`_coupled_trial`, which re-couples the solve union and re-selects the scored set at that
    orientation -- the deliberately non-stationary objective (see the module docstring).
    """
    grid = plan.structure_factor_grid
    n_trials = 0
    # One rotation's search revisits the same excitation unions across many nearby trials, so the
    # orientation-free per-segment F-gathers are memoized by beam set for the search's duration.
    # Scoped per rotation: bounded, and trivially thread-safe if rotations ever fit in parallel.
    gather_cache: dict[bytes, StructureFactorGather] = {}
    if coupling is None:
        current = op
    else:
        current = _coupled_trial(
            grid,
            op,
            np.asarray(op.orientation, dtype=np.float64),
            coupling,
            gather_cache,
            validate=validate,
        )
    current_score = float(engine.score_orientation(current, fgb))
    search_angle = search.max_search_angle
    n_passes = 0
    for _ in range(search.max_iterations):
        if search_angle <= search.min_search_angle:
            r_obs = float(engine.score_orientation_r_obs(current, fgb))
            return current, current_score, r_obs, n_trials, n_passes
        n_passes += 1  # noqa: SIM113 -- not enumerate: the floor-check iteration above returns before this, so this counts executed sweeps only
        improved = False
        for n in range(search.n_steps):
            azimuth = n * 360.0 / search.n_steps  # hexagonal: 0, 60, ..., 300 deg at n_steps = 6
            orientation = np.asarray(current.orientation, dtype=np.float64) @ hexagonal_tilt(
                azimuth, search_angle
            )
            if coupling is None:
                trial = current.with_orientation(grid, orientation)
            else:
                trial = _coupled_trial(
                    grid, op, orientation, coupling, gather_cache, validate=validate
                )
            n_trials += 1
            trial_score = float(engine.score_orientation(trial, fgb))
            accepted = trial_score < current_score
            if accepted:  # greedy first-improvement; restart at this radius
                current, current_score = trial, trial_score
                improved = True
                break
        if not improved:
            search_angle /= 2.0
    raise RuntimeError(
        f"fit_orientation search did not converge within {search.max_iterations} iterations"
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
    """Re-couple the solve union while pinning the scored set at ``orientation`` (one trial).

    One objective evaluation: (1) ``build_coupling_segments`` re-derives
    the per-tilt-segment excitation union at ``orientation`` (the changing SOLVE set); (2) the seed
    alignment is added to every segment and retained as the fixed SCORED set. Thus every trial's
    wR2 compares the same observations. The observed ``pattern``, ``thickness``, and
    ``tilt_reduction`` are carried from ``op`` unchanged. The atomic ``F_gb`` is
    untouched either way; ``gather_cache`` (keyed by a segment's beam-set bytes) reuses the
    orientation-free per-segment F-gathers across a search's trials -- identical beam set, identical
    gather -- collapsing the per-trial rebuild cost.

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
    scored_hkl = np.asarray(op.alignment.hkl, dtype=np.int64)
    # A trial must not improve merely by dropping scored reflections. Keep the seed objective
    # domain fixed and include it in every segment so scored ⊆ solved holds throughout the curve.
    segments = tuple(
        replace(
            segment,
            union_hkl=np.unique(
                np.concatenate([segment.union_hkl, scored_hkl]),
                axis=0,
            ),
        )
        for segment in segments
    )
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
