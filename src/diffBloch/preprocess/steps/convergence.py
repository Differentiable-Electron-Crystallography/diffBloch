"""Convergence testing: grow a simulation-accuracy knob until the diffraction pattern stops moving.

A convergence sweep is *self-referential* -- unlike ``fit_orientation`` / ``fit_thickness`` (which
match the simulation to *observed* data), it asks whether two *consecutive* simulations still
differ,
so it is a numerical resolution study (has the calculation stopped depending on the knob?), run
before and orthogonally to the accuracy fit. This module has three layers:

- :func:`simulation_rfactor` -- the measurement: ``(previous, current) -> float``, the mean
  per-orientation R-factor between two Plans' simulations (0 when they are identical).
- :func:`converge_scalar` -- the parameter-agnostic driver: given a ``build(value) -> object`` and a
  ``measure`` it clicks a scalar knob upward until two consecutive builds stop changing (the
  R-factor drops below threshold), or a hard cap raises. It knows nothing about beams (or even
  Plans); adapters instantiate it.
- :func:`converge_beams` -- the beam-window adapter: a ``Plan -> Plan`` step that widens
  ``integration_semiangle`` until the pattern stabilises, re-running ``select_beams`` from the seed.
- :func:`converge_pool` -- the coupled second beam lever: widens the ``g_max_refine`` candidate
  pool, re-seeding each orientation from the shared grid, until the pattern stabilises. It is the
  standalone lever; the joint window+pool fixpoint needs the driver to thread their shared scalar
  state (the naive ``iterate_until(pipeline([...]))`` does not compose -- see ``converge_pool``).
- :func:`converge_sampling` -- the forward-model lever: refines the rocking-curve tilt count
  (``rocking_curve_sampling``) until the integrated pattern stabilises. Independent of the beam
  levers (the tilt count does not touch the ``Fgb`` support), so it composes as an ordinary lever.

:func:`simulation_converged` wraps :func:`simulation_rfactor` with a threshold to give the boolean
:data:`~diffBloch.preprocess.pipeline.ConvergenceCheck` that
:func:`~diffBloch.preprocess.pipeline.iterate_until` drives to a fixpoint.

The convergence sweep grows each numeric knob (beam-pool radius, excitation window, tilt count) and
stops the first time the per-orientation Bragg R-factor between consecutive simulations drops below
``r_factor_threshold`` -- a deliberately simple stopping rule (no patience, no skip-null).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import torch
from torch import Tensor

from diffBloch.core.losses import optimal_scale, rbragg
from diffBloch.core.products import BlochSolution
from diffBloch.core.solver import SolverMethod
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.pipeline import ConvergenceCheck, PlanStep
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.scoring import build_engine
from diffBloch.preprocess.steps.beams import (
    build_orientation_plans,
    reseed_pool,
    select_beams,
)
from diffBloch.preprocess.steps.rocking_curve import integrate_rocking_curve
from diffBloch.specs import BeamSelection, ConvergenceTolerance, RockingCurve

__all__ = [
    "converge_beams",
    "converge_pool",
    "converge_sampling",
    "converge_scalar",
    "simulation_converged",
    "simulation_rfactor",
]

# ``(previous, current) -> mean consecutive-simulation R-factor``.
type SimulationRfactor = Callable[[Plan, Plan], float]

# The R-factor compares two simulations, not simulation-vs-data, so there is no measurement noise to
# weight by; a near-zero sigma makes ``rbragg`` effectively unweighted while keeping its
# ``I > 3*sigma`` mask inclusive.
_UNWEIGHTED_SIGMA = 1e-10


def simulation_rfactor(
    refinement: RefinementSetup,
    *,
    method: SolverMethod = "matrix_exp",
) -> SimulationRfactor:
    """Return ``(previous, current) -> float``: the mean consecutive-simulation R-factor.

    ``refinement`` (the read-only structure context) is captured and rejoined to each Plan via
    :func:`build_engine`; ``method`` configures the solver. The returned measure simulates both
    Plans, computes the scale-optimised ``rbragg`` R-factor between them on each orientation's
    shared
    reflections, and averages over orientations. The comparison is a control-flow decision, not a
    gradient path, so the simulated intensities are detached. It is 0 exactly when the two Plans
    produce identical simulations (the *null*-step signal the sweep skips).

    The two Plans must describe the same orientations in the same order (a convergence step rebuilds
    each orientation, changing only its beam set), and each pair must share at least one reflection
    (the retained 000 guarantees this in practice).
    """

    def measure(previous: Plan, current: Plan) -> float:
        previous_solutions = _simulate(previous, refinement, method)
        current_solutions = _simulate(current, refinement, method)
        if len(previous_solutions) != len(current_solutions):
            raise ValueError("convergence check requires the two Plans to share their orientations")
        r_factors = [
            _orientation_rfactor(prev, curr)
            for prev, curr in zip(previous_solutions, current_solutions, strict=True)
        ]
        return float(np.mean(r_factors))

    return measure


def simulation_converged(
    refinement: RefinementSetup,
    tolerance: ConvergenceTolerance,
    *,
    method: SolverMethod = "matrix_exp",
) -> ConvergenceCheck:
    """Return a ``(previous, current) -> bool`` check: have consecutive simulations stabilised?

    Thin threshold wrapper over :func:`simulation_rfactor`: the mean per-orientation R-factor is
    compared against ``tolerance.r_factor_threshold``. This is the boolean
    :data:`~diffBloch.preprocess.pipeline.ConvergenceCheck` that
    :func:`~diffBloch.preprocess.pipeline.iterate_until` drives to a fixpoint (the cross-lever
    composition); :func:`converge_scalar` uses the underlying float measure directly, applying the
    same threshold inline.
    """
    measure = simulation_rfactor(refinement, method=method)

    def check(previous: Plan, current: Plan) -> bool:
        return measure(previous, current) < tolerance.r_factor_threshold

    return check


def converge_scalar[T](
    build: Callable[[float], T],
    measure: Callable[[T, T], float],
    tolerance: ConvergenceTolerance,
    *,
    start: float,
    step: float,
) -> T:
    """Grow a scalar knob until two consecutive builds stop changing; return the converged object.

    The parameter-agnostic convergence driver -- it knows nothing about beams or Plans.
    ``build(value)`` rebuilds the object at a knob value; ``measure(previous, candidate)`` is the
    consecutive-output R-factor (0 when identical). Starting from ``start`` and clicking by ``step``
    each iteration, it returns the first candidate whose R-factor against the previous build is
    below ``tolerance.r_factor_threshold``. The stopping rule is deliberately simple -- the first
    dip stops the sweep, and an unchanged build (R = 0) counts as converged -- with no patience and
    no null-step handling. Raises
    ``RuntimeError`` if ``tolerance.max_iterations`` steps pass without a dip below threshold
    (silent non-convergence is never returned, matching
    :func:`~diffBloch.preprocess.pipeline.iterate_until`).
    """
    current = build(start)
    value = start
    for _ in range(tolerance.max_iterations):
        value += step
        candidate = build(value)
        r = measure(current, candidate)
        current = candidate
        if r < tolerance.r_factor_threshold:
            return current
    raise RuntimeError(f"converge_scalar did not converge within {tolerance.max_iterations} steps")


def converge_beams(
    selection: BeamSelection,
    refinement: RefinementSetup,
    tolerance: ConvergenceTolerance,
    *,
    step: float,
    method: SolverMethod = "matrix_exp",
) -> PlanStep:
    """Return a ``Plan -> Plan`` step: widen ``integration_semiangle`` until the pattern stabilises.

    The window lever of beam-set convergence (the physically primary "how many near-Ewald beams").
    Each candidate re-runs :func:`~diffBloch.preprocess.steps.beams.select_beams` from the incoming
    *seed*
    Plan at a wider ``integration_semiangle`` -- selecting from the fixed seed each time, not from
    the
    previous (already-pruned) candidate, so widening can admit beams a narrower window dropped. The
    sweep starts at ``selection.integration.semiangle`` and clicks up by ``step`` (degrees) until
    :func:`converge_scalar` settles the pattern (first sub-threshold step wins); ``rsg`` / ``dsg``
    are
    held fixed. ``step`` must be positive.
    """
    if step <= 0.0:
        raise ValueError("step must be positive")
    measure = simulation_rfactor(refinement, method=method)

    def run(seed: Plan) -> Plan:
        def build(semiangle: float) -> Plan:
            geometry = replace(selection.integration, semiangle=semiangle)
            selected = select_beams(replace(selection, integration=geometry))(seed)
            return build_orientation_plans()(selected)

        return converge_scalar(
            build, measure, tolerance, start=selection.integration.semiangle, step=step
        )

    return run


def converge_pool(
    selection: BeamSelection,
    refinement: RefinementSetup,
    tolerance: ConvergenceTolerance,
    *,
    start_g_max_refine: float,
    step: float,
    method: SolverMethod = "matrix_exp",
) -> PlanStep:
    """Return a ``Plan -> Plan`` step: widen the ``g_max_refine`` pool until the pattern settles.

    The pool lever of beam-set convergence -- the second, coupled beam knob (the window lever is
    :func:`converge_beams`). Each candidate re-seeds every orientation's *candidate* reflections
    from the shared grid at a wider ``g_max_refine``
    (:func:`~diffBloch.preprocess.experiment.seed_beam_hkl`), rebuilds each
    :class:`~diffBloch.engine.plan.OrientationPlan` on that seed, then re-applies the
    fixed Klar window via :func:`~diffBloch.preprocess.steps.beams.select_beams` -- so the active
    set is
    ``seed(g_max_refine) intersect Klar-window(selection)`` at each step. The sweep starts at
    ``start_g_max_refine`` and clicks up by ``step`` until :func:`converge_scalar` settles the
    pattern (first sub-threshold step wins); it settles when the widened pool stops admitting beams
    the
    window keeps. Re-seeding from the fixed grid (not the previous pruned set) lets widening recover
    beams a narrower pool clipped, mirroring :func:`converge_beams`.

    ``step`` must be positive. The pool stays inside the existing ``Fgb`` difference support while
    ``2 * g_max_refine <= grid.g_max``; growing past that needs a dependent grid-resize that is not
    implemented (the anchor never reaches it -- ``g_max_refine`` 1.6 vs ``g_max`` 4.5), so a
    candidate that would exceed the grid raises rather than silently truncating.

    This is the *standalone* pool lever. The joint window+pool fixpoint is **not** a naive
    ``iterate_until(pipeline([converge_beams, converge_pool]))``: :func:`converge_beams` re-selects
    from an *unpruned* seed while this step *emits a window-pruned* Plan, and the two levers share
    scalar state (the window ``integration_semiangle`` and the pool ``g_max_refine``) the ``Plan``
    does not carry. Threading that shared state across levers is the preprocess driver's job (block
    coordinate descent: converge one lever, feed its settled scalar to the other, repeat).
    """
    if step <= 0.0:
        raise ValueError("step must be positive")
    measure = simulation_rfactor(refinement, method=method)

    def run(seed: Plan) -> Plan:
        def build(g_max_refine: float) -> Plan:
            return reseed_pool(seed, selection, g_max_refine=g_max_refine)

        return converge_scalar(build, measure, tolerance, start=start_g_max_refine, step=step)

    return run


def converge_sampling(
    rocking: RockingCurve,
    refinement: RefinementSetup,
    tolerance: ConvergenceTolerance,
    *,
    step: float,
    method: SolverMethod = "matrix_exp",
) -> PlanStep:
    """Return a ``Plan -> Plan`` step: refine the rocking-curve tilt count until the pattern stops.

    The forward-model convergence lever (independent of the two beam levers): each candidate bakes
    the rocking-curve integration geometry at a finer ``rocking_curve_sampling`` (the tilt count)
    via :func:`~diffBloch.preprocess.steps.rocking_curve.integrate_rocking_curve`, so the summed
    ``|psi|^2`` over tilts approaches the continuous rotation-frame integral. The sweep starts at
    ``rocking.sampling`` and clicks up by ``step`` (rounded to a whole tilt count) until
    :func:`converge_scalar` settles the pattern (first sub-threshold step wins): it settles when a
    finer tilt grid stops moving the integrated intensities. Only ``sampling`` is swept -- the tilt
    span (``rocking.integration.semiangle``) and ``rocking.integration.geometry`` are held fixed.
    Re-integrating from the
    incoming seed each step (``integrate_rocking_curve`` rebuilds tilts from each nominal
    orientation, discarding any prior tilts) makes the sweep independent of the seed's tilt state.

    ``step`` must be positive. Unlike the beam levers this needs no grid guard (the tilt count does
    not touch the ``Fgb`` support) and does not couple to them, so it composes as an ordinary extra
    lever.
    """
    if step <= 0.0:
        raise ValueError("step must be positive")
    measure = simulation_rfactor(refinement, method=method)

    def run(seed: Plan) -> Plan:
        def build(sampling: float) -> Plan:
            tilts = replace(rocking, sampling=int(round(sampling)))
            return integrate_rocking_curve(tilts)(seed)

        return converge_scalar(build, measure, tolerance, start=float(rocking.sampling), step=step)

    return run


def _simulate(
    plan: Plan, refinement: RefinementSetup, method: SolverMethod
) -> tuple[BlochSolution, ...]:
    return build_engine(plan, refinement, method=method).simulate(refinement.params)


def _orientation_rfactor(previous: BlochSolution, current: BlochSolution) -> float:
    """Scale-optimised ``rbragg`` between two simulations on their shared reflections.

    Each table is ``(T, N)`` over its own beam set; the beam sets differ between the two
    simulations, so the comparison is restricted to the reflections both contain. A single intensity
    scale (shared across thicknesses, matching ``optimal_scale``) maps ``current`` onto ``previous``
    before the R-factor, since the two simulations have no common normalization.
    """
    previous_index, current_index = _shared_reflections(previous.beam_hkl, current.beam_hkl)
    previous_intensity = previous.intensities.detach().cpu()[:, previous_index].reshape(-1)
    current_intensity = current.intensities.detach().cpu()[:, current_index].reshape(-1)
    sigmas = torch.full_like(previous_intensity, _UNWEIGHTED_SIGMA)
    _, r_value = optimal_scale(current_intensity, previous_intensity, sigmas, metric=rbragg)
    return float(r_value)


def _shared_reflections(previous_hkl: Tensor, current_hkl: Tensor) -> tuple[Tensor, Tensor]:
    """Indices into each beam set selecting the reflections present in both, in a shared order."""
    previous_rows = previous_hkl.detach().cpu().numpy()
    current_rows = current_hkl.detach().cpu().numpy()
    previous_position = {tuple(row): i for i, row in enumerate(previous_rows)}
    previous_index: list[int] = []
    current_index: list[int] = []
    for j, row in enumerate(current_rows):
        i = previous_position.get(tuple(row))
        if i is not None:
            previous_index.append(i)
            current_index.append(j)
    if not previous_index:
        raise ValueError("convergence check found no reflections shared by the two simulations")
    return torch.tensor(previous_index), torch.tensor(current_index)
