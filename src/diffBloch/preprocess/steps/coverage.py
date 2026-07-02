"""Coverage sweep: grow a beam knob to the minimum that recovers the most matched reflections.

The *second* convergence operation (the first is self-stability in ``convergence.py``). Where a
``converge_*`` sweep asks whether two consecutive *simulations* agree, a coverage sweep asks a
cheaper, purely geometric question: how many of each orientation's *observed* reflections does the
current beam set actually include? Growing a beam knob admits more beams; a candidate is accepted
only while it *increases* that matched count, and the sweep stops at the first knob step that buys
no new match -- the minimal beam set that still covers the data.

- :func:`plan_coverage` -- the objective: ``sum over orientations |beam_hkl & observed hkl|``.
  It is a *pure function of the Plan* (no engine, no structure factors): a "match" is set
  membership, exactly the private's ``match == "yes"`` (an experimental hkl present in the filtered
  simulated beam set, with no intensity gate).
- :func:`maximize_scalar` -- the parameter-agnostic driver: click a scalar knob upward, keep the
  build while the objective strictly increases, return the last build at the first non-increase, or
  raise at a hard cap. Mirrors :func:`~diffBloch.preprocess.steps.convergence.converge_scalar` for
  the
  match-count objective.
- :func:`cover_beams` / :func:`cover_pool` -- the ``Plan -> Plan`` adapters for the two beam levers
  (Klar window ``integration_semiangle`` and seed pool ``g_max_refine``).

Faithful to ``diffBloch_private`` ``convergence_testing._run_initial_minimum_param_sweep`` (branch
``pattern-vis-convergence-testing``): sequential per-knob sweeps accepting a candidate only when
``_count_unique_matches`` increases, ``for ... else`` raising at ``MAX_SWEEP_ITERATIONS``. See
``design/decisions/stage11-convergence.md`` ("two operations are two kinds of objective").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from diffBloch.engine.plan import OrientationPlan
from diffBloch.preprocess.experiment import seed_beam_hkl
from diffBloch.preprocess.pipeline import PlanStep
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.steps.beams import select_beams
from diffBloch.specs import BeamSelection

__all__ = [
    "cover_beams",
    "cover_pool",
    "maximize_scalar",
    "plan_coverage",
]


def plan_coverage(plan: Plan) -> int:
    """Count matched reflections: ``sum over orientations |beam_hkl intersect observed hkl|``.

    A pure, engine-free measure of how much of the observed data the current beam set covers. A
    "match" is an observed reflection whose hkl is present in that orientation's active beam set --
    the 2.0 analog of the private's ``match == "yes"`` (set membership, no intensity threshold).
    """
    total = 0
    for op in plan.orientations:
        beams = {tuple(int(x) for x in hkl) for hkl in np.asarray(op.beam_hkl, dtype=np.int64)}
        obs_hkl = np.asarray(op.pattern.hkl, dtype=np.int64)
        observed = {tuple(int(x) for x in hkl) for hkl in obs_hkl}
        total += len(beams & observed)
    return total


def maximize_scalar[T](
    build: Callable[[float], T],
    objective: Callable[[T], float],
    *,
    start: float,
    step: float,
    max_iterations: int = 100,
) -> T:
    """Grow a scalar knob while ``objective`` strictly increases; return the last accepted build.

    The parameter-agnostic coverage driver -- it knows nothing about beams or Plans.
    ``build(value)``
    rebuilds the object at a knob value; ``objective(obj)`` is the score to maximise (for coverage,
    :func:`plan_coverage`). Starting from ``start`` and clicking by ``step``, it keeps a candidate
    while its score is strictly greater than the best so far and returns the best at the first step
    that does not improve it -- the private's ``if candidate > best: accept; else: break``. Raises
    ``RuntimeError`` if ``max_iterations`` steps pass while the score is still increasing (the score
    never plateaus), matching the private's ``for ... else`` cap. ``max_iterations`` must be >= 1.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    best = build(start)
    best_score = objective(best)
    value = start
    for _ in range(max_iterations):
        value += step
        candidate = build(value)
        score = objective(candidate)
        if score > best_score:
            best = candidate
            best_score = score
        else:
            return best
    raise RuntimeError(f"maximize_scalar did not plateau within {max_iterations} steps")


def cover_beams(selection: BeamSelection, *, step: float, max_iterations: int = 100) -> PlanStep:
    """Return a ``Plan -> Plan`` step: widen the Klar window to the minimum that maximises coverage.

    The window (``integration_semiangle``) lever of the coverage sweep: each candidate re-runs
    :func:`~diffBloch.preprocess.steps.beams.select_beams` from the incoming seed at a wider window,
    and
    :func:`maximize_scalar` keeps widening while :func:`plan_coverage` strictly increases, stopping
    at the first window that admits no new matched reflection. ``step`` must be positive. See
    ``design/decisions/stage11-convergence.md``.
    """
    if step <= 0.0:
        raise ValueError("step must be positive")

    def run(seed: Plan) -> Plan:
        def build(semiangle: float) -> Plan:
            return select_beams(replace(selection, integration_semiangle=semiangle))(seed)

        return maximize_scalar(
            build,
            plan_coverage,
            start=selection.integration_semiangle,
            step=step,
            max_iterations=max_iterations,
        )

    return run


def cover_pool(
    selection: BeamSelection,
    *,
    start_g_max_refine: float,
    step: float,
    max_iterations: int = 100,
) -> PlanStep:
    """Return a ``Plan -> Plan`` step: widen the ``g_max_refine`` seed pool to maximise coverage.

    The pool lever of the coverage sweep: each candidate re-seeds every orientation from the shared
    grid at a wider ``g_max_refine`` (:func:`~diffBloch.preprocess.experiment.seed_beam_hkl`),
    rebuilds each :class:`~diffBloch.engine.plan.OrientationPlan`, then re-applies the fixed Klar
    window (``selection``); :func:`maximize_scalar` keeps widening while :func:`plan_coverage`
    strictly increases. Guards the ``Fgb`` difference support exactly like
    :func:`~diffBloch.preprocess.steps.convergence.converge_pool`: a candidate with
    ``2 * g_max_refine > grid.g_max`` raises (dependent grid resizing is unimplemented; see
    ``KNOWN_ISSUES.md``). ``step`` must be positive.
    """
    if step <= 0.0:
        raise ValueError("step must be positive")

    def run(seed: Plan) -> Plan:
        def build(g_max_refine: float) -> Plan:
            if 2.0 * g_max_refine > seed.grid.g_max:
                raise ValueError(
                    f"g_max_refine={g_max_refine:.4g} exceeds the grid's beam-difference support "
                    f"(g_max={seed.grid.g_max:.4g}); dependent grid resizing is not implemented"
                )
            beam_hkl = seed_beam_hkl(seed.grid, g_max_refine=g_max_refine)
            reseeded = tuple(
                OrientationPlan.build(
                    seed.grid,
                    beam_hkl,
                    op.pattern,
                    energy=op.energy,
                    thickness=op.thickness,
                    u0=op.u0,
                    orientation=op.orientation,
                )
                for op in seed.orientations
            )
            return select_beams(selection)(replace(seed, orientations=reseeded))

        return maximize_scalar(
            build,
            plan_coverage,
            start=start_g_max_refine,
            step=step,
            max_iterations=max_iterations,
        )

    return run
