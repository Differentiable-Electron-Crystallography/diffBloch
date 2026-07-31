"""Coverage sweep: grow a beam knob to the minimum that recovers the most matched reflections.

The *second* convergence operation (the first is self-stability in ``convergence.py``). Where a
``converge_*`` sweep asks whether two consecutive *simulations* agree, a coverage sweep asks a
cheaper, purely geometric question: how many of each orientation's *observed* reflections does the
current beam set actually include? Growing a beam knob admits more beams; a candidate is accepted
only while it *increases* that matched count, and the sweep stops at the first knob step that buys
no new match -- the minimal beam set that still covers the data.

- :func:`plan_coverage` -- the objective: ``sum over orientations |beam_hkl & observed hkl|``.
  It is a *pure function of the Plan* (no engine, no structure factors): a "match" is set
  membership (an experimental hkl present in the filtered simulated beam set, with no intensity
  gate).
- :func:`maximize_scalar` -- the parameter-agnostic driver: click a scalar knob upward, keep the
  build while the objective strictly increases, return the last build at the first non-increase, or
  raise at a hard cap. The match-count analogue of
  :func:`~diffBloch.preprocess.steps.convergence.converge_scalar`.
- :func:`cover_beams` -- the ``Plan -> Plan`` adapter for the Klar window
  (``integration_semiangle``) lever.

The two convergence operations are two kinds of objective: coverage maximises *observed matches*
(sequential per-knob sweeps, accepting a candidate only while the match count increases, capped),
while self-stability (``convergence.py``) settles *simulations*.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from diffBloch.preprocess.pipeline import PlanStep
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.steps.beams import (
    build_orientation_plans,
    select_beams,
)
from diffBloch.specs import BeamSelection

__all__ = [
    "cover_beams",
    "maximize_scalar",
    "plan_coverage",
]


def plan_coverage(plan: Plan) -> int:
    """Count matched reflections: ``sum over orientations |beam_hkl intersect observed hkl|``.

    A pure, engine-free measure of how much of the observed data the current beam set covers. A
    "match" is an observed reflection whose hkl is present in that orientation's active beam set
    (set membership, no intensity threshold).
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
    that does not improve it (accept while ``candidate > best``, else stop). Raises
    ``RuntimeError`` if ``max_iterations`` steps pass while the score is still increasing (the score
    never plateaus). ``max_iterations`` must be >= 1.
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
    at the first window that admits no new matched reflection. ``step`` must be positive.
    """
    if step <= 0.0:
        raise ValueError("step must be positive")

    def run(seed: Plan) -> Plan:
        def build(semiangle: float) -> Plan:
            geometry = replace(selection.integration, semiangle=semiangle)
            selected = select_beams(replace(selection, integration=geometry))(seed)
            return build_orientation_plans()(selected)

        return maximize_scalar(
            build,
            plan_coverage,
            start=selection.integration.semiangle,
            step=step,
            max_iterations=max_iterations,
        )

    return run
