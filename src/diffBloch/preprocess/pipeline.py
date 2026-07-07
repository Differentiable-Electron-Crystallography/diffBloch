"""The preprocess composition combinators: sequence and fixpoint over ``Plan -> Plan`` steps.

A *step* is a pure ``Plan -> Plan`` transformer (it *fits* something -- numerics, orientation,
thickness -- and returns a sharpened :class:`~diffBloch.preprocess.plan.Plan`). :func:`pipeline`
chains steps left to right; :func:`iterate_until` drives one step to a fixpoint (for convergence
testing or alternating fits). Both return a ``Plan -> Plan`` step, so they nest -- a fixpoint of
steps is itself a step. ``refine`` is deliberately *not* expressible here: it is the terminal
``Plan -> Result`` estimator, not a ``Plan -> Plan`` transform.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from diffBloch.preprocess.plan import Plan

__all__ = [
    "NO_PIPELINES",
    "ConvergenceCheck",
    "Pipelines",
    "PlanStep",
    "identity",
    "iterate_until",
    "pipeline",
]

# A step sharpens the Plan; a convergence check compares a step's (previous, just-produced) Plans.
type PlanStep = Callable[[Plan], Plan]
type ConvergenceCheck = Callable[[Plan, Plan], bool]


def identity(plan: Plan) -> Plan:
    """The no-op step: return ``plan`` unchanged (the pipeline identity element)."""
    return plan


@dataclass(frozen=True)
class Pipelines:
    """The two declared phases of a run: plan-shaping, then parameter refinement.

    ``preprocess`` shapes the :class:`~diffBloch.preprocess.plan.Plan` -- beam coupling,
    rocking-curve
    integration, mosaicity -- and ``refine`` fits parameters (orientation, thickness) under the
    already-shaped plan. Both are ordinary ``Plan -> Plan`` steps (compose each with
    :func:`pipeline`), and a run applies ``refine(preprocess(plan))``. Splitting the two phases is
    what keeps a faithful default legible: the run reads as the exact list of physical choices, each
    independently removable, rather than one opaque step list mixing plan-shaping with fitting.
    """

    preprocess: PlanStep = identity
    refine: PlanStep = identity

    def apply(self, plan: Plan) -> Plan:
        """Shape the plan then refine its parameters: ``refine(preprocess(plan))``."""
        return self.refine(self.preprocess(plan))


# Shared identity default (both phases no-op), so signatures avoid a call-in-default.
NO_PIPELINES: Pipelines = Pipelines()


def pipeline(steps: Sequence[PlanStep]) -> PlanStep:
    """Compose ``steps`` left to right into one step (an empty list yields the identity)."""

    def run(plan: Plan) -> Plan:
        for step in steps:
            plan = step(plan)
        return plan

    return run


def iterate_until(step: PlanStep, *, until: ConvergenceCheck, max_iterations: int = 50) -> PlanStep:
    """Drive ``step`` to a fixpoint: re-apply it until ``until(previous, current)`` holds.

    Returns a ``Plan -> Plan`` step that applies ``step`` repeatedly, checking ``until`` against the
    (previous, just-produced) Plan pair after each application, and returns the first Plan that
    satisfies it. Raises ``RuntimeError`` if ``max_iterations`` is reached without convergence --
    silent non-convergence is never returned. ``max_iterations`` must be >= 1.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    def run(plan: Plan) -> Plan:
        current = plan
        for _ in range(max_iterations):
            nxt = step(current)
            if until(current, nxt):
                return nxt
            current = nxt
        raise RuntimeError(f"iterate_until did not converge within {max_iterations} iterations")

    return run
