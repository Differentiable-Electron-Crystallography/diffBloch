"""``report_coupling``: emit the settled plan's coupled solve geometry as observability events.

A pure ``Plan -> Plan`` identity whose only effect is reporting -- the consumer-boundary companion
to the per-step :class:`~diffBloch.observability.PlanStepCompleted` stream. Because it inspects the
*settled* plan rather than running a recipe step, it fires on every run, including a reuse refine
(which executes no pipeline steps), so the coupling the refinement/inference loop is about to repeat
every step -- the unions, their tilt covers, and their beam counts -- is legible up front.
"""

from __future__ import annotations

from diffBloch.observability import CouplingSummary, Logger, RotationCoupling
from diffBloch.preprocess.pipeline import PlanStep
from diffBloch.preprocess.plan import Plan, coupling_stats, summarize_plan

__all__ = ["report_coupling"]


def report_coupling(logger: Logger) -> PlanStep:
    """Return an identity ``Plan -> Plan`` step emitting the plan's coupling geometry to ``logger``.

    One :class:`~diffBloch.observability.RotationCoupling` per rotation (its unions /
    tilts-per-union / beams-per-union) then one :class:`~diffBloch.observability.CouplingSummary`
    (structure-factor support size + radius and the cross-rotation aggregates). The plan is returned
    unchanged, so this is a boundary observation, not a recipe step -- it is applied outside the
    pipeline (never stamped into provenance or the checkpoint lock).
    """

    def run(plan: Plan) -> Plan:
        for index, op in enumerate(plan.orientations):
            logger.report(RotationCoupling(index=index, **coupling_stats(op)))
        logger.report(CouplingSummary(measurements=summarize_plan(plan)))
        return plan

    return run
