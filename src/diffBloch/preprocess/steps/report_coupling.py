"""``report_coupling``: emit the settled plan's coupled solve geometry as observability events.

A pure ``Plan -> Plan`` identity whose only effect is reporting -- the consumer-boundary companion
to the per-step :class:`~diffBloch.observability.PlanStepCompleted` stream. Because it inspects the
*settled* plan rather than running a recipe step, it fires on every run, including a reuse refine
(which executes no pipeline steps), so the coupling the refinement/inference loop is about to repeat
every step -- the unions, their tilt covers, and their beam counts -- is legible up front.
"""

from __future__ import annotations

from collections.abc import Callable

from diffBloch.engine.plan import CoupledOrientationPlan, OrientationPlan
from diffBloch.observability import (
    CouplingSummary,
    Logger,
    RotationCoupling,
    RotationCouplingSegments,
)
from diffBloch.preprocess.pipeline import PlanStep
from diffBloch.preprocess.plan import Plan, coupling_stats, summarize_plan

__all__ = ["report_coupling"]


def report_coupling(
    logger: Logger, *, dataset_for_rotation: Callable[[int], str] | None = None
) -> PlanStep:
    """Return an identity ``Plan -> Plan`` step emitting the plan's coupling geometry to ``logger``.

    One :class:`~diffBloch.observability.RotationCoupling` per rotation (its unions /
    tilts-per-union / beams-per-union), one batched
    :class:`~diffBloch.observability.RotationCouplingSegments` per rotation (the per-segment rows),
    then one :class:`~diffBloch.observability.CouplingSummary` (structure-factor support size +
    radius and the cross-rotation aggregates). The plan is returned unchanged, so this is a boundary
    observation, not a recipe step -- it is applied outside the pipeline (never stamped into
    provenance or the checkpoint lock).
    """

    def run(plan: Plan) -> Plan:
        for index, op in enumerate(plan.orientations):
            rotation_index = int(op.pattern.rotation_index)
            dataset = "" if dataset_for_rotation is None else dataset_for_rotation(rotation_index)
            logger.report(
                RotationCoupling(
                    index=index,
                    dataset=dataset,
                    rotation_index=rotation_index,
                    **coupling_stats(op),
                )
            )
            if isinstance(op, OrientationPlan | CoupledOrientationPlan):
                logger.report(_segment_event(op, dataset=dataset))
        logger.report(CouplingSummary(measurements=summarize_plan(plan)))
        return plan

    return run


def _segment_event(
    op: OrientationPlan | CoupledOrientationPlan, *, dataset: str
) -> RotationCouplingSegments:
    rotation_index = int(op.pattern.rotation_index)
    if isinstance(op, CoupledOrientationPlan):
        segment_index = tuple(range(len(op.segments)))
        first_tilt_index = tuple(int(segment.cover.min()) for segment in op.segments)
        last_tilt_index = tuple(int(segment.cover.max()) for segment in op.segments)
        n_tilts = tuple(int(segment.cover.shape[0]) for segment in op.segments)
        n_segment_beams = tuple(int(segment.plan.beam_hkl.shape[0]) for segment in op.segments)
    else:
        n_total_tilts = int(op.tilts.shape[0])
        segment_index = (0,)
        first_tilt_index = (0,)
        last_tilt_index = (max(0, n_total_tilts - 1),)
        n_tilts = (n_total_tilts,)
        n_segment_beams = (int(op.beam_hkl.shape[0]),)
    return RotationCouplingSegments(
        rotation_index=rotation_index,
        segment_index=segment_index,
        first_tilt_index=first_tilt_index,
        last_tilt_index=last_tilt_index,
        n_tilts=n_tilts,
        n_segment_beams=n_segment_beams,
        n_union_beams=int(op.beam_hkl.shape[0]),
        n_total_tilts=int(op.tilts.shape[0]),
        dataset=dataset,
    )
