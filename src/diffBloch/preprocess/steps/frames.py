"""``select_frames``: drop whole rotations whose observed pattern is too sparse to inform the fit.

The whole-frame sibling of :mod:`~diffBloch.preprocess.steps.beams`: ``select_beams`` prunes the
*reflections within* each frame, whereas ``select_frames`` drops *entire frames/orientations*. It is
the public analog to ``diffBloch_private``'s per-dataset ``ignore_orientations`` -- for the
beam-damaged tail of a rotation scan, where late frames carry almost no measurable signal.

The drop criterion is **model-independent**: it counts each frame's *observed* strong reflections
(``intensity > 3 * sigma``) from the stored ``pattern`` and never the calculated fit, so it cannot
circularly keep the frames the current model already explains (which would bias the reported
R-factor downward). Because it reads only the observed ``pattern`` -- present the moment
``from_experiment`` lays down the candidates -- it is composed *before* ``select_beams`` so dead
frames never pay beam-selection or solve cost.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from diffBloch.core.solver import Method
from diffBloch.engine.forward import LossFn
from diffBloch.engine.plan import OrientationPlanLike
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.pipeline import PlanStep, as_step
from diffBloch.preprocess.plan import CandidatePlan, Plan
from diffBloch.preprocess.scoring import build_engine
from diffBloch.specs import FrameSelection

__all__ = ["select_finite_loss_frames", "select_frames"]


def select_frames(selection: FrameSelection) -> PlanStep:
    """Return a ``Plan -> Plan`` step that drops frames with too few observed strong reflections.

    Every orientation whose observed pattern has fewer than ``selection.min_observed`` reflections
    at ``intensity > 3 * sigma`` (strict) is removed; the kept frames retain their original order
    and the shared ``grid`` is untouched. ``selection`` is a pre-validated
    :class:`~diffBloch.specs.FrameSelection` (``min_observed >= 0``, ``0`` keeps all), so this never
    re-validates.

    Reads the observed ``pattern`` only (phase-agnostic: both a :class:`CandidatePlan` and a built
    :class:`~diffBloch.engine.plan.OrientationPlan` carry it), so it is composed early -- before
    ``select_beams`` -- keeping the criterion model-independent and sparing dropped frames the solve
    cost. Raises if the criterion would drop *every* frame: an empty plan is a caller error that
    otherwise surfaces far downstream.
    """

    def run(plan: Plan) -> Plan:
        kept = tuple(op for op in plan.orientations if _n_observed(op) >= selection.min_observed)
        if not kept:
            raise ValueError(
                f"select_frames(min_observed={selection.min_observed}) dropped every frame "
                f"of {len(plan.orientations)}; loosen the threshold"
            )
        return replace(plan, orientations=kept)

    return as_step("select_frames", selection, run)


def select_finite_loss_frames(
    refinement: RefinementSetup,
    *,
    loss: LossFn,
    method: Method = "matrix_exp",
) -> PlanStep:
    """Return a ``Plan -> Plan`` step that keeps frames with finite initial objective loss.

    This is an explicit model-dependent filter for compact demos or diagnostic recipes where a
    settled Plan may contain orientations that produce non-finite diffraction loss under the chosen
    refinement objective. It evaluates each orientation independently at ``refinement.params`` and
    keeps only finite totals. Prefer model-independent :func:`select_frames` for scientific frame
    exclusion policies.
    """

    def run(plan: Plan) -> Plan:
        kept = []
        for orientation in plan.orientations:
            one = replace(plan, orientations=(orientation,))
            engine = build_engine(one, refinement, loss=loss, method=method)
            if torch.isfinite(engine.objective_value(refinement.params).total):
                kept.append(orientation)
        if not kept:
            raise ValueError("select_finite_loss_frames dropped every frame")
        return replace(plan, orientations=tuple(kept))

    return as_step("select_finite_loss_frames", None, run)


def _n_observed(op: CandidatePlan | OrientationPlanLike) -> int:
    """Count a frame's *observed* strong reflections (``intensity > 3 * sigma``, strict)."""
    pattern = op.pattern
    return int((pattern.intensities > 3.0 * pattern.sigmas).sum())
