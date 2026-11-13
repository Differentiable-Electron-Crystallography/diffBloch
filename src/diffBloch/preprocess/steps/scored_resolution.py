"""``restrict_scored_resolution``: cap the scored reflection set at a maximum ``|g|``.

A ``Plan -> Plan`` step that drops every scored reflection with ``|g| > cap.g_max`` from each
rotation's ``alignment``, leaving the solve beam set (and so the Bloch solve itself) untouched: it
narrows *what is scored*, not *what is solved*. ``|g| = |hkl . B|`` is read from the shared grid's
reciprocal basis, so it is orientation-independent and the cut holds through any later orientation
change. ``with_orientation`` and the checkpoint round trip carry ``alignment.hkl`` forward, but the
orientation fit re-selects the scored set per trial, so its ``TrialCoupling.scored.g_max`` must carry
the same cap (it uses the same orientation-independent ``|g|``).
"""

from __future__ import annotations

from dataclasses import replace

import torch

from diffBloch.core.products import AlignmentPlan
from diffBloch.engine.plan import OrientationPlanLike
from diffBloch.preprocess.pipeline import PlanStep, as_step
from diffBloch.preprocess.plan import Plan, require_built_plans
from diffBloch.specs import ScoredResolutionCap

__all__ = ["restrict_scored_resolution"]


def restrict_scored_resolution(cap: ScoredResolutionCap) -> PlanStep:
    """Return a ``Plan -> Plan`` step scoring only reflections with ``|g| <= cap.g_max``.

    Requires built plans (it narrows ``alignment``, which a :class:`CandidatePlan` does not have).
    Raises ``ValueError`` if the cap leaves any rotation with no scored reflection.
    """

    def run(plan: Plan) -> Plan:
        basis = torch.as_tensor(plan.structure_factor_grid.reciprocal_basis, dtype=torch.float64)
        orientations = tuple(_restrict(op, basis, cap.g_max) for op in require_built_plans(plan))
        return replace(plan, orientations=orientations)

    return as_step("restrict_scored_resolution", cap, run)


def _restrict(op: OrientationPlanLike, basis: torch.Tensor, g_max: float) -> OrientationPlanLike:
    alignment = op.alignment
    hkl = alignment.hkl
    g = torch.linalg.norm(hkl.to(torch.float64) @ basis.to(hkl.device), dim=-1)
    keep = g <= g_max
    if not bool(keep.any()):
        raise ValueError(
            f"scored resolution cap g_max={g_max} leaves rotation "
            f"{op.pattern.rotation_index} with no scored reflection"
        )
    return replace(
        op,
        alignment=AlignmentPlan(
            hkl=hkl[keep],
            solution_index=alignment.solution_index[keep],
            pattern_index=alignment.pattern_index[keep],
        ),
    )
