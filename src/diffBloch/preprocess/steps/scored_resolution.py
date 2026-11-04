"""``restrict_scored_resolution``: cap the scored reflection set at a maximum ``|g|``.

A ``Plan -> Plan`` step that drops every scored reflection with ``|g| > cap.g_max`` from each
rotation's ``alignment``, leaving the solve beam set (and so the Bloch solve itself) untouched: it
narrows *what is scored*, not *what is solved*. ``|g| = |hkl . B|`` is read from the shared grid's
reciprocal basis, so it is orientation-independent and the cut holds through any later orientation
change. Every later rebuild pins the scored set it is given (``with_orientation``, coupling, and
the checkpoint round trip all carry ``alignment.hkl`` forward), so running this once right after
``build_orientation_plans`` is enough for the fits, the refinement objective, and the report.
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
    A rotation left with no scored reflection keeps an empty alignment; the fits and the objective
    already handle an unmatched rotation.
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
    return replace(
        op,
        alignment=AlignmentPlan(
            hkl=hkl[keep],
            solution_index=alignment.solution_index[keep],
            pattern_index=alignment.pattern_index[keep],
        ),
    )
