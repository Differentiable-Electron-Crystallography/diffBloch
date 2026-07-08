"""``fit_thickness``: per-rotation specimen-thickness calibration by grid search.

A ``Plan -> Plan`` step that replaces each rotation's thickness with the value that best fits its
observed pattern. The specimen's 3D shape is irregular, so each orientation presents a different
beam path length; this fits that length per rotation rather than assuming one shared thickness.

For each orientation it evaluates ``n_steps`` candidate thicknesses spaced evenly from
``min_thickness`` to ``max_thickness`` and keeps the candidate with the lowest scaling-optimised
weighted R-factor (wR2). All candidates are simulated in a single forward Bloch pass: the expensive
eigendecomposition depends only on the orientation and the structure factors, while thickness enters
only the cheap propagation tail, so scoring 100 thicknesses costs barely more than scoring one
(:meth:`~diffBloch.engine.forward.RefinementEngine.score_orientation_per_thickness`).

The captured ``refinement`` is read-only context the step never mutates; the simulation inside is
deterministic and depends only on its inputs, so it is ordinary computation, not a side effect.

Faithful to ``diffBloch_private``'s ``thickness_optim``: an evenly-spaced (``np.linspace``) grid of
candidate thicknesses, per-candidate wR2 via the scaling factor, then the per-rotation minimum.

Plan-agnostic: ``replace(op, thickness=...)`` swaps the thickness on either an
:class:`OrientationPlan` or a :class:`~diffBloch.engine.plan.SegmentedOrientationPlan` (whose
``_solve_segmented`` reads the top-level thickness, ignoring the stale sub-plan copies), so a
coupled plan is fit unchanged.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from diffBloch.core.solver import Method
from diffBloch.engine import RefinementEngine
from diffBloch.engine.plan import OrientationPlanLike
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.pipeline import PlanStep, as_step
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.scoring import build_engine
from diffBloch.specs import ThicknessGrid

__all__ = ["fit_thickness"]


def fit_thickness(
    refinement: RefinementSetup,
    grid: ThicknessGrid,
    *,
    method: Method = "matrix_exp",
) -> PlanStep:
    """Return a ``Plan -> Plan`` step fitting each rotation's thickness by grid search.

    ``refinement`` (constraint spec, ASU expansion, atomic numbers, seeded params) is captured
    read-only and rejoined to the geometry ``Plan`` via :func:`build_engine`; the
    orientation-invariant ``F_gb`` is computed once and reused across every orientation. Each
    rotation is then assigned the lowest-wR2 of ``grid.n_steps`` candidate thicknesses spaced evenly
    from ``grid.min_thickness`` to ``grid.max_thickness`` (inclusive, Angstroms). ``grid`` is a
    pre-validated :class:`~diffBloch.specs.ThicknessGrid` (invalid bounds are unrepresentable, so
    this function never re-validates); ``method`` configures the engine's solver.
    """

    def run(plan: Plan) -> Plan:
        engine = build_engine(plan, refinement, method=method)
        fgb = engine.fgb(refinement.params)
        candidates = torch.linspace(
            grid.min_thickness, grid.max_thickness, grid.n_steps, dtype=torch.float64
        )
        orientations = tuple(_fit_one(engine, fgb, op, candidates) for op in plan.orientations)
        return replace(plan, orientations=orientations)

    # method rides in the config digest (cfg.solver.refine); the grid is the step's own param.
    return as_step("fit_thickness", grid, run)


def _fit_one(
    engine: RefinementEngine,
    fgb: Tensor,
    op: OrientationPlanLike,
    candidates: Tensor,
) -> OrientationPlanLike:
    """Score every candidate thickness for one orientation; bake the lowest-wR2 winner."""
    trial = replace(op, thickness=candidates)  # geometry unchanged; only the (T,) thickness swaps
    scores = engine.score_orientation_per_thickness(trial, fgb)  # one pass over all candidates
    best = int(torch.argmin(scores))
    return replace(op, thickness=candidates[best : best + 1])  # (1,) baked thickness
