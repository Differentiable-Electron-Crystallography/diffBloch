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
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from diffBloch.core.solver import Method
from diffBloch.engine import RefinementEngine
from diffBloch.engine.plan import OrientationPlan
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.pipeline import PlanStep
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.scoring import build_engine

__all__ = ["fit_thickness"]


def fit_thickness(
    refinement: RefinementSetup,
    *,
    min_thickness: float = 5.0,
    max_thickness: float = 2000.0,
    n_steps: int = 100,
    method: Method = "matrix_exp",
) -> PlanStep:
    """Return a ``Plan -> Plan`` step fitting each rotation's thickness by grid search.

    ``refinement`` (constraint spec, ASU expansion, atomic numbers, seeded params) is captured
    read-only and rejoined to the geometry ``Plan`` via :func:`build_engine`; the
    orientation-invariant ``F_gb`` is computed once and reused across every orientation. Each
    rotation is then assigned the lowest-wR2 of ``n_steps`` candidate thicknesses spaced evenly from
    ``min_thickness`` to ``max_thickness`` (inclusive, in Angstroms); defaults are the faithful
    ``diffBloch_private`` values. ``method`` configures the engine's solver.

    ``max_thickness`` must exceed ``min_thickness``, both must be positive, and ``n_steps`` must be
    >= 1 (with ``n_steps == 1`` the single candidate is ``min_thickness``).
    """
    if min_thickness <= 0.0 or max_thickness <= 0.0:
        raise ValueError("thickness bounds must be positive")
    if max_thickness <= min_thickness:
        raise ValueError("max_thickness must exceed min_thickness")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    def run(plan: Plan) -> Plan:
        engine = build_engine(plan, refinement, method=method)
        fgb = engine.fgb(refinement.params)
        candidates = torch.linspace(min_thickness, max_thickness, n_steps, dtype=torch.float64)
        orientations = tuple(_fit_one(engine, fgb, op, candidates) for op in plan.orientations)
        return replace(plan, orientations=orientations)

    return run


def _fit_one(
    engine: RefinementEngine,
    fgb: Tensor,
    op: OrientationPlan,
    candidates: Tensor,
) -> OrientationPlan:
    """Score every candidate thickness for one orientation; bake the lowest-wR2 winner."""
    trial = replace(op, thickness=candidates)  # geometry unchanged; only the (T,) thickness swaps
    scores = engine.score_orientation_per_thickness(trial, fgb)  # one pass over all candidates
    best = int(torch.argmin(scores))
    return replace(op, thickness=candidates[best : best + 1])  # (1,) baked thickness
