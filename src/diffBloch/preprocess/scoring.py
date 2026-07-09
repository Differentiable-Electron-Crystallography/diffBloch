"""Assemble a :class:`RefinementEngine` and score orientations against data (computes wR2).

Bridges the two products :func:`diffBloch.preprocess.from_experiment` returns -- the geometry
:class:`~diffBloch.preprocess.plan.Plan` and the structure-side
:class:`~diffBloch.preprocess.experiment.RefinementSetup` -- into a runnable
:class:`~diffBloch.engine.RefinementEngine`, then exposes the per-orientation scaling-optimised wR2
the orientation refinement (``fit_orientation``, slice 5b) minimises.

``build_engine`` is the general ``Plan + RefinementSetup -> engine`` assembly (the same engine
``refine`` will consume); ``score_orientations`` is the thin convenience that computes the
orientation-invariant ``F_gb`` once and scores every orientation of a ``Plan``. The forward
simulation inside is deterministic and depends only on its inputs (same inputs always give the same
result), so it does not change any shared state -- it is ordinary computation reading captured
read-only context, not a side effect.
"""

from __future__ import annotations

from torch import Tensor

from diffBloch.core.solver import Method
from diffBloch.engine import LossFn, RefinementEngine, w_rbragg_loss
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.plan import Plan, require_built_plans, require_orientation_plans

__all__ = ["build_engine", "score_orientations"]


def build_engine(
    plan: Plan,
    refinement: RefinementSetup,
    *,
    loss: LossFn = w_rbragg_loss,
    method: Method = "matrix_exp",
) -> RefinementEngine:
    """Wire a geometry ``plan`` and a structure ``refinement`` into a runnable engine (no compute).

    Pure assembly, not a forward pass. ``Plan`` and ``RefinementSetup`` are kept deliberately
    separate -- the ``Plan`` (shared grid + per-rotation orientations) flows through the
    ``Plan -> Plan`` preprocess steps, while ``refinement`` (constraint spec, ASU-expansion plan,
    ASU atomic numbers) is static structure context. ``build_engine`` is the single place that
    rejoins them when a simulation is actually needed; both ``score_orientations``
    here and ``refine`` later go through it. ``loss`` is the per-orientation term ``refine`` would
    minimise (irrelevant to :meth:`RefinementEngine.fgb` /
    :meth:`RefinementEngine.score_orientation`, which use a scaling-optimised wR2 internally).
    """
    return RefinementEngine(
        spec=refinement.spec,
        asu_plan=refinement.asu_plan,
        numbers=refinement.numbers,
        grid=plan.grid,
        orientations=require_built_plans(plan),
        loss=loss,
        method=method,
    )


def score_orientations(
    plan: Plan, refinement: RefinementSetup, *, method: Method = "matrix_exp"
) -> tuple[Tensor, ...]:
    """Scaling-optimised wR2 for every orientation in ``plan`` at the seeded ``refinement.params``.

    Computes the orientation-invariant ``F_gb`` once and reuses it across orientations. This is the
    objective surface ``fit_orientation`` (slice 5b) searches over per rotation; here it evaluates
    the current (seed) orientations, returning one scalar score per rotation.
    """
    engine = build_engine(plan, refinement, method=method)
    fgb = engine.fgb(refinement.params)
    return tuple(
        engine.score_orientation(orientation, fgb)
        for orientation in require_orientation_plans(plan)
    )
