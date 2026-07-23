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

from diffBloch.core.solver import FloatFormat, Method
from diffBloch.engine import LossFn, RefinementEngine, scaled_w_rbragg_loss
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.plan import Plan, require_built_plans, require_orientation_plans

__all__ = ["build_engine", "score_orientations"]


def build_engine(
    plan: Plan,
    refinement: RefinementSetup,
    *,
    loss: LossFn = scaled_w_rbragg_loss,
    method: Method = "matrix_exp",
    precision: FloatFormat = "fp64",
    max_batch: int | None = None,
) -> RefinementEngine:
    """Wire a geometry ``plan`` and a structure ``refinement`` into a runnable engine (no compute).

    Pure assembly, not a forward pass. ``Plan`` and ``RefinementSetup`` are kept deliberately
    separate -- the ``Plan`` (shared grid + per-rotation orientations) flows through the
    ``Plan -> Plan`` preprocess steps, while ``refinement`` (constraint spec, ASU-expansion plan,
    ASU atomic numbers) is static structure context. ``build_engine`` is the single place that
    rejoins them when a simulation is actually needed; both ``score_orientations``
    here and ``refine`` later go through it. ``loss`` is the per-orientation term ``refine`` would
    minimise; it defaults to :func:`~diffBloch.engine.scaled_w_rbragg_loss` (wR2 after matching
    calculated total intensity to observed -- calc and obs are on different scales, so the raw
    metric would be flat/gradient-free). It is irrelevant to :meth:`RefinementEngine.fgb` /
    :meth:`RefinementEngine.score_orientation`, which apply their own scaling-optimised wR2.

    ``precision`` defaults to ``"fp64"`` (complex128 -- byte-identical to today). ``"fp32"``
    (complex64) is a speed/precision knob: preprocess fits use it for transient coarse-search
    engines, and the app refinement path may opt into it explicitly via config. Terminal inference
    omits it and stays fp64. See :func:`diffBloch.core.solver.propagate`.

    ``max_batch`` (default ``None``) caps the ``matrix_exp`` propagator block; ``None`` lets each
    solve pick a memory-safe block from its beam count, bounding peak memory while matching the
    unbounded solve to machine precision (a pin is only needed for a specific device budget).
    Execution-only, like ``precision``/``method``.
    """
    return RefinementEngine(
        spec=refinement.spec,
        asu_plan=refinement.asu_plan,
        numbers=refinement.numbers,
        grid=plan.grid,
        orientations=require_built_plans(plan),
        loss=loss,
        method=method,
        precision=precision,
        max_batch=max_batch,
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
