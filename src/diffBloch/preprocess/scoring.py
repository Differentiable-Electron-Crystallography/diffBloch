"""Assemble a :class:`RefinementEngine` and score orientations against data (computes wR2).

Bridges the two products :func:`diffBloch.preprocess.from_experiment` returns -- the geometry
:class:`~diffBloch.preprocess.plan.Plan` and the structure-side
:class:`~diffBloch.preprocess.experiment.RefinementSetup` -- into a runnable
:class:`~diffBloch.engine.RefinementEngine`, then exposes the per-orientation scaling-optimised wR2
the orientation refinement (``fit_orientation``) minimises.

``build_engine`` is the general ``Plan + RefinementSetup -> engine`` assembly (the same engine
``refine`` will consume); ``score_orientations`` is the thin convenience that computes the
orientation-invariant ``F_gb`` once and scores every orientation of a ``Plan``. The forward
simulation inside is deterministic and depends only on its inputs (same inputs always give the same
result), so it does not change any shared state -- it is ordinary computation reading captured
read-only context, not a side effect.
"""

from __future__ import annotations

from math import prod
from typing import cast

import torch
from torch import Tensor

from diffBloch.core import StructureFactorGather
from diffBloch.core.solver import SolverMethod
from diffBloch.engine import (
    CoupledOrientationPlan,
    LossFn,
    OrientationPlanLike,
    RefinementEngine,
    wr2_loss,
)
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.plan import Plan, require_built_plans, require_orientation_plans
from diffBloch.specs import NO_ABSORPTION, Absorption

__all__ = ["active_structure_factor_indices", "build_engine", "score_orientations"]


def build_engine(
    plan: Plan,
    refinement: RefinementSetup,
    *,
    loss: LossFn = wr2_loss,
    method: SolverMethod = "matrix_exp",
    max_batch: int | None = None,
    absorption: Absorption = NO_ABSORPTION,
    compact_structure_factors: bool = True,
    profile: bool = False,
    checkpoint_activations: bool = True,
) -> RefinementEngine:
    """Wire a geometry ``plan`` and a structure ``refinement`` into a runnable engine (no compute).

    Pure assembly, not a forward pass. ``Plan`` and ``RefinementSetup`` are kept deliberately
    separate -- the ``Plan`` (shared grid + per-rotation orientations) flows through the
    ``Plan -> Plan`` preprocess steps, while ``refinement`` (constraint spec, ASU-expansion plan,
    ASU atomic numbers) is static structure context. ``build_engine`` is the single place that
    rejoins them when a simulation is actually needed; both ``score_orientations``
    here and ``refine`` later go through it. ``loss`` is the per-orientation term ``refine`` would
    minimise; it defaults to :func:`~diffBloch.engine.wr2_loss` (wR2 after matching
    calculated total intensity to observed -- calc and obs are on different scales, so the raw
    metric would be flat/gradient-free). It is irrelevant to :meth:`RefinementEngine.fgb` /
    :meth:`RefinementEngine.score_orientation`, which apply their own scaling-optimised wR2.

    ``max_batch`` (default ``None``) caps the ``matrix_exp`` propagator block; ``None`` lets each
    solve pick a memory-safe block from its beam count, bounding peak memory while matching the
    unbounded solve to machine precision (a pin is only needed for a specific device budget).
    Execution-only, like ``method``.

    ``compact_structure_factors`` computes only support-grid rows referenced by the settled solve
    gathers and scatters them into the unchanged grid-shaped interface. It changes neither the
    solve nor its gradients; callers whose beam sets change dynamically may disable it or extend
    the support lazily.

    ``profile`` logs per-phase wall time (structure factors, each rotation's solve) on the built
    engine; see :class:`~diffBloch.engine.forward.RefinementEngine`. Execution-only and off by
    default -- it forces a CUDA sync around every measured block.

    ``checkpoint_activations`` (default ``True``) trades peak memory for one extra forward
    recompute per per-orientation/per-segment solve on the refinement backward pass; disabling it
    removes that recompute at the cost of retaining every solve's intermediates until backward.
    Execution-only -- gradients are identical either way. See
    :class:`~diffBloch.engine.forward.RefinementEngine`.
    """
    orientations = require_built_plans(plan)
    active_indices = (
        active_structure_factor_indices(orientations, plan.structure_factor_grid.gpts)
        if compact_structure_factors
        else None
    )
    return RefinementEngine(
        spec=refinement.spec,
        asu_plan=refinement.asu_plan,
        numbers=refinement.numbers,
        grid=plan.structure_factor_grid,
        orientations=orientations,
        loss=loss,
        method=method,
        max_batch=max_batch,
        absorption=absorption,
        active_structure_factor_indices=active_indices,
        profile=profile,
        checkpoint_activations=checkpoint_activations,
    )


def active_structure_factor_indices(
    orientations: tuple[OrientationPlanLike, ...], gpts: tuple[int, int, int]
) -> Tensor:
    """Return grid rows referenced by the settled plans' structure-factor gathers."""
    gathers: list[StructureFactorGather] = []
    for orientation in orientations:
        if isinstance(orientation, CoupledOrientationPlan):
            gathers.extend(segment.plan.beam_plans[0].gather for segment in orientation.segments)
        else:
            gathers.append(orientation.beam_plans[0].gather)
    source = gathers[0].structure_factor_indices
    inverse = torch.full((prod(gpts),), -1, dtype=torch.long)
    inverse[source] = torch.arange(source.numel(), dtype=torch.long)
    used_destinations = torch.unique(
        torch.cat([gather.beam_difference_indices for gather in gathers])
    )
    active = inverse[used_destinations]
    if bool(torch.any(active < 0)):
        raise ValueError("a solve gather references an hkl outside the structure-factor grid")
    return cast(Tensor, torch.unique(active, sorted=True))


def score_orientations(
    plan: Plan, refinement: RefinementSetup, *, method: SolverMethod = "matrix_exp"
) -> tuple[Tensor, ...]:
    """Scaling-optimised wR2 for every orientation in ``plan`` at the seeded ``refinement.params``.

    Computes the orientation-invariant ``F_gb`` once and reuses it across orientations. This is the
    objective surface ``fit_orientation`` searches over per rotation; here it evaluates
    the current (seed) orientations, returning one scalar score per rotation.
    """
    engine = build_engine(plan, refinement, method=method)
    fgb = engine.fgb(refinement.params)
    return tuple(
        engine.score_orientation(orientation, fgb)
        for orientation in require_orientation_plans(plan)
    )
