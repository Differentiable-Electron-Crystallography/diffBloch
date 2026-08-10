"""``optimize_thickness``: per-rotation specimen-thickness calibration by grid search.

A ``Plan -> Plan`` step that replaces each rotation's thickness with the value that best matches its
observed pattern. The specimen's 3D shape is irregular, so each orientation presents a different
beam path length; this optimizes that length per rotation rather than assuming one shared thickness.

For each orientation it evaluates ``n_steps`` candidate thicknesses spaced evenly from
``min_thickness`` to ``max_thickness`` and keeps the candidate with the lowest scaling-optimised
weighted R-factor (wR2). All candidates are simulated in a single forward Bloch pass: the expensive
eigendecomposition depends only on the orientation and the structure factors, while thickness enters
only the cheap propagation tail, so scoring 100 thicknesses costs barely more than scoring one
(:meth:`~diffBloch.engine.forward.RefinementEngine.score_orientation_per_thickness`).

The captured ``refinement`` is read-only context the step never mutates; the simulation inside is
deterministic and depends only on its inputs, so it is ordinary computation, not a side effect.

The search is an evenly-spaced (``np.linspace``) grid of
candidate thicknesses, per-candidate wR2 via the scaling factor, then the per-rotation minimum.

Plan-agnostic: ``replace(op, thickness=...)`` swaps the thickness on either an
:class:`OrientationPlan` or a :class:`~diffBloch.engine.plan.CoupledOrientationPlan` (whose
``_solve_segmented`` reads the top-level thickness, ignoring the stale sub-plan copies), so a
coupled plan is optimized unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch
from torch import Tensor

from diffBloch.core.solver import SolverMethod
from diffBloch.engine import RefinementEngine, ScoresFn, wr2_scores
from diffBloch.engine.plan import OrientationPlanLike
from diffBloch.observability import (
    NULL_LOGGER,
    Logger,
    ThicknessOptimizationStarted,
    ThicknessOptimized,
)
from diffBloch.params import Device
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.pipeline import PlanStep, as_step
from diffBloch.preprocess.plan import Plan, require_built_plans
from diffBloch.preprocess.scoring import build_engine
from diffBloch.specs import NO_ABSORPTION, Absorption, ThicknessGrid

__all__ = ["optimize_thickness"]


def optimize_thickness(
    refinement: RefinementSetup,
    grid: ThicknessGrid | Mapping[str, ThicknessGrid],
    *,
    method: SolverMethod = "matrix_exp",
    device: Device | None = None,
    max_batch: int | None = None,
    logger: Logger = NULL_LOGGER,
    absorption: Absorption = NO_ABSORPTION,
    scores: ScoresFn = wr2_scores,
    residual: str = "wr2",
) -> PlanStep:
    """Return a ``Plan -> Plan`` step optimizing each rotation's thickness by grid search.

    ``scores`` (default :func:`~diffBloch.engine.wr2_scores`) is the per-thickness metric the grid
    search argmins over -- pass ``cfg.loss_metrics.to_scores()`` to search the same residual the
    gradient refinement stage minimises (:func:`~diffBloch.config.schema.LossMetricsConfig.to_scores`).
    Execution-only like ``method``: the resolved ``ExperimentConfig.loss_metrics`` already rides in
    :func:`~diffBloch.config.manifest.config_digest`. ``residual`` (default ``"wr2"``) is the
    display name for ``scores`` -- pass ``cfg.loss_metrics.residual`` alongside it so
    :class:`~diffBloch.observability.ThicknessOptimized` reports the score under its real name.

    ``refinement`` (constraint spec, ASU expansion, atomic numbers, seeded params) is captured
    read-only and rejoined to the geometry ``Plan`` via :func:`build_engine`; the
    orientation-invariant ``F_gb`` is computed once and reused across every orientation. Each
    rotation is then assigned the lowest-wR2 of ``grid.n_steps`` candidate thicknesses spaced evenly
    from ``grid.min_thickness`` to ``grid.max_thickness`` (inclusive, Angstroms). ``grid`` is a
    pre-validated :class:`~diffBloch.specs.ThicknessGrid` (invalid bounds are unrepresentable, so
    this function never re-validates); ``method`` configures the engine's solver.

    ``grid`` may instead be a mapping from ``str(rotation_index)`` to a :class:`ThicknessGrid` --
    for ``inputs.multi_dataset``, where each combined file gets its own search range rather than one
    shared range that may not bracket every dataset's specimen thickness equally well. Keyed by
    string (not ``int``) so the recipe's provenance -- serialized to JSON for the checkpoint lock --
    round-trips to the same value it started as; every rotation must have an entry.

    ``device`` (default ``None`` = CPU) places the grid search's forward solve on the given
    accelerator by moving the seed params there; the engine co-locates every invariant onto the
    param device at the use site. Execution-only (kept out of the recipe identity), exactly as in
    :func:`optimize_orientation`.

    ``max_batch`` (default ``None``) caps the ``matrix_exp`` propagator block. ``None`` lets each
    solve derive a memory-safe block from its beam count -- it matters most here because the grid
    search evaluates ``grid.n_steps`` thicknesses at once, so a wide coupled segment's
    ``(C, T, N, N)`` propagator can be tens of GiB if left unbounded. Raise it to fill a larger GPU.
    The bound matches the unbounded solve to machine precision (memory only) and is execution-only,
    like ``device``.

    ``logger`` (default the null sink) receives a :class:`~diffBloch.observability.ThicknessOptimized`
    per rotation as its grid search completes -- the progress stream for this phase (mirroring
    ``optimize_orientation``); the memory-heavy thickness search is otherwise silent under a console logger.
    """

    def run(plan: Plan) -> Plan:
        engine = build_engine(
            plan,
            refinement,
            method=method,
            max_batch=max_batch,
            absorption=absorption,
            scores=scores,
        )
        params = refinement.params if device is None else refinement.params.to(device)
        fgb = engine.fgb(params)

        def candidates_for(rotation_index: int) -> Tensor:
            this_grid = grid[str(rotation_index)] if isinstance(grid, Mapping) else grid
            return torch.linspace(
                this_grid.min_thickness,
                this_grid.max_thickness,
                this_grid.n_steps,
                dtype=torch.float64,
            )

        built = require_built_plans(plan)
        logger.report(ThicknessOptimizationStarted(total_rotations=len(built)))
        fitted = []
        for op in built:
            candidates = candidates_for(op.pattern.rotation_index)
            candidate_thicknesses = tuple(float(value) for value in candidates.tolist())
            orientation, score, thickness, candidate_scores = _fit_one(engine, fgb, op, candidates)
            logger.report(
                ThicknessOptimized(
                    rotation_index=orientation.pattern.rotation_index,
                    score=score,
                    residual=residual,
                    thickness=thickness,
                    candidate_thicknesses=candidate_thicknesses,
                    candidate_score=candidate_scores,
                )
            )
            fitted.append(orientation)
        return replace(plan, orientations=tuple(fitted))

    # method rides in the config digest (cfg.solver.refine); the grid is the step's own param.
    return as_step("optimize_thickness", {"grid": grid, "absorption": absorption}, run)


def _fit_one(
    engine: RefinementEngine,
    fgb: Tensor,
    op: OrientationPlanLike,
    candidates: Tensor,
) -> tuple[OrientationPlanLike, float, float, tuple[float, ...]]:
    """Score every candidate thickness for one orientation; bake the argmin winner.

    Returns the baked orientation, the winner's ``(score, thickness)`` under this engine's
    configured ``scores`` (whichever residual ``ExperimentConfig.loss_metrics`` sets), and every
    candidate's score (same order as ``candidates``) for the progress event/plot.
    """
    trial = replace(op, thickness=candidates)  # geometry unchanged; only the (T,) thickness swaps
    candidate_scores = engine.score_orientation_per_thickness(
        trial, fgb
    )  # one pass, all candidates
    if torch.isnan(candidate_scores).any():
        raise FloatingPointError(
            f"thickness search score is nan for {int(torch.isnan(candidate_scores).sum())}/"
            f"{candidate_scores.numel()} candidate thicknesses -- torch.argmin's result is "
            "undefined with nan present, so this would otherwise silently pick a meaningless "
            "winner. The reference simulation's intensity is exactly zero across every scored "
            "reflection for those candidates (0/0 in the residual): check that the structure and "
            "experimental-data files describe the same crystal setting."
        )
    best = int(torch.argmin(candidate_scores))
    baked = replace(op, thickness=candidates[best : best + 1])  # (1,) baked thickness
    all_scores = tuple(float(value) for value in candidate_scores.tolist())
    return baked, float(candidate_scores[best]), float(candidates[best]), all_scores
