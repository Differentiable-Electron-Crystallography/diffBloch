"""The inference terminal: run the forward model over every rotation and score it, no refinement.

``run_inference`` is the eval-only member of the preprocess pipeline's terminal family (the other is
``engine.refine``, which optimizes structure): it runs one forward Bloch pass per orientation under
``no_grad`` and reports a per-rotation :class:`RotationInference` (the Bragg R-factor ``R_obs`` and
two diagnostics).

Built entirely from the public forward spine -- ``engine.simulate`` + :func:`core.products.align` +
:func:`core.losses.rbragg`/:func:`core.losses.optimal_scale` -- so callers never reach into engine
internals. Preprocess is composed in optionally via the ``preprocess``
``PlanStep``; the solver is swappable via ``method``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from diffBloch.core.losses import optimal_scale, rbragg, w_rbragg
from diffBloch.core.products import BlochSolution, align
from diffBloch.core.solver import SolverMethod
from diffBloch.engine.plan import OrientationPlanLike
from diffBloch.observability import (
    NULL_LOGGER,
    InferenceCompleted,
    Logger,
    RotationScored,
)
from diffBloch.params import Device
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.pipeline import PlanStep, identity
from diffBloch.preprocess.plan import Plan, require_built_plans
from diffBloch.preprocess.scoring import build_engine
from diffBloch.specs import NO_ABSORPTION, Absorption

__all__ = [
    "InferenceResult",
    "RotationInference",
    "run_inference",
]


@dataclass(frozen=True)
class RotationInference:
    """One rotation's forward-inference metrics.

    ``r_obs`` is the scaling-optimised Bragg R-factor of calculated vs observed intensities over the
    reflections with ``I > 3*sigma`` (``core.losses.rbragg``); it is ``nan`` when no reflection
    passes that cut. ``n_observed`` counts those reflections and ``n_beams`` the active beam set --
    both diagnostics for why an ``r_obs`` is what it is.
    """

    r_obs: float
    wr2: float
    n_observed: int
    n_beams: int


@dataclass(frozen=True)
class InferenceResult:
    """Per-rotation forward-inference metrics for a whole :class:`Plan`."""

    per_rotation: tuple[RotationInference, ...]

    @property
    def n_evaluated(self) -> int:
        """Rotations with a finite ``r_obs`` (i.e. at least one ``I > 3*sigma`` reflection)."""
        return sum(1 for row in self.per_rotation if math.isfinite(row.r_obs))

    @property
    def mean_wr2(self) -> float:
        """Mean weighted-R2 over rotations with a finite value; ``nan`` when none has one.

        The companion to :attr:`mean_r_obs`, filtered independently: a rotation can produce a finite
        score under one metric and not the other, so the two means need not share a denominator.
        """
        finite = [row.wr2 for row in self.per_rotation if math.isfinite(row.wr2)]
        return sum(finite) / len(finite) if finite else float("nan")

    @property
    def mean_r_obs(self) -> float:
        """Mean ``R_obs`` over the finite rotations.

        ``nan`` when no rotation has a finite ``r_obs``. The per-rotation ``R_obs`` values are
        averaged, skipping rotations with no reflections.
        """
        finite = [row.r_obs for row in self.per_rotation if math.isfinite(row.r_obs)]
        if not finite:
            return math.nan
        return sum(finite) / len(finite)


def run_inference(
    plan: Plan,
    refinement: RefinementSetup,
    *,
    prepare: PlanStep = identity,
    method: SolverMethod = "matrix_exp",
    device: Device | None = None,
    max_batch: int | None = None,
    absorption: Absorption = NO_ABSORPTION,
    logger: Logger = NULL_LOGGER,
) -> InferenceResult:
    """Run the forward model once per orientation and score each against its observed pattern.

    First applies ``prepare`` to ``plan`` -- one composed ``Plan -> Plan`` pipeline (compose the
    run's steps with :func:`~diffBloch.preprocess.pipeline.pipeline`, e.g. ``select_beams`` ->
    ``integrate_rocking_curve`` -> ``mosaicity`` -> ``optimize_orientation`` -> ``optimize_thickness``);
    it defaults to the identity (evaluate the plan as given). Then builds
    a :class:`RefinementEngine`, simulates every orientation under ``no_grad`` with the swappable
    ``method`` solver, and returns per-rotation :class:`RotationInference`.

    Emits a :class:`~diffBloch.observability.RotationScored` per rotation and one
    :class:`~diffBloch.observability.InferenceCompleted` aggregate to ``logger`` (the
    :data:`~diffBloch.observability.NULL_LOGGER` default discards them, so the returned value is
    unchanged whether or not a sink is attached). Attach a console/wandb logger at the boundary to
    watch per-rotation ``R_obs`` live -- e.g. while chasing a residual.

    ``device`` (default ``None`` = CPU, unchanged) runs the forward solve on the given accelerator:
    the seed params are moved there, and the engine co-locates every invariant onto the param device
    at the use site (:meth:`RefinableParams.to`), so the whole eigensolve runs on-device. The
    scoring tail (``align`` / ``optimal_scale``) is device-safe (observed data is co-located there),
    so the returned ``R_obs`` is identical (to solver tolerance) across devices.

    ``max_batch`` (default ``None``) caps the ``matrix_exp`` propagator block on the terminal solve;
    ``None`` lets the engine pick a memory-safe block per beam count. Execution-only (memory), like
    ``device``. See :func:`~diffBloch.engine.build_engine`.
    """
    plan = prepare(plan)
    params = refinement.params if device is None else refinement.params.to(device)
    engine = build_engine(
        plan, refinement, method=method, max_batch=max_batch, absorption=absorption
    )
    with torch.no_grad():
        solutions = engine.simulate(params)
    rows = tuple(
        _score_rotation(orientation, solution)
        for orientation, solution in zip(require_built_plans(plan), solutions, strict=True)
    )
    for index, row in enumerate(rows):
        logger.report(
            RotationScored(
                index=index, r_obs=row.r_obs, n_observed=row.n_observed, n_beams=row.n_beams
            )
        )
    result = InferenceResult(per_rotation=rows)
    logger.report(
        InferenceCompleted(
            n_rotations=len(rows),
            n_evaluated=result.n_evaluated,
            mean_r_obs=result.mean_r_obs,
        )
    )
    return result


def _score_rotation(orientation: OrientationPlanLike, solution: BlochSolution) -> RotationInference:
    """Bragg R-factor + diagnostics for one already-simulated orientation."""
    aligned = align(solution, orientation.pattern, orientation.alignment)

    # One forward pass covers all thicknesses; take the best-fitting thickness's R (a nuisance
    # here, and the anchor uses a single thickness so this is a no-op there).
    def best_over_thickness(metric: Callable[[Tensor, Tensor, Tensor], Tensor]) -> float:
        """The best-fitting thickness's score under one metric, each independently scaled."""
        per_thickness = torch.stack(
            [
                optimal_scale(
                    aligned.calculated[t], aligned.observed[t], aligned.sigmas[t], metric=metric
                )[1]
                for t in range(aligned.calculated.shape[0])
            ]
        )
        return float(per_thickness.min())

    # observed/sigmas are thickness-independent, so the I > 3*sigma count is taken at t = 0.
    n_observed = int((aligned.observed[0] > 3.0 * aligned.sigmas[0]).sum())
    return RotationInference(
        r_obs=best_over_thickness(rbragg),
        # Free alongside r_obs: the same aligned intensities under the other metric, no extra
        # solve. Reported so the CI anchor can track both -- a drift can show in one and not the
        # other, and R_obs alone would not say which.
        wr2=best_over_thickness(w_rbragg),
        n_observed=n_observed,
        n_beams=int(orientation.beam_hkl.shape[0]),
    )
