"""The inference terminal: run the forward model over every rotation and score it, no refinement.

``run_inference`` is the eval-only member of the preprocess pipeline's terminal family (the other is
``engine.refine``, which optimizes structure): it runs one forward Bloch pass per orientation under
``no_grad`` and reports a per-rotation :class:`RotationInference` (the Bragg R-factor ``R_obs`` and
two diagnostics). It is the 2.0 analog of the private ``evaluate_over_rotations`` and the terminal
the executable quartz anchor calls.

Built entirely from the public forward spine -- ``engine.simulate`` + :func:`core.products.align` +
:func:`core.losses.rbragg`/:func:`core.losses.optimal_scale` -- so callers (and the anchor test)
never reach into engine internals. Preprocess is composed in optionally via the ``preprocess``
``PlanStep``; the solver is swappable via ``method``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from diffBloch.core.losses import optimal_scale, rbragg
from diffBloch.core.products import BlochSolution, align
from diffBloch.core.solver import Method
from diffBloch.engine.plan import OrientationPlan
from diffBloch.observability import (
    NULL_LOGGER,
    InferenceCompleted,
    Logger,
    RotationScored,
)
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.pipeline import PlanStep, identity
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.scoring import build_engine

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
    def mean_r_obs(self) -> float:
        """Mean ``R_obs`` over the finite rotations (the private aggregate convention).

        ``nan`` when no rotation has a finite ``r_obs``. The private reference summary averages the
        per-rotation ``R_obs`` (rotations with no reflections are skipped), so this mirrors it.
        """
        finite = [row.r_obs for row in self.per_rotation if math.isfinite(row.r_obs)]
        if not finite:
            return math.nan
        return sum(finite) / len(finite)


def run_inference(
    plan: Plan,
    refinement: RefinementSetup,
    *,
    preprocess: PlanStep = identity,
    method: Method = "bloch_eigen",
    logger: Logger = NULL_LOGGER,
) -> InferenceResult:
    """Run the forward model once per orientation and score each against its observed pattern.

    Optionally sharpens ``plan`` with the ``preprocess`` step first (e.g. ``select_beams``; identity
    by default), builds a :class:`RefinementEngine`, simulates every orientation under ``no_grad``
    with the swappable ``method`` solver, and returns per-rotation :class:`RotationInference`.

    Emits a :class:`~diffBloch.observability.RotationScored` per rotation and one
    :class:`~diffBloch.observability.InferenceCompleted` aggregate to ``logger`` (the
    :data:`~diffBloch.observability.NULL_LOGGER` default discards them, so the returned value is
    unchanged whether or not a sink is attached). Attach a console/wandb logger at the boundary to
    watch per-rotation ``R_obs`` live -- e.g. while chasing a residual.
    """
    plan = preprocess(plan)
    engine = build_engine(plan, refinement, method=method)
    with torch.no_grad():
        solutions = engine.simulate(refinement.params)
    rows = tuple(
        _score_rotation(orientation, solution)
        for orientation, solution in zip(plan.orientations, solutions, strict=True)
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


def _score_rotation(orientation: OrientationPlan, solution: BlochSolution) -> RotationInference:
    """Bragg R-factor + diagnostics for one already-simulated orientation."""
    aligned = align(solution, orientation.pattern, orientation.alignment)
    # One forward pass covers all thicknesses; take the best-fitting thickness's R (a nuisance
    # here, and the anchor uses a single thickness so this is a no-op there).
    per_thickness = torch.stack(
        [
            optimal_scale(
                aligned.calculated[t], aligned.observed[t], aligned.sigmas[t], metric=rbragg
            )[1]
            for t in range(aligned.calculated.shape[0])
        ]
    )
    # observed/sigmas are thickness-independent, so the I > 3*sigma count is taken at t = 0.
    n_observed = int((aligned.observed[0] > 3.0 * aligned.sigmas[0]).sum())
    return RotationInference(
        r_obs=float(per_thickness.min()),
        n_observed=n_observed,
        n_beams=int(orientation.beam_hkl.shape[0]),
    )
