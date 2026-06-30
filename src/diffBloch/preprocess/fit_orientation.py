"""``fit_orientation``: per-rotation crystal-orientation refinement (Palatinus hexagonal simplex).

A ``Plan -> Plan`` step that sharpens each orientation by minimising the scaling-optimised wR2 of
the full dynamical simulation against the observed pattern -- the objective exposed by
:meth:`~diffBloch.engine.RefinementEngine.score_orientation`. Per rotation it runs the modified
simplex of Palatinus et al. (*Acta Cryst.* A69, 171-188, 2013): a hexagonal search of ``n_steps``
azimuths at a shrinking tilt radius, greedily accepting the first tilt that lowers wR2 and halving
the radius when none does, until the radius falls below ``min_search_angle``.

The trial orientation is ``orientation @ hexagonal_tilt(azimuth, radius)`` -- a right-multiplied
true rotation, so the non-orthonormal ``U`` measured-cell correction is preserved (no
re-orthonormalisation; see ``KNOWN_ISSUES.md``). The captured ``refinement`` is read-only context
(the Reader pattern; ``design/decisions/stage11-fit-orientation.md``); the deterministic simulation
inside is referentially transparent compute, not a side effect.

The active beam set is held fixed at each orientation's seed selection across the search -- it is
*not* re-filtered per trial as ``diffBloch_private`` does (see ``DIVERGENCE.md``).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from torch import Tensor

from diffBloch.core.solver import Method
from diffBloch.engine import LossFn, RefinementEngine, w_rbragg_loss
from diffBloch.engine.plan import OrientationPlan
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.orientation import hexagonal_tilt
from diffBloch.preprocess.pipeline import PlanStep
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.scoring import build_engine

__all__ = ["fit_orientation"]


def fit_orientation(
    refinement: RefinementSetup,
    *,
    max_search_angle: float = 0.4,
    min_search_angle: float = 0.001,
    n_steps: int = 6,
    loss: LossFn = w_rbragg_loss,
    method: Method = "matrix_exp",
) -> PlanStep:
    """Return a ``Plan -> Plan`` step refining each orientation by Palatinus hexagonal search.

    ``refinement`` (constraint spec, ASU expansion, atomic numbers, thicknesses, seeded params) is
    captured read-only and rejoined to the geometry ``Plan`` via :func:`build_engine`; the
    orientation-invariant ``F_gb`` is computed once and reused across every orientation and trial.
    ``max_search_angle`` / ``min_search_angle`` (degrees) bound the shrinking tilt radius, and
    ``n_steps`` is the number of hexagonal azimuths (6 -> 0, 60, ..., 300 deg, matching the
    private); defaults are the faithful ``diffBloch_private`` values. ``loss`` / ``method``
    configure the engine (``score_orientation`` uses a scaling-optimised wR2 internally,
    independent of ``loss``).
    """

    def run(plan: Plan) -> Plan:
        engine = build_engine(plan, refinement, loss=loss, method=method)
        fgb = engine.fgb(refinement.params)
        orientations = tuple(
            _refine_one(
                engine,
                fgb,
                plan,
                op,
                max_search_angle=max_search_angle,
                min_search_angle=min_search_angle,
                n_steps=n_steps,
            )
            for op in plan.orientations
        )
        return replace(plan, orientations=orientations)

    return run


def _refine_one(
    engine: RefinementEngine,
    fgb: Tensor,
    plan: Plan,
    op: OrientationPlan,
    *,
    max_search_angle: float,
    min_search_angle: float,
    n_steps: int,
) -> OrientationPlan:
    """Palatinus hexagonal search over one orientation; returns the best-scoring OrientationPlan."""
    current = op
    current_score = float(engine.score_orientation(current, fgb))
    beam_hkl = np.asarray(
        op.beam_hkl, dtype=np.int64
    )  # fixed across the search (see DIVERGENCE.md)
    search_angle = max_search_angle
    while search_angle > min_search_angle:
        improved = False
        for n in range(n_steps):
            azimuth = n * 360.0 / n_steps  # hexagonal: 0, 60, ..., 300 deg at n_steps = 6
            orientation = np.asarray(current.orientation, dtype=np.float64) @ hexagonal_tilt(
                azimuth, search_angle
            )
            trial = OrientationPlan.build(
                plan.grid,
                beam_hkl,
                current.pattern,
                energy=current.energy,
                u0=current.u0,
                orientation=orientation,
            )
            trial_score = float(engine.score_orientation(trial, fgb))
            if trial_score < current_score:  # greedy first-improvement; restart at this radius
                current, current_score = trial, trial_score
                improved = True
                break
        if not improved:
            search_angle /= 2.0
    return current
