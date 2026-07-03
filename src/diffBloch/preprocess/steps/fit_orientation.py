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
the step never mutates (``design/decisions/stage11-fit-orientation.md``); the simulation inside is
deterministic and depends only on its inputs, so it is ordinary computation, not a side effect.

The active beam set is held fixed at each orientation's seed selection across the search -- it is
*not* re-filtered per trial as ``diffBloch_private`` does (see ``DIVERGENCE.md``). The rocking-curve
tilt set carried by the ``Plan`` is likewise threaded through every trial unchanged, so each
candidate is scored under the *same* integration as the seed -- the fit/eval consistency invariant
(the private fits under integration too, passing ``tilts`` into every simplex trial). Ordering
``integrate_rocking_curve`` before this step therefore couples the fit to the integrated model; with
rocking off the tilt set is a single identity, byte-identical to a static fit.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from torch import Tensor

from diffBloch.core.solver import Method
from diffBloch.engine import RefinementEngine
from diffBloch.engine.plan import OrientationPlan
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.orientation import hexagonal_tilt
from diffBloch.preprocess.pipeline import PlanStep
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.scoring import build_engine
from diffBloch.specs import HexagonalSearch

__all__ = ["fit_orientation"]


def fit_orientation(
    refinement: RefinementSetup,
    search: HexagonalSearch,
    *,
    method: Method = "matrix_exp",
) -> PlanStep:
    """Return a ``Plan -> Plan`` step refining each orientation by Palatinus hexagonal search.

    ``refinement`` (constraint spec, ASU expansion, atomic numbers, seeded params) is captured
    read-only and rejoined to the geometry ``Plan`` via :func:`build_engine`; the
    orientation-invariant ``F_gb`` is computed once and reused across every orientation and trial.
    ``search`` is a pre-validated :class:`~diffBloch.specs.HexagonalSearch` (invalid bounds are
    unrepresentable, so this function never re-validates): ``max_search_angle`` /
    ``min_search_angle`` (degrees) bound the shrinking tilt radius, and ``n_steps`` is the number of
    hexagonal azimuths (6 -> 0, 60, ..., 300 deg, matching the private). ``method`` configures the
    engine's solver (``score_orientation`` uses a scaling-optimised wR2 internally).

    The greedy search restarts at the same radius on every accepting (improving) pass, so the
    radius schedule alone does not bound the pass count. Mirroring :func:`iterate_until`,
    ``search.max_iterations`` caps the total passes *per orientation* and a ``RuntimeError`` is
    raised if it is reached -- silent non-convergence is never returned.

    The cap is a 2.0 addition: ``diffBloch_private``'s search has none (it relies on monotone wR2
    descent + the radius floor; see DIVERGENCE.md). The search does terminate by construction for a
    non-degenerate objective -- the cap only guards pathological ridge-walking on (near-)degenerate
    landscapes. Its default of ``600`` is **calibrated on the quartz anchor** (slowest legitimate
    search: 526 passes across 99 rotations, so 600 has headroom); raise it via config if a dataset
    with shallower minima trips it (see KNOWN_ISSUES.md).
    """

    def run(plan: Plan) -> Plan:
        engine = build_engine(plan, refinement, method=method)
        fgb = engine.fgb(refinement.params)
        orientations = tuple(
            _refine_one(engine, fgb, plan, op, search=search) for op in plan.orientations
        )
        return replace(plan, orientations=orientations)

    return run


def _refine_one(
    engine: RefinementEngine,
    fgb: Tensor,
    plan: Plan,
    op: OrientationPlan,
    *,
    search: HexagonalSearch,
) -> OrientationPlan:
    """Palatinus hexagonal search over one orientation; returns the best-scoring OrientationPlan."""
    current = op
    current_score = float(engine.score_orientation(current, fgb))
    beam_hkl = np.asarray(
        op.beam_hkl, dtype=np.int64
    )  # fixed across the search (see DIVERGENCE.md)
    # The rocking-curve tilt set is threaded through every trial unchanged, so each candidate is
    # scored under the same integration as the seed (fit/eval consistency). Identity (N=1) when
    # rocking is off -> I @ orientation == orientation, byte-identical to the untilted build.
    tilts = np.asarray(op.tilts, dtype=np.float64)
    reduction = op.tilt_reduction  # mosaicity broadening (if any) is likewise fit under, not after
    gather = op.beam_plans[0].gather  # beam set is fixed across the search; reuse (avoid rebuild)
    search_angle = search.max_search_angle
    for _ in range(search.max_iterations):
        if search_angle <= search.min_search_angle:
            return current
        improved = False
        for n in range(search.n_steps):
            azimuth = n * 360.0 / search.n_steps  # hexagonal: 0, 60, ..., 300 deg at n_steps = 6
            orientation = np.asarray(current.orientation, dtype=np.float64) @ hexagonal_tilt(
                azimuth, search_angle
            )
            trial = OrientationPlan.build(
                plan.grid,
                beam_hkl,
                current.pattern,
                energy=current.energy,
                thickness=current.thickness,
                u0=current.u0,
                orientation=orientation,
                tilts=tilts,
                tilt_reduction=reduction,
                gather=gather,
            )
            trial_score = float(engine.score_orientation(trial, fgb))
            if trial_score < current_score:  # greedy first-improvement; restart at this radius
                current, current_score = trial, trial_score
                improved = True
                break
        if not improved:
            search_angle /= 2.0
    raise RuntimeError(
        f"fit_orientation search did not converge within {search.max_iterations} iterations"
    )
