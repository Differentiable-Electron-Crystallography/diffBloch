"""``fit_orientation``: per-rotation crystal-orientation refinement (Palatinus hexagonal simplex).

A ``Plan -> Plan`` step that sharpens each orientation by minimising the scaling-optimised wR2 of
the full dynamical simulation against the observed pattern -- the objective exposed by
:meth:`~diffBloch.engine.RefinementEngine.score_orientation`. Per rotation it runs the modified
simplex of Palatinus et al. (*Acta Cryst.* A69, 171-188, 2013): a hexagonal search of ``n_steps``
azimuths at a shrinking tilt radius, greedily accepting the first tilt that lowers wR2 and halving
the radius when none does, until the radius falls below ``min_search_angle``.

The trial orientation is ``orientation @ hexagonal_tilt(azimuth, radius)`` -- a right-multiplied
true rotation, so the non-orthonormal ``U`` measured-cell correction is preserved (no
re-orthonormalisation). The captured ``refinement`` is read-only context
the step never mutates; the simulation inside is
deterministic and depends only on its inputs, so it is ordinary computation, not a side effect.

The active beam set is held fixed at each orientation's seed selection across the search -- it is
*not* re-filtered per trial (a divergence from ``diffBloch_private``) *unless* the fit is opted into
coupling via ``coupling=`` (see below). The rocking-curve tilt set carried by the ``Plan`` is
threaded through every trial unchanged, so each candidate is scored under the *same* integration as
the seed -- the fit/eval consistency invariant (the private fits under integration too, passing
``tilts`` into every simplex trial). Ordering ``integrate_rocking_curve`` before this step therefore
couples the fit to the integrated model; with rocking off the tilt set is a single identity,
byte-identical to a static fit.

**Coupling (opt-in, ``coupling=TrialCoupling(...)``)** reproduces the private's objective exactly:
at *every* trial orientation both the SOLVE union (the per-tilt-segment excitation coupling) and the
SCORED set (the Klar window intersected with a resolution cap) are re-derived from scratch, so both
track the trial rather than staying pinned to the seed. The seed is rebuilt through the same builder
so the greedy comparison is always coupled-vs-coupled, and the last accepted trial is already the
coupled-at-fitted-orientation plan -- no separate ``couple_beams`` step is needed. Per-trial
re-selection makes the objective **deliberately non-stationary**: consecutive trials score different
reflection sets (different wR2 denominators). This is faithful -- the private's objective returns
wR2 over each trial's own filtered set -- not a bug. It is affordable because the atomic ``F_gb`` is
computed once and every segment's structure-factor matrix is a cheap gather-index into it (~55 ms of
re-coupling per trial, measured), not a re-derivation.

With ``coupling=None`` (the default) each trial is ``current.with_orientation(...)``, defined on
both the tilt-independent :class:`OrientationPlan` and the
:class:`~diffBloch.engine.plan.SegmentedOrientationPlan`, so an already-segmented plan is fit under
its frozen union.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from diffBloch.core.crystal import orientation_basis
from diffBloch.core.reciprocal import g_vectors, gmax_mask
from diffBloch.core.solver import Method
from diffBloch.engine import RefinementEngine
from diffBloch.engine.plan import OrientationPlanLike, ScatteringGrid, SegmentedOrientationPlan
from diffBloch.preprocess.coupling import tilt_segment_coupling
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.orientation import hexagonal_tilt
from diffBloch.preprocess.pipeline import PlanStep
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.scoring import build_engine
from diffBloch.preprocess.steps.beams import klar_beam_mask
from diffBloch.specs import HexagonalSearch, TrialCoupling

__all__ = ["fit_orientation"]


def fit_orientation(
    refinement: RefinementSetup,
    search: HexagonalSearch,
    *,
    method: Method = "matrix_exp",
    coupling: TrialCoupling | None = None,
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

    ``coupling`` (default ``None``) opts the fit into the private's per-trial re-coupling: a
    :class:`~diffBloch.specs.TrialCoupling` re-derives the solve union and re-selects the scored set
    at every trial orientation (see the module docstring for the non-stationary-objective nuance).
    ``None`` keeps the tilt-independent fit (one fixed beam set across the search).

    The greedy search restarts at the same radius on every accepting (improving) pass, so the
    radius schedule alone does not bound the pass count. Mirroring :func:`iterate_until`,
    ``search.max_iterations`` caps the total passes *per orientation* and a ``RuntimeError`` is
    raised if it is reached -- silent non-convergence is never returned.

    The cap is a 2.0 addition: ``diffBloch_private``'s search has none (it relies on monotone wR2
    descent + the radius floor). The search does terminate by construction for a
    non-degenerate objective -- the cap only guards pathological ridge-walking on (near-)degenerate
    landscapes. Its default of ``2000`` is **calibrated on the quartz anchor under the integrated
    recipe** (slowest legitimate search: 1288 passes across 99 rotations, so 2000 has headroom);
    raise it via config if a dataset with shallower minima trips it.
    """

    def run(plan: Plan) -> Plan:
        engine = build_engine(plan, refinement, method=method)
        fgb = engine.fgb(refinement.params)
        orientations = tuple(
            _refine_one(engine, fgb, plan, op, search=search, coupling=coupling)
            for op in plan.orientations
        )
        return replace(plan, orientations=orientations)

    return run


def _refine_one(
    engine: RefinementEngine,
    fgb: Tensor,
    plan: Plan,
    op: OrientationPlanLike,
    *,
    search: HexagonalSearch,
    coupling: TrialCoupling | None,
) -> OrientationPlanLike:
    """Palatinus hexagonal search over one orientation; returns the best-scoring plan.

    With ``coupling=None`` each trial is ``current.with_orientation(grid, o)`` -- it rebuilds only
    the orientation-dependent bases and reuses the F-gather, so the beam set, rocking-curve tilts,
    and reduction are held fixed across the search (byte-identical to the pre-coupling behaviour).
    With a :class:`~diffBloch.specs.TrialCoupling`, the seed *and* every trial are rebuilt through
    :func:`_coupled_trial`, which re-couples the solve union and re-selects the scored set at that
    orientation -- the deliberately non-stationary faithful objective (see the module docstring).
    """
    grid = plan.grid
    if coupling is None:
        current = op
    else:
        current = _coupled_trial(grid, op, np.asarray(op.orientation, dtype=np.float64), coupling)
    current_score = float(engine.score_orientation(current, fgb))
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
            if coupling is None:
                trial = current.with_orientation(grid, orientation)
            else:
                trial = _coupled_trial(grid, op, orientation, coupling)
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


def _coupled_trial(
    grid: ScatteringGrid,
    op: OrientationPlanLike,
    orientation: NDArray[np.float64],
    coupling: TrialCoupling,
) -> SegmentedOrientationPlan:
    """Re-couple the solve union and re-select the scored set at ``orientation`` (faithful trial).

    Ports one ``diffBloch_private`` objective evaluation: (1) ``tilt_segment_coupling`` re-derives
    the per-tilt-segment excitation union at ``orientation`` (the SOLVE set); (2) the Klar window
    (:func:`klar_beam_mask`) intersected with the scoring-resolution cap
    (:func:`~diffBloch.core.reciprocal.gmax_mask`, an ideal-cell ``|g|`` metric mirroring the
    private's ``resolution_filter``) selects the SCORED set from that union, with ``000`` retained
    (it anchors ``psi0`` and is dropped later on pattern intersection). The observed ``pattern``,
    ``thickness``, and ``tilt_reduction`` are carried from ``op`` unchanged. Fresh per-segment
    gathers are the measured-cheap path -- the atomic ``F_gb`` is untouched.
    """
    cell = np.asarray(grid.cell, dtype=np.float64)
    tilts = np.asarray(op.tilts, dtype=np.float64)
    segments = tilt_segment_coupling(
        coupling.policy,
        np.asarray(grid.grid_hkl, dtype=np.int64),
        cell=cell,
        orientation=orientation,
        tilts=tilts,
        energy=op.energy,
        u0=op.u0,
    )
    union = np.unique(np.concatenate([segment.beam_hkl for segment in segments]), axis=0)
    scored = coupling.scored
    basis = orientation_basis(cell, orientation)
    keep = klar_beam_mask(
        g_vectors(union, basis),
        energy=op.energy,
        u0=op.u0,
        rsg=scored.klar.rsg,
        dsg=scored.klar.dsg,
        semiangle=scored.klar.integration.semiangle,
        geometry=scored.klar.integration.geometry,
    )
    keep |= (union == 0).all(axis=1)  # 000 anchors psi0; dropped later on pattern intersection
    keep &= gmax_mask(union, np.asarray(grid.reciprocal_basis), scored.g_max)  # resolution cap
    return SegmentedOrientationPlan.build(
        grid,
        [(segment.beam_hkl, segment.cover) for segment in segments],
        op.pattern,
        energy=op.energy,
        thickness=op.thickness,
        u0=op.u0,
        orientation=orientation,
        tilts=tilts,
        tilt_reduction=op.tilt_reduction,
        scored_hkl=union[keep],
    )
