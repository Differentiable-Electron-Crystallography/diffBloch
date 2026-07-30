"""``couple_beams``: choose the rocking curve's beam-coupling policy across its tilts.

A ``Plan -> Plan`` step selecting how each rotation couples beams over its rocking-curve tilts, a
discriminated union rather than a boolean toggle:

- :class:`~diffBloch.specs.TiltIndependent` -- the default: one beam set (the ``select_beams``
  active set) shared across every tilt. A no-op here (the shared set is already on the plan).
- :class:`~diffBloch.specs.UnionCoupling` -- a policy that partitions
  the tilts into contiguous chunks and gives each its own boundary-union beam set
  (:func:`~diffBloch.preprocess.coupling.build_coupling_segments`), replacing each
  :class:`~diffBloch.engine.plan.OrientationPlan` with a
  :class:`~diffBloch.engine.plan.CoupledOrientationPlan` the engine reassembles + reduces.

It is the tilt-dependent generalization of ``select_beams``. The default app recipe does not use
this step -- it couples *per trial* inside ``fit_orientation`` (``coupling=...``); ``couple_beams``
is the explicit composable step when a caller wants to settle a coupled ``Plan`` directly. It is
only meaningful once ``select_beams`` has established the Klar-selected scored set (before it, an
orientation's
``alignment`` is the seed-pool intersection, too wide). It accepts either a plain
:class:`~diffBloch.engine.plan.OrientationPlan` (the first application) or an already-coupled
:class:`~diffBloch.engine.plan.CoupledOrientationPlan` (the re-couple), re-deriving each
rotation's segments from its current source fields (``orientation`` / ``tilts`` / ``energy`` /
``u0``). The
candidate pool is the shared grid (``plan.structure_factor_grid.structure_factor_hkl``, which spans
the coupling cap), and the
tilt set is the one ``integrate_rocking_curve`` already baked, so this raises if a rotation has
fewer than two tilts. The upstream ``tilt_reduction`` (e.g. a ``mosaicity`` broadening) is carried
through unchanged.

Crucially it decouples the two reflection sets the plan otherwise conflates: the *solve* set
expands to the excitation coupling union, while the *scored* set stays pinned to
the pre-couple ``select_beams`` selection (``op.alignment.hkl``), intersected with the union.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from diffBloch.engine.plan import CoupledOrientationPlan, OrientationPlanLike, StructureFactorGrid
from diffBloch.preprocess.coupling import build_coupling_segments
from diffBloch.preprocess.pipeline import PlanStep, as_step, identity
from diffBloch.preprocess.plan import Plan, require_built_plans
from diffBloch.specs import CouplingPolicy, PerTiltCoupling, TiltIndependent, UnionCoupling

__all__ = ["couple_beams"]


def couple_beams(policy: CouplingPolicy) -> PlanStep:
    """Return a ``Plan -> Plan`` step applying the ``policy`` beam-coupling to every rotation.

    ``policy`` is a :class:`~diffBloch.specs.CouplingPolicy` selected by construction:
    :class:`~diffBloch.specs.TiltIndependent` yields the identity (the shared beam set is kept), and
    :class:`~diffBloch.specs.UnionCoupling` replaces each rotation with its per-chunk
    :class:`~diffBloch.engine.plan.CoupledOrientationPlan`. Pre-validated, so this never
    re-validates its bounds.
    """
    match policy:
        case TiltIndependent():
            # A no-op on the plan, but recorded as couple_beams(TiltIndependent) so provenance
            # records the coupling decision (distinct from a plain unrecorded identity).
            return as_step("couple_beams", policy, identity.run)
        case UnionCoupling() | PerTiltCoupling():

            def run(plan: Plan) -> Plan:
                orientations = tuple(
                    _couple_one(plan.structure_factor_grid, op, policy)
                    for op in require_built_plans(plan)
                )
                return replace(plan, orientations=orientations)

            return as_step("couple_beams", policy, run)


def _couple_one(
    grid: StructureFactorGrid,
    op: OrientationPlanLike,
    policy: UnionCoupling | PerTiltCoupling,
) -> CoupledOrientationPlan:
    """Partition one rotation's rocking curve into boundary-union segments at its orientation.

    Accepts either plan type: the segments are re-derived at the plan's *current* ``orientation``
    from its source fields, so this serves both as the initial coupling (a plain
    :class:`OrientationPlan`) and as a re-couple at the fitted orientation (an already-coupled
    :class:`CoupledOrientationPlan`). Scoring stays pinned to ``op.alignment.hkl`` -- the
    ``select_beams`` set, which both plan types carry and which the fits preserve.
    """
    tilts = np.asarray(op.tilts, dtype=np.float64)
    if tilts.shape[0] < 2:
        raise ValueError(
            "couple_beams(UnionCoupling) requires a rocking-curve tilt set; compose "
            f"integrate_rocking_curve first (found {tilts.shape[0]} tilt)"
        )
    segments = build_coupling_segments(
        policy,
        np.asarray(grid.structure_factor_hkl, dtype=np.int64),
        cell=np.asarray(grid.cell, dtype=np.float64),
        orientation=np.asarray(op.orientation, dtype=np.float64),
        tilts=tilts,
        energy=op.energy,
        u0=op.u0,
    )
    return CoupledOrientationPlan.build(
        grid,
        [(segment.union_hkl, segment.covered_tilt_indices) for segment in segments],
        op.pattern,
        energy=op.energy,
        thickness=op.thickness,
        u0=op.u0,
        orientation=op.orientation,
        tilts=tilts,
        tilt_reduction=op.tilt_reduction,
        # Pin scoring to the pre-couple select_beams selection (Klar set), not the enlarged union.
        scored_hkl=np.asarray(op.alignment.hkl, dtype=np.int64),
    )
