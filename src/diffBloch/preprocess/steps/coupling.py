"""``couple_beams``: choose the rocking curve's beam-coupling policy across its tilts.

A ``Plan -> Plan`` step selecting how each rotation couples beams over its rocking-curve tilts, a
discriminated union rather than a boolean toggle:

- :class:`~diffBloch.specs.TiltIndependent` -- the default: one beam set (the ``select_beams``
  active set) shared across every tilt. A no-op here (the shared set is already on the plan).
- :class:`~diffBloch.specs.TiltSegmentUnion` -- the ``diffBloch_private`` policy: partition the
  tilts into contiguous chunks and give each its own boundary-union beam set
  (:func:`~diffBloch.preprocess.coupling.tilt_segment_coupling`), replacing each
  :class:`~diffBloch.engine.plan.OrientationPlan` with a
  :class:`~diffBloch.engine.plan.SegmentedOrientationPlan` the engine reassembles + reduces.

It is the tilt-dependent generalization of ``select_beams``, but unlike it, ``couple_beams`` is
ordered **last** -- after ``fit_orientation`` / ``fit_thickness``. The fits rebuild each orientation
as a plain ``OrientationPlan`` (fixed shared beam set, no per-trial re-coupling -- a recorded
divergence from the private), so any segmentation placed before them would be overwritten; the
segmented geometry is a final, post-fit plan shaping applied only for evaluation. The candidate pool
is the shared grid (``plan.grid.grid_hkl``, which spans the coupling cap), and the tilt set is the
one ``integrate_rocking_curve`` already baked, so this raises if a rotation has fewer than two
tilts.
The upstream ``tilt_reduction`` (e.g. a ``mosaicity`` broadening) is carried through unchanged.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from diffBloch.engine.plan import OrientationPlan, ScatteringGrid, SegmentedOrientationPlan
from diffBloch.preprocess.coupling import tilt_segment_coupling
from diffBloch.preprocess.pipeline import PlanStep, identity
from diffBloch.preprocess.plan import Plan
from diffBloch.specs import CouplingPolicy, TiltIndependent, TiltSegmentUnion

__all__ = ["couple_beams"]


def couple_beams(policy: CouplingPolicy) -> PlanStep:
    """Return a ``Plan -> Plan`` step applying the ``policy`` beam-coupling to every rotation.

    ``policy`` is a :class:`~diffBloch.specs.CouplingPolicy` selected by construction:
    :class:`~diffBloch.specs.TiltIndependent` yields the identity (the shared beam set is kept), and
    :class:`~diffBloch.specs.TiltSegmentUnion` replaces each rotation with its per-chunk
    :class:`~diffBloch.engine.plan.SegmentedOrientationPlan`. Pre-validated, so this never
    re-validates its bounds.
    """
    match policy:
        case TiltIndependent():
            return identity
        case TiltSegmentUnion():

            def run(plan: Plan) -> Plan:
                orientations = tuple(_couple_one(plan.grid, op, policy) for op in plan.orientations)
                return replace(plan, orientations=orientations)

            return run


def _couple_one(
    grid: ScatteringGrid, op: OrientationPlan | SegmentedOrientationPlan, policy: TiltSegmentUnion
) -> SegmentedOrientationPlan:
    """Partition one rotation's rocking curve into boundary-union segments (grid candidates)."""
    if not isinstance(op, OrientationPlan):
        raise TypeError("couple_beams runs on tilt-independent OrientationPlans; already segmented")
    tilts = np.asarray(op.tilts, dtype=np.float64)
    if tilts.shape[0] < 2:
        raise ValueError(
            "couple_beams(TiltSegmentUnion) requires a rocking-curve tilt set; compose "
            f"integrate_rocking_curve first (found {tilts.shape[0]} tilt)"
        )
    segments = tilt_segment_coupling(
        policy,
        np.asarray(grid.grid_hkl, dtype=np.int64),
        cell=np.asarray(grid.cell, dtype=np.float64),
        orientation=np.asarray(op.orientation, dtype=np.float64),
        tilts=tilts,
        energy=op.energy,
        u0=op.u0,
    )
    return SegmentedOrientationPlan.build(
        grid,
        [(segment.beam_hkl, segment.cover) for segment in segments],
        op.pattern,
        energy=op.energy,
        thickness=op.thickness,
        u0=op.u0,
        orientation=op.orientation,
        tilts=tilts,
        tilt_reduction=op.tilt_reduction,
    )
