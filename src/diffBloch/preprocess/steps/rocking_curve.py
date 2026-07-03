"""``integrate_rocking_curve``: bake each rotation's rocking-curve tilt set into the geometry.

A ``Plan -> Plan`` step that replaces each rotation's single static geometry with ``N`` tilted
sub-orientations spanning the integration semi-angle -- the forward model then sums ``|psi|^2`` over
the tilts (an incoherent rotation-frame integration; see
:meth:`diffBloch.core.products.BlochSolution.integrate`). The rocking curve is *the* scientific
enabling structure of the rotation-electron-diffraction forward model, so it is a composable,
toggleable step rather than baked into ``from_experiment``: composing it in with
``rocking.sampling == 1`` (a single angle-0 tilt) is the identity, so appending it off leaves the
``Plan`` byte-identical (see ``design/decisions/stage11-rocking-curve.md``).

It is pure geometry -- no engine, no structure factors, no refinement: the tilts depend only on the
fixed ``RockingCurve`` and each settled nominal orientation, so they are precompiled into the
``Plan`` exactly like the per-orientation beam plans. Ordered *last* in the pipeline (after
``select_beams`` / ``fit_orientation`` / ``fit_thickness``, which score on the fast single-solve):
the fits settle the nominal orientation and the one shared beam set, then this bakes the integration
geometry those results are held fixed at, reusing that beam set across every tilt.

Faithful to ``diffBloch_private``'s ``generate_integration_rotation_matrices`` +
``get_integrated_intensities`` (which the private wires in unconditionally); 2.0 exposes it as a
composed unit.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from diffBloch.engine.plan import OrientationPlan
from diffBloch.preprocess.orientation import rocking_curve_tilts
from diffBloch.preprocess.pipeline import PlanStep
from diffBloch.preprocess.plan import Plan
from diffBloch.specs import RockingCurve

__all__ = ["integrate_rocking_curve"]


def integrate_rocking_curve(rocking: RockingCurve) -> PlanStep:
    """Return a ``Plan -> Plan`` step baking each rotation's rocking-curve tilt geometry.

    ``rocking`` is a pre-validated :class:`~diffBloch.specs.RockingCurve` (invalid bounds are
    unrepresentable, so this never re-validates): ``rocking.sampling`` tilts about the goniometer
    axis spanning ``+/- rocking.semiangle`` degrees (``rocking.geometry`` selects the sweep). The
    tilt matrices are orientation-independent, so they are generated once and left-multiplied onto
    every rotation's nominal orientation (``R_tilt @ orientation``); each rotation is rebuilt with
    its ``N`` sub-orientations sharing its one existing beam set.
    """
    tilts = rocking_curve_tilts(rocking.semiangle, rocking.sampling, geometry=rocking.geometry)

    def run(plan: Plan) -> Plan:
        orientations = tuple(_integrate_one(plan, op, tilts) for op in plan.orientations)
        return replace(plan, orientations=orientations)

    return run


def _integrate_one(plan: Plan, op: OrientationPlan, tilts: np.ndarray) -> OrientationPlan:
    """Rebuild one orientation with the tilt set baked in, reusing its beam set and thickness."""
    return OrientationPlan.build(
        plan.grid,
        np.asarray(op.beam_hkl, dtype=np.int64),
        op.pattern,
        energy=op.energy,
        thickness=op.thickness,
        u0=op.u0,
        orientation=op.orientation,
        tilts=tilts,
        gather=op.beam_plans[0].gather,  # same beam set; reuse the seed's gather (avoid rebuild)
    )
