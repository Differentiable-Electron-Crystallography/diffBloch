"""``integrate_rocking_curve``: the composable ``Plan -> Plan`` tilt-integration step (geometry).

Pins the step's geometry effect (each orientation gains its ``N``-tilt set, sharing its beam set)
and the off-by-default invariant (``sampling = 1`` is the untilted identity). The engine's
sum-over-tilts ``|psi|^2`` reduction is pinned in ``test_engine`` /
``test_core_products::test_integrate_*``.
"""

from __future__ import annotations

import numpy as np
import torch

from diffBloch.core.products import PatternBatch
from diffBloch.engine import OrientationPlan, StructureFactorGrid
from diffBloch.preprocess import Plan, integrate_rocking_curve
from diffBloch.specs import IntegrationGeometry, RockingCurve

_CELL = np.eye(3, dtype=np.float64) * 5.0
_ENERGY = 200e3
_BEAM_HKL = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.int64)  # off the x tilt-axis


def _plan() -> Plan:
    grid = StructureFactorGrid.from_cell(_CELL, g_max=0.45)
    hkl = torch.tensor(_BEAM_HKL, dtype=torch.int64)
    pattern = PatternBatch(
        hkl=hkl,
        intensities=torch.zeros(3, dtype=torch.float64),
        sigmas=torch.ones(3, dtype=torch.float64),
    )
    op = OrientationPlan.build(grid, _BEAM_HKL, pattern, energy=_ENERGY, thickness=(300.0,))
    return Plan(structure_factor_grid=grid, orientations=(op,))


def test_integrate_bakes_the_tilt_set_into_every_orientation() -> None:
    plan = _plan()
    integrated = integrate_rocking_curve(
        RockingCurve(sampling=3, integration=IntegrationGeometry(semiangle=0.5))
    )(plan)
    (op,) = integrated.orientations
    assert op.tilts.shape == (3, 3, 3)
    assert len(op.beam_plans) == 3
    # The tilted sub-orientations are distinct geometry: tilting about x moves these off-axis
    # k-reflections, so the +/- 0.5 deg endpoints give different excitation errors.
    d0, _, d2 = (bp.diagonal for bp in op.beam_plans)
    assert not torch.allclose(d0, d2)
    # Source/compiled preserved: the beam set, thickness, and shared grid are untouched.
    assert torch.equal(op.beam_hkl, plan.orientations[0].beam_hkl)
    assert torch.equal(op.thickness, plan.orientations[0].thickness)
    assert integrated.structure_factor_grid is plan.structure_factor_grid


def test_integrate_unit_sampling_is_the_identity() -> None:
    plan = _plan()
    integrated = integrate_rocking_curve(RockingCurve(sampling=1))(plan)
    (op,) = integrated.orientations
    assert op.tilts.shape == (1, 3, 3)
    assert torch.allclose(op.tilts[0], torch.eye(3, dtype=torch.float64))
    assert len(op.beam_plans) == 1
    # A single angle-0 tilt reproduces the untilted geometry (the off-by-default invariant).
    assert torch.allclose(op.beam_plans[0].diagonal, plan.orientations[0].beam_plans[0].diagonal)
