"""``mosaicity``: the composable ``Plan -> Plan`` step that broadens the rocking-curve reduction.

Pins that the step swaps each orientation's tilt reduction to ``MosaicSmoothed`` without touching
geometry (a ``replace``, like ``fit_thickness``), the off-by-default invariant (no step = the plain
sum), and the ordering guards (mosaicity needs a rocking-curve tilt set, and the window may not
exceed it). The reduction *arithmetic* (moving-average then sum) is pinned in
``test_core_products::test_integrate_mosaic_*``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diffBloch.core.products import MosaicSmoothed, PatternBatch, PlainSum
from diffBloch.engine import OrientationPlan, ScatteringGrid
from diffBloch.preprocess import Plan, integrate_rocking_curve, mosaicity
from diffBloch.specs import IntegrationGeometry, Mosaicity, RockingCurve

_CELL = np.eye(3, dtype=np.float64) * 5.0
_ENERGY = 200e3
_BEAM_HKL = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.int64)


def _plan() -> Plan:
    grid = ScatteringGrid.from_cell(_CELL, g_max=0.45)
    hkl = torch.tensor(_BEAM_HKL, dtype=torch.int64)
    pattern = PatternBatch(
        hkl=hkl,
        intensities=torch.zeros(3, dtype=torch.float64),
        sigmas=torch.ones(3, dtype=torch.float64),
    )
    op = OrientationPlan.build(grid, _BEAM_HKL, pattern, energy=_ENERGY, thickness=(300.0,))
    return Plan(grid=grid, orientations=(op,))


def _integrated(sampling: int) -> Plan:
    return integrate_rocking_curve(
        RockingCurve(sampling=sampling, integration=IntegrationGeometry(semiangle=0.5))
    )(_plan())


def test_mosaicity_sets_the_reduction_without_touching_geometry() -> None:
    integrated = _integrated(5)
    mosaic = mosaicity(Mosaicity(window=3))(integrated)
    (before,) = integrated.orientations
    (after,) = mosaic.orientations
    assert isinstance(before.tilt_reduction, PlainSum)  # integration alone leaves the plain sum
    assert after.tilt_reduction == MosaicSmoothed(3)  # mosaicity swaps in the broadened reduction
    # Geometry is untouched -- only the reduction descriptor changed (replace, like fit_thickness).
    assert after.beam_plans is before.beam_plans
    assert torch.equal(after.tilts, before.tilts)
    assert torch.equal(after.beam_hkl, before.beam_hkl)


def test_mosaicity_requires_a_rocking_curve_tilt_set() -> None:
    # On an un-integrated plan every orientation has a single (identity) tilt, so a moving average
    # is degenerate; the step fails fast rather than silently no-op inside the forward pass.
    with pytest.raises(ValueError, match="requires a rocking-curve tilt set"):
        mosaicity(Mosaicity())(_plan())


def test_mosaicity_window_may_not_exceed_the_tilt_count() -> None:
    with pytest.raises(ValueError, match="window 7 exceeds the 5 rocking-curve tilts"):
        mosaicity(Mosaicity(window=7))(_integrated(5))
