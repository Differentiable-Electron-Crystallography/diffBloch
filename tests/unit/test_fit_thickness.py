"""Slice 11 (6b): ``fit_thickness`` -- per-rotation thickness grid search.

Recovery + invariants on the same fast synthetic silicon system as ``test_scoring`` /
``test_fit_orientation`` (no heavy fixture sim). The observed pattern is built by simulating at a
known thickness, so the grid search has a ground truth to recover.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diffBloch.core.products import PatternBatch
from diffBloch.core.symmetry import build_asu_expansion_plan
from diffBloch.engine import OrientationPlan, RefinementEngine, ScatteringGrid, w_rbragg_loss
from diffBloch.params import ConstraintSpec, RefinableParams
from diffBloch.preprocess import RefinementSetup, fit_thickness
from diffBloch.preprocess.plan import Plan

_ENERGY = 200e3
_CELL = np.eye(3, dtype=np.float64) * 5.0
_BEAM_HKL = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)
_TRUE_THICKNESS = 300.0


def _silicon() -> tuple[ScatteringGrid, object, ConstraintSpec, torch.Tensor]:
    grid = ScatteringGrid.from_cell(_CELL, g_max=0.45)
    asu_plan = build_asu_expansion_plan(np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, 3)))
    spec = ConstraintSpec(
        fixed_positions=torch.zeros((1, 3), dtype=torch.float64),
        refinable_position_mask=torch.ones((1, 3), dtype=torch.float64),
        occupancies=torch.ones(1, dtype=torch.float64),
        reciprocal_basis=grid.reciprocal_basis,
    )
    numbers = torch.tensor([14], dtype=torch.int64)
    return grid, asu_plan, spec, numbers


def _params() -> RefinableParams:
    return RefinableParams(
        asu_positions=torch.zeros((1, 3), dtype=torch.float64),
        uij_raw=torch.eye(3, dtype=torch.float64)[None] * 0.1,
    )


def _engine(
    grid: ScatteringGrid,
    asu_plan: object,
    spec: ConstraintSpec,
    numbers: torch.Tensor,
    orientation: OrientationPlan,
) -> RefinementEngine:
    return RefinementEngine(
        spec=spec,
        asu_plan=asu_plan,  # type: ignore[arg-type]
        numbers=numbers,
        grid=grid,
        orientations=(orientation,),
        loss=w_rbragg_loss,
    )


def _refinement(asu_plan: object, spec: ConstraintSpec, numbers: torch.Tensor) -> RefinementSetup:
    return RefinementSetup(
        asu_plan=asu_plan,  # type: ignore[arg-type]
        spec=spec,
        params=_params(),
        numbers=numbers,
    )


def _observed_at(
    grid: ScatteringGrid,
    asu_plan: object,
    spec: ConstraintSpec,
    numbers: torch.Tensor,
    thickness: float,
) -> PatternBatch:
    """The pattern the engine simulates at ``thickness`` -- the ground truth the fit recovers."""
    dummy = PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=torch.zeros(3, dtype=torch.float64),
        sigmas=torch.ones(3, dtype=torch.float64),
    )
    seed = OrientationPlan.build(grid, _BEAM_HKL, dummy, energy=_ENERGY, thickness=(thickness,))
    (solution,) = _engine(grid, asu_plan, spec, numbers, seed).simulate(_params())
    return PatternBatch(
        hkl=solution.beam_hkl,
        intensities=solution.intensities[0].detach(),
        sigmas=torch.full((3,), 0.01, dtype=torch.float64),
    )


def test_fit_thickness_recovers_the_simulated_thickness() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    observed = _observed_at(grid, asu_plan, spec, numbers, _TRUE_THICKNESS)
    # Seed the orientation at a deliberately wrong thickness: the fit must overwrite it.
    op = OrientationPlan.build(grid, _BEAM_HKL, observed, energy=_ENERGY, thickness=(900.0,))
    plan = Plan(grid=grid, orientations=(op,))

    # grid = [200, 250, 300, 350, 400] includes the true 300 A.
    fitted = fit_thickness(
        _refinement(asu_plan, spec, numbers), min_thickness=200.0, max_thickness=400.0, n_steps=5
    )(plan)

    baked = fitted.orientations[0].thickness
    assert baked.shape == (1,)
    assert float(baked[0]) == _TRUE_THICKNESS


def test_fit_thickness_leaves_geometry_untouched() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    observed = _observed_at(grid, asu_plan, spec, numbers, _TRUE_THICKNESS)
    op = OrientationPlan.build(grid, _BEAM_HKL, observed, energy=_ENERGY, thickness=(900.0,))
    plan = Plan(grid=grid, orientations=(op,))

    fitted = fit_thickness(
        _refinement(asu_plan, spec, numbers), min_thickness=200.0, max_thickness=400.0, n_steps=5
    )(plan)

    assert fitted.grid is plan.grid
    assert len(fitted.orientations) == 1
    fop = fitted.orientations[0]
    # Only thickness changed: orientation / energy / beam set / observed pattern are preserved.
    assert torch.equal(fop.orientation, op.orientation)
    assert fop.energy == op.energy
    assert torch.equal(fop.beam_hkl, op.beam_hkl)
    assert torch.equal(fop.pattern.intensities, op.pattern.intensities)
    assert not torch.equal(fop.thickness, op.thickness)


def test_fit_thickness_single_step_bakes_the_lower_bound() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    observed = _observed_at(grid, asu_plan, spec, numbers, _TRUE_THICKNESS)
    op = OrientationPlan.build(grid, _BEAM_HKL, observed, energy=_ENERGY, thickness=(900.0,))
    plan = Plan(grid=grid, orientations=(op,))

    # n_steps == 1 -> the only candidate is min_thickness.
    fitted = fit_thickness(
        _refinement(asu_plan, spec, numbers), min_thickness=123.0, max_thickness=400.0, n_steps=1
    )(plan)
    assert float(fitted.orientations[0].thickness[0]) == 123.0


def test_fit_thickness_rejects_invalid_bounds() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    refinement = _refinement(asu_plan, spec, numbers)
    with pytest.raises(ValueError, match="thickness bounds must be positive"):
        fit_thickness(refinement, min_thickness=-1.0, max_thickness=400.0)
    with pytest.raises(ValueError, match="max_thickness must exceed min_thickness"):
        fit_thickness(refinement, min_thickness=400.0, max_thickness=400.0)
    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        fit_thickness(refinement, min_thickness=5.0, max_thickness=400.0, n_steps=0)
