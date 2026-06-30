"""Slice 11 (5b): ``fit_orientation`` -- the Palatinus hexagonal-tilt orientation refinement.

``hexagonal_tilt`` is checked as a proper rotation (and that right-multiplying it preserves a
non-orthonormal determinant -- the re-orthonormalisation trap). ``fit_orientation`` is then
exercised end-to-end on the same fast synthetic silicon system as ``test_scoring``: an already
self-consistent orientation must be left essentially unchanged. Recovery of a *specific* perturbed
orientation needs a unique minimum, which this trivial high-symmetry system does not have -- that is
deferred to the real quartz e2e.
"""

from __future__ import annotations

import numpy as np
import torch

from diffBloch.core.products import PatternBatch
from diffBloch.core.symmetry import build_asu_expansion_plan
from diffBloch.engine import OrientationPlan, RefinementEngine, ScatteringGrid, w_rbragg_loss
from diffBloch.params import ConstraintSpec, RefinableParams
from diffBloch.preprocess import RefinementSetup, fit_orientation, hexagonal_tilt
from diffBloch.preprocess.plan import Plan

_ENERGY = 200e3
_CELL = np.eye(3, dtype=np.float64) * 5.0
_BEAM_HKL = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)


# --- hexagonal_tilt (pure) ------------------------------------------------------------------------


def test_hexagonal_tilt_is_a_proper_rotation() -> None:
    tilt = hexagonal_tilt(120.0, 0.3)
    assert np.allclose(tilt @ tilt.T, np.eye(3))  # orthogonal
    assert abs(np.linalg.det(tilt) - 1.0) < 1e-12  # proper (det = +1)


def test_hexagonal_tilt_zero_azimuth_is_a_pure_x_rotation() -> None:
    theta = np.deg2rad(0.4)
    rx = np.array(
        [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    )
    assert np.allclose(hexagonal_tilt(0.0, 0.4), rx)


def test_hexagonal_tilt_right_multiply_preserves_a_non_orthonormal_determinant() -> None:
    m = np.diag([1.01, 1.0, 1.0])  # non-orthonormal (det != 1)
    assert abs(np.linalg.det(m @ hexagonal_tilt(60.0, 0.25)) - np.linalg.det(m)) < 1e-12


# --- fit_orientation (synthetic silicon) ----------------------------------------------------------


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


def _refinement(asu_plan: object, spec: ConstraintSpec, numbers: torch.Tensor) -> RefinementSetup:
    return RefinementSetup(
        asu_plan=asu_plan,  # type: ignore[arg-type]
        spec=spec,
        params=_params(),
        numbers=numbers,
        thicknesses=torch.tensor([300.0], dtype=torch.float64),
    )


def _self_consistent(
    grid: ScatteringGrid,
    asu_plan: object,
    spec: ConstraintSpec,
    numbers: torch.Tensor,
    orientation: np.ndarray,
) -> OrientationPlan:
    """An OrientationPlan whose observed pattern is what the engine simulates at ``orientation``."""
    dummy = PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=torch.zeros(3, dtype=torch.float64),
        sigmas=torch.ones(3, dtype=torch.float64),
    )
    seed = OrientationPlan.build(grid, _BEAM_HKL, dummy, energy=_ENERGY, orientation=orientation)
    engine = RefinementEngine(
        spec=spec,
        asu_plan=asu_plan,  # type: ignore[arg-type]
        numbers=numbers,
        grid=grid,
        orientations=(seed,),
        thicknesses=torch.tensor([300.0], dtype=torch.float64),
        loss=w_rbragg_loss,
    )
    (solution,) = engine.simulate(_params())
    observed = PatternBatch(
        hkl=solution.beam_hkl,
        intensities=solution.intensities[0].detach(),
        sigmas=torch.full((3,), 0.01, dtype=torch.float64),
    )
    return OrientationPlan.build(grid, _BEAM_HKL, observed, energy=_ENERGY, orientation=orientation)


def test_fit_orientation_leaves_a_self_consistent_orientation_unchanged() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    true_orientation = np.eye(3, dtype=np.float64)
    matched = _self_consistent(grid, asu_plan, spec, numbers, true_orientation)
    refinement = _refinement(asu_plan, spec, numbers)

    (refined,) = fit_orientation(refinement)(Plan(grid=grid, orientations=(matched,))).orientations

    # Already optimal: the search must not wander it away from the seed.
    assert np.linalg.norm(np.asarray(refined.orientation) - true_orientation) < 1e-2
