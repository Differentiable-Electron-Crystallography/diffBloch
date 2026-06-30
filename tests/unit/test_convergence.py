"""Slice 11: ``simulation_converged`` -- the sim-vs-sim convergence check.

Uses the same fast synthetic silicon system as ``test_fit_thickness`` (no heavy fixture sim). The
check compares two *simulations*, so the orientations' observed patterns are irrelevant
placeholders;
what matters is that two Plans with different beam sets produce a non-zero R-factor that the
tolerance threshold gates, and that identical Plans read as converged.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diffBloch.core.products import PatternBatch
from diffBloch.core.symmetry import build_asu_expansion_plan
from diffBloch.engine import OrientationPlan, ScatteringGrid
from diffBloch.params import ConstraintSpec, RefinableParams
from diffBloch.preprocess import RefinementSetup, simulation_converged
from diffBloch.preprocess.plan import Plan
from diffBloch.specs import ConvergenceTolerance

_ENERGY = 200e3
_CELL = np.eye(3, dtype=np.float64) * 5.0
_FULL_BEAMS = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)
_PRUNED_BEAMS = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)  # one fewer coupled beam
_THICKNESS = 300.0


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
    )


def _orientation(grid: ScatteringGrid, beam_hkl: np.ndarray) -> OrientationPlan:
    pattern = PatternBatch(
        hkl=torch.tensor(beam_hkl, dtype=torch.int64),
        intensities=torch.zeros(len(beam_hkl), dtype=torch.float64),
        sigmas=torch.ones(len(beam_hkl), dtype=torch.float64),
    )
    return OrientationPlan.build(grid, beam_hkl, pattern, energy=_ENERGY, thickness=(_THICKNESS,))


def test_identical_plans_read_as_converged() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    refinement = _refinement(asu_plan, spec, numbers)
    plan = Plan(grid=grid, orientations=(_orientation(grid, _FULL_BEAMS),))

    check = simulation_converged(refinement, ConvergenceTolerance(r_factor_threshold=0.005))

    # Comparing a Plan against itself: R-factor is ~0, well under any positive threshold.
    assert check(plan, plan) is True


def test_changed_beam_set_is_gated_by_the_threshold() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    refinement = _refinement(asu_plan, spec, numbers)
    previous = Plan(grid=grid, orientations=(_orientation(grid, _FULL_BEAMS),))
    current = Plan(grid=grid, orientations=(_orientation(grid, _PRUNED_BEAMS),))

    # Dropping a coupled beam changes the dynamical intensities on the shared reflections, so the
    # consecutive-simulation R-factor is non-zero: a tight threshold rejects it, a loose one
    # accepts.
    tight = simulation_converged(refinement, ConvergenceTolerance(r_factor_threshold=1e-6))
    loose = simulation_converged(refinement, ConvergenceTolerance(r_factor_threshold=10.0))
    assert tight(previous, current) is False
    assert loose(previous, current) is True


def test_mismatched_orientation_count_is_rejected() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    refinement = _refinement(asu_plan, spec, numbers)
    one = Plan(grid=grid, orientations=(_orientation(grid, _FULL_BEAMS),))
    two = Plan(grid=grid, orientations=(_orientation(grid, _FULL_BEAMS),) * 2)

    check = simulation_converged(refinement, ConvergenceTolerance())
    with pytest.raises(ValueError, match="share their orientations"):
        check(one, two)
