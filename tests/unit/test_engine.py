"""Stateless refinement forward (``diffBloch.engine``): params -> simulated diffraction -> loss."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diffBloch.core.losses import mse
from diffBloch.core.products import BlochSolution, PatternBatch
from diffBloch.core.symmetry import build_asu_expansion_plan
from diffBloch.engine import OrientationPlan, RefinementEngine, ScatteringGrid
from diffBloch.params import ConstraintSpec, RefinableParams

_ENERGY = 200e3
_CELL = np.eye(3, dtype=np.float64) * 5.0  # 5 A cubic -> reciprocal basis (1/5) I
_BEAM_HKL = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)


def _engine(objective=lambda aligned: mse(aligned.calculated, aligned.observed).sum()):
    grid = ScatteringGrid.from_cell(_CELL, g_max=0.45)  # spans the beam differences (h up to +-2)
    asu_plan = build_asu_expansion_plan(
        np.zeros((1, 3)),
        np.eye(3)[None],
        np.zeros((1, 3)),  # P1: single atom, identity symop
    )
    pattern = PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=torch.tensor([0.9, 0.05, 0.05], dtype=torch.float64),
        sigmas=torch.full((3,), 0.01, dtype=torch.float64),
    )
    orientation = OrientationPlan.build(grid, _BEAM_HKL, pattern, energy=_ENERGY)
    spec = ConstraintSpec(
        fixed_positions=torch.zeros((1, 3), dtype=torch.float64),
        position_mask=torch.ones((1, 3), dtype=torch.float64),
        occupancies=torch.ones(1, dtype=torch.float64),
        reciprocal_basis=grid.reciprocal_basis,
    )
    return RefinementEngine(
        spec=spec,
        asu_plan=asu_plan,
        numbers=torch.tensor([14], dtype=torch.int64),  # silicon
        grid=grid,
        orientations=(orientation,),
        thicknesses=torch.tensor([30.0], dtype=torch.float64),
        objective=objective,
    )


def _params(*, requires_grad: bool = False):
    return RefinableParams(
        asu_positions=torch.zeros((1, 3), dtype=torch.float64, requires_grad=requires_grad),
        uij_raw=(torch.eye(3, dtype=torch.float64)[None] * 0.1).requires_grad_(requires_grad),
    )


def test_simulate_returns_a_solution_per_orientation() -> None:
    engine = _engine()
    solutions = engine.simulate(_params())

    assert len(solutions) == len(engine.orientations)
    solution = solutions[0]
    assert isinstance(solution, BlochSolution)
    assert solution.intensities.shape == (1, _BEAM_HKL.shape[0])  # (T=1, N=3)
    # matrix_exp is unitary on this Hermitian system -> incident flux conserved
    assert torch.allclose(solution.intensities.sum(dim=1), torch.ones(1, dtype=torch.float64))


def test_forward_returns_scalar_objective() -> None:
    loss = _engine().forward(_params())
    assert loss.shape == ()
    assert torch.isfinite(loss) and loss >= 0.0


def test_forward_is_differentiable_through_the_whole_chain() -> None:
    engine = _engine()
    params = _params(requires_grad=True)

    engine.forward(params).backward()

    # gradient flows back through align -> intensity -> propagate -> A -> Fgb -> expand -> constrain
    for grad in (params.asu_positions.grad, params.uij_raw.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
    # the 000-atom structure-factor depends on the ADP; positions at a fixed special site need not
    assert params.uij_raw.grad.abs().sum() > 0


def test_forward_co_locates_invariants_on_the_param_device() -> None:
    # On CPU this is a no-op, but it pins the contract: engine-owned invariants (numbers, grid_hkl,
    # reciprocal_basis, thicknesses, beam_hkl) are moved to the params device at the use site, so a
    # simulated solution lands on the same device as the parameter-derived tensors.
    engine = _engine()
    params = _params()
    (solution,) = engine.simulate(params)
    assert solution.intensities.device == params.asu_positions.device
    assert solution.beam_hkl.device == params.asu_positions.device


def test_forward_rejects_engine_without_orientations() -> None:
    engine = _engine()
    empty = RefinementEngine(
        spec=engine.spec,
        asu_plan=engine.asu_plan,
        numbers=engine.numbers,
        grid=engine.grid,
        orientations=(),
        thicknesses=engine.thicknesses,
        objective=engine.objective,
    )
    with pytest.raises(ValueError, match="no orientations"):
        empty.forward(_params())


def test_scattering_grid_from_cell_spans_difference_support() -> None:
    grid = ScatteringGrid.from_cell(_CELL, g_max=0.45)
    # building beam plans validates the grid covers hkl_j - hkl_i; too-small g_max must raise.
    pattern = PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=torch.zeros(3, dtype=torch.float64),
        sigmas=torch.ones(3, dtype=torch.float64),
    )
    OrientationPlan.build(grid, _BEAM_HKL, pattern, energy=_ENERGY)  # ok

    tiny = ScatteringGrid.from_cell(_CELL, g_max=0.15)  # |g|<=0.15 -> only h=0, no differences
    with pytest.raises(ValueError, match="difference support|gpts is too small"):
        OrientationPlan.build(tiny, _BEAM_HKL, pattern, energy=_ENERGY)
