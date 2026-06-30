"""Slice 11 (5a): the wR2-scoring seam (``optimal_scale``, ``score_orientation``, ``build_engine``).

Pure scaling on hand-checkable intensities, then the per-orientation scaling-optimised wR2 on a fast
synthetic silicon system (mirroring ``test_engine``'s setup so no heavy fixture sim is needed).
"""

from __future__ import annotations

import numpy as np
import torch

from diffBloch.core.losses import optimal_scale, w_rbragg
from diffBloch.core.products import PatternBatch
from diffBloch.core.symmetry import build_asu_expansion_plan
from diffBloch.engine import OrientationPlan, RefinementEngine, ScatteringGrid, w_rbragg_loss
from diffBloch.params import ConstraintSpec, RefinableParams
from diffBloch.preprocess import RefinementSetup, build_engine, score_orientations
from diffBloch.preprocess.plan import Plan

_ENERGY = 200e3
_CELL = np.eye(3, dtype=np.float64) * 5.0
_BEAM_HKL = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)


# --- optimal_scale (pure) -------------------------------------------------------------------------


def test_optimal_scale_recovers_a_known_constant_factor() -> None:
    observed = torch.tensor([0.9, 0.05, 0.05], dtype=torch.float64)
    sigmas = torch.full((3,), 0.01, dtype=torch.float64)

    # calculated = 2 x observed -> the absolute scale that recovers observed is 0.5, wR2 -> 0.
    scale, value = optimal_scale(2 * observed, observed, sigmas)
    assert abs(float(scale) - 0.5) < 0.02
    assert float(value) < 1e-6
    assert torch.allclose(2 * observed * scale, observed, atol=1e-3)


def test_optimal_scale_returns_the_grid_minimum() -> None:
    observed = torch.tensor([1.0, 0.4, 0.2, 0.1], dtype=torch.float64)
    sigmas = torch.full((4,), 0.02, dtype=torch.float64)
    calculated = torch.tensor([0.7, 0.5, 0.25, 0.05], dtype=torch.float64)

    scale, value = optimal_scale(calculated, observed, sigmas)
    # No other absolute scale on the search grid beats the returned one.
    ratio = float(observed.sum() / calculated.sum())
    for factor in np.linspace(0.02, 2.0, 100):
        other = w_rbragg(factor * ratio * calculated, observed, sigmas)
        assert float(value) <= float(other) + 1e-12


# --- score_orientation + seam (synthetic silicon) -------------------------------------------------


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
        thicknesses=torch.tensor([300.0], dtype=torch.float64),
        loss=w_rbragg_loss,
    )


def _self_consistent_orientation(
    grid: ScatteringGrid, asu_plan: object, spec: ConstraintSpec, numbers: torch.Tensor
) -> tuple[OrientationPlan, torch.Tensor]:
    """An orientation whose observed pattern is what the engine simulates at ``_params()``."""
    dummy = PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=torch.zeros(3, dtype=torch.float64),
        sigmas=torch.ones(3, dtype=torch.float64),
    )
    seed = OrientationPlan.build(grid, _BEAM_HKL, dummy, energy=_ENERGY)
    (solution,) = _engine(grid, asu_plan, spec, numbers, seed).simulate(_params())
    observed = PatternBatch(
        hkl=solution.beam_hkl,
        intensities=solution.intensities[0].detach(),
        sigmas=torch.full((3,), 0.01, dtype=torch.float64),
    )
    return OrientationPlan.build(grid, _BEAM_HKL, observed, energy=_ENERGY), solution.intensities[0]


def test_score_orientation_vanishes_at_a_self_consistent_pattern() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    orientation, _ = _self_consistent_orientation(grid, asu_plan, spec, numbers)
    engine = _engine(grid, asu_plan, spec, numbers, orientation)

    score = engine.score_orientation(orientation, engine.fgb(_params()))
    assert score.shape == ()
    assert torch.isfinite(score) and float(score) < 1e-4


def test_score_orientation_penalises_a_mismatched_pattern() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    matched, intensities = _self_consistent_orientation(grid, asu_plan, spec, numbers)
    # Perturb the observed pattern: a worse fit must score strictly higher than the matched one.
    perturbed_pattern = PatternBatch(
        hkl=matched.pattern.hkl,
        intensities=(intensities.detach().flip(0) + 0.05),
        sigmas=torch.full((3,), 0.01, dtype=torch.float64),
    )
    perturbed = OrientationPlan.build(grid, _BEAM_HKL, perturbed_pattern, energy=_ENERGY)
    engine = _engine(grid, asu_plan, spec, numbers, matched)
    fgb = engine.fgb(_params())

    assert float(engine.score_orientation(perturbed, fgb)) > float(
        engine.score_orientation(matched, fgb)
    )


def test_build_engine_wires_plan_geometry_and_structure_context() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    orientation, _ = _self_consistent_orientation(grid, asu_plan, spec, numbers)
    plan = Plan(grid=grid, orientations=(orientation,))
    refinement = RefinementSetup(
        asu_plan=asu_plan,  # type: ignore[arg-type]
        spec=spec,
        params=_params(),
        numbers=numbers,
        thicknesses=torch.tensor([300.0], dtype=torch.float64),
    )

    engine = build_engine(plan, refinement)
    assert engine.grid is plan.grid
    assert engine.orientations is plan.orientations
    assert engine.spec is refinement.spec
    assert engine.asu_plan is refinement.asu_plan
    assert engine.numbers is refinement.numbers
    assert engine.thicknesses is refinement.thicknesses

    scores = score_orientations(plan, refinement)
    assert len(scores) == len(plan.orientations)
    assert all(torch.isfinite(s) and s.shape == () for s in scores)
