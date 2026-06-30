"""Stage 11 (1/n): the ``Plan`` spine + the preprocess ``Plan -> Plan`` pipeline combinators."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from diffBloch.core.products import PatternBatch
from diffBloch.engine import OrientationPlan, ScatteringGrid
from diffBloch.preprocess import Plan, identity, iterate_until, pipeline

_ENERGY = 200e3
_CELL = np.eye(3, dtype=np.float64) * 5.0
_BEAM_HKL = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)


def _orientation() -> tuple[ScatteringGrid, OrientationPlan]:
    grid = ScatteringGrid.from_cell(_CELL, g_max=0.45)
    pattern = PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=torch.zeros(3, dtype=torch.float64),
        sigmas=torch.ones(3, dtype=torch.float64),
    )
    return grid, OrientationPlan.build(grid, _BEAM_HKL, pattern, energy=_ENERGY, thickness=(300.0,))


def _plan(n_orientations: int = 1) -> Plan:
    grid, orient = _orientation()
    return Plan(grid=grid, orientations=(orient,) * n_orientations)


# Synthetic steps using the orientation count as the observable: grow appends a copy of the first
# orientation; truncate keeps only the first. (Real steps -- converge_numerics/fit_* -- come later.)
def _grow(plan: Plan) -> Plan:
    return dataclasses.replace(plan, orientations=plan.orientations + plan.orientations[:1])


def _truncate(plan: Plan) -> Plan:
    return dataclasses.replace(plan, orientations=plan.orientations[:1])


def test_plan_bundles_grid_and_orientations() -> None:
    grid, orient = _orientation()
    plan = Plan(grid=grid, orientations=(orient,))
    assert plan.grid is grid
    assert plan.orientations == (orient,)


def test_identity_is_a_noop() -> None:
    plan = _plan(2)
    assert identity(plan) is plan


def test_pipeline_applies_steps_in_order() -> None:
    start = _plan(1)
    # order matters: grow->truncate collapses back to 1; truncate->grow ends at 2.
    assert len(pipeline([_grow, _truncate])(start).orientations) == 1
    assert len(pipeline([_truncate, _grow])(start).orientations) == 2
    assert len(pipeline([_grow, _grow])(start).orientations) == 3


def test_pipeline_empty_is_identity() -> None:
    plan = _plan(1)
    assert pipeline([])(plan) is plan


def test_iterate_until_reaches_a_fixpoint() -> None:
    out = iterate_until(_grow, until=lambda prev, cur: len(cur.orientations) >= 4)(_plan(1))
    assert len(out.orientations) == 4


def test_iterate_until_raises_when_not_converged_within_max_iterations() -> None:
    never = iterate_until(_grow, until=lambda prev, cur: False, max_iterations=3)
    with pytest.raises(RuntimeError, match="did not converge"):
        never(_plan(1))


def test_iterate_until_rejects_non_positive_max_iterations() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        iterate_until(_grow, until=lambda prev, cur: True, max_iterations=0)
