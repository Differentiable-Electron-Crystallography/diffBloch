"""Slice 11 (C1): the inference terminal ``run_inference`` on a fast synthetic silicon system.

Mirrors ``test_scoring``'s setup (no heavy fixture simulation): a self-consistent orientation whose
observed pattern *is* what the engine simulates scores ``R_obs ~ 0``; a mismatched one scores
higher; the optional ``preprocess`` step is applied; and rotations with no ``I > 3*sigma``
reflection drop out of the aggregate (``nan`` ``r_obs``). The quartz aggregate ``R_obs`` pin lives
in the e2e anchor (fit orientations), where the number is meaningful.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from diffBloch.core.products import PatternBatch
from diffBloch.core.symmetry import build_asu_expansion_plan
from diffBloch.engine import OrientationPlan, RefinementEngine, ScatteringGrid, w_rbragg_loss
from diffBloch.params import ConstraintSpec, RefinableParams
from diffBloch.preprocess import RefinementSetup, run_inference
from diffBloch.preprocess.plan import Plan

_ENERGY = 200e3
_CELL = np.eye(3, dtype=np.float64) * 5.0
_BEAM_HKL = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)
_METHOD = "matrix_exp"  # observed patterns below use this; keep run_inference matched to it


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


def _simulated_intensities(
    grid: ScatteringGrid, asu_plan: object, spec: ConstraintSpec, numbers: torch.Tensor
) -> torch.Tensor:
    """The intensities the engine simulates at ``_params()`` (the self-consistent target)."""
    dummy = PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=torch.zeros(3, dtype=torch.float64),
        sigmas=torch.ones(3, dtype=torch.float64),
    )
    seed = OrientationPlan.build(grid, _BEAM_HKL, dummy, energy=_ENERGY, thickness=(300.0,))
    engine = RefinementEngine(
        spec=spec,
        asu_plan=asu_plan,  # type: ignore[arg-type]
        numbers=numbers,
        grid=grid,
        orientations=(seed,),
        loss=w_rbragg_loss,
        method=_METHOD,
    )
    (solution,) = engine.simulate(_params())
    return solution.intensities[0].detach()


def _orientation(grid: ScatteringGrid, intensities: torch.Tensor, sigma: float) -> OrientationPlan:
    pattern = PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=intensities,
        sigmas=torch.full((3,), sigma, dtype=torch.float64),
    )
    return OrientationPlan.build(grid, _BEAM_HKL, pattern, energy=_ENERGY, thickness=(300.0,))


def test_run_inference_reports_low_r_obs_at_a_self_consistent_pattern() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    intensities = _simulated_intensities(grid, asu_plan, spec, numbers)
    plan = Plan(grid=grid, orientations=(_orientation(grid, intensities, 0.01),))

    result = run_inference(plan, _refinement(asu_plan, spec, numbers), method=_METHOD)

    assert len(result.per_rotation) == 1
    row = result.per_rotation[0]
    assert row.n_beams == 3
    assert 0 < row.n_observed <= 3
    assert math.isfinite(row.r_obs) and row.r_obs < 1e-3
    assert result.n_evaluated == 1
    assert result.mean_r_obs == row.r_obs


def test_run_inference_penalises_a_mismatched_pattern() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    intensities = _simulated_intensities(grid, asu_plan, spec, numbers)
    matched = _orientation(grid, intensities, 0.01)
    mismatched = _orientation(grid, intensities.flip(0) + 0.05, 0.01)
    plan = Plan(grid=grid, orientations=(matched, mismatched))

    result = run_inference(plan, _refinement(asu_plan, spec, numbers), method=_METHOD)

    assert result.per_rotation[1].r_obs > result.per_rotation[0].r_obs


def test_run_inference_applies_the_prepare_step() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    intensities = _simulated_intensities(grid, asu_plan, spec, numbers)
    plan = Plan(
        grid=grid,
        orientations=(_orientation(grid, intensities, 0.01), _orientation(grid, intensities, 0.01)),
    )

    # A prepare step that keeps only the first orientation must shrink the reported rotations.
    keep_first = lambda p: Plan(grid=p.grid, orientations=p.orientations[:1])  # noqa: E731
    result = run_inference(
        plan,
        _refinement(asu_plan, spec, numbers),
        prepare=keep_first,
        method=_METHOD,
    )

    assert len(result.per_rotation) == 1


def test_inference_result_aggregates_only_finite_rotations() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    intensities = _simulated_intensities(grid, asu_plan, spec, numbers)
    observed = _orientation(grid, intensities, 0.01)
    # A huge sigma leaves no reflection above I > 3*sigma, so this rotation's r_obs is nan.
    unobserved = _orientation(grid, intensities, 1e6)
    plan = Plan(grid=grid, orientations=(observed, unobserved))

    result = run_inference(plan, _refinement(asu_plan, spec, numbers), method=_METHOD)

    assert math.isnan(result.per_rotation[1].r_obs)
    assert result.per_rotation[1].n_observed == 0
    assert result.n_evaluated == 1
    assert result.mean_r_obs == result.per_rotation[0].r_obs


def test_run_inference_emits_events_to_the_logger() -> None:
    from diffBloch.observability import InferenceCompleted, RecordingLogger, RotationScored

    grid, asu_plan, spec, numbers = _silicon()
    intensities = _simulated_intensities(grid, asu_plan, spec, numbers)
    plan = Plan(
        grid=grid,
        orientations=(_orientation(grid, intensities, 0.01), _orientation(grid, intensities, 0.01)),
    )
    logger = RecordingLogger()

    result = run_inference(
        plan, _refinement(asu_plan, spec, numbers), method=_METHOD, logger=logger
    )

    # One RotationScored per rotation (in order), then one InferenceCompleted aggregate.
    rotations = [e for e in logger.events if isinstance(e, RotationScored)]
    completed = [e for e in logger.events if isinstance(e, InferenceCompleted)]
    assert [e.index for e in rotations] == [0, 1]
    assert rotations[0].r_obs == result.per_rotation[0].r_obs
    assert len(completed) == 1
    assert completed[0].n_rotations == 2
    assert completed[0].mean_r_obs == result.mean_r_obs
    assert isinstance(logger.events[-1], InferenceCompleted)  # aggregate emitted last
