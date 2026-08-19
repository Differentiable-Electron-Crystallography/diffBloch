"""Slice 11 (6b): ``optimize_thickness`` -- per-rotation thickness grid search.

Recovery + invariants on the same fast synthetic silicon system as ``test_scoring`` /
``test_fit_orientation`` (no heavy fixture sim). The observed pattern is built by simulating at a
known thickness, so the grid search has a ground truth to recover.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from tests.unit.synthetic import make_constraint_spec

from diffBloch.app.loggers import ReportLogger
from diffBloch.core.products import PatternBatch
from diffBloch.core.symmetry import build_asu_expansion_plan
from diffBloch.engine import OrientationPlan, RefinementEngine, StructureFactorGrid, w_rbragg_loss
from diffBloch.observability import EventRecord
from diffBloch.params import ConstraintSpec, RefinableParams
from diffBloch.preprocess import RefinementSetup, optimize_thickness
from diffBloch.preprocess.plan import Plan
from diffBloch.specs import ThicknessGrid

_ENERGY = 200e3
_CELL = np.eye(3, dtype=np.float64) * 5.0
_BEAM_HKL = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)
_TRUE_THICKNESS = 300.0


def _records(path: Path) -> list[EventRecord]:
    return [EventRecord.model_validate_json(line) for line in path.read_text().splitlines()]


def _silicon() -> tuple[StructureFactorGrid, object, ConstraintSpec, torch.Tensor]:
    grid = StructureFactorGrid.from_cell(_CELL, g_max=0.45)
    asu_plan = build_asu_expansion_plan(np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, 3)))
    spec = make_constraint_spec(reciprocal_basis=grid.reciprocal_basis)
    numbers = torch.tensor([14], dtype=torch.int64)
    return grid, asu_plan, spec, numbers


def _params() -> RefinableParams:
    return RefinableParams(
        asu_positions=torch.zeros((1, 3), dtype=torch.float64),
        uij_raw=torch.eye(3, dtype=torch.float64)[None] * 0.1,
    )


def _engine(
    grid: StructureFactorGrid,
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
    grid: StructureFactorGrid,
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
    plan = Plan(structure_factor_grid=grid, orientations=(op,))

    # grid = [200, 250, 300, 350, 400] includes the true 300 A.
    fitted = optimize_thickness(
        _refinement(asu_plan, spec, numbers),
        ThicknessGrid(min_thickness=200.0, max_thickness=400.0, n_steps=5),
    )(plan)

    baked = fitted.orientations[0].thickness
    assert baked.shape == (1,)
    assert float(baked[0]) == _TRUE_THICKNESS


def test_fit_thickness_leaves_geometry_untouched() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    observed = _observed_at(grid, asu_plan, spec, numbers, _TRUE_THICKNESS)
    op = OrientationPlan.build(grid, _BEAM_HKL, observed, energy=_ENERGY, thickness=(900.0,))
    plan = Plan(structure_factor_grid=grid, orientations=(op,))

    fitted = optimize_thickness(
        _refinement(asu_plan, spec, numbers),
        ThicknessGrid(min_thickness=200.0, max_thickness=400.0, n_steps=5),
    )(plan)

    assert fitted.structure_factor_grid is plan.structure_factor_grid
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
    plan = Plan(structure_factor_grid=grid, orientations=(op,))

    # n_steps == 1 -> the only candidate is min_thickness.
    fitted = optimize_thickness(
        _refinement(asu_plan, spec, numbers),
        ThicknessGrid(min_thickness=123.0, max_thickness=400.0, n_steps=1),
    )(plan)
    assert float(fitted.orientations[0].thickness[0]) == 123.0


def test_fit_thickness_emits_one_thicknessfitted_per_rotation(tmp_path: Path) -> None:
    """The progress stream: one ThicknessOptimized per rotation, in plan order, tied to the result."""
    grid, asu_plan, spec, numbers = _silicon()
    observed = _observed_at(grid, asu_plan, spec, numbers, _TRUE_THICKNESS)
    observed = replace(observed, rotation_index=42)
    op = OrientationPlan.build(grid, _BEAM_HKL, observed, energy=_ENERGY, thickness=(900.0,))
    plan = Plan(structure_factor_grid=grid, orientations=(op, op))  # two rotations -> indices 0, 1

    path = tmp_path / "report.jsonl"
    log = ReportLogger(path)
    fitted = optimize_thickness(
        _refinement(asu_plan, spec, numbers),
        ThicknessGrid(min_thickness=200.0, max_thickness=400.0, n_steps=5),
        logger=log,
    )(plan)

    events = [event for event in _records(path) if event.event_type == "ThicknessOptimized"]
    assert [event.rotation_index for event in events] == [42, 42]
    # each event's thickness is the value baked onto the matching returned orientation
    for index, event in enumerate(events):
        assert event.payload["thickness"] == float(fitted.orientations[index].thickness[0])


# --- device knob (the grid search runs on the accelerator; params.device is authoritative) --------


def test_fit_thickness_device_cpu_is_a_no_op() -> None:
    """``device='cpu'`` reproduces the default fit exactly -- device plumbing inert on CPU."""
    grid, asu_plan, spec, numbers = _silicon()
    observed = _observed_at(grid, asu_plan, spec, numbers, _TRUE_THICKNESS)
    op = OrientationPlan.build(grid, _BEAM_HKL, observed, energy=_ENERGY, thickness=(900.0,))
    plan = Plan(structure_factor_grid=grid, orientations=(op,))
    thickness_grid = ThicknessGrid(min_thickness=200.0, max_thickness=400.0, n_steps=5)

    base = optimize_thickness(_refinement(asu_plan, spec, numbers), thickness_grid)(plan)
    cpu = optimize_thickness(_refinement(asu_plan, spec, numbers), thickness_grid, device="cpu")(
        plan
    )
    assert torch.equal(cpu.orientations[0].thickness, base.orientations[0].thickness)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required (runs on the A100)")
def test_fit_thickness_cuda_matches_cpu() -> None:
    """CPU<->GPU parity: the on-device grid search bakes the same thickness the CPU fit picks.

    Skipped locally (this Mac has no complex-capable GPU); the real assertion runs on the SSEC A100.
    Thickness is picked by argmin over a shared candidate grid, so a *well-separated* winner
    matches exactly. The ``==`` assumes that separation (it holds for this clean synthetic): a
    genuine near-tie between adjacent candidates could let the ~1e-11 cross-device score shift pick
    the neighbour -- search-path divergence under FP perturbation, not a device bug, and harmless
    (the terminal re-scores whatever the fit bakes; reproducibility is anchored at the
    checkpoint boundary, not here).
    """
    grid, asu_plan, spec, numbers = _silicon()
    observed = _observed_at(grid, asu_plan, spec, numbers, _TRUE_THICKNESS)
    op = OrientationPlan.build(grid, _BEAM_HKL, observed, energy=_ENERGY, thickness=(900.0,))
    plan = Plan(structure_factor_grid=grid, orientations=(op,))
    thickness_grid = ThicknessGrid(min_thickness=200.0, max_thickness=400.0, n_steps=5)

    cpu = optimize_thickness(_refinement(asu_plan, spec, numbers), thickness_grid, device="cpu")(
        plan
    )
    cuda = optimize_thickness(_refinement(asu_plan, spec, numbers), thickness_grid, device="cuda")(
        plan
    )
    assert float(cuda.orientations[0].thickness[0]) == float(cpu.orientations[0].thickness[0])


# Bound validation now lives in ThicknessGrid (parse, don't validate); see test_specs.py.
