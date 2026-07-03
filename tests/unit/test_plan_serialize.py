"""Round-trip tests for the ``Plan`` ``.npz`` checkpoint (``preprocess.serialize``).

A checkpoint must reproduce the ``Plan``'s **source of truth** exactly (the fit outputs a cached
run reuses) and rebuild its **derived geometry** faithfully. The tests pin both: source tensors
compare equal, the ``tilt_reduction`` discriminant survives, and the rebuilt compiled geometry
matches (so a cache hit is byte-faithful to a fresh preprocess, not merely close).
"""

from __future__ import annotations

import numpy as np
import torch

from diffBloch.core.products import MosaicSmoothed, PatternBatch, PlainSum
from diffBloch.engine.plan import OrientationPlan, ScatteringGrid
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.serialize import read_plan, write_plan

_CELL = np.diag([4.9, 4.9, 5.4]).astype(np.float64)
_ENERGY = 200_000.0
_THICKNESS = 320.0


def _orientation(
    grid: ScatteringGrid,
    beam_hkl: np.ndarray,
    *,
    orientation: np.ndarray | None = None,
    tilts: np.ndarray | None = None,
    tilt_reduction: object = PlainSum(),
) -> OrientationPlan:
    pattern = PatternBatch(
        hkl=torch.tensor(beam_hkl, dtype=torch.int64),
        intensities=torch.arange(len(beam_hkl), dtype=torch.float64),
        sigmas=torch.ones(len(beam_hkl), dtype=torch.float64),
    )
    return OrientationPlan.build(
        grid,
        beam_hkl,
        pattern,
        energy=_ENERGY,
        thickness=(_THICKNESS,),
        orientation=orientation,
        tilts=tilts,
        tilt_reduction=tilt_reduction,  # type: ignore[arg-type]
    )


def _plan() -> Plan:
    grid = ScatteringGrid.from_cell(_CELL, g_max=1.6)
    beams_a = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]], dtype=np.int64)
    beams_b = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    tilts = np.stack([np.eye(3), np.eye(3)]).astype(np.float64)  # a 2-tilt rocking set
    return Plan(
        grid=grid,
        orientations=(
            _orientation(grid, beams_a),
            _orientation(
                grid,
                beams_b,
                tilts=tilts,
                tilt_reduction=MosaicSmoothed(window=2),
            ),
        ),
    )


def test_checkpoint_round_trips_the_plan_source_and_rebuilds_geometry(tmp_path) -> None:
    plan = _plan()
    path = tmp_path / "plan.npz"

    write_plan(plan, path)
    restored = read_plan(path)

    assert len(restored.orientations) == len(plan.orientations)
    assert restored.grid.g_max == plan.grid.g_max
    assert torch.equal(restored.grid.cell, plan.grid.cell)
    assert restored.grid.gpts == plan.grid.gpts
    for original, back in zip(plan.orientations, restored.orientations, strict=True):
        # source of truth: exact
        assert torch.equal(back.orientation, original.orientation)
        assert torch.equal(back.tilts, original.tilts)
        assert torch.equal(back.thickness, original.thickness)
        assert torch.equal(back.beam_hkl, original.beam_hkl)
        assert torch.equal(back.pattern.hkl, original.pattern.hkl)
        assert torch.equal(back.pattern.intensities, original.pattern.intensities)
        assert torch.equal(back.pattern.sigmas, original.pattern.sigmas)
        assert back.energy == original.energy
        assert back.u0 == original.u0
        # derived geometry: rebuilt faithfully (one beam_plan per tilt, matching diagonal 2kSgMii)
        assert len(back.beam_plans) == len(original.beam_plans)
        assert torch.allclose(back.beam_plans[0].diagonal, original.beam_plans[0].diagonal)


def test_checkpoint_preserves_the_tilt_reduction_discriminant(tmp_path) -> None:
    plan = _plan()
    path = tmp_path / "plan.npz"

    write_plan(plan, path)
    restored = read_plan(path)

    assert isinstance(restored.orientations[0].tilt_reduction, PlainSum)
    mosaic = restored.orientations[1].tilt_reduction
    assert isinstance(mosaic, MosaicSmoothed)
    assert mosaic.window == 2
