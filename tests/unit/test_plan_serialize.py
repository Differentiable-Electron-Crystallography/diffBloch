"""``write_plan`` / ``read_plan`` round-trip: the checkpoint persists source, rebuilds compiled.

The checkpoint stores only each plan's source (orientation/tilts/thickness/beam set(s)/pattern/
reduction/scored set) + the recipe provenance, and rebuilds all compiled geometry on read. The
load-bearing property is a *byte-identical* forward simulation after a round trip -- for both the
tilt-independent :class:`OrientationPlan` and the coupled :class:`CoupledOrientationPlan` (the
path the reverted serializer never handled) -- plus faithful provenance.
"""

from __future__ import annotations

import numpy as np
import torch
from tests.unit.test_inference import (
    _BEAM_HKL,
    _ENERGY,
    _METHOD,
    _refinement,
    _silicon,
    _simulated_intensities,
)

from diffBloch.core.products import MosaicSmoothed, PatternBatch
from diffBloch.engine import CoupledOrientationPlan
from diffBloch.engine.plan import OrientationPlan
from diffBloch.preprocess.orientation import rocking_curve_tilts
from diffBloch.preprocess.pipeline import StepRecord
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.scoring import build_engine
from diffBloch.preprocess.serialize import read_plan, write_plan

_TILTS = rocking_curve_tilts(1.0, 4, geometry="continuous_rotation")  # (4, 3, 3)


def _pattern() -> PatternBatch:
    grid, asu_plan, spec, numbers = _silicon()
    return PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=_simulated_intensities(grid, asu_plan, spec, numbers),
        sigmas=torch.full((len(_BEAM_HKL),), 0.01, dtype=torch.float64),
    )


def _simulated(plan: Plan) -> torch.Tensor:
    grid, asu_plan, spec, numbers = _silicon()
    refinement = _refinement(asu_plan, spec, numbers)
    engine = build_engine(plan, refinement, method=_METHOD)
    return engine.simulate(refinement.params)[0].intensities


def _assert_round_trips(plan: Plan, tmp_path) -> Plan:
    path = tmp_path / "plan.npz"
    write_plan(plan, path)
    loaded = read_plan(path)
    assert len(loaded.orientations) == len(plan.orientations)
    torch.testing.assert_close(_simulated(loaded), _simulated(plan))  # byte-identical forward
    assert loaded.provenance == plan.provenance
    assert [op.pattern.rotation_index for op in loaded.orientations] == [
        op.pattern.rotation_index for op in plan.orientations
    ]
    return loaded


def test_plain_plan_round_trips(tmp_path) -> None:
    grid, *_ = _silicon()
    op = OrientationPlan.build(
        grid, _BEAM_HKL, _pattern(), energy=_ENERGY, thickness=(300.0,), tilts=_TILTS
    )
    plan = Plan(
        structure_factor_grid=grid,
        orientations=(op,),
        provenance=(StepRecord("select_beams", None),),
    )
    _assert_round_trips(plan, tmp_path)


def test_segmented_plan_round_trips_with_scored_set_and_reduction(tmp_path) -> None:
    """The coupled path: per-segment (beam_hkl, cover), pinned scored set, mosaicity reduction."""
    grid, *_ = _silicon()
    pattern = _pattern()
    chunk_a = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    chunk_b = np.array([[0, 0, 0], [-1, 0, 0]], dtype=np.int64)
    scored = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)  # -100 in union but excluded here
    op = CoupledOrientationPlan.build(
        grid,
        [(chunk_a, (0, 1)), (chunk_b, (2, 3))],
        pattern,
        energy=_ENERGY,
        thickness=(300.0,),
        u0=0.0,
        orientation=np.eye(3),
        tilts=_TILTS,
        tilt_reduction=MosaicSmoothed(3),
        scored_hkl=scored,
    )
    plan = Plan(
        structure_factor_grid=grid,
        orientations=(op,),
        provenance=(StepRecord("couple_beams", {"__type__": "UnionCoupling"}),),
    )
    loaded = _assert_round_trips(plan, tmp_path)

    (reloaded_op,) = loaded.orientations
    assert isinstance(reloaded_op, CoupledOrientationPlan)
    assert len(reloaded_op.segments) == 2
    assert reloaded_op.alignment.hkl.tolist() == op.alignment.hkl.tolist()  # scored set preserved
    assert isinstance(reloaded_op.tilt_reduction, MosaicSmoothed)
    assert reloaded_op.tilt_reduction.window == 3


def test_mixed_plan_round_trips(tmp_path) -> None:
    """A Plan holding a plain AND a segmented orientation (the per-rotation kind discriminant)."""
    grid, *_ = _silicon()
    pattern = _pattern()
    plain = OrientationPlan.build(
        grid, _BEAM_HKL, pattern, energy=_ENERGY, thickness=(300.0,), tilts=_TILTS
    )
    segmented = CoupledOrientationPlan.build(
        grid,
        [(_BEAM_HKL, (0, 1)), (_BEAM_HKL, (2, 3))],
        pattern,
        energy=_ENERGY,
        thickness=(300.0,),
        u0=0.0,
        orientation=np.eye(3),
        tilts=_TILTS,
    )
    plan = Plan(structure_factor_grid=grid, orientations=(plain, segmented))
    loaded = _assert_round_trips(plan, tmp_path)
    assert isinstance(loaded.orientations[0], OrientationPlan)
    assert isinstance(loaded.orientations[1], CoupledOrientationPlan)
