"""``restrict_scored_resolution``: the scored set is cut at ``|g| <= cap``; the solve set is not.

The silicon fixture's 5 A cubic cell puts ``000`` at ``|g| = 0`` and ``+/-100`` at ``|g| = 0.2``, so
a 0.1 cap keeps only ``000`` scored while all three beams still solve. The cut must also survive
everything that later rebuilds a plan's alignment: an orientation change and a checkpoint round
trip.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from tests.unit.synthetic import seed_system
from tests.unit.test_inference import _BEAM_HKL, _ENERGY, _silicon
from tests.unit.test_plan_serialize import _TILTS, _pattern

from diffBloch.config.schema import ExperimentConfig
from diffBloch.engine import CoupledOrientationPlan
from diffBloch.engine.plan import OrientationPlan
from diffBloch.preprocess import pipeline, restrict_scored_resolution
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.serialize import read_plan, write_plan
from diffBloch.specs import ScoredResolutionCap


def _plain(grid: object) -> OrientationPlan:
    return OrientationPlan.build(
        grid, _BEAM_HKL, _pattern(), energy=_ENERGY, thickness=(300.0,), tilts=_TILTS
    )


def _coupled(grid: object) -> CoupledOrientationPlan:
    return CoupledOrientationPlan.build(
        grid,
        [(_BEAM_HKL, (0, 1)), (_BEAM_HKL, (2, 3))],
        _pattern(),
        energy=_ENERGY,
        thickness=(300.0,),
        u0=0.0,
        orientation=np.eye(3),
        tilts=_TILTS,
    )


def _restricted(g_max: float) -> Plan:
    grid, *_ = _silicon()
    plan = Plan(structure_factor_grid=grid, orientations=(_plain(grid), _coupled(grid)))
    return pipeline([restrict_scored_resolution(ScoredResolutionCap(g_max=g_max))])(plan)


def test_cap_narrows_the_scored_set_and_leaves_the_solve_set() -> None:
    plan = _restricted(0.1)

    for op in plan.orientations:
        assert op.alignment.hkl.tolist() == [[0, 0, 0]]
        # The kept row still points at 000 in both the solution and the pattern.
        assert op.beam_hkl[op.alignment.solution_index].tolist() == [[0, 0, 0]]
        assert op.pattern.hkl[op.alignment.pattern_index].tolist() == [[0, 0, 0]]
        assert sorted(op.beam_hkl.tolist()) == sorted(_BEAM_HKL.tolist())  # still solved
    assert plan.provenance[-1].name == "restrict_scored_resolution"
    assert plan.provenance[-1].params == {"__type__": "ScoredResolutionCap", "g_max": 0.1}


def test_cap_at_or_above_every_reflection_changes_nothing() -> None:
    grid, *_ = _silicon()
    unrestricted = _plain(grid).alignment.hkl.tolist()

    plan = _restricted(0.2)

    assert plan.orientations[0].alignment.hkl.tolist() == unrestricted


def test_cap_survives_an_orientation_rebuild_and_a_checkpoint_round_trip(tmp_path) -> None:
    plan = _restricted(0.1)
    grid = plan.structure_factor_grid
    tilt = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, np.cos(0.01), -np.sin(0.01)], [0.0, np.sin(0.01), np.cos(0.01)]],
        dtype=torch.float64,
    )

    for op in plan.orientations:
        rotated = op.with_orientation(grid, tilt @ op.orientation)
        assert rotated.alignment.hkl.tolist() == [[0, 0, 0]]
    path = tmp_path / "plan.npz"
    write_plan(plan, path)
    for op in read_plan(path).orientations:
        assert op.alignment.hkl.tolist() == [[0, 0, 0]]


def test_cap_needs_built_plans() -> None:
    _, candidates = seed_system()

    with pytest.raises(TypeError, match="needs a built plan"):
        restrict_scored_resolution(ScoredResolutionCap(g_max=1.0))(candidates)


def test_cap_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ScoredResolutionCap(g_max=0.0)


def test_config_cap_parses_and_may_not_exceed_the_solve_cutoff() -> None:
    base = {"name": "q", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}

    capped = ExperimentConfig.model_validate(
        {**base, "blochwave": {"g_max": 2.0, "gmax_refine": 1.5}}
    )
    assert capped.blochwave.to_scored_resolution_cap() == ScoredResolutionCap(g_max=1.5)
    assert ExperimentConfig.model_validate(base).blochwave.to_scored_resolution_cap() is None
    with pytest.raises(ValidationError, match="gmax_refine must not exceed g_max"):
        ExperimentConfig.model_validate({**base, "blochwave": {"g_max": 2.0, "gmax_refine": 2.5}})
