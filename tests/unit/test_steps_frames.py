"""``select_frames`` -- the whole-frame drop step (the public analog to ``ignore_orientations``).

Pins the model-independent criterion (observed ``intensity > 3 * sigma``, strict), order/grid
preservation, the empty-plan guard, the keep-all default, spec validation, and self-provenance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diffBloch.core.products import PatternBatch
from diffBloch.engine.plan import StructureFactorGrid
from diffBloch.preprocess import (
    FrameSelection,
    Plan,
    select_finite_loss_frames,
    select_frames,
    step_records,
)
from diffBloch.preprocess.plan import CandidatePlan

_CELL = np.eye(3, dtype=np.float64) * 5.0
_ENERGY = 200e3


def _frame(intensities: list[float], sigmas: list[float] | None = None) -> CandidatePlan:
    """A candidate whose observed pattern carries the given per-reflection intensities/sigmas."""
    inten = torch.tensor(intensities, dtype=torch.float64)
    sig = torch.ones_like(inten) if sigmas is None else torch.tensor(sigmas, dtype=torch.float64)
    hkl = torch.zeros((len(intensities), 3), dtype=torch.int64)
    pattern = PatternBatch(hkl=hkl, intensities=inten, sigmas=sig)
    return CandidatePlan.seed(
        np.zeros((1, 3), dtype=np.int64), pattern, energy=_ENERGY, thickness=(500.0,)
    )


def _plan(*frames: CandidatePlan) -> Plan:
    return Plan(
        structure_factor_grid=StructureFactorGrid.from_cell(_CELL, g_max=2.0), orientations=frames
    )


def _strong(n: int) -> CandidatePlan:
    """A frame with exactly ``n`` strong reflections (I=10 > 3*1) plus 3 weak ones (I=1)."""
    return _frame([10.0] * n + [1.0] * 3)


def test_drops_frames_below_threshold_and_preserves_order() -> None:
    plan = _plan(_strong(50), _strong(2), _strong(0), _strong(30))
    kept = select_frames(FrameSelection(min_observed=5))(plan)
    counts = [
        int((op.pattern.intensities > 3.0 * op.pattern.sigmas).sum()) for op in kept.orientations
    ]
    assert counts == [50, 30]  # only the two frames at/above the floor, in their original order


def test_threshold_is_strict_greater_than_three_sigma() -> None:
    # sigma == 1 everywhere: I=3 is exactly 3*sigma (excluded, strict >), 3+eps included, 3-eps not.
    frame = _frame([3.0, 3.0 - 1e-9, 3.0 + 1e-9, 3.0 + 1e-9])
    plan = _plan(frame)
    assert len(select_frames(FrameSelection(min_observed=2))(plan).orientations) == 1  # 2 strong
    with pytest.raises(ValueError):
        select_frames(FrameSelection(min_observed=3))(plan)  # only 2 clear 3*sigma, not 3


def test_min_observed_zero_keeps_every_frame() -> None:
    plan = _plan(_strong(0), _strong(1), _strong(50))
    kept = select_frames(FrameSelection(min_observed=0))(plan)
    assert len(kept.orientations) == 3


def test_dropping_every_frame_raises() -> None:
    plan = _plan(_strong(1), _strong(2))
    with pytest.raises(ValueError, match="dropped every frame"):
        select_frames(FrameSelection(min_observed=10))(plan)


def test_grid_is_preserved() -> None:
    plan = _plan(_strong(50), _strong(1))
    kept = select_frames(FrameSelection(min_observed=5))(plan)
    assert kept.structure_factor_grid is plan.structure_factor_grid


def test_select_finite_loss_frames_drops_nonfinite_initial_objectives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(_frame([1.0]), _frame([float("nan")]), _frame([2.0]))

    class _FakeObjective:
        def __init__(self, total: torch.Tensor) -> None:
            self.total = total

    class _FakeEngine:
        def __init__(self, one: Plan) -> None:
            self._one = one

        def objective_value(self, params: object) -> _FakeObjective:
            _ = params
            value = self._one.orientations[0].pattern.intensities[0]
            return _FakeObjective(value)

    class _FakeRefinement:
        params = object()

    def fake_build_engine(plan: Plan, *args: object, **kwargs: object) -> _FakeEngine:
        _ = args, kwargs
        return _FakeEngine(plan)

    monkeypatch.setattr("diffBloch.preprocess.steps.frames.build_engine", fake_build_engine)

    kept = select_finite_loss_frames(
        _FakeRefinement(),
        loss=lambda *_: torch.tensor(0.0),  # type: ignore[arg-type]
    )(plan)

    assert [float(op.pattern.intensities[0]) for op in kept.orientations] == [1.0, 2.0]


def test_step_record_round_trips_for_provenance() -> None:
    # Assert through the public step_records helper (not step.record): select_frames is typed to
    # return PlanStep, whose structural type carries no .record -- only the concrete Step does.
    (record,) = step_records([select_frames(FrameSelection(min_observed=7))])
    assert record.name == "select_frames"
    assert record.params == {"__type__": "FrameSelection", "min_observed": 7}


def test_negative_min_observed_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_observed must be >= 0"):
        FrameSelection(min_observed=-1)
