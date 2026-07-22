"""``report_coupling`` + ``coupling_stats`` / ``summarize_plan``: the plan-shape observability.

The consumer-boundary step emits one :class:`RotationCoupling` per rotation and one
:class:`CouplingSummary`, returning the plan unchanged. ``coupling_stats`` reads the per-rotation
solve geometry across all three plan phases (segmented, tilt-independent, pre-build candidate).
"""

from __future__ import annotations

import numpy as np
import torch
from tests.unit.test_inference import _BEAM_HKL, _ENERGY, _silicon

from diffBloch.core.products import PatternBatch
from diffBloch.engine import OrientationPlan, SegmentedOrientationPlan
from diffBloch.observability import CouplingSummary, RecordingLogger, RotationCoupling
from diffBloch.preprocess.orientation import rocking_curve_tilts
from diffBloch.preprocess.plan import CandidatePlan, Plan, coupling_stats, summarize_plan
from diffBloch.preprocess.steps.report_coupling import report_coupling

_TILTS = rocking_curve_tilts(1.0, 4, geometry="continuous_rotation")  # (4, 3, 3)


def _pattern() -> PatternBatch:
    return PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=torch.zeros(len(_BEAM_HKL), dtype=torch.float64),
        sigmas=torch.full((len(_BEAM_HKL),), 0.01, dtype=torch.float64),
    )


def _segmented(grid: object) -> SegmentedOrientationPlan:
    # Two chunks (2 beams each, one shared 000) over 4 tilts -> union of 3 beams, cover_max 2.
    chunk_a = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    chunk_b = np.array([[0, 0, 0], [-1, 0, 0]], dtype=np.int64)
    return SegmentedOrientationPlan.build(
        grid,
        [(chunk_a, (0, 1)), (chunk_b, (2, 3))],
        _pattern(),
        energy=_ENERGY,
        thickness=(300.0,),
        u0=0.0,
        orientation=np.eye(3),
        tilts=_TILTS,
    )


def test_coupling_stats_reads_each_plan_phase() -> None:
    grid, *_ = _silicon()

    segmented = coupling_stats(_segmented(grid))
    assert segmented == {
        "n_segments": 2,
        "n_tilts": 4,
        "cover_max": 2,
        "beams_union": 3,  # {000, 100, -100} deduped
        "beams_seg_max": 2,  # each chunk carries 2 beams
    }

    tilt_independent = OrientationPlan.build(
        grid, _BEAM_HKL, _pattern(), energy=_ENERGY, thickness=(300.0,), tilts=_TILTS
    )
    stats = coupling_stats(tilt_independent)
    assert stats["n_segments"] == 1  # one implicit union
    assert stats["n_tilts"] == 4 and stats["cover_max"] == 4
    assert stats["beams_union"] == len(_BEAM_HKL) == stats["beams_seg_max"]

    candidate = CandidatePlan.seed(_BEAM_HKL, _pattern(), energy=_ENERGY, thickness=(300.0,))
    assert coupling_stats(candidate) == {
        "n_segments": 0,  # no tilts/segments before the build
        "n_tilts": 0,
        "cover_max": 0,
        "beams_union": len(_BEAM_HKL),
        "beams_seg_max": len(_BEAM_HKL),
    }


def test_report_coupling_is_identity_and_emits_per_rotation_plus_a_summary() -> None:
    grid, *_ = _silicon()
    segmented = _segmented(grid)
    tilt_independent = OrientationPlan.build(
        grid, _BEAM_HKL, _pattern(), energy=_ENERGY, thickness=(300.0,), tilts=_TILTS
    )
    plan = Plan(grid=grid, orientations=(segmented, tilt_independent))

    log = RecordingLogger()
    out = report_coupling(log)(plan)

    assert out is plan  # identity: a boundary observation, not a transform
    rotations = [event for event in log.events if isinstance(event, RotationCoupling)]
    summaries = [event for event in log.events if isinstance(event, CouplingSummary)]
    assert [event.index for event in rotations] == [0, 1]
    assert (
        rotations[0].n_segments == 2 and rotations[0].beams_seg_max == 2
    )  # the segmented rotation
    assert rotations[1].n_segments == 1  # the tilt-independent rotation
    assert len(summaries) == 1
    assert summaries[0].measurements["n_orientations"] == 2.0
    assert summaries[0].measurements["segments_total"] == 3.0  # 2 + 1
    assert summaries[0].measurements["n_grid_hkl"] == float(grid.grid_hkl.shape[0])


def test_summarize_plan_aggregates_the_widest_and_largest() -> None:
    grid, *_ = _silicon()
    plan = Plan(grid=grid, orientations=(_segmented(grid),))
    summary = summarize_plan(plan)
    assert summary["n_orientations"] == 1.0
    assert summary["cover_max"] == 2.0
    assert summary["beams_seg_max"] == 2.0
    assert summary["beams_union_max"] == 3.0
    assert summary["g_max"] == float(grid.g_max)
