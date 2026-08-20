"""``report_coupling`` + ``coupling_stats`` / ``summarize_plan``: the plan-shape observability.

The consumer-boundary step emits one :class:`RotationCoupling` per rotation and one
:class:`CouplingSummary`, returning the plan unchanged. ``coupling_stats`` reads the per-rotation
solve geometry across all three plan phases (segmented, tilt-independent, pre-build candidate).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tests.unit.test_inference import _BEAM_HKL, _ENERGY, _silicon

from diffBloch.app.loggers import ReportLogger
from diffBloch.core.products import PatternBatch
from diffBloch.engine import CoupledOrientationPlan, OrientationPlan
from diffBloch.observability import EventRecord
from diffBloch.preprocess.orientation import rocking_curve_tilts
from diffBloch.preprocess.plan import CandidatePlan, Plan, coupling_stats, summarize_plan
from diffBloch.preprocess.steps.report_coupling import report_coupling

_TILTS = rocking_curve_tilts(1.0, 4, geometry="continuous_rotation")  # (4, 3, 3)


def _records(path: Path) -> list[EventRecord]:
    return [EventRecord.model_validate_json(line) for line in path.read_text().splitlines()]


def _pattern() -> PatternBatch:
    return PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=torch.zeros(len(_BEAM_HKL), dtype=torch.float64),
        sigmas=torch.full((len(_BEAM_HKL),), 0.01, dtype=torch.float64),
    )


def _segmented(grid: object) -> CoupledOrientationPlan:
    # Two chunks (2 beams each, one shared 000) over 4 tilts -> union of 3 beams, 2 tilts/segment.
    chunk_a = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    chunk_b = np.array([[0, 0, 0], [-1, 0, 0]], dtype=np.int64)
    return CoupledOrientationPlan.build(
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
        "n_coupling_segments": 2,
        "n_tilts": 4,
        "max_tilts_per_segment": 2,
        "n_union_beams": 3,  # {000, 100, -100} deduped
        "max_beams_per_segment": 2,  # each chunk carries 2 beams
    }

    tilt_independent = OrientationPlan.build(
        grid, _BEAM_HKL, _pattern(), energy=_ENERGY, thickness=(300.0,), tilts=_TILTS
    )
    stats = coupling_stats(tilt_independent)
    assert stats["n_coupling_segments"] == 1  # one implicit union
    assert stats["n_tilts"] == 4 and stats["max_tilts_per_segment"] == 4
    assert stats["n_union_beams"] == len(_BEAM_HKL) == stats["max_beams_per_segment"]

    candidate = CandidatePlan.seed(_BEAM_HKL, _pattern(), energy=_ENERGY, thickness=(300.0,))
    assert coupling_stats(candidate) == {
        "n_coupling_segments": 0,  # no tilts/segments before the build
        "n_tilts": 0,
        "max_tilts_per_segment": 0,
        "n_union_beams": len(_BEAM_HKL),
        "max_beams_per_segment": len(_BEAM_HKL),
    }


def test_report_coupling_is_identity_and_emits_per_rotation_plus_a_summary(tmp_path: Path) -> None:
    grid, *_ = _silicon()
    segmented = _segmented(grid)
    tilt_independent = OrientationPlan.build(
        grid, _BEAM_HKL, _pattern(), energy=_ENERGY, thickness=(300.0,), tilts=_TILTS
    )
    plan = Plan(structure_factor_grid=grid, orientations=(segmented, tilt_independent))

    path = tmp_path / "report.jsonl"
    log = ReportLogger(path)
    out = report_coupling(
        log, dataset_for_rotation=lambda rotation_index: f"dataset-{rotation_index}"
    )(plan)

    assert out is plan  # identity: a boundary observation, not a transform
    records = _records(path)
    rotations = [event for event in records if event.event_type == "RotationCoupling"]
    segments = [event for event in records if event.event_type == "RotationCouplingSegments"]
    summaries = [event for event in records if event.event_type == "CouplingSummary"]
    assert [event.payload["index"] for event in rotations] == [0, 1]
    assert [event.payload["rotation_index"] for event in rotations] == [0, 0]
    assert [event.dataset for event in rotations] == ["dataset-0", "dataset-0"]
    assert (
        rotations[0].payload["n_coupling_segments"] == 2
        and rotations[0].payload["max_beams_per_segment"] == 2
    )  # the segmented rotation
    assert rotations[1].payload["n_coupling_segments"] == 1  # the tilt-independent rotation
    assert len(segments) == 2
    # Row position is the segment index; the event stores no `range(n)` column for it.
    assert segments[0].series["n_segment_beams"] == [2.0, 2.0]
    assert segments[0].series["first_tilt_index"] == [0.0, 2.0]
    assert segments[0].measurements["n_segments"] == 2.0
    assert segments[1].series["n_segment_beams"] == [3.0]  # tilt-independent: one whole-plan row
    assert segments[1].measurements["n_segments"] == 1.0
    assert len(summaries) == 1
    assert summaries[0].measurements["n_orientations"] == 2.0
    assert summaries[0].measurements["n_grid_hkl"] == float(grid.structure_factor_hkl.shape[0])


def test_summarize_plan_reports_solve_and_scored_counts_for_a_built_plan() -> None:
    grid, *_ = _silicon()
    built = _segmented(grid)
    plan = Plan(structure_factor_grid=grid, orientations=(built,))
    summary = summarize_plan(plan)
    assert summary == {
        "n_orientations": 1.0,
        "n_grid_hkl": float(grid.structure_factor_hkl.shape[0]),
        # SOLVE (which beams couple dynamically) and SCORED (which reflections enter the R-factor)
        # are separate sets and are reported under separately scoped names.
        "n_solve_beams_total": 3.0,
        "n_solve_beams_max": 3.0,
        "n_observed_hkl": float(len(_BEAM_HKL)),
        "n_matched_hkl": float(built.alignment.hkl.shape[0]),
    }


def test_summarize_plan_omits_the_matched_count_before_the_build() -> None:
    """A candidate has no alignment: absent, not zero -- 0 would mean "matched nothing"."""
    grid, *_ = _silicon()
    candidate = CandidatePlan.seed(
        np.asarray(_BEAM_HKL, dtype=np.int64), _pattern(), energy=_ENERGY, thickness=(300.0,)
    )
    summary = summarize_plan(Plan(structure_factor_grid=grid, orientations=(candidate,)))
    assert "n_matched_hkl" not in summary
    assert summary["n_solve_beams_total"] == float(len(_BEAM_HKL))
    assert summary["n_observed_hkl"] == float(len(_BEAM_HKL))
