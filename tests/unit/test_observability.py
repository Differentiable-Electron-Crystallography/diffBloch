"""The domain-observation channel: typed events and pluggable logger sinks.

Unit-level (no engine): the event surface (``channel`` + ``measurements``), the null/fan-out
loggers, and the two boundary backends. ``run_inference`` emission is covered in
``test_inference`` (it needs the forward model). See the effects-and-observability decision doc.
"""

from __future__ import annotations

import csv
import logging
import sys
import types
from pathlib import Path

import pytest

from diffBloch.app.loggers import ConsoleLogger, CSVLogger, EarlyAbortLogger, FitAbortedError
from diffBloch.app.loggers.comet import CometLogger
from diffBloch.app.loggers.wandb import WandbLogger
from diffBloch.observability import (
    CouplingSummary,
    Event,
    InferenceCompleted,
    Logger,
    MultiLogger,
    NullLogger,
    OrientationFitted,
    PlanStepCompleted,
    RecordingLogger,
    RefinementCompleted,
    RefinementStep,
    RotationCoupling,
    RotationScored,
    ThicknessFitted,
)


def _fitted(index: int, wr2: float) -> OrientationFitted:
    return OrientationFitted(index=index, wr2=wr2, n_trials=10, n_passes=3, pass_cap=2000)


def test_events_expose_a_uniform_channel_and_measurements_surface() -> None:
    rotation = RotationScored(index=3, r_obs=0.42, n_observed=12, n_beams=20)
    assert rotation.channel == "rotation"
    assert rotation.step == 3  # a rotation's step is its index
    assert rotation.measurements == {"r_obs": 0.42, "n_observed": 12.0, "n_beams": 20.0}

    completed = InferenceCompleted(n_rotations=99, n_evaluated=97, mean_r_obs=0.065)
    assert completed.channel == "inference"
    assert completed.step is None  # a run-level aggregate has no per-rotation position
    assert completed.measurements == {
        "n_rotations": 99.0,
        "n_evaluated": 97.0,
        "mean_r_obs": 0.065,
    }

    fitted = _fitted(index=5, wr2=0.12)
    assert fitted.channel == "fit_orientation"  # shared with the step's PlanStepCompleted line
    assert fitted.step == 5  # an orientation fit's step is its rotation index
    assert fitted.measurements == {
        "wr2": 0.12,
        "n_trials": 10.0,
        "n_passes": 3.0,
        "pass_cap": 2000.0,
    }

    thickness = ThicknessFitted(index=7, wr2=0.031, thickness=1460.0)
    assert thickness.channel == "fit_thickness"
    assert thickness.step == 7  # a thickness fit's step is its rotation index
    assert thickness.measurements == {"wr2": 0.031, "thickness": 1460.0}

    coupled = RotationCoupling(
        index=2,
        n_coupling_segments=8,
        n_tilts=42,
        max_tilts_per_segment=15,
        n_union_beams=700,
        max_beams_per_segment=641,
    )
    assert coupled.channel == "coupling"
    assert coupled.step == 2
    assert coupled.measurements == {
        "n_coupling_segments": 8.0,
        "n_tilts": 42.0,
        "max_tilts_per_segment": 15.0,
        "n_union_beams": 700.0,
        "max_beams_per_segment": 641.0,
    }

    coupling_summary = CouplingSummary(measurements={"n_orientations": 55.0, "g_max": 5.0})
    assert coupling_summary.channel == "coupling"
    assert coupling_summary.step is None  # a run-level aggregate shares the channel, step None
    assert coupling_summary.measurements == {"n_orientations": 55.0, "g_max": 5.0}

    # PlanStepCompleted carries the step NAME as a per-instance channel (not a class constant).
    plan_step = PlanStepCompleted(channel="fit_orientation", index=4, measurements={"beams": 641.0})
    assert plan_step.channel == "fit_orientation"
    assert plan_step.step == 4
    assert plan_step.measurements == {"beams": 641.0}

    refinement = RefinementStep(iteration=4, loss=1.5)
    assert refinement.channel == "refinement"
    assert refinement.step == 4  # a refinement step's step is its iteration
    assert refinement.measurements == {"loss": 1.5}

    structured_refinement = RefinementStep(
        iteration=5,
        loss=2.0,
        objective_total=2.0,
        components={"diffraction": {"raw": 1.0, "weight": 2.0, "contribution": 2.0}},
    )
    assert structured_refinement.measurements == {
        "loss": 2.0,
        "objective_total": 2.0,
        "component.diffraction.raw": 1.0,
        "component.diffraction.weight": 2.0,
        "component.diffraction.contribution": 2.0,
    }

    refinement_done = RefinementCompleted(n_steps=20, best_step=17, best_loss=0.3)
    assert refinement_done.channel == "refinement"  # shares the stream's channel
    assert refinement_done.step is None  # separated from the stream by step, not channel
    assert refinement_done.measurements == {"n_steps": 20.0, "best_step": 17.0, "best_loss": 0.3}


def test_events_and_loggers_satisfy_the_protocols_structurally() -> None:
    assert isinstance(RotationScored(index=0, r_obs=0.1, n_observed=1, n_beams=2), Event)
    assert isinstance(NullLogger(), Logger)
    assert isinstance(RecordingLogger(), Logger)


def test_null_logger_discards_events() -> None:
    logger = NullLogger()
    assert logger.report(RotationScored(index=0, r_obs=0.1, n_observed=1, n_beams=2)) is None


def test_multi_logger_fans_each_event_out_to_every_logger() -> None:
    a, b = RecordingLogger(), RecordingLogger()
    fanout = MultiLogger(loggers=(a, b))
    event = RotationScored(index=1, r_obs=0.2, n_observed=3, n_beams=5)

    fanout.report(event)

    assert a.events == [event]
    assert b.events == [event]


def test_console_logger_logs_channel_step_and_measurements(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = ConsoleLogger(level=logging.INFO)
    with caplog.at_level(logging.INFO, logger="diffBloch.loggers"):
        logger.report(RotationScored(index=47, r_obs=0.5, n_observed=4, n_beams=7))
        logger.report(InferenceCompleted(n_rotations=99, n_evaluated=97, mean_r_obs=0.06))

    rotation_msg, inference_msg = (r.getMessage() for r in caplog.records)
    assert "rotation[47]" in rotation_msg  # the step pins the line to a rotation
    assert "r_obs=0.5" in rotation_msg
    assert inference_msg.startswith("inference ")  # aggregate has no step bracket


def test_csv_logger_appends_events_in_long_format(tmp_path: Path) -> None:
    path = tmp_path / "run.csv"
    logger = CSVLogger(path=path)
    logger.report(RotationScored(index=0, r_obs=0.5, n_observed=4, n_beams=7))
    logger.report(InferenceCompleted(n_rotations=1, n_evaluated=1, mean_r_obs=0.5))

    with path.open() as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["channel", "step", "metric", "value"]  # header written once
    assert ["rotation", "0", "r_obs", "0.5"] in rows  # step = the rotation index
    assert ["inference", "", "mean_r_obs", "0.5"] in rows  # step None -> empty cell


def test_wandb_logger_maps_measurements_to_a_namespaced_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[tuple[dict[str, float], int | None]] = []
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.log = lambda payload, step=None: logged.append((payload, step))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    WandbLogger().report(RotationScored(index=5, r_obs=0.5, n_observed=4, n_beams=7))

    assert logged == [
        ({"rotation/r_obs": 0.5, "rotation/n_observed": 4.0, "rotation/n_beams": 7.0}, 5)
    ]


def test_comet_logger_forwards_namespaced_metrics_to_the_experiment() -> None:
    logged: list[tuple[dict[str, float], int | None]] = []

    class _FakeExperiment:  # duck-types comet_ml.Experiment.log_metrics
        def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
            logged.append((metrics, step))

    CometLogger(experiment=_FakeExperiment()).report(
        RotationScored(index=5, r_obs=0.5, n_observed=4, n_beams=7)
    )

    assert logged == [
        ({"rotation/r_obs": 0.5, "rotation/n_observed": 4.0, "rotation/n_beams": 7.0}, 5)
    ]


# --- EarlyAbortLogger: abort a from-scratch fit that is not tracking the data --------------------


def test_early_abort_fires_when_no_rotation_clears_the_ceiling() -> None:
    """patience rotations all above the ceiling -> FitAbortedError at the patience-th event."""
    guard = EarlyAbortLogger(wr2_ceiling=0.6, patience=3)
    guard.report(_fitted(0, 0.9))
    guard.report(_fitted(1, 0.8))  # still only 2 seen, no decision yet
    with pytest.raises(FitAbortedError, match=r"best wr2 is 0.8.*above the 0.6 ceiling"):
        guard.report(_fitted(2, 0.85))


def test_early_abort_does_not_fire_once_a_rotation_clears_the_ceiling() -> None:
    """One good rotation within patience -> promising; never aborts, even on later high wr2."""
    guard = EarlyAbortLogger(wr2_ceiling=0.6, patience=3)
    guard.report(_fitted(0, 0.9))
    guard.report(_fitted(1, 0.05))  # clears the ceiling -> promising
    for i in range(2, 10):
        guard.report(_fitted(i, 0.95))  # later poor rotations do not resurrect the abort


def test_early_abort_waits_for_patience_before_deciding() -> None:
    """Below patience, a high wr2 never aborts -- the guard needs patience rotations of evidence."""
    guard = EarlyAbortLogger(wr2_ceiling=0.6, patience=5)
    for i in range(4):  # 4 < patience 5
        guard.report(_fitted(i, 0.99))  # no raise
    with pytest.raises(FitAbortedError):
        guard.report(_fitted(4, 0.99))  # the 5th tips it over


def test_early_abort_forwards_every_event_to_inner_including_the_aborting_one() -> None:
    """The guard is also a pass-through: inner sees all events, including the one that aborts."""
    inner = RecordingLogger()
    guard = EarlyAbortLogger(wr2_ceiling=0.6, patience=2, inner=inner)
    guard.report(RotationScored(index=0, r_obs=0.3, n_observed=1, n_beams=2))  # non-fit, forwarded
    guard.report(_fitted(0, 0.9))
    with pytest.raises(FitAbortedError):
        guard.report(_fitted(1, 0.9))
    # all three forwarded (the aborting event was reported to inner before the raise)
    assert len(inner.events) == 3
    assert isinstance(inner.events[0], RotationScored)


def test_early_abort_ignores_non_fit_events_for_the_decision() -> None:
    """Only the fit stream counts toward patience -- a flood of rotation events never aborts."""
    guard = EarlyAbortLogger(wr2_ceiling=0.6, patience=2)
    for i in range(20):
        guard.report(RotationScored(index=i, r_obs=0.99, n_observed=1, n_beams=2))  # never aborts


def test_early_abort_rejects_nonpositive_patience() -> None:
    with pytest.raises(ValueError, match="patience must be >= 1"):
        EarlyAbortLogger(patience=0)
