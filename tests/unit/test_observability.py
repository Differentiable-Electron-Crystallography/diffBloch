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

from diffBloch.app.loggers import (
    ConsoleLogger,
    CSVLogger,
    EarlyAbortLogger,
    FitAbortedError,
    _format_eta,
)
from diffBloch.app.loggers.comet import CometLogger
from diffBloch.app.loggers.wandb import WandbLogger
from diffBloch.observability import (
    CouplingSummary,
    DeviceSelected,
    Event,
    InferenceCompleted,
    Logger,
    MultiLogger,
    NullLogger,
    ObjectiveManifest,
    ObjectiveTerm,
    OrientationOptimizationStarted,
    OrientationOptimizationSummary,
    OrientationOptimized,
    PlanStepCompleted,
    RecordingLogger,
    RefinementCompleted,
    RefinementOrientationStep,
    RefinementStarted,
    RefinementStep,
    RotationCoupling,
    RotationScored,
    ThicknessOptimizationStarted,
    ThicknessOptimized,
)


def _fitted(index: int, score: float, residual: str = "wr2") -> OrientationOptimized:
    return OrientationOptimized(
        rotation_index=index,
        score=score,
        residual=residual,
        n_matched_hkl=42,
        n_trials=10,
        n_passes=3,
        pass_cap=2000,
    )


def test_events_expose_a_uniform_channel_and_measurements_surface() -> None:

    device = DeviceSelected(requested="cuda", selected="cpu", cuda_available=False)
    assert device.channel == "device"
    assert device.step is None
    assert device.measurements == {"cuda_available": 0.0, "selected_cuda": 0.0}

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

    thickness = ThicknessOptimized(
        rotation_index=7,
        score=0.031,
        residual="wr2",
        thickness=1460.0,
        candidate_thicknesses=(1400.0, 1460.0, 1520.0),
        candidate_score=(0.05, 0.031, 0.045),
    )
    assert thickness.channel == "optimize_thickness"
    assert thickness.step == 7  # a thickness fit's step is its rotation index

    thickness_started = ThicknessOptimizationStarted(total_rotations=99)
    assert thickness_started.channel == "thickness_started"
    assert thickness_started.step is None
    assert thickness_started.measurements == {"total_rotations": 99.0}
    assert thickness_started.channel != thickness.channel
    assert thickness.measurements == {"wr2": 0.031, "thickness": 1460.0}

    robs_thickness = ThicknessOptimized(
        rotation_index=7,
        score=0.028,
        residual="robs",
        thickness=1460.0,
        candidate_thicknesses=(1400.0, 1460.0, 1520.0),
        candidate_score=(0.045, 0.028, 0.05),
    )
    assert robs_thickness.measurements == {"robs": 0.028, "thickness": 1460.0}

    orientation_summary = OrientationOptimizationSummary(
        n_orientations=2,
        mean_score=0.03,
        residual="wr2",
        total_matched_hkl=80,
        total_strong_hkl=60,
        total_weak_hkl=20,
        total_observed_hkl=100,
        total_trials=50,
        max_passes=8,
    )
    assert orientation_summary.measurements == {
        "n_orientations": 2.0,
        "mean_wr2": 0.03,
        "total_matched_hkl": 80.0,
        "total_strong_hkl": 60.0,
        "total_weak_hkl": 20.0,
        "total_observed_hkl": 100.0,
        "total_unmatched_hkl": 20.0,
        "total_trials": 50.0,
        "max_passes": 8.0,
    }

    refinement_started = RefinementStarted(total_steps=40)
    assert refinement_started.channel == "refinement_started"
    assert refinement_started.step is None
    assert refinement_started.measurements == {"total_steps": 40.0}
    assert refinement_started.channel != RefinementStep(iteration=0, loss=0.0).channel

    orientation_started = OrientationOptimizationStarted(total_rotations=52)
    assert orientation_started.channel == "orientation_started"
    assert orientation_started.step is None
    assert orientation_started.measurements == {"total_rotations": 52.0}
    assert orientation_started.channel != _fitted(index=0, score=0.0).channel

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

    coupling_summary = CouplingSummary(measurements={"n_orientations": 55.0})
    assert coupling_summary.channel == "coupling"
    assert coupling_summary.step is None  # a run-level aggregate shares the channel, step None
    assert coupling_summary.measurements == {"n_orientations": 55.0}

    # PlanStepCompleted carries the step NAME as a per-instance channel (not a class constant).
    plan_step = PlanStepCompleted(
        channel="optimize_orientation", index=4, measurements={"beams": 641.0}
    )
    assert plan_step.channel == "optimize_orientation"
    assert plan_step.step == 4
    assert plan_step.measurements == {"beams": 641.0}

    refinement = RefinementStep(iteration=4, loss=1.5)
    assert refinement.channel == "refinement"
    assert refinement.step == 4  # a refinement step's step is its iteration
    assert refinement.measurements == {"loss": 1.5}

    structured_refinement = RefinementStep(
        iteration=5,
        loss=2.0,
        wr2=0.05,
        objective_total=2.0,
        components={"diffraction": {"raw": 1.0, "weight": 2.0, "contribution": 2.0}},
    )
    assert structured_refinement.measurements == {
        "wr2": 0.05,
        "diffraction/raw": 1.0,
        "diffraction/weight": 2.0,
        "diffraction/contribution": 2.0,
    }

    orientation_step = RefinementOrientationStep(
        iteration=5, rotation_index=3, wr2=0.04, r_obs=0.05, diff_loss=0.02
    )
    assert orientation_step.channel == "refinement orientation"
    assert orientation_step.step == 3  # a refinement orientation step's step is its rotation_index
    assert orientation_step.measurements == {
        "iteration": 5.0,
        "wr2": 0.04,
        "r_obs": 0.05,
        "diff_loss": 0.02,
    }

    refinement_done = RefinementCompleted(n_steps=20, best_step=17, best_loss=0.3)
    assert refinement_done.channel == "refinement"  # shares the stream's channel
    assert refinement_done.step is None  # separated from the stream by step, not channel
    assert refinement_done.selection == "training"  # the default selection objective
    assert refinement_done.measurements == {
        "n_steps": 20.0,
        "best_step": 17.0,
        "best_training_loss": 0.3,
    }

    # A validation-selected run reports the same number under a *different* key, so a generic
    # backend cannot plot the two populations as one series -- the key is absent, not wrong.
    validation_done = RefinementCompleted(
        n_steps=20, best_step=17, best_loss=0.3, selection="validation"
    )
    assert validation_done.measurements == {
        "n_steps": 20.0,
        "best_step": 17.0,
        "best_validation_loss": 0.3,
    }
    assert "best_training_loss" not in validation_done.measurements


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


def test_console_logger_formats_device_selection(caplog: pytest.LogCaptureFixture) -> None:
    logger = ConsoleLogger(level=logging.INFO)
    with caplog.at_level(logging.INFO, logger="diffBloch.loggers"):
        logger.report(DeviceSelected(requested="cuda", selected="cpu", cuda_available=False))

    assert caplog.records[-1].getMessage() == (
        "No CUDA detected, using CPU, diffBloch is optimized for CUDA"
    )


def test_console_logger_formats_refinement_epoch_metrics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = ConsoleLogger(level=logging.INFO)
    with caplog.at_level(logging.INFO, logger="diffBloch.loggers"):
        logger.report(RefinementStep(iteration=7, loss=4.95, wr2=0.05))

    assert caplog.records[-1].getMessage() == (
        "Refinement epoch   8 │ wR2 0.050000 │ R_obs n/a │ diffraction loss n/a"
    )


def test_objective_manifest_declares_composition_including_the_empty_case() -> None:
    empty = ObjectiveManifest()
    assert empty.channel == "objective"
    assert empty.step is None  # a run-level declaration, not a per-step observation
    assert empty.measurements == {"n_penalties": 0.0, "n_constraints": 0.0, "n_components": 0.0}

    composed = ObjectiveManifest(
        penalties=(ObjectiveTerm(name="bond_length", weight=3.0),),
        constraints=("hydrogen_riding",),
        components=("apparent_thickness",),
    )
    assert composed.measurements == {
        "n_penalties": 1.0,
        "n_constraints": 1.0,
        "n_components": 1.0,
        "bond_length/weight": 3.0,
    }


def test_console_logger_states_an_empty_objective_as_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The default path composes no penalties; the report says so instead of omitting the line."""
    logger = ConsoleLogger(level=logging.INFO)
    with caplog.at_level(logging.INFO, logger="diffBloch.loggers"):
        logger.report(ObjectiveManifest(components=("apparent_thickness",)))

    assert [record.getMessage() for record in caplog.records] == [
        "Objective │ penalties  : none",
        "Objective │ constraints: none",
        "Objective │ components : apparent_thickness",
    ]


def test_refinement_step_reports_each_mean_with_its_own_denominator() -> None:
    """wR2 and R_obs are NaN-filtered independently, so they carry separate counts."""
    step = RefinementStep(
        iteration=0,
        loss=1.0,
        wr2=0.05,
        r_obs=0.07,
        n_rotations=99,
        n_wr2_evaluated=97,
        n_r_obs_evaluated=95,
    )
    assert step.measurements["n_rotations"] == 99.0
    assert step.measurements["n_wr2_evaluated"] == 97.0
    assert step.measurements["n_r_obs_evaluated"] == 95.0


def test_console_logger_prints_the_epoch_mean_denominators(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = ConsoleLogger(level=logging.INFO)
    with caplog.at_level(logging.INFO, logger="diffBloch.loggers"):
        logger.report(
            RefinementStep(
                iteration=0,
                loss=1.0,
                wr2=0.05,
                r_obs=0.07,
                n_rotations=99,
                n_wr2_evaluated=97,
                n_r_obs_evaluated=95,
            )
        )

    assert caplog.records[-1].getMessage() == (
        "Refinement epoch   1 │ wR2 0.050000 [97/99] │ R_obs 0.070000 [95/99] │ "
        "diffraction loss n/a"
    )


def test_refinement_step_measurements_carry_only_composed_objective_terms() -> None:
    """An absent restraint has no measurement at all -- it can never read as a satisfied zero."""
    without_penalty = RefinementStep(
        iteration=0,
        loss=1.0,
        wr2=0.05,
        components={"diffraction": {"raw": 1.0, "weight": 1.0, "contribution": 1.0}},
    )
    with_penalty = RefinementStep(
        iteration=0,
        loss=1.0,
        wr2=0.05,
        components={
            "diffraction": {"raw": 1.0, "weight": 1.0, "contribution": 1.0},
            "bond_length": {"raw": 0.0, "weight": 3.0, "contribution": 0.0},
        },
    )

    assert not any(key.startswith("bond_length/") for key in without_penalty.measurements)
    # Composed but currently satisfied: the raw value is 0.0, yet the weight proves it was applied.
    assert with_penalty.measurements["bond_length/raw"] == 0.0
    assert with_penalty.measurements["bond_length/weight"] == 3.0


def test_console_logger_reports_composed_penalties_beneath_the_epoch_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = ConsoleLogger(level=logging.INFO)
    with caplog.at_level(logging.INFO, logger="diffBloch.loggers"):
        logger.report(
            RefinementStep(
                iteration=0,
                loss=1.25,
                wr2=0.05,
                diff_loss=1.0,
                components={
                    "diffraction": {"raw": 1.0, "weight": 1.0, "contribution": 1.0},
                    "bond_length": {"raw": 0.125, "weight": 2.0, "contribution": 0.25},
                },
            )
        )

    messages = [record.getMessage() for record in caplog.records]
    # diffraction is not repeated as a penalty line -- it is already the epoch line's diff_loss.
    assert messages[-1] == (
        "  penalty bond_length          │ raw 0.125 │ weight 2 │ contribution 0.25"
    )
    assert not any("penalty diffraction" in message for message in messages)


def test_console_logger_formats_refinement_orientation_step(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = ConsoleLogger(level=logging.INFO)
    with caplog.at_level(logging.INFO, logger="diffBloch.loggers"):
        logger.report(
            RefinementOrientationStep(iteration=7, rotation_index=3, wr2=0.05, r_obs=0.09)
        )

    assert caplog.records[-1].getMessage() == (
        "  epoch   8 rotation   3 │ wR2 0.050000 │ R_obs 0.090000 │ diffraction loss n/a"
    )


def test_console_logger_formats_preprocess_stage(caplog: pytest.LogCaptureFixture) -> None:
    logger = ConsoleLogger(level=logging.INFO)
    event = PlanStepCompleted(
        channel="build_orientation_plans",
        index=0,
        measurements={"n_orientations": 2.0},
    )
    with caplog.at_level(logging.INFO, logger="diffBloch.loggers"):
        logger.report(event)

    assert caplog.records[-1].getMessage() == (
        "Preprocess stage  1 │ Build Orientation Plans     │ n_orientations=2"
    )


def test_console_logger_labels_orientation_refinement_index(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = ConsoleLogger(level=logging.INFO)
    with caplog.at_level(logging.INFO, logger="diffBloch.loggers"):
        logger.report(_fitted(index=10, score=0.025))

    assert (
        caplog.records[-1]
        .getMessage()
        .startswith("orientation optimization[rotation_index=10] wr2=0.025 n_matched_hkl=42")
    )


def test_console_logger_labels_thickness_refinement_index(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = ConsoleLogger(level=logging.INFO)
    with caplog.at_level(logging.INFO, logger="diffBloch.loggers"):
        logger.report(
            ThicknessOptimized(
                rotation_index=10,
                score=0.025,
                residual="wr2",
                thickness=820.0,
                candidate_thicknesses=(780.0, 820.0, 860.0),
                candidate_score=(0.04, 0.025, 0.038),
            )
        )

    assert caplog.records[-1].getMessage() == (
        "thickness optimization[rotation_index=10] wr2=0.025 thickness=820"
    )


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "0:00"),
        (45.0, "0:45"),
        (75.0, "1:15"),
        (3661.0, "1:01:01"),
    ],
)
def test_format_eta(seconds: float, expected: str) -> None:
    assert _format_eta(seconds) == expected


def test_console_logger_renders_a_refinement_progress_bar_on_a_tty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    logger = ConsoleLogger(level=logging.INFO)

    logger.report(RefinementStarted(total_steps=4))
    logger.report(RefinementStep(iteration=0, loss=1.0, wr2=0.05, r_obs=0.06))
    logger.report(RefinementStep(iteration=3, loss=0.5, wr2=0.03, r_obs=0.04))

    out = capsys.readouterr().out
    assert "1/4" in out and "4/4" in out
    assert "eta" in out  # mid-run (1/4): an ETA is extrapolated from the observed rate
    assert out.count("\r") == 2  # one write per RefinementStep, no per-event logging fallback
    assert out.endswith("\n")  # the completed (current >= total) bar ends the line
    assert "eta" not in out.rsplit("\r", 1)[-1]  # the final (4/4, complete) line has none


def test_console_logger_renders_an_orientation_progress_bar_on_a_tty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    logger = ConsoleLogger(level=logging.INFO)

    logger.report(OrientationOptimizationStarted(total_rotations=2))
    logger.report(_fitted(index=5, score=0.05))
    logger.report(_fitted(index=9, score=0.02))

    out = capsys.readouterr().out
    assert "1/2" in out and "2/2" in out
    assert out.endswith("\n")


def test_console_logger_renders_a_thickness_progress_bar_on_a_tty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    logger = ConsoleLogger(level=logging.INFO)

    logger.report(ThicknessOptimizationStarted(total_rotations=2))
    logger.report(
        ThicknessOptimized(
            rotation_index=5,
            score=0.05,
            residual="wr2",
            thickness=1000.0,
            candidate_thicknesses=(900.0, 1000.0),
            candidate_score=(0.1, 0.05),
        )
    )
    logger.report(
        ThicknessOptimized(
            rotation_index=9,
            score=0.02,
            residual="wr2",
            thickness=1100.0,
            candidate_thicknesses=(1000.0, 1100.0),
            candidate_score=(0.04, 0.02),
        )
    )

    out = capsys.readouterr().out
    assert "1/2" in out and "2/2" in out
    assert out.endswith("\n")


def test_console_logger_falls_back_to_plain_lines_off_a_tty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The default (non-tty, e.g. piped/CI) path is untouched: no ``*Started`` event required."""
    logger = ConsoleLogger(level=logging.INFO)
    with caplog.at_level(logging.INFO, logger="diffBloch.loggers"):
        logger.report(RefinementStarted(total_steps=4))
        logger.report(RefinementStep(iteration=0, loss=1.0, wr2=0.05, r_obs=0.06))

    assert caplog.records[-1].getMessage().startswith("Refinement epoch   1")


def test_early_abort_logger_ignores_the_started_events(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression: OrientationOptimizationStarted shares no channel with OrientationOptimized, so
    a channel-filtering consumer like EarlyAbortLogger must not try to read its (absent) wr2."""
    logger = EarlyAbortLogger(wr2_ceiling=0.6, patience=5)
    logger.report(OrientationOptimizationStarted(total_rotations=52))  # must not raise
    logger.report(RefinementStarted(total_steps=40))  # must not raise
    logger.report(ThicknessOptimizationStarted(total_rotations=52))  # must not raise


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
