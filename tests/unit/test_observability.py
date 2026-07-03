"""The domain-observation channel: typed events and pluggable logger sinks.

Unit-level (no engine): the event surface (``channel`` + ``measurements``), the null/fan-out
loggers, and the two boundary backends. ``run_inference`` emission is covered in
``test_inference`` (it needs the forward model). See the effects-and-observability decision doc.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

from diffBloch.app.loggers import ConsoleLogger
from diffBloch.app.loggers.wandb import WandbLogger
from diffBloch.observability import (
    Event,
    InferenceCompleted,
    Logger,
    MultiLogger,
    NullLogger,
    RecordingLogger,
    RotationScored,
)


def test_events_expose_a_uniform_channel_and_measurements_surface() -> None:
    rotation = RotationScored(index=3, r_obs=0.42, n_observed=12, n_beams=20)
    assert rotation.channel == "rotation"
    assert rotation.measurements == {"r_obs": 0.42, "n_observed": 12.0, "n_beams": 20.0}

    completed = InferenceCompleted(n_rotations=99, n_evaluated=97, mean_r_obs=0.065)
    assert completed.channel == "inference"
    assert completed.measurements == {
        "n_rotations": 99.0,
        "n_evaluated": 97.0,
        "mean_r_obs": 0.065,
    }


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


def test_console_logger_logs_channel_and_measurements(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = ConsoleLogger(level=logging.INFO)
    with caplog.at_level(logging.INFO, logger="diffBloch.loggers"):
        logger.report(RotationScored(index=0, r_obs=0.5, n_observed=4, n_beams=7))

    (record,) = caplog.records
    assert "rotation" in record.getMessage()
    assert "r_obs=0.5" in record.getMessage()


def test_wandb_logger_maps_measurements_to_a_namespaced_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[dict[str, float]] = []
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.log = logged.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    WandbLogger().report(InferenceCompleted(n_rotations=2, n_evaluated=2, mean_r_obs=0.06))

    assert logged == [
        {"inference/n_rotations": 2.0, "inference/n_evaluated": 2.0, "inference/mean_r_obs": 0.06}
    ]
