"""Domain-observation events and the pluggable logger sink (effects-as-data observability).

This is the *domain observations* channel of ``design/decisions/effects-and-observability.md`` --
distinct from stdlib ``logging``, which carries solver *diagnostics*. The pure core **emits** typed
events as plain values; a :class:`Logger` attached at the ``app/`` boundary interprets them. The
core installs no sink and runs correctly with the :data:`NULL_LOGGER` default, so it stays pure,
testable, and vendor-free: Weights & Biases / Comet ML / CSV live only in logger *backends* at the
boundary (``diffBloch.app.loggers``), never in the maths.

The name follows the PyTorch-Lightning convention (``WandbLogger`` / ``CometLogger`` / ``CSVLogger``
plug into a common ``Logger``); it is the experiment-tracking sink, orthogonal to the stdlib
``logging.Logger`` used for diagnostics. Every :class:`Event` exposes a uniform
``(channel, measurements)`` surface -- the Phoenix ``:telemetry`` "named event + measurements" idea
-- so a generic logger consumes *any* event without knowing its concrete type; adding an event never
touches a logger. Callers wanting richer handling can still pattern-match the concrete dataclass.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

__all__ = [
    "NULL_LOGGER",
    "Event",
    "InferenceCompleted",
    "Logger",
    "MultiLogger",
    "NullLogger",
    "RecordingLogger",
    "RotationScored",
]


@runtime_checkable
class Event(Protocol):
    """A named domain observation carrying numeric measurements.

    ``channel`` is the event's stable name (a class constant); ``measurements`` maps metric name to
    value. Together they let a generic logger record any event with no per-type knowledge.
    """

    channel: ClassVar[str]

    @property
    def measurements(self) -> Mapping[str, float]: ...


@runtime_checkable
class Logger(Protocol):
    """A sink for domain-observation events, attached at the app boundary.

    A logger performs I/O (print, CSV row, ``wandb.log``); the core only hands it values. The core
    defaults to :data:`NULL_LOGGER` so it installs no sink and can run with none attached. Implement
    a single method to add a backend -- see ``diffBloch.app.loggers``.
    """

    def report(self, event: Event) -> None: ...


@dataclass(frozen=True)
class RotationScored:
    """One rotation's forward-inference score, emitted per rotation by ``run_inference``."""

    channel: ClassVar[str] = "rotation"
    index: int
    r_obs: float
    n_observed: int
    n_beams: int

    @property
    def measurements(self) -> Mapping[str, float]:
        return {
            "r_obs": self.r_obs,
            "n_observed": float(self.n_observed),
            "n_beams": float(self.n_beams),
        }


@dataclass(frozen=True)
class InferenceCompleted:
    """The run-level aggregate, emitted once when ``run_inference`` finishes."""

    channel: ClassVar[str] = "inference"
    n_rotations: int
    n_evaluated: int
    mean_r_obs: float

    @property
    def measurements(self) -> Mapping[str, float]:
        return {
            "n_rotations": float(self.n_rotations),
            "n_evaluated": float(self.n_evaluated),
            "mean_r_obs": self.mean_r_obs,
        }


class NullLogger:
    """The default sink: discards every event, so the core runs with no logger attached."""

    def report(self, event: Event) -> None:
        return None


NULL_LOGGER = NullLogger()


@dataclass(frozen=True)
class MultiLogger:
    """Fan each event out to several loggers (e.g. console and wandb at once)."""

    loggers: tuple[Logger, ...]

    def report(self, event: Event) -> None:
        for logger in self.loggers:
            logger.report(event)


@dataclass
class RecordingLogger:
    """An in-memory logger that keeps every event (the doc's "in-memory history" sink).

    A shippable backend -- useful for post-hoc inspection of a run and as the natural test double
    (assert on ``events`` instead of scraping a console). Unlike the vendor backends it performs no
    external I/O, so it stays vendor-free here beside :class:`NullLogger` / :class:`MultiLogger`.
    """

    events: list[Event] = field(default_factory=list)

    def report(self, event: Event) -> None:
        self.events.append(event)
