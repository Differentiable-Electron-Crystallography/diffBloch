"""Domain-observation events and the pluggable logger sink (effects-as-data observability).

This is the *domain observations* channel -- distinct from stdlib ``logging``, which carries
solver *diagnostics*. The pure core **emits** typed
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

Events fall into two families: a *per-unit stream* (a :class:`RotationScored` per rotation, a
:class:`RefinementStep` per optimizer iteration -- each carries a ``step``) and a *run-level
aggregate* (:class:`InferenceCompleted`, :class:`RefinementCompleted` -- ``step`` is ``None``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Protocol, runtime_checkable

__all__ = [
    "NULL_LOGGER",
    "Event",
    "InferenceCompleted",
    "Logger",
    "MultiLogger",
    "NullLogger",
    "OrientationFitted",
    "RecordingLogger",
    "RefinementCompleted",
    "RefinementStep",
    "RotationScored",
]


@runtime_checkable
class Event(Protocol):
    """A named domain observation carrying numeric measurements.

    ``channel`` is the event's stable name (a class constant); ``measurements`` maps metric name to
    value; ``step`` is the optional position on the run's x-axis (a rotation index, later a
    refinement iteration) or ``None`` for a run-level aggregate. Together they let a generic logger
    record and *place* any event with no per-type knowledge.
    """

    channel: ClassVar[str]

    @property
    def step(self) -> int | None: ...

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
    def step(self) -> int | None:
        return self.index

    @property
    def measurements(self) -> Mapping[str, float]:
        return {
            "r_obs": self.r_obs,
            "n_observed": float(self.n_observed),
            "n_beams": float(self.n_beams),
        }


@dataclass(frozen=True)
class OrientationFitted:
    """One rotation's finished orientation search, emitted per rotation by ``fit_orientation``.

    The fit is the long phase of a run (a coupled search solves ~100+ trials per rotation), so this
    is the progress stream that makes it observable: ``index`` is the rotation's position in the
    plan, ``wr2`` the final scaling-optimised objective at the fitted orientation, ``n_trials`` the
    number of trial orientations the search scored, ``n_passes`` the number of hexagonal-ring sweeps
    the search took to converge (the quantity ``HexagonalSearch.max_iterations`` caps), and
    ``pass_cap`` that cap itself -- carried per event so a plot can show each rotation's headroom
    (``n_passes`` vs ``pass_cap``) and flag any rotation that ran to the cap. With ``workers > 1``
    events arrive in *completion* order (the plan itself stays ordered).
    """

    channel: ClassVar[str] = "fit"
    index: int
    wr2: float
    n_trials: int
    n_passes: int
    pass_cap: int

    @property
    def step(self) -> int | None:
        return self.index

    @property
    def measurements(self) -> Mapping[str, float]:
        return {
            "wr2": self.wr2,
            "n_trials": float(self.n_trials),
            "n_passes": float(self.n_passes),
            "pass_cap": float(self.pass_cap),
        }


@dataclass(frozen=True)
class InferenceCompleted:
    """The run-level aggregate, emitted once when ``run_inference`` finishes."""

    channel: ClassVar[str] = "inference"
    n_rotations: int
    n_evaluated: int
    mean_r_obs: float

    @property
    def step(self) -> int | None:
        return None  # a run-level aggregate has no position on the per-rotation axis

    @property
    def measurements(self) -> Mapping[str, float]:
        return {
            "n_rotations": float(self.n_rotations),
            "n_evaluated": float(self.n_evaluated),
            "mean_r_obs": self.mean_r_obs,
        }


@dataclass(frozen=True)
class RefinementStep:
    """One optimizer iteration, emitted per step by ``run_refinement``.

    ``loss`` is the scalar value recorded by the optimizer loop. When available,
    ``objective_total`` and ``components`` expose the structured
    :class:`diffBloch.engine.refine.ObjectiveValue` diagnostics as plain numeric measurements:
    every component contributes ``component.<name>.raw``, ``.weight``, and ``.contribution``.
    """

    channel: ClassVar[str] = "refinement"
    iteration: int
    loss: float
    objective_total: float | None = None
    components: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        copied = {name: MappingProxyType(dict(values)) for name, values in self.components.items()}
        object.__setattr__(self, "components", MappingProxyType(copied))

    @property
    def step(self) -> int | None:
        return self.iteration

    @property
    def measurements(self) -> Mapping[str, float]:
        measurements = {"loss": self.loss}
        if self.objective_total is not None:
            measurements["objective_total"] = self.objective_total
        for name, values in self.components.items():
            for field_name, value in values.items():
                measurements[f"component.{name}.{field_name}"] = value
        return measurements


@dataclass(frozen=True)
class RefinementCompleted:
    """The refinement-run aggregate, emitted once when ``run_refinement`` finishes.

    Shares the ``"refinement"`` channel with :class:`RefinementStep`: unlike inference (where a
    per-rotation score and the run mean are distinct entities on distinct channels), a run's
    per-step loss and its best-loss summary are the same quantity at different granularities, so
    ``step`` -- the iteration index vs ``None`` -- is what separates the stream from the summary.
    """

    channel: ClassVar[str] = "refinement"
    n_steps: int
    best_step: int
    best_loss: float

    @property
    def step(self) -> int | None:
        return None  # a run-level aggregate has no position on the per-iteration axis

    @property
    def measurements(self) -> Mapping[str, float]:
        return {
            "n_steps": float(self.n_steps),
            "best_step": float(self.best_step),
            "best_loss": self.best_loss,
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
