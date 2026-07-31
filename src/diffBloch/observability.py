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
    "CouplingSummary",
    "ConvergenceTrial",
    "ConvergencePassStarted",
    "ConvergenceSweepStarted",
    "Event",
    "InferenceCompleted",
    "Logger",
    "MultiLogger",
    "NullLogger",
    "OrientationOptimized",
    "OrientationOptimizationStarted",
    "OrientationOptimizationSummary",
    "PlanStepCompleted",
    "RecordingLogger",
    "RefinementCompleted",
    "RefinementOrientationStep",
    "RefinementStarted",
    "RefinementStep",
    "RotationCoupling",
    "RotationScored",
    "ThicknessOptimized",
]


@runtime_checkable
class Event(Protocol):
    """A named domain observation carrying numeric measurements.

    ``channel`` is the event's stable name -- usually a class constant, but read as a plain
    attribute so an event may set it per instance (e.g. :class:`PlanStepCompleted` uses the pipeline
    step's name). ``measurements`` maps metric name to value; ``step`` is the optional position on
    the run's x-axis (a rotation index, later a refinement iteration) or ``None`` for a run-level
    aggregate. Together they let a generic logger record and *place* any event, no per-type view.
    """

    @property
    def channel(self) -> str: ...

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
class ConvergenceTrial:
    """One comparison between consecutive numerical settings in a convergence sweep."""

    control: str
    trial_index: int
    pass_index: int
    previous: float
    candidate: float
    r_factor: float
    n_compared_hkl: int

    @property
    def channel(self) -> str:
        return f"convergence {self.control}"

    @property
    def step(self) -> int | None:
        return self.trial_index

    @property
    def measurements(self) -> Mapping[str, float]:
        return {
            "pass": float(self.pass_index),
            "previous": self.previous,
            "candidate": self.candidate,
            "r_factor": self.r_factor,
            "n_compared_hkl": float(self.n_compared_hkl),
        }


@dataclass(frozen=True)
class ConvergencePassStarted:
    """Starting settings for one coordinated convergence pass."""

    pass_index: int
    g_max: float
    sg_max: float
    tilt_steps: int
    r_factor_threshold: float
    n_orientations: int

    channel: ClassVar[str] = "convergence pass"

    @property
    def step(self) -> int | None:
        return self.pass_index

    @property
    def measurements(self) -> Mapping[str, float]:
        return {
            "g_max": self.g_max,
            "sg_max": self.sg_max,
            "tilt_steps": float(self.tilt_steps),
            "r_factor_threshold": self.r_factor_threshold,
            "n_orientations": float(self.n_orientations),
        }


@dataclass(frozen=True)
class ConvergenceSweepStarted:
    """Announcement emitted before one parameter sweep begins."""

    control: str
    pass_index: int

    channel: ClassVar[str] = "convergence sweep"

    @property
    def step(self) -> int | None:
        return self.pass_index

    @property
    def measurements(self) -> Mapping[str, float]:
        return {"pass": float(self.pass_index)}


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
class OrientationOptimizationStarted:
    """The rotation count ``optimize_orientation`` is about to search, emitted once before any of it.

    Exists so a progress display can show a countdown (``n_seen / total_rotations``) against
    :class:`OrientationOptimized` without needing to know the plan size in advance -- the plan is
    only assembled deep inside the step itself. Deliberately a distinct channel from
    ``OrientationOptimized`` (not merely a different type) -- a consumer such as
    :class:`~diffBloch.app.loggers.EarlyAbortLogger` that filters by ``event.channel`` alone must
    not mistake this for a per-rotation result.
    """

    channel: ClassVar[str] = "orientation_started"
    total_rotations: int

    @property
    def step(self) -> int | None:
        return None

    @property
    def measurements(self) -> Mapping[str, float]:
        return {"total_rotations": float(self.total_rotations)}


@dataclass(frozen=True)
class OrientationOptimized:
    """One rotation's finished orientation search, emitted per rotation by ``optimize_orientation``.

    The fit is the long phase of a run (a coupled search solves ~100+ trials per rotation), so this
    is the progress stream that makes it observable: ``rotation_index`` is the original zero-based
    PETS rotation index, ``wr2`` the final scaling-optimised objective, ``n_trials`` the
    number of trial orientations the search scored, ``n_passes`` scipy's reported iteration count
    (the quantity ``NelderMeadSearch.max_iterations`` caps), and
    ``pass_cap`` that cap itself -- carried per event so a plot can show each rotation's headroom
    (``n_passes`` vs ``pass_cap``) and flag any rotation that ran to the cap. With ``workers > 1``
    events arrive in *completion* order (the plan itself stays ordered). The channel is shared
    with the step's ``PlanStepCompleted`` summary line, like the refinement stream's events.
    """

    channel: ClassVar[str] = "orientation"
    rotation_index: int
    wr2: float
    n_matched_hkl: int
    n_trials: int
    n_passes: int
    pass_cap: int

    @property
    def step(self) -> int | None:
        return self.rotation_index

    @property
    def measurements(self) -> Mapping[str, float]:
        return {"wr2": self.wr2, "n_matched_hkl": float(self.n_matched_hkl)}


@dataclass(frozen=True)
class OrientationOptimizationSummary:
    """Aggregate statistics after every rotation's orientation fit has completed."""

    n_orientations: int
    mean_wr2: float
    mean_r_obs: float
    total_matched_hkl: int
    total_strong_hkl: int
    total_weak_hkl: int
    total_observed_hkl: int
    total_trials: int
    max_passes: int

    channel: ClassVar[str] = "orientation summary"

    @property
    def step(self) -> int | None:
        return None

    @property
    def measurements(self) -> Mapping[str, float]:
        return {
            "n_orientations": float(self.n_orientations),
            "mean_wr2": self.mean_wr2,
            "mean_r_obs": self.mean_r_obs,
            "total_matched_hkl": float(self.total_matched_hkl),
            "total_strong_hkl": float(self.total_strong_hkl),
            "total_weak_hkl": float(self.total_weak_hkl),
            "total_observed_hkl": float(self.total_observed_hkl),
            "total_unmatched_hkl": float(self.total_observed_hkl - self.total_matched_hkl),
            "total_trials": float(self.total_trials),
            "max_passes": float(self.max_passes),
        }


@dataclass(frozen=True)
class ThicknessOptimized:
    """One rotation's finished thickness grid search, emitted per rotation by ``optimize_thickness``.

    The thickness fit is the memory-heavy tail phase (each rotation scores the whole
    ``ThicknessGrid`` in one segmented solve), so like :class:`OrientationOptimized` this makes it a
    progress stream rather than a silent block: ``rotation_index`` is the original zero-based PETS
    rotation index, ``wr2`` the scaling-optimised objective at the baked thickness, and
    ``thickness`` that winning candidate (Angstrom). ``candidate_thicknesses``/``candidate_wr2``
    carry the whole scored grid (same order, one entry per :class:`~diffBloch.specs.ThicknessGrid`
    step) -- deliberately excluded from ``measurements`` (which stays flat-scalar for the
    generic console/CSV/wandb/comet backends); a plotting backend such as
    :class:`~diffBloch.app.loggers.plotting.ThicknessPlotLogger` pattern-matches the concrete
    dataclass to read them. Emitted in plan order (the fit is sequential).
    """

    channel: ClassVar[str] = "optimize_thickness"
    rotation_index: int
    wr2: float
    thickness: float
    candidate_thicknesses: tuple[float, ...]
    candidate_wr2: tuple[float, ...]

    @property
    def step(self) -> int | None:
        return self.rotation_index

    @property
    def measurements(self) -> Mapping[str, float]:
        return {"wr2": self.wr2, "thickness": self.thickness}


@dataclass(frozen=True)
class PlanStepCompleted:
    """The Plan produced by one preprocess pipeline step, summarised as the recipe runs.

    Unlike the other events its ``channel`` is the *step name* (``select_beams``,
    ``optimize_orientation``, ...), set per instance rather than a class constant -- so the console reads
    ``optimize_orientation[4] n_orientations=55 ...``, carrying the categorical
    step identity a fixed channel cannot. ``index`` is the step's ordinal in the recipe (its
    ``step`` on the run's x-axis); ``measurements`` is
    :func:`diffBloch.preprocess.plan.summarize_plan` of the resulting plan. Emitted only on a
    *fresh* preprocess run -- a reused checkpoint runs no steps.
    """

    channel: str
    index: int
    measurements: Mapping[str, float]

    @property
    def step(self) -> int | None:
        return self.index


@dataclass(frozen=True)
class RotationCoupling:
    """One rotation's coupled solve geometry, emitted per rotation at the consumer boundary.

    The shape the refinement loop repeats every step: ``n_coupling_segments`` coupled unions over
    ``n_tilts`` rocking-curve tilts, the widest union spanning ``max_tilts_per_segment`` tilts, the
    deduped union carrying ``n_union_beams`` beams, and the largest single segment
    ``max_beams_per_segment`` beams -- the ``N`` of the dominant per-segment eigensolve. Fires on
    every run (fresh or checkpoint-reuse), so the coupling a long refine is about to chew on is
    legible before the first step.
    """

    channel: ClassVar[str] = "coupling"
    index: int
    n_coupling_segments: int
    n_tilts: int
    max_tilts_per_segment: int
    n_union_beams: int
    max_beams_per_segment: int

    @property
    def step(self) -> int | None:
        return self.index

    @property
    def measurements(self) -> Mapping[str, float]:
        return {
            "n_coupling_segments": float(self.n_coupling_segments),
            "n_tilts": float(self.n_tilts),
            "max_tilts_per_segment": float(self.max_tilts_per_segment),
            "n_union_beams": float(self.n_union_beams),
            "max_beams_per_segment": float(self.max_beams_per_segment),
        }


@dataclass(frozen=True)
class CouplingSummary:
    """Run-level summary of the plan the refinement/inference consumes (on the coupling channel).

    The aggregate companion to the per-rotation :class:`RotationCoupling` (``step`` ``None`` vs a
    rotation index separates the two on one channel): ``measurements`` is
    :func:`diffBloch.preprocess.plan.summarize_plan` -- the structure-factor support size/radius
    plus the coupling aggregates across rotations. Emitted once at the consumer boundary.
    """

    channel: ClassVar[str] = "coupling"
    measurements: Mapping[str, float]

    @property
    def step(self) -> int | None:
        return None


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
class RefinementStarted:
    """The epoch budget ``run_refinement_model`` is about to run, emitted once before the loop.

    Exists so a progress display can show a countdown (``iteration / total_steps``) against
    :class:`RefinementStep` without needing the config's ``refinement.steps`` passed in separately.
    Deliberately a distinct channel from ``RefinementStep`` (not merely a different type) -- a
    consumer that filters by ``event.channel`` alone must not mistake this for a per-epoch result.
    """

    channel: ClassVar[str] = "refinement_started"
    total_steps: int

    @property
    def step(self) -> int | None:
        return None

    @property
    def measurements(self) -> Mapping[str, float]:
        return {"total_steps": float(self.total_steps)}


@dataclass(frozen=True)
class RefinementStep:
    """One refinement epoch.

    ``wr2`` is the mean weighted-R2 across orientations when weighted-R2 is the data term.
    Otherwise the optimizer loss is reported.
    """

    channel: ClassVar[str] = "refinement"
    iteration: int
    loss: float
    wr2: float | None = None
    r_obs: float | None = None
    diff_loss: float | None = None
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
        values: dict[str, float] = {}
        if self.wr2 is not None:
            values["wr2"] = self.wr2
        if self.r_obs is not None:
            values["r_obs"] = self.r_obs
        if self.diff_loss is not None:
            values["diff_loss"] = self.diff_loss
        if not values:
            values["loss"] = self.loss
        return values


@dataclass(frozen=True)
class RefinementOrientationStep:
    """One rotation's wR2/R_obs/diffraction-loss diagnostics within a refinement epoch.

    The per-orientation companion to :class:`RefinementStep`'s epoch mean:
    ``run_refinement_model`` emits one of these per rotation per step only when its ``verbose``
    flag is set (the "verbose refinement" reporting mode) -- the per-rotation stream is
    ``n_orientations``x louder than the epoch summary, so it is a diagnosis tool, not the default
    reporting shape. ``iteration`` places it on the same x-axis as :class:`RefinementStep`;
    ``rotation_index`` (the original zero-based PETS rotation index) is this event's ``step``,
    matching the per-rotation convention of :class:`RotationScored` / :class:`OrientationOptimized`.
    """

    channel: ClassVar[str] = "refinement orientation"
    iteration: int
    rotation_index: int
    wr2: float | None = None
    r_obs: float | None = None
    diff_loss: float | None = None

    @property
    def step(self) -> int | None:
        return self.rotation_index

    @property
    def measurements(self) -> Mapping[str, float]:
        values: dict[str, float] = {"iteration": float(self.iteration)}
        if self.wr2 is not None:
            values["wr2"] = self.wr2
        if self.r_obs is not None:
            values["r_obs"] = self.r_obs
        if self.diff_loss is not None:
            values["diff_loss"] = self.diff_loss
        return values


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
