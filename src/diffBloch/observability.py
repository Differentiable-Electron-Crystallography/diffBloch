"""Domain-observation events and the pluggable logger sink (effects-as-data observability).

This is the *domain observations* channel -- distinct from stdlib ``logging``, which carries
solver *diagnostics*. The pure core **emits** typed events as plain values; a :class:`Logger`
attached at the ``app/`` boundary interprets them. The
core installs no sink and runs correctly with the :data:`NULL_LOGGER` default, so it stays pure,
testable, and vendor-free: file/vendor sinks live only in logger *backends* at the boundary
(``diffBloch.app.loggers``), never in the maths.

The name follows the PyTorch-Lightning convention (``WandbLogger`` / ``CometLogger`` /
``ReportLogger`` plug into a common ``Logger``); it is the experiment-tracking sink, orthogonal to
the stdlib ``logging.Logger`` used for diagnostics. Every :class:`Event` exposes a uniform
``(channel, measurements)`` surface -- the Phoenix ``:telemetry`` "named event + measurements" idea
-- so a generic logger consumes *any* event without knowing its concrete type; adding an event never
touches a logger. Callers wanting richer handling can still pattern-match the concrete dataclass.

Events fall into two families: a *per-unit stream* (a :class:`RotationScored` per rotation, a
:class:`RefinementStep` per optimizer iteration -- each carries a ``step``) and a *run-level
aggregate* (:class:`InferenceCompleted`, :class:`RefinementCompleted` -- ``step`` is ``None``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "NULL_LOGGER",
    "CouplingSummary",
    "ConvergenceTrial",
    "ConvergencePassStarted",
    "ConvergenceSweepStarted",
    "DeviceSelected",
    "Event",
    "EventRecord",
    "ExperimentDeclared",
    "InferenceCompleted",
    "Logger",
    "MultiLogger",
    "NullLogger",
    "ObjectiveManifest",
    "ObjectiveTerm",
    "OrientationOptimized",
    "OrientationOptimizationStarted",
    "OrientationOptimizationSummary",
    "OrientationSearchTrace",
    "PlanSeeded",
    "PlanStepCompleted",
    "PreprocessCompleted",
    "RefinedRotationMetrics",
    "RefinementCompleted",
    "RefinementOrientationStep",
    "RefinementOutputsWritten",
    "RefinementStarted",
    "RefinementStep",
    "RotationCouplingSegments",
    "RotationCoupling",
    "RotationScored",
    "RunStage",
    "RunStageStarted",
    "RunStageStopped",
    "RunStageStatus",
    "ThicknessOptimized",
    "ThicknessOptimizationStarted",
    "ThicknessProfile",
    "event_record_from_event",
]

_EVENT_SCHEMA_VERSION: Literal[1] = 1
RunStage = Literal["converge", "preprocess", "infer", "refine"]
RunStageStatus = Literal["completed", "failed"]


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

    A logger performs I/O (print, JSONL row, ``wandb.log``); the core only hands it values. The core
    defaults to :data:`NULL_LOGGER` so it installs no sink and can run with none attached. Implement
    a single method to add a backend -- see ``diffBloch.app.loggers``.
    """

    def report(self, event: Event) -> None: ...


class EventRecord(BaseModel):
    """Versioned, durable event-log envelope for visualizers and post-run tools.

    The live :class:`Event` protocol stays deliberately small: ``channel`` + ``step`` +
    scalar ``measurements`` are enough for console, W&B, and Comet sinks. Post-run
    visualizers need the richer concrete event data too, such as
    :class:`ThicknessOptimized`'s full thickness grid or :class:`ThicknessProfile`'s curve. This
    envelope preserves both: generic scalar measurements for easy pivoting, and a JSON payload made
    from the event dataclass fields for type-aware renderers.

    This is a data contract only. File writing, WebSocket broadcasting, notebooks, and plotting live
    outside the functional core.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="strings")

    schema_version: Literal[1] = _EVENT_SCHEMA_VERSION
    run_id: str
    sequence: int = Field(ge=0)
    timestamp_utc: str
    event_type: str
    channel: str
    step: int | None
    dataset: str | None = None
    rotation_index: int | None = None
    measurements: dict[str, float]
    series: dict[str, list[float]] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    payload: dict[str, Any]


def event_record_from_event(
    event: Event,
    *,
    run_id: str,
    sequence: int,
    timestamp: datetime | None = None,
) -> EventRecord:
    """Wrap one live event in the durable event-log schema.

    ``sequence`` is assigned by the sink, not by the event producer, so multiple sinks can consume
    the same pure event stream without sharing mutable state. ``timestamp`` is injectable for tests;
    callers normally let it default to the current UTC time.
    """
    emitted = datetime.now(UTC) if timestamp is None else timestamp.astimezone(UTC)
    payload = _event_payload(event)
    return EventRecord(
        run_id=run_id,
        sequence=sequence,
        timestamp_utc=emitted.isoformat(),
        event_type=type(event).__name__,
        channel=event.channel,
        step=event.step,
        dataset=_optional_str(_first_present(payload, "dataset", "label")),
        rotation_index=_optional_int(_first_present(payload, "rotation_index", "index")),
        measurements=dict(event.measurements),
        series=_numeric_series(payload),
        artifacts=_artifact_paths(payload),
        payload=payload,
    )


def _event_payload(event: Event) -> dict[str, Any]:
    if not is_dataclass(event):
        return {}
    return {field.name: _jsonable(getattr(event, field.name)) for field in fields(event)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(entry) for key, entry in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(entry) for entry in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _numeric_series(payload: Mapping[str, Any]) -> dict[str, list[float]]:
    """Extract plot-ready numeric arrays from the full payload."""
    out: dict[str, list[float]] = {}
    for key, value in payload.items():
        if (
            isinstance(value, list)
            and value
            and all(isinstance(item, int | float) for item in value)
        ):
            out[key] = [float(item) for item in value]
    return out


def _artifact_paths(payload: Mapping[str, Any]) -> dict[str, str]:
    value = payload.get("artifacts")
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): str(path)
        for key, path in value.items()
        if isinstance(key, str) and isinstance(path, str)
    }


@dataclass(frozen=True)
class DeviceSelected:
    """Execution-device selection for an app run.

    Device placement is an execution knob, not scientific provenance. This run-level event makes the
    selected backend visible to console/event/vendor sinks without entering config or checkpoint
    identity. Presentation wording stays with concrete logger backends; this event carries only
    stable selection data plus numeric measurements for generic metric sinks.
    """

    requested: str
    selected: str
    cuda_available: bool

    channel: ClassVar[str] = "device"

    @property
    def step(self) -> int | None:
        return None

    @property
    def measurements(self) -> Mapping[str, float]:
        return {
            "cuda_available": float(self.cuda_available),
            "selected_cuda": float(self.selected.startswith("cuda")),
        }


@dataclass(frozen=True)
class RunStageStarted:
    """A declared app-stage boundary before work starts.

    Report visualizers should use these lifecycle events to create command/stage sections rather
    than inferring boundaries from whichever result events happened to be emitted. The stage is an
    app-level workflow name, not a scientific knob, so it is report payload only and never part of
    config or lock identity.
    """

    stage: RunStage
    experiment_directory: str = ""

    channel: ClassVar[str] = "run_stage"

    @property
    def step(self) -> int | None:
        return None

    @property
    def measurements(self) -> Mapping[str, float]:
        return {}


@dataclass(frozen=True)
class RunStageStopped:
    """A declared app-stage boundary after work stops.

    ``status`` records whether the stage completed normally or stopped because an exception was
    raised. The optional error fields stay in the structured payload; scalar measurements carry only
    machine-friendly timing and status flags for generic sinks.
    """

    stage: RunStage
    status: RunStageStatus
    elapsed_seconds: float
    experiment_directory: str = ""
    error_type: str = ""
    error_message: str = ""

    channel: ClassVar[str] = "run_stage"

    @property
    def step(self) -> int | None:
        return None

    @property
    def measurements(self) -> Mapping[str, float]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "failed": float(self.status == "failed"),
        }


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
    dataset: str = ""
    rotation_index: int | None = None

    @property
    def step(self) -> int | None:
        return self.rotation_index if self.rotation_index is not None else self.index

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
    ``OrientationOptimized`` (not merely a different type), so channel-filtering consumers do not
    mistake this for a per-rotation result.
    """

    channel: ClassVar[str] = "orientation_started"
    total_rotations: int
    dataset: str = ""

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
    PETS rotation index, ``score`` the final orientation's value under ``residual`` -- the
    :class:`~diffBloch.config.schema.LossMetricsConfig` name (``"wr2"``/``"robs"``) that produced
    it, carried alongside so a consumer can label the number correctly
    (:attr:`measurements` keys on it directly, e.g. ``{"wr2": ...}`` or ``{"robs": ...}``) rather
    than a generic, misleading ``wr2`` field under a different residual. ``n_trials`` the number of
    trial orientations the search scored, ``n_passes`` scipy's reported iteration count (the
    quantity ``NelderMeadSearch.max_iterations`` caps), and ``pass_cap`` that cap itself -- carried
    per event so a plot can show each rotation's headroom (``n_passes`` vs ``pass_cap``) and flag
    any rotation that ran to the cap. With ``workers > 1`` events arrive in *completion* order (the
    plan itself stays ordered). The channel is shared with the step's ``PlanStepCompleted`` summary
    line, like the refinement stream's events.
    """

    channel: ClassVar[str] = "orientation"
    rotation_index: int
    score: float
    residual: str
    n_matched_hkl: int
    n_trials: int
    n_passes: int
    pass_cap: int
    dataset: str = ""
    seed_score: float | None = None
    alpha: float | None = None
    beta: float | None = None
    omega: float | None = None

    @property
    def step(self) -> int | None:
        return self.rotation_index

    @property
    def measurements(self) -> Mapping[str, float]:
        values = {self.residual: self.score, "n_matched_hkl": float(self.n_matched_hkl)}
        if self.seed_score is not None:
            values[f"seed_{self.residual}"] = self.seed_score
        if self.alpha is not None:
            values["delta_alpha_deg"] = self.alpha
        if self.beta is not None:
            values["delta_beta_deg"] = self.beta
        if self.omega is not None:
            values["delta_omega_deg"] = self.omega
        return values


@dataclass(frozen=True)
class OrientationOptimizationSummary:
    """Aggregate statistics after every rotation's orientation fit has completed.

    ``unique_*`` counts are deduplicated distinct ``(h, k, l)`` counts across every rotation's own
    set (:func:`~diffBloch.preprocess.plan.unique_hkl_count`), not a sum of each rotation's own
    count -- a reflection re-observed (or matched) in more than one rotation is counted once, not
    once per rotation. ``unique_strong_hkl`` is "matched *and* I > 3*sigma in at least one rotation"
    -- the same reflection can be strong in one rotation and weak in another, so this is a
    lower bound on "genuinely always weak," not a claim every occurrence was strong.
    """

    n_orientations: int
    mean_score: float
    residual: str
    unique_matched_hkl: int
    unique_strong_hkl: int
    unique_observed_hkl: int
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
            f"mean_{self.residual}": self.mean_score,
            "unique_matched_hkl": float(self.unique_matched_hkl),
            "unique_strong_hkl": float(self.unique_strong_hkl),
            "unique_weak_hkl": float(self.unique_matched_hkl - self.unique_strong_hkl),
            "unique_observed_hkl": float(self.unique_observed_hkl),
            "unique_unmatched_hkl": float(self.unique_observed_hkl - self.unique_matched_hkl),
            "total_trials": float(self.total_trials),
            "max_passes": float(self.max_passes),
        }


@dataclass(frozen=True)
class OrientationSearchTrace:
    """The scored orientation-search path for one rotation, batched for post-run visualization.

    ``OrientationOptimized`` is the progress/result event. This trace is the richer notebook
    contract: a compact numeric table of the seed, every scipy objective evaluation, and the final
    best point. The event is emitted once per rotation, after the search completes, to avoid calling
    logger backends inside the objective's hot loop.
    """

    channel: ClassVar[str] = "orientation trace"
    rotation_index: int
    residual: str
    trial_index: tuple[int, ...]
    alpha: tuple[float, ...]
    beta: tuple[float, ...]
    omega: tuple[float, ...]
    score: tuple[float, ...]
    comparable_score: tuple[float, ...]
    n_matched_hkl: tuple[int, ...]
    is_seed: tuple[int, ...]
    is_final: tuple[int, ...]
    dataset: str = ""

    def __post_init__(self) -> None:
        lengths = {
            len(self.trial_index),
            len(self.alpha),
            len(self.beta),
            len(self.omega),
            len(self.score),
            len(self.comparable_score),
            len(self.n_matched_hkl),
            len(self.is_seed),
            len(self.is_final),
        }
        if len(lengths) != 1:
            raise ValueError("orientation trace columns must have equal length")

    @property
    def step(self) -> int | None:
        return self.rotation_index

    @property
    def measurements(self) -> Mapping[str, float]:
        values = {
            "n_trials": float(len(self.trial_index)),
            f"best_{self.residual}": min(self.score) if self.score else float("nan"),
        }
        final_scores = [
            value for value, is_final in zip(self.score, self.is_final, strict=True) if is_final
        ]
        if final_scores:
            values[f"final_{self.residual}"] = final_scores[-1]
        return values


@dataclass(frozen=True)
class ThicknessOptimizationStarted:
    """The rotation count ``optimize_thickness`` is about to grid-search, emitted once up front.

    Exists so a progress display can show a countdown (``n_seen / total_rotations``) against
    :class:`ThicknessOptimized` without needing to know the plan size in advance -- mirrors
    :class:`OrientationOptimizationStarted`. Deliberately a distinct channel from
    ``ThicknessOptimized`` (not merely a different type), so channel-filtering consumers do not
    mistake this for a per-rotation result.
    """

    channel: ClassVar[str] = "thickness_started"
    total_rotations: int
    dataset: str = ""

    @property
    def step(self) -> int | None:
        return None

    @property
    def measurements(self) -> Mapping[str, float]:
        return {"total_rotations": float(self.total_rotations)}


@dataclass(frozen=True)
class ThicknessOptimized:
    """One rotation's finished thickness grid search, emitted per rotation by ``optimize_thickness``.

    The thickness fit is the memory-heavy tail phase (each rotation scores the whole
    ``ThicknessGrid`` in one segmented solve), so like :class:`OrientationOptimized` this makes it a
    progress stream rather than a silent block: ``rotation_index`` is the original zero-based PETS
    rotation index, ``score`` the baked thickness's value under ``residual`` -- the
    :class:`~diffBloch.config.schema.LossMetricsConfig` name (``"wr2"``/``"robs"``) that produced
    it, carried alongside so a consumer can label the number correctly
    (:attr:`measurements` keys on it directly) rather than a generic, misleading ``wr2`` field
    under a different residual, and ``thickness`` that winning candidate (Angstrom).
    ``candidate_thicknesses``/``candidate_score`` carry the whole scored grid (same order, one
    entry per :class:`~diffBloch.specs.ThicknessGrid` step) -- deliberately excluded from
    ``measurements`` (which stays flat-scalar for the generic console/wandb/comet backends) and
    preserved by :class:`EventRecord` for post-run visualizers. Emitted in plan order (the fit is
    sequential).
    """

    channel: ClassVar[str] = "optimize_thickness"
    rotation_index: int
    score: float
    residual: str
    thickness: float
    candidate_thicknesses: tuple[float, ...]
    candidate_score: tuple[float, ...]
    dataset: str = ""

    @property
    def step(self) -> int | None:
        return self.rotation_index

    @property
    def measurements(self) -> Mapping[str, float]:
        return {self.residual: self.score, "thickness": self.thickness}


@dataclass(frozen=True)
class PreprocessCompleted:
    """Run-level summary emitted after all per-dataset plans have been pooled.

    This is the app/setup boundary saying the settled :class:`~diffBloch.preprocess.Plan` is ready
    for infer/refine. It intentionally carries only data already materialized by preprocessing:
    no visual artifacts are written here, and post-run renderers build any figures from the JSONL
    event stream.
    """

    channel: ClassVar[str] = "preprocess"
    n_rotations: int
    total_hkl: int
    matched_hkl: int

    @property
    def step(self) -> int | None:
        return None

    @property
    def measurements(self) -> Mapping[str, float]:
        return {
            "n_rotations": float(self.n_rotations),
            "total_hkl": float(self.total_hkl),
            "matched_hkl": float(self.matched_hkl),
        }


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
class PlanSeeded:
    """The Plan a preprocess pipeline is about to run on, summarised before the first step.

    Exists so every :class:`PlanStepCompleted` has a predecessor to be read against: a step's counts
    are only a *survival* count if the incoming counts were reported too, and the seed is produced
    by ``from_experiment`` (or loaded from a checkpoint on resume) rather than by any step, so no
    ``PlanStepCompleted`` covers it. ``measurements`` is
    :func:`diffBloch.preprocess.plan.summarize_plan` of that incoming plan.

    Deliberately a distinct channel from the per-step stream, and ``step`` is ``None``: a consumer
    filtering on channel alone must not mistake the baseline for a stage result.
    """

    channel: ClassVar[str] = "plan_seeded"
    measurements: Mapping[str, float]

    @property
    def step(self) -> int | None:
        return None  # the baseline sits before the recipe's x-axis, not on it


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
    dataset: str = ""
    rotation_index: int | None = None

    @property
    def step(self) -> int | None:
        return self.rotation_index if self.rotation_index is not None else self.index

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
class RotationCouplingSegments:
    """Segment-level coupled solve geometry for one rotation, batched for heatmap visualizers."""

    channel: ClassVar[str] = "coupling segments"
    rotation_index: int
    segment_index: tuple[int, ...]
    first_tilt_index: tuple[int, ...]
    last_tilt_index: tuple[int, ...]
    n_tilts: tuple[int, ...]
    n_segment_beams: tuple[int, ...]
    n_union_beams: int
    n_total_tilts: int
    dataset: str = ""

    def __post_init__(self) -> None:
        lengths = {
            len(self.segment_index),
            len(self.first_tilt_index),
            len(self.last_tilt_index),
            len(self.n_tilts),
            len(self.n_segment_beams),
        }
        if len(lengths) != 1:
            raise ValueError("coupling segment columns must have equal length")

    @property
    def step(self) -> int | None:
        return self.rotation_index

    @property
    def measurements(self) -> Mapping[str, float]:
        return {
            "n_segments": float(len(self.segment_index)),
            "n_union_beams": float(self.n_union_beams),
            "n_total_tilts": float(self.n_total_tilts),
            "max_segment_beams": float(max(self.n_segment_beams, default=0)),
            "max_segment_tilts": float(max(self.n_tilts, default=0)),
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
class ExperimentDeclared:
    """The run's identity and its result-determining knobs, declared once before any compute.

    The counterpart to :class:`ObjectiveManifest` for everything the objective does *not* cover: which
    inputs are being refined and under which simulation/optimizer settings. A sink that writes a
    standalone artifact (the refinement report) needs this to describe the run without being handed
    the :class:`~diffBloch.config.schema.ExperimentConfig` directly -- which is what keeps such a sink
    an ordinary :class:`Logger` rather than a component wired into the app's orchestration.

    Paths, the optimizer name, and the per-dataset seed-thickness declarations ride on the
    dataclass rather than in ``measurements``, which stays flat-scalar for the generic backends --
    the same split :class:`ThicknessOptimized` makes for its candidate grid.
    """

    channel: ClassVar[str] = "experiment"
    name: str
    structure: str
    experimental_data: str
    optimizer: str
    # One effective seed-thickness tuple per inputs.exp_data entry, preserving the dataset labels
    # when ``sample.mean_thickness_by_dataset`` overrides the shared default.
    seed_thicknesses_by_dataset: tuple[tuple[str, tuple[float, ...]], ...]
    # One semiangle per inputs.exp_data entry, in that order -- pooled datasets may differ.
    integration_semiangles: tuple[float, ...]
    rocking_curve_sampling: int
    dsg: float
    rsg: float
    solve_g_max: float
    sg_max: float
    absorption: bool
    steps: int
    learning_rate: float
    experiment_directory: str = ""

    @property
    def step(self) -> int | None:
        return None

    @property
    def measurements(self) -> Mapping[str, float]:
        # Single dataset keeps the flat unindexed key (the common case, and stable across
        # dashboards); a pooled experiment emits one indexed key per dataset in exp_data order.
        semiangles: dict[str, float] = (
            {"integration_semiangle": self.integration_semiangles[0]}
            if len(self.integration_semiangles) == 1
            else {
                f"integration_semiangle_{index}": semiangle
                for index, semiangle in enumerate(self.integration_semiangles)
            }
        )
        return {
            **semiangles,
            "rocking_curve_sampling": float(self.rocking_curve_sampling),
            "dsg": self.dsg,
            "rsg": self.rsg,
            # Scoped per the SOLVE/SCORED/support rule: a bare g_max has no owning object here.
            "solve_g_max": self.solve_g_max,
            "sg_max": self.sg_max,
            "absorption": float(self.absorption),
            "steps": float(self.steps),
            "learning_rate": self.learning_rate,
        }


@dataclass(frozen=True)
class RefinedRotationMetrics:
    """One rotation's wR2/R_obs under the *final refined* model, emitted after the loop.

    Distinct from :class:`RefinementOrientationStep`, which is a per-epoch training diagnostic: this
    is the settled result, scored once on the best model by the *reporting* engine, so it covers
    every rotation including the held-out ones (``is_validation`` marks those). The refinement loop
    cannot emit it -- the loop only ever sees the training engine -- so the app boundary emits it
    once the run has finished.
    """

    channel: ClassVar[str] = "refined rotation"
    rotation_index: int
    wr2: float
    r_obs: float
    n_matched: int
    is_validation: bool
    dataset: str = ""

    @property
    def step(self) -> int | None:
        return self.rotation_index

    @property
    def measurements(self) -> Mapping[str, float]:
        return {
            "wr2": self.wr2,
            "r_obs": self.r_obs,
            "n_matched": float(self.n_matched),
            "is_validation": float(self.is_validation),
        }


@dataclass(frozen=True)
class ThicknessProfile:
    """One dataset's trained apparent-thickness curve, sampled at its rotations' tilt angles.

    Emitted once per composed thickness network after refinement -- one event per dataset, each
    labeled by its ``inputs.exp_data`` ref. The whole curve rides on the dataclass as parallel
    tuples (one entry per rotation, in plan order) rather than as ~100 separate events or ~300
    flat measurement keys -- the shape :class:`ThicknessOptimized` already uses for its candidate
    grid. ``measurements`` carries only the scalar summary.

    ``channel`` embeds the label (the per-instance form the :class:`Event` protocol anticipates)
    so metric sinks that key series on ``channel/name`` keep pooled datasets' curves apart.
    """

    form: str
    min_thickness: float
    max_thickness: float
    rotation_indices: tuple[int, ...]
    alphas: tuple[float, ...]
    thicknesses: tuple[float, ...]
    label: str

    def __post_init__(self) -> None:
        lengths = {len(self.rotation_indices), len(self.alphas), len(self.thicknesses)}
        if len(lengths) != 1:
            raise ValueError("thickness profile columns must have equal length")
        if not self.label:
            raise ValueError("thickness profile label must name its dataset")

    @property
    def channel(self) -> str:
        return f"thickness_profile[{self.label}]"

    @property
    def step(self) -> int | None:
        return None

    @property
    def measurements(self) -> Mapping[str, float]:
        return {
            "n_rotations": float(len(self.rotation_indices)),
            "min_thickness": self.min_thickness,
            "max_thickness": self.max_thickness,
        }


@dataclass(frozen=True)
class RefinementOutputsWritten:
    """The refined artifacts are on disk -- the run's terminal event.

    The event closes the refinement event stream with the files the run produced. JSONL report
    readers and notebook visualizers can use it as the artifact manifest without the refinement
    runner needing a separate summary-writing side effect.

    ``structure`` is the path to the written ``refined_structure.cif``.
    """

    channel: ClassVar[str] = "outputs"
    structure: str
    artifacts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))

    @property
    def step(self) -> int | None:
        return None

    @property
    def measurements(self) -> Mapping[str, float]:
        return {"n_artifacts": float(len(self.artifacts))}


@dataclass(frozen=True)
class ObjectiveTerm:
    """One declared soft-penalty term: the objective name it reports under and its weight."""

    name: str
    weight: float


@dataclass(frozen=True)
class ObjectiveManifest:
    """What the refinement objective is composed of, declared once before the first step.

    The refinement-side counterpart to the preprocess pipeline's ``StepRecord`` provenance: penalties,
    constraints, and components are typed Python composition rather than config, so nothing else in a
    run states which of them are actually in play. This says so up front, before any compute -- the
    "startup summary listing which restraints are active with which weights" that a bare per-epoch
    loss cannot provide.

    Reporting the *empty* case is the point as much as the populated one: the default CLI path
    composes no penalties at all, and a run that says ``penalties: none`` is making a scientific fact
    legible rather than leaving it to be inferred from a missing line. ``measurements`` carries the
    three counts plus each penalty's declared weight; the categorical names ride on the dataclass for
    a backend that pattern-matches it (as :class:`ThicknessOptimized` does for its candidate grid).

    This is a *report*, not an identity: it is deliberately not folded into ``refinement.lock`` or
    :func:`~diffBloch.config.manifest.refinement_config_digest`. Refinement outputs are not
    checkpoint-reused, so hashing a composed-recipe axis would be identity infrastructure built ahead
    of the need for it.
    """

    channel: ClassVar[str] = "objective"
    penalties: tuple[ObjectiveTerm, ...] = ()
    constraints: tuple[str, ...] = ()
    components: tuple[str, ...] = ()

    @property
    def step(self) -> int | None:
        return None  # a run-level declaration has no position on the per-iteration axis

    @property
    def measurements(self) -> Mapping[str, float]:
        values: dict[str, float] = {
            "n_penalties": float(len(self.penalties)),
            "n_constraints": float(len(self.constraints)),
            "n_components": float(len(self.components)),
        }
        for term in self.penalties:
            values[f"{term.name}/weight"] = term.weight
        return values


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

    ``wr2``/``r_obs`` are always-computed reporting diagnostics (mean weighted-R2 / R_obs across
    orientations), free regardless of ``ExperimentConfig.loss_metrics`` (which decides what
    ``loss`` actually minimises, not what gets reported here) -- so both are always shown. Contrast
    the preprocessing search's events (:class:`OrientationOptimized` / :class:`ThicknessOptimized`),
    which report only the configured residual, since computing the other would cost an extra solve.

    ``components`` carries each named objective term's ``raw`` scientific diagnostic, its
    ``weight``, and the ``contribution`` that weight produces, and :attr:`measurements` flattens
    every one of them to a ``"{term}/{field}"`` key so the generic backends (console, W&B, Comet)
    report a restraint's state without knowing any term by name. A term that was never
    composed into the objective has **no entry**, so it cannot surface as a satisfied ``0.0``; that
    absence is the reportable fact, and it is why the flattening is unconditional rather than keyed
    on a fixed term list.

    ``wr2``/``r_obs`` are means over the rotations that produced a finite score, so each carries its
    own denominator: ``n_rotations`` is how many the objective covered (the *training* set when a
    validation split is on) and ``n_wr2_evaluated``/``n_r_obs_evaluated`` how many actually entered
    each mean. They are separate counts because the two metrics are NaN-filtered independently -- a
    rotation can contribute to one and not the other -- and a mean whose denominator is implicit can
    improve simply by evaluating fewer rotations. Compare
    :class:`InferenceCompleted`, which has always reported ``n_evaluated`` beside its mean.
    """

    channel: ClassVar[str] = "refinement"
    iteration: int
    loss: float
    wr2: float | None = None
    r_obs: float | None = None
    diff_loss: float | None = None
    objective_total: float | None = None
    components: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    n_rotations: int | None = None
    n_wr2_evaluated: int | None = None
    n_r_obs_evaluated: int | None = None
    val_wr2: float | None = None
    val_r_obs: float | None = None
    val_n_rotations: int | None = None
    val_n_wr2_evaluated: int | None = None
    val_n_r_obs_evaluated: int | None = None

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
        if self.n_rotations is not None:
            values["n_rotations"] = float(self.n_rotations)
        if self.n_wr2_evaluated is not None:
            values["n_wr2_evaluated"] = float(self.n_wr2_evaluated)
        if self.n_r_obs_evaluated is not None:
            values["n_r_obs_evaluated"] = float(self.n_r_obs_evaluated)
        if self.val_wr2 is not None:
            values["val_wr2"] = self.val_wr2
        if self.val_r_obs is not None:
            values["val_r_obs"] = self.val_r_obs
        if self.val_n_rotations is not None:
            values["val_n_rotations"] = float(self.val_n_rotations)
        if self.val_n_wr2_evaluated is not None:
            values["val_n_wr2_evaluated"] = float(self.val_n_wr2_evaluated)
        if self.val_n_r_obs_evaluated is not None:
            values["val_n_r_obs_evaluated"] = float(self.val_n_r_obs_evaluated)
        for term, entries in self.components.items():
            for name, value in entries.items():
                values[f"{term}/{name}"] = value
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
    dataset: str = ""

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

    Shares the ``"refinement"`` channel with :class:`RefinementStep`, separated from the stream by
    ``step`` (the iteration index vs ``None``). The two are *not* the same quantity at different
    granularities: :class:`RefinementStep` always reports the training objective, whereas
    ``best_loss`` is whichever objective actually selected the epoch. ``selection`` names that
    objective -- ``"training"`` by default, or ``"validation"`` when ``run_refinement_model`` was
    given a held-out selection engine.

    Because those two populations are not comparable, ``measurements`` emits ``best_loss`` under a
    *different key* per mode (``best_training_loss`` / ``best_validation_loss``) rather than one
    shared key plus a flag. A generic backend cannot then plot a train-selected and a val-selected
    run as one series: the key is absent instead of silently wrong.
    """

    channel: ClassVar[str] = "refinement"
    n_steps: int
    best_step: int
    best_loss: float
    selection: Literal["training", "validation"] = "training"
    # The reflection counts the best model was scored over, on the *training* engine -- the same set
    # the per-step means range over. Carried here rather than as a separate event because they are
    # facts about this completed run, and the summary is the one place they belong.
    reflection_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reflection_counts", MappingProxyType(dict(self.reflection_counts))
        )

    @property
    def step(self) -> int | None:
        return None  # a run-level aggregate has no position on the per-iteration axis

    @property
    def measurements(self) -> Mapping[str, float]:
        best_key = (
            "best_validation_loss" if self.selection == "validation" else "best_training_loss"
        )
        values = {
            "n_steps": float(self.n_steps),
            "best_step": float(self.best_step),
            best_key: self.best_loss,
        }
        values.update({name: float(count) for name, count in self.reflection_counts.items()})
        return values


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
