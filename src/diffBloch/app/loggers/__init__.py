"""Logger backends -- the ``app/`` boundary where sinks and vendor SDKs live, never the core.

Each backend consumes the uniform :class:`~diffBloch.observability.Event` surface. The console sink
uses the scalar ``measurements`` view, while :class:`ReportLogger` writes the full versioned
event-record payload for post-run tools. Each third-party backend lives in its own confined submodule
that imports its SDK lazily, so importing this package never requires an optional dependency:

- :class:`~diffBloch.app.loggers.wandb.WandbLogger` (``diffBloch.app.loggers.wandb``)
- :class:`~diffBloch.app.loggers.comet.CometLogger` (``diffBloch.app.loggers.comet``)

Writing your own backend is a single method: implement ``report(event)`` for the events you care
about.
"""

from __future__ import annotations

import logging
import math
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from diffBloch.io import ParseDiagnostic
from diffBloch.observability import (
    ConvergencePassStarted,
    ConvergenceSweepStarted,
    ConvergenceTrial,
    DeviceSelected,
    Event,
    ExperimentDeclared,
    ObjectiveManifest,
    OrientationOptimizationStarted,
    OrientationOptimized,
    PlanSeeded,
    PlanStepCompleted,
    PreprocessCompleted,
    RefinedRotationMetrics,
    RefinementOrientationStep,
    RefinementStarted,
    RefinementStep,
    RunStageStarted,
    RunStageStopped,
    ThicknessOptimizationStarted,
    ThicknessOptimized,
    event_record_from_event,
)

__all__ = [
    "ConsoleLogger",
    "ReportLogger",
    "format_measurements",
    "namespaced_measurements",
    "residual_label",
]

_log = logging.getLogger("diffBloch.loggers")


def format_measurements(event: Event) -> str:
    """Render an event's measurements as space-joined ``name=value`` pairs (shared by backends)."""
    return " ".join(f"{name}={value:g}" for name, value in event.measurements.items())


def namespaced_measurements(event: Event) -> dict[str, float]:
    """Map an event to ``{channel}/{metric}: value`` (the series convention shared by the
    wandb/comet backends)."""
    return {f"{event.channel}/{name}": value for name, value in event.measurements.items()}


_BAR_WIDTH = 30

# Display label per LossMetricsConfig.residual name, for progress-bar text that doesn't go
# through .measurements (which already keys dynamically on the raw residual string).
_RESIDUAL_LABELS = {"wr2": "wR2", "robs": "R_obs"}


def residual_label(residual: str) -> str:
    return _RESIDUAL_LABELS.get(residual, residual)


def _mean_over(value: float | None, evaluated: int | None, total: int | None) -> str:
    """Format an epoch mean with the denominator it was actually taken over.

    ``0.061 [97/99]`` rather than a bare ``0.061``: the mean covers only the rotations that produced
    a finite score, so a run that quietly evaluates fewer of them would otherwise look like a run
    that got better.

    A non-finite mean renders ``n/a [0/99]``, matching the report visualizer's per-rotation means.
    The objective averages only the finite per-rotation scores, so its mean is non-finite in exactly
    one case -- nothing was evaluated -- which the ``[0/99]`` already states; the bare ``nan`` adds
    nothing and reads as a numerical failure rather than an empty denominator. ``None`` is the
    distinct case of a metric not reported at all, and carries no counts to attach.
    """
    if value is None:
        return "n/a"
    rendered = f"{value:.6f}" if math.isfinite(value) else "n/a"
    if evaluated is None or total is None:
        return rendered
    return f"{rendered} [{evaluated}/{total}]"


def _penalty_components(event: RefinementStep) -> tuple[tuple[str, Mapping[str, float]], ...]:
    """The epoch's composed soft-penalty terms, in objective order.

    ``diffraction`` is dropped because the same number is already reported as ``diff_loss``; what
    remains is exactly the restraints the objective actually composed. An absent restraint has no
    entry at all, so an empty result means *no penalty was applied*, never *a penalty evaluated to
    zero* -- the distinction the console would otherwise be unable to draw.
    """
    return tuple(
        (term, values) for term, values in event.components.items() if term != "diffraction"
    )


def _format_eta(seconds: float) -> str:
    """``H:MM:SS``/``M:SS`` for a non-negative duration (tqdm-style, no fractional seconds)."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _format_device_selection(event: DeviceSelected) -> str:
    if event.selected.startswith("cuda"):
        return "CUDA detected, using CUDA"
    if event.selected == "cpu" and event.requested.startswith("cuda"):
        return "No CUDA detected, using CPU, diffBloch is optimized for CUDA"
    if event.selected == "cpu":
        return "Using CPU, diffBloch is optimized for CUDA"
    return f"Using {event.selected}"


def _render_progress_bar(current: int, total: int, elapsed: float, suffix: str) -> None:
    """Write an in-place (``\\r``-updated) progress bar + ETA to stdout; newline once it completes.

    ETA is extrapolated linearly from the observed rate so far (``elapsed / current``) -- a rough
    estimate, not a scheduler guarantee, but the standard tqdm-style approach and good enough for a
    console progress indicator.
    """
    fraction = min(1.0, current / total) if total > 0 else 0.0
    filled = int(_BAR_WIDTH * fraction)
    bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
    eta = ""
    if current > 0 and current < total:
        remaining = (elapsed / current) * (total - current)
        eta = f" eta {_format_eta(remaining)}"
    # `\r` returns the cursor but does not erase, so a shorter line leaves the tail of a longer one
    # behind ("wR2 0.057427" over "wR2 0.0398710.039671"). `\033[K` clears to end of line; it is the
    # one escape used here, and only on the tty path this function is already gated behind.
    sys.stdout.write(f"\r[{bar}] {current}/{total} ({100.0 * fraction:.0f}%){eta} {suffix}\033[K")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


@dataclass
class ConsoleLogger:
    """Log each event to stdlib ``logging`` at ``level`` (console/file handlers attached by app).

    The bridge from the domain-observation channel to the diagnostics channel: handy for a live
    scroll of per-rotation ``R_obs`` while chasing a residual. Attach a handler (or call
    :func:`logging.basicConfig`) at the app boundary to see it.

    On a real terminal (``sys.stdout.isatty()``), refinement epochs, orientation-search rotations,
    and thickness-search rotations render as an in-place progress bar (with a linearly-extrapolated
    ETA) instead of one scrolling line per event -- their respective ``*Started`` event supplies
    the total up front. Off a
    terminal (piped to a file, CI logs) this falls back to the plain per-event line, since
    ``\\r``-based in-place updates are meaningless there and would just corrupt a log file with
    control characters.
    """

    level: int = logging.INFO
    # The settled per-rotation stream: one line per rotation, once, after the run. On by default
    # -- unlike the per-epoch verbose stream (n_orientations x n_steps lines) this fires once, and
    # the final score of every rotation is a result worth reading, not a diagnostic.
    per_rotation: bool = True
    # The active convergence pass's acceptance threshold, remembered from its header so each trial
    # line can say whether it cleared it. None until a pass announces itself.
    _r_factor_threshold: float | None = field(default=None, init=False, repr=False)
    _refinement_total: int = field(default=0, init=False, repr=False)
    _refinement_started_at: float = field(default=0.0, init=False, repr=False)
    _orientation_total: int = field(default=0, init=False, repr=False)
    _orientation_seen: int = field(default=0, init=False, repr=False)
    _orientation_started_at: float = field(default=0.0, init=False, repr=False)
    _thickness_total: int = field(default=0, init=False, repr=False)
    _thickness_seen: int = field(default=0, init=False, repr=False)
    _thickness_started_at: float = field(default=0.0, init=False, repr=False)

    def report(self, event: Event) -> None:
        if isinstance(event, DeviceSelected):
            _log.log(self.level, _format_device_selection(event))
            return
        if isinstance(event, ParseDiagnostic):
            _log.log(
                self.level,
                "Input │ %s │ %s │ %s",
                event.input_kind.replace("_", " "),
                "" if event.source_path is None else str(event.source_path),
                event.message,
            )
            return
        if isinstance(event, ExperimentDeclared):
            _log.log(
                self.level,
                "Experiment │ %s │ %s + %s",
                event.name,
                event.structure,
                event.experimental_data,
            )
            _log.log(
                self.level,
                "Experiment │ %s lr=%g │ %d epoch(s) │ g_max(solve)=%g sg_max=%g │ absorption %s",
                event.optimizer,
                event.learning_rate,
                event.steps,
                event.solve_g_max,
                event.sg_max,
                "on" if event.absorption else "off",
            )
            return
        if isinstance(event, RunStageStarted):
            _log.log(self.level, "Stage │ %s started", event.stage)
            return
        if isinstance(event, RunStageStopped):
            status = event.status
            suffix = (
                f" │ {event.error_type}: {event.error_message}"
                if event.status == "failed" and event.error_type
                else ""
            )
            _log.log(
                self.level,
                "Stage │ %s stopped │ %s │ %.3fs%s",
                event.stage,
                status,
                event.elapsed_seconds,
                suffix,
            )
            return
        if isinstance(event, RefinedRotationMetrics):
            # Gated at the sink, not the emitter: the event must always be emitted -- the
            # ReportLogger and vendor sinks want the settled scores, so a console that wants less
            # says so here.
            if self.per_rotation:
                _log.log(
                    self.level,
                    "  rotation %3d │ wR2 %.6f │ R_obs %.6f │ %d matched%s",
                    event.rotation_index,
                    event.wr2,
                    event.r_obs,
                    event.n_matched,
                    " │ validation" if event.is_validation else "",
                )
            return
        if isinstance(event, ObjectiveManifest):
            # "none" is printed rather than the line being dropped: an objective composing no
            # restraints is a scientific fact worth stating, not an absence to be inferred.
            _log.log(
                self.level,
                "Objective │ penalties  : %s",
                ", ".join(f"{term.name} (weight {term.weight:g})" for term in event.penalties)
                or "none",
            )
            _log.log(
                self.level, "Objective │ constraints: %s", ", ".join(event.constraints) or "none"
            )
            _log.log(
                self.level, "Objective │ components : %s", ", ".join(event.components) or "none"
            )
            return
        if isinstance(event, RefinementStarted):
            self._refinement_total = event.total_steps
            self._refinement_started_at = time.perf_counter()
            return
        if isinstance(event, OrientationOptimizationStarted):
            label = (
                f"Orientation optimization │ {event.dataset} │"
                if event.dataset
                else "Orientation optimization │"
            )
            _log.log(self.level, "%s %d rotation(s)", label, event.total_rotations)
            self._orientation_total = event.total_rotations
            self._orientation_seen = 0
            self._orientation_started_at = time.perf_counter()
            return
        if (
            isinstance(event, OrientationOptimized)
            and self._orientation_total > 0
            and sys.stdout.isatty()
        ):
            self._orientation_seen += 1
            _render_progress_bar(
                self._orientation_seen,
                self._orientation_total,
                time.perf_counter() - self._orientation_started_at,
                f"rotation {event.rotation_index} │ "
                f"{residual_label(event.residual)} {event.score:.6f}",
            )
            return
        if isinstance(event, ThicknessOptimizationStarted):
            label = (
                f"Thickness optimization │ {event.dataset} │"
                if event.dataset
                else "Thickness optimization │"
            )
            _log.log(self.level, "%s %d rotation(s)", label, event.total_rotations)
            self._thickness_total = event.total_rotations
            self._thickness_seen = 0
            self._thickness_started_at = time.perf_counter()
            return
        if (
            isinstance(event, ThicknessOptimized)
            and self._thickness_total > 0
            and sys.stdout.isatty()
        ):
            self._thickness_seen += 1
            _render_progress_bar(
                self._thickness_seen,
                self._thickness_total,
                time.perf_counter() - self._thickness_started_at,
                f"rotation {event.rotation_index} │ "
                f"{residual_label(event.residual)} {event.score:.6f}",
            )
            return
        if isinstance(event, RefinementStep) and self._refinement_total > 0 and sys.stdout.isatty():
            wr2 = _mean_over(event.wr2, event.n_wr2_evaluated, event.n_rotations)
            r_obs = _mean_over(event.r_obs, event.n_r_obs_evaluated, event.n_rotations)
            suffix = f"epoch │ wR2 {wr2} │ R_obs {r_obs}"
            if event.val_wr2 is not None or event.val_r_obs is not None:
                val_wr2 = _mean_over(
                    event.val_wr2, event.val_n_wr2_evaluated, event.val_n_rotations
                )
                val_r_obs = _mean_over(
                    event.val_r_obs, event.val_n_r_obs_evaluated, event.val_n_rotations
                )
                suffix += f" │ val wR2 {val_wr2} │ val R_obs {val_r_obs}"
            # The bar owns its line (``\r``, no newline), so penalties ride in the suffix rather
            # than as extra log lines that would overwrite it.
            for term, values in _penalty_components(event):
                suffix += f" │ {term} {values['contribution']:.4g}"
            _render_progress_bar(
                event.iteration + 1,
                self._refinement_total,
                time.perf_counter() - self._refinement_started_at,
                suffix,
            )
            return
        if isinstance(event, ConvergencePassStarted):
            _log.log(
                self.level,
                "=== Hyperparameter Optimization Pass %d ===",
                event.pass_index,
            )
            _log.log(
                self.level,
                "start: gmax=%g sgmax=%g tilt_steps=%d r_threshold=%.6f orientations=%d",
                event.g_max,
                event.sg_max,
                event.tilt_steps,
                event.r_factor_threshold,
                event.n_orientations,
            )
            self._r_factor_threshold = event.r_factor_threshold
            return
        if isinstance(event, ConvergenceSweepStarted):
            label = "gmax" if event.control == "g_max" else event.control
            _log.log(self.level, "sweep: %s", label)
            return
        if isinstance(event, ConvergenceTrial):
            label = "gmax" if event.control == "g_max" else event.control
            # Mark the trial that fell under the pass threshold -- the comparison converge_scalar
            # actually settles on. Printing the threshold once in the pass header and then leaving
            # the reader to check each R against it by eye is the arithmetic this saves.
            settled = (
                self._r_factor_threshold is not None and event.r_factor < self._r_factor_threshold
            )
            _log.log(
                self.level,
                "  %s %g -> %g | R=%.6f | fixed_hkls=%d%s",
                label,
                event.previous,
                event.candidate,
                event.r_factor,
                event.n_compared_hkl,
                " | settled" if settled else "",
            )
            return
        if isinstance(event, OrientationOptimized):
            label = f"orientation optimization[rotation_index={event.rotation_index}]"
        elif isinstance(event, ThicknessOptimized):
            label = f"thickness optimization[rotation_index={event.rotation_index}]"
        elif isinstance(event, RefinementStep):
            wr2 = _mean_over(event.wr2, event.n_wr2_evaluated, event.n_rotations)
            r_obs = _mean_over(event.r_obs, event.n_r_obs_evaluated, event.n_rotations)
            diff_loss = "n/a" if event.diff_loss is None else f"{event.diff_loss:.6f}"
            val_suffix = ""
            if event.val_wr2 is not None or event.val_r_obs is not None:
                val_wr2 = _mean_over(
                    event.val_wr2, event.val_n_wr2_evaluated, event.val_n_rotations
                )
                val_r_obs = _mean_over(
                    event.val_r_obs, event.val_n_r_obs_evaluated, event.val_n_rotations
                )
                val_suffix = f" │ val wR2 {val_wr2} │ val R_obs {val_r_obs}"
            _log.log(
                self.level,
                "Refinement epoch %3d │ wR2 %s │ R_obs %s%s │ diffraction loss %s",
                event.iteration + 1,
                wr2,
                r_obs,
                val_suffix,
                diff_loss,
            )
            for term, values in _penalty_components(event):
                _log.log(
                    self.level,
                    "  penalty %-20s │ raw %.6g │ weight %g │ contribution %.6g",
                    term,
                    values["raw"],
                    values["weight"],
                    values["contribution"],
                )
            return
        elif isinstance(event, RefinementOrientationStep):
            wr2 = "n/a" if event.wr2 is None else f"{event.wr2:.6f}"
            r_obs = "n/a" if event.r_obs is None else f"{event.r_obs:.6f}"
            diff_loss = "n/a" if event.diff_loss is None else f"{event.diff_loss:.6f}"
            _log.log(
                self.level,
                "  epoch %3d rotation %3d │ wR2 %s │ R_obs %s │ diffraction loss %s",
                event.iteration + 1,
                event.rotation_index,
                wr2,
                r_obs,
                diff_loss,
            )
            return
        elif isinstance(event, PlanSeeded):
            _log.log(
                self.level,
                "Preprocess seed  │ %-27s │ %s",
                "(incoming plan)",
                format_measurements(event),
            )
            return
        elif isinstance(event, PlanStepCompleted):
            stage = event.channel.replace("_", " ").title()
            _log.log(
                self.level,
                "Preprocess stage %2d │ %-27s │ %s",
                event.index + 1,
                stage,
                format_measurements(event),
            )
            return
        elif isinstance(event, PreprocessCompleted):
            _log.log(
                self.level,
                "PREPROCESS COMPLETE │ rotations=%d │ matched_hkl=%d │ total_hkl=%d",
                event.n_rotations,
                event.matched_hkl,
                event.total_hkl,
            )
            return
        else:
            label = event.channel if event.step is None else f"{event.channel}[{event.step}]"
        _log.log(self.level, "%s %s", label, format_measurements(event))


@dataclass
class ReportLogger:
    """Write a versioned JSONL event stream for post-run visualizers.

    This supersedes the old scalar file sink: each line preserves the whole event payload, including
    structured fields that are intentionally absent from the flat scalar ``measurements`` surface. It
    is still an ordinary app-boundary logger: producers emit typed event values, and this sink alone
    performs filesystem I/O.
    """

    path: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    completed_only: bool = False
    _active_path: Path = field(default=Path(), init=False, repr=False)
    _temporary_dir: Path | None = field(default=None, init=False, repr=False)
    _sequence: int = field(default=0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @staticmethod
    def timestamped_path(directory: Path, *, prefix: str = "report") -> Path:
        """Return ``directory/report-YYYYMMDDTHHMMSSZ.jsonl`` for a new report artifact."""
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return Path(directory) / f"{prefix}-{stamp}.jsonl"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.completed_only:
            temporary_dir = Path(tempfile.mkdtemp(prefix="diffbloch-report-"))
            self._temporary_dir = temporary_dir
            self._active_path = temporary_dir / self.path.name
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._active_path = self.path
        self._active_path.write_text("")

    def report(self, event: Event) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed ReportLogger")
        record = event_record_from_event(event, run_id=self.run_id, sequence=self._sequence)
        self._sequence += 1
        with self._active_path.open("a") as handle:
            handle.write(record.model_dump_json() + "\n")

    def finalize(self) -> None:
        """Promote the streamed report to its declared output path."""
        if self._closed:
            return
        if self.completed_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(self._active_path), self.path)
            if self._temporary_dir is not None:
                shutil.rmtree(self._temporary_dir, ignore_errors=True)
        self._closed = True

    def discard(self) -> None:
        """Drop a streamed report that did not complete successfully."""
        if self._closed:
            return
        if self.completed_only and self._temporary_dir is not None:
            shutil.rmtree(self._temporary_dir, ignore_errors=True)
        self._closed = True
