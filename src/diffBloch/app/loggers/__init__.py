"""Logger backends -- the ``app/`` boundary where sinks and vendor SDKs live, never the core.

Each backend consumes the uniform :class:`~diffBloch.observability.Event` surface (``channel`` +
``measurements``), so none knows the concrete event types and new events need no backend change.
:class:`ConsoleLogger` (here, no vendor dependency) routes events to stdlib ``logging`` and
:class:`CSVLogger` (here too) appends them to a file; each third-party backend lives in its own
confined submodule that imports its SDK lazily, so importing this package never requires an optional
dependency:

- :class:`~diffBloch.app.loggers.wandb.WandbLogger` (``diffBloch.app.loggers.wandb``)
- :class:`~diffBloch.app.loggers.comet.CometLogger` (``diffBloch.app.loggers.comet``)
- :class:`~diffBloch.app.loggers.plotting.ThicknessPlotLogger` (``diffBloch.app.loggers.plotting``)

Writing your own backend is a single method: implement ``report(event)`` for the events you care
about.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from diffBloch.observability import (
    NULL_LOGGER,
    ConvergencePassStarted,
    ConvergenceSweepStarted,
    ConvergenceTrial,
    Event,
    Logger,
    OrientationOptimized,
    PlanStepCompleted,
    RefinementOrientationStep,
    RefinementStep,
    ThicknessOptimized,
)

__all__ = [
    "CSVLogger",
    "ConsoleLogger",
    "EarlyAbortLogger",
    "FitAbortedError",
    "format_measurements",
    "namespaced_measurements",
]

_log = logging.getLogger("diffBloch.loggers")

_CSV_HEADER = ("channel", "step", "metric", "value")


def format_measurements(event: Event) -> str:
    """Render an event's measurements as space-joined ``name=value`` pairs (shared by backends)."""
    return " ".join(f"{name}={value:g}" for name, value in event.measurements.items())


def namespaced_measurements(event: Event) -> dict[str, float]:
    """Map an event to ``{channel}/{metric}: value`` (the series convention shared by the
    wandb/comet backends)."""
    return {f"{event.channel}/{name}": value for name, value in event.measurements.items()}


@dataclass
class ConsoleLogger:
    """Log each event to stdlib ``logging`` at ``level`` (console/file handlers attached by app).

    The bridge from the domain-observation channel to the diagnostics channel: handy for a live
    scroll of per-rotation ``R_obs`` while chasing a residual. Attach a handler (or call
    :func:`logging.basicConfig`) at the app boundary to see it.
    """

    level: int = logging.INFO

    def report(self, event: Event) -> None:
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
            return
        if isinstance(event, ConvergenceSweepStarted):
            label = "gmax" if event.control == "g_max" else event.control
            _log.log(self.level, "sweep: %s", label)
            return
        if isinstance(event, ConvergenceTrial):
            label = "gmax" if event.control == "g_max" else event.control
            _log.log(
                self.level,
                "  %s -> %g | R=%.6f | fixed_hkls=%d",
                label,
                event.candidate,
                event.r_factor,
                event.n_compared_hkl,
            )
            return
        if isinstance(event, OrientationOptimized):
            label = f"orientation optimization[rotation_index={event.rotation_index}]"
        elif isinstance(event, ThicknessOptimized):
            label = f"thickness optimization[rotation_index={event.rotation_index}]"
        elif isinstance(event, RefinementStep):
            wr2 = "n/a" if event.wr2 is None else f"{event.wr2:.6f}"
            r_obs = "n/a" if event.r_obs is None else f"{event.r_obs:.6f}"
            diff_loss = "n/a" if event.diff_loss is None else f"{event.diff_loss:.6f}"
            _log.log(
                self.level,
                "Refinement epoch %3d │ wR2 %s │ R_obs %s │ diffraction loss %s",
                event.iteration + 1,
                wr2,
                r_obs,
                diff_loss,
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
        else:
            label = event.channel if event.step is None else f"{event.channel}[{event.step}]"
        _log.log(self.level, "%s %s", label, format_measurements(event))


@dataclass
class CSVLogger:
    """Append each event's measurements to a CSV file in long format (Lightning-style sink).

    One row per measurement -- ``channel, step, metric, value`` -- so a heterogeneous event stream
    (rotation, inference, refinement) shares a single flat table with no sparse columns, ready to
    filter by ``channel`` or pivot by ``step``. The header is written once at construction (a fresh
    file per run); each :meth:`report` appends and flushes, so the file is crash-safe and tailable.

    This is an *observation log*, not persistence: run state is checkpointed by serialising the
    whole ``Plan``, never reconstructed from these rows.
    """

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        with self.path.open("w", newline="") as handle:
            csv.writer(handle).writerow(_CSV_HEADER)

    def report(self, event: Event) -> None:
        with self.path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            for metric, value in event.measurements.items():
                writer.writerow([event.channel, event.step, metric, value])


class FitAbortedError(RuntimeError):
    """Raised by :class:`EarlyAbortLogger` to unwind a fit judged unpromising and stop it early.

    Carries the diagnostic that triggered the abort (rotations seen, best ``wr2``, the ceiling), so
    the caller running the fit sees *why* it stopped, not just that it did.
    """


@dataclass
class EarlyAbortLogger:
    """Watch the per-rotation fit stream and abort a run that is not tracking the data.

    A fit-quality guard for a long, oracle-less **from-scratch** fit -- one with no committed
    checkpoint and no reference ``R_obs`` to pin against, so a mis-set-up
    run (wrong energy / ``g_max``, bad data lineage) would otherwise burn its whole budget producing
    a bad answer. ``optimize_orientation`` emits one :class:`~diffBloch.observability.OrientationOptimized`
    per rotation as it finishes, carrying the scaling-optimised ``wr2`` at the fitted orientation. A
    healthy run reaches a low ``wr2`` on essentially every rotation; a fundamentally broken one stays
    high on all of them. This guard gives the run
    ``patience`` rotations to show *at least one* orientation reaching ``wr2 <= wr2_ceiling``; if
    none does, it raises :class:`FitAbortedError`, unwinding the fit before the remaining rotations
    run. Pick ``wr2_ceiling`` generously (well above a healthy fit, well below a garbage one) so a
    real run clears it within the first rotation or two and never false-aborts.

    Only the fit stream drives the decision; **every** event is forwarded verbatim to ``inner``
    (default :data:`~diffBloch.observability.NULL_LOGGER`), so this composes with a
    :class:`ConsoleLogger` / :class:`CSVLogger` for the live scroll --
    ``EarlyAbortLogger(inner=ConsoleLogger())``. Raising from :meth:`report` is the abort mechanism:
    the fit loop's only per-rotation hook is the logger, and both the sequential and ``workers > 1``
    paths call ``report`` from the driving thread, so the raise unwinds the run cleanly. Compute
    saved: **all** remaining rotations under ``workers = 1`` (sequential -- nothing further starts);
    under ``workers > 1`` the queued rotations are cancelled but the ``<= workers`` already running
    cannot be interrupted and finish first (``optimize_orientation`` cancels the rest on abort).
    """

    wr2_ceiling: float = 0.6
    patience: int = 5
    inner: Logger = NULL_LOGGER
    _seen: int = field(default=0, init=False, repr=False)
    _best_wr2: float = field(default=math.inf, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be >= 1")

    def report(self, event: Event) -> None:
        self.inner.report(event)  # forward first: the guard never swallows an observation
        if event.channel != OrientationOptimized.channel:
            return
        self._seen += 1
        self._best_wr2 = min(self._best_wr2, event.measurements["wr2"])
        if self._seen >= self.patience and self._best_wr2 > self.wr2_ceiling:
            raise FitAbortedError(
                f"fit aborted early: after {self._seen} rotation(s) the best wr2 is "
                f"{self._best_wr2:.4g}, above the {self.wr2_ceiling:g} ceiling -- the run is not "
                "tracking the data (check energy / g_max / data lineage). Raise wr2_ceiling or "
                "patience to allow a slower/looser fit."
            )
