"""Live terminal display backend (the ``diffBloch[tui]`` extra).

Confines the ``rich`` dependency to this module, imported lazily on first use, mirroring
``app.loggers.wandb``/``comet``/``plotting`` -- importing ``diffBloch.app.loggers`` never requires
rich to be installed.

Consumes the same :class:`~diffBloch.observability.Event` stream every other backend does and
performs no computation of its own. It replaces :class:`~diffBloch.app.loggers.ConsoleLogger`
rather than composing beside it: that one bridges events onto stdlib ``logging``, and a rich
``Live`` display owns the terminal for the duration of a run, so two writers would interleave and
corrupt each other. Compose it with the file/vendor sinks freely -- those do not touch the terminal.

Off a terminal (piped to a file, CI logs) it degrades to nothing: ``Live`` is not started and events
are dropped, because an in-place display has no meaning there. Use ``ConsoleLogger`` for that case.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from diffBloch.observability import (
    ConvergencePassStarted,
    ConvergenceSweepStarted,
    ConvergenceTrial,
    Event,
    ExperimentDeclared,
    ObjectiveManifest,
    OrientationOptimizationStarted,
    OrientationOptimized,
    PlanSeeded,
    PlanStepCompleted,
    RefinedRotationMetrics,
    RefinementCompleted,
    RefinementOutputsWritten,
    RefinementStarted,
    RefinementStep,
    ThicknessOptimizationStarted,
    ThicknessOptimized,
)

__all__ = ["TuiLogger"]

# Rows reserved for the panels, headers and progress bar that frame the tables, so the live window
# can be sized against the real terminal height instead of a magic number.
_CHROME_ROWS = 26

# Floor for the live window when the terminal is very short: fewer rows than this is not a table.
_MIN_WINDOW = 4


@dataclass
class _Phase:
    """One tracked phase's progress: a label, how far along, and its most recent detail line."""

    label: str = ""
    current: int = 0
    total: int = 0
    detail: str = ""
    started_at: float = 0.0


@dataclass
class TuiLogger:
    """Render the run as a live-updating terminal dashboard.

    Shows the declared experiment and objective as header panels, the active phase as a progress
    bar with an ETA, the epoch stream as a rolling table, and -- once the run settles -- the
    per-rotation table. Every value comes from the event stream; nothing is recomputed here.

    ``refresh_per_second`` caps redraw rate (rich coalesces updates between frames), so a fast
    per-rotation stream costs a bounded amount of terminal I/O rather than one repaint per event.
    """

    refresh_per_second: float = 8.0
    _live: Any = field(default=None, init=False, repr=False)
    _console: Any = field(default=None, init=False, repr=False)
    _experiment: ExperimentDeclared | None = field(default=None, init=False, repr=False)
    _manifest: ObjectiveManifest | None = field(default=None, init=False, repr=False)
    _phase: _Phase = field(default_factory=_Phase, init=False, repr=False)
    _stages: list[tuple[str, Mapping[str, float]]] = field(
        default_factory=list, init=False, repr=False
    )
    _epochs: list[RefinementStep] = field(default_factory=list, init=False, repr=False)
    _rotations: list[RefinedRotationMetrics] = field(default_factory=list, init=False, repr=False)
    _completed: RefinementCompleted | None = field(default=None, init=False, repr=False)
    _pass: ConvergencePassStarted | None = field(default=None, init=False, repr=False)
    _sweep: str = field(default="", init=False, repr=False)
    _trials: list[ConvergenceTrial] = field(default_factory=list, init=False, repr=False)
    _other: list[tuple[str, int | None, str]] = field(default_factory=list, init=False, repr=False)

    def report(self, event: Event) -> None:
        if not self._absorb(event):
            return
        if self._live is None:
            self._start()
        if self._live is not None:
            self._live.update(self._render())
        if isinstance(event, RefinementOutputsWritten):
            self.close()

    def close(self) -> None:
        """Release the terminal, then print the complete tables. Idempotent.

        A live in-place display cannot exceed the terminal height, so while the run is going the
        tables show a trailing window. That constraint disappears the moment ``Live`` stops, and the
        settled result is exactly what a reader wants in full -- so the final render is unwindowed
        and lands in the scrollback like ordinary output.
        """
        if self._live is None:
            return
        self._live.stop()
        self._live = None
        if self._console is not None:
            self._console.print(self._render(window=None))

    # -- state ---------------------------------------------------------------------------------

    def _absorb(self, event: Event) -> bool:
        """Fold one event into the display state; ``False`` if it changes nothing worth redrawing."""
        match event:
            case ExperimentDeclared():
                self._experiment = event
            case ObjectiveManifest():
                self._manifest = event
            case ConvergencePassStarted():
                self._pass = event
            case ConvergenceSweepStarted():
                self._sweep = event.control
            case ConvergenceTrial():
                self._trials.append(event)
            case PlanSeeded():
                self._stages.append(("seed (incoming plan)", event.measurements))
            case PlanStepCompleted():
                self._stages.append((event.channel, event.measurements))
            case OrientationOptimizationStarted():
                self._phase = _Phase("orientation fit", 0, event.total_rotations, "", time.time())
            case ThicknessOptimizationStarted():
                self._phase = _Phase("thickness fit", 0, event.total_rotations, "", time.time())
            case RefinementStarted():
                self._phase = _Phase("refinement", 0, event.total_steps, "", time.time())
            case OrientationOptimized() | ThicknessOptimized():
                self._phase.current += 1
                self._phase.detail = f"rotation {event.rotation_index} · {event.score:.6f}"
            case RefinementStep():
                self._epochs.append(event)
                self._phase.current = event.iteration + 1
            case RefinedRotationMetrics():
                self._rotations.append(event)
            case RefinementCompleted():
                self._completed = event
            case RefinementOutputsWritten():
                self._phase.detail = "outputs written"
            case _:
                # No silent drop. An event this dashboard does not model is still an observation,
                # and for some of them (the convergence sweep) the terminal is the only sink a user
                # has attached -- so it falls back to a generic row rather than vanishing. The cost
                # of a new event going unstyled is a plain line; the cost of it going unseen is the
                # class of defect this whole observability effort exists to remove.
                from diffBloch.app.loggers import format_measurements

                self._other.append((event.channel, event.step, format_measurements(event)))
        return True

    # -- rendering -----------------------------------------------------------------------------

    def _start(self) -> None:
        from rich.console import Console
        from rich.live import Live

        console = Console()
        if not console.is_terminal:
            return  # an in-place display off a tty would just emit control characters
        self._console = console
        self._live = Live(console=console, refresh_per_second=self.refresh_per_second)
        self._live.start()

    def _window(self) -> int | None:
        """Rows each table may show while live, from the real terminal height (``None`` = all).

        Split across however many tables are currently on screen, so three tables on a short
        terminal each shrink rather than the last one being pushed off the bottom.
        """
        height = self._console.size.height if self._console is not None else 24
        on_screen = (self._trials, self._stages, self._epochs, self._rotations, self._other)
        tables = sum(bool(x) for x in on_screen) or 1
        return max(_MIN_WINDOW, (height - _CHROME_ROWS) // tables)

    def _render(self, window: int | None = -1) -> Any:
        """Compose the dashboard. ``window`` rows per table; ``None`` shows every row."""
        from rich.console import Group

        rows = self._window() if window == -1 else window
        blocks: list[Any] = [block for block in (self._header(), self._objective()) if block]
        if self._trials or self._pass:
            blocks.append(self._convergence_table(rows))
        if self._stages:
            blocks.append(self._stage_table(rows))
        if self._phase.total:
            blocks.append(self._progress())
        if self._epochs:
            blocks.append(self._epoch_table(rows))
        if self._rotations:
            blocks.append(self._rotation_table(rows))
        if self._other:
            blocks.append(self._other_table(rows))
        return Group(*blocks)

    def _header(self) -> Any:
        from rich.panel import Panel

        experiment = self._experiment
        if experiment is None:
            return None
        body = (
            f"[bold]{experiment.name}[/]   {experiment.structure} + {experiment.experimental_data}\n"
            f"{experiment.optimizer} lr={experiment.learning_rate:g} · "
            f"{experiment.steps} epoch(s) · g_max(solve)={experiment.solve_g_max:g} "
            f"sg_max={experiment.sg_max:g} · absorption "
            f"{'on' if experiment.absorption else 'off'}"
        )
        return Panel(body, title="experiment", border_style="cyan")

    def _objective(self) -> Any:
        from rich.panel import Panel

        manifest = self._manifest
        if manifest is None:
            return None
        penalties = ", ".join(f"{t.name} (w={t.weight:g})" for t in manifest.penalties) or "none"
        constraints = ", ".join(manifest.constraints) or "none"
        components = ", ".join(manifest.components) or "none"
        # "none" is rendered, never omitted: an objective composing no restraints is a fact.
        body = f"penalties   {penalties}\nconstraints {constraints}\ncomponents  {components}"
        return Panel(body, title="objective", border_style="magenta")

    def _convergence_table(self, window: int | None) -> Any:
        """The numerical-convergence sweep: each trial's setting change and what it cost in R.

        A trial is starred once its R-factor falls under the pass threshold -- that is the
        comparison ``converge_scalar`` actually settles on, so showing the threshold without
        showing which trial cleared it would leave the reader to do the arithmetic.
        """
        from rich.table import Table

        started = self._pass
        label = "convergence"
        if started is not None:
            label += (
                f"  (pass {started.pass_index}, threshold {started.r_factor_threshold:g}, "
                f"{started.n_orientations} rotation(s))"
            )
        if self._sweep:
            label += f"  sweeping {self._sweep}"
        table = Table(title=_titled(label, self._trials, window), title_justify="left")
        table.add_column("control")
        for column in ("trial", "from", "to", "R", "compared hkl"):
            table.add_column(column, justify="right")
        table.add_column("")
        for trial in _tail(self._trials, window):
            settled = started is not None and trial.r_factor < started.r_factor_threshold
            table.add_row(
                trial.control,
                str(trial.trial_index),
                f"{trial.previous:g}",
                f"{trial.candidate:g}",
                f"{trial.r_factor:.6f}",
                str(trial.n_compared_hkl),
                "settled" if settled else "",
            )
        return table

    def _stage_table(self, window: int | None) -> Any:
        from rich.table import Table

        table = Table(
            title=_titled("preprocess", self._stages, window), title_justify="left", expand=False
        )
        for column in ("stage", "rotations", "solve beams", "max/rot", "observed", "matched"):
            table.add_column(column, justify="right" if column != "stage" else "left")
        for name, values in _tail(self._stages, window):
            table.add_row(
                name.replace("_", " "),
                *(
                    # "-" rather than "0" for a key the stage does not report: n_matched_hkl is
                    # absent before the build, and zero would mean "matched nothing".
                    "-" if values.get(key) is None else f"{int(values[key])}"
                    for key in (
                        "n_orientations",
                        "n_solve_beams_total",
                        "n_solve_beams_max",
                        "n_observed_hkl",
                        "n_matched_hkl",
                    )
                ),
            )
        return table

    def _progress(self) -> Any:
        from rich.progress_bar import ProgressBar
        from rich.table import Table

        phase = self._phase
        fraction = min(1.0, phase.current / phase.total) if phase.total else 0.0
        elapsed = time.time() - phase.started_at if phase.started_at else 0.0
        eta = ""
        if 0 < phase.current < phase.total and elapsed > 0:
            remaining = (elapsed / phase.current) * (phase.total - phase.current)
            eta = f"  eta {int(remaining) // 60}:{int(remaining) % 60:02d}"
        grid = Table.grid(padding=(0, 1))
        grid.add_row(
            f"[bold]{phase.label}[/]",
            ProgressBar(total=max(1, phase.total), completed=phase.current, width=32),
            f"{phase.current}/{phase.total} ({100 * fraction:.0f}%){eta}",
            phase.detail,
        )
        return grid

    def _epoch_table(self, window: int | None) -> Any:
        from rich.table import Table

        table = Table(
            title=_titled("refinement", self._epochs, window), title_justify="left", expand=False
        )
        table.add_column("epoch", justify="right")
        table.add_column("wR2", justify="right")
        table.add_column("R_obs", justify="right")
        table.add_column("diffraction", justify="right")
        penalty_names = [name for name in self._epochs[-1].components if name != "diffraction"]
        for name in penalty_names:
            table.add_column(name, justify="right")
        best = self._completed.best_step if self._completed else None
        for epoch in _tail(self._epochs, window):
            marker = " *" if best is not None and epoch.iteration == best else ""
            row = [
                f"{epoch.iteration + 1}{marker}",
                _mean(epoch.wr2, epoch.n_wr2_evaluated, epoch.n_rotations),
                _mean(epoch.r_obs, epoch.n_r_obs_evaluated, epoch.n_rotations),
                "-" if epoch.diff_loss is None else f"{epoch.diff_loss:.6f}",
            ]
            row += [
                f"{epoch.components[name]['contribution']:.4g}" if name in epoch.components else "-"
                for name in penalty_names
            ]
            table.add_row(*row)
        return table

    def _rotation_table(self, window: int | None) -> Any:
        from rich.table import Table

        finite_wr2 = [row.wr2 for row in self._rotations if row.wr2 == row.wr2]
        mean = f"{sum(finite_wr2) / len(finite_wr2):.6f}" if finite_wr2 else "n/a"
        table = Table(
            title=_titled(
                f"settled per-rotation  (mean wR2 {mean} "
                f"[{len(finite_wr2)}/{len(self._rotations)}])",
                self._rotations,
                window,
            ),
            title_justify="left",
            expand=False,
        )
        for column in ("rotation", "wR2", "R_obs", "matched", ""):
            table.add_column(column, justify="right" if column else "left")
        for row in _tail(self._rotations, window):
            table.add_row(
                str(row.rotation_index),
                f"{row.wr2:.6f}",
                f"{row.r_obs:.6f}",
                str(row.n_matched),
                "validation" if row.is_validation else "",
            )
        return table

    def _other_table(self, window: int | None) -> Any:
        """Events the dashboard has no dedicated view for, rendered generically.

        Its presence is the point: a channel showing up here says the display is behind the event
        vocabulary, which is a prompt to give it a real view -- not a reason to have hidden it.
        """
        from rich.table import Table

        table = Table(
            title=_titled("other events", self._other, window),
            title_justify="left",
            expand=False,
            style="dim",
        )
        table.add_column("channel")
        table.add_column("step", justify="right")
        table.add_column("measurements")
        for channel, step, measurements in _tail(self._other, window):
            table.add_row(channel, "-" if step is None else str(step), measurements)
        return table


def _mean(value: float | None, evaluated: int | None, total: int | None) -> str:
    """A mean with the denominator it was taken over; ``n/a`` when nothing was evaluated."""
    if value is None or value != value:
        return "n/a" if evaluated is None else f"n/a [{evaluated}/{total}]"
    if evaluated is None or total is None:
        return f"{value:.6f}"
    return f"{value:.6f} [{evaluated}/{total}]"


def _tail[T](rows: list[T], window: int | None) -> list[T]:
    """The last ``window`` rows, or every row when ``window`` is ``None``."""
    return rows if window is None else rows[-window:]


def _titled(label: str, rows: list[Any], window: int | None) -> str:
    """A table title that admits when it is showing a trailing window rather than everything.

    A silently truncated table misreads as a complete one -- the same failure as a mean printed
    without the denominator it was taken over.
    """
    if window is None or len(rows) <= window:
        return f"{label}  ({len(rows)})"
    return f"{label}  (last {window} of {len(rows)})"
