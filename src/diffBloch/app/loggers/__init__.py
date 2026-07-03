"""Logger backends -- the ``app/`` boundary where sinks and vendor SDKs live, never the core.

Each backend consumes the uniform :class:`~diffBloch.observability.Event` surface (``channel`` +
``measurements``), so none knows the concrete event types and new events need no backend change.
:class:`ConsoleLogger` (here, no vendor dependency) routes events to stdlib ``logging``; each
third-party backend lives in its own confined submodule that imports its SDK lazily, so importing
this package never requires an optional dependency:

- :class:`~diffBloch.app.loggers.wandb.WandbLogger` (``diffBloch.app.loggers.wandb``)
- Comet ML: ``diffBloch.app.loggers.comet`` *(planned, same shape)*

Writing your own backend is one method -- see the "extension" section of the logging tutorial.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from diffBloch.observability import Event

__all__ = ["ConsoleLogger", "format_measurements"]

_log = logging.getLogger("diffBloch.loggers")


def format_measurements(event: Event) -> str:
    """Render an event's measurements as space-joined ``name=value`` pairs (shared by backends)."""
    return " ".join(f"{name}={value:g}" for name, value in event.measurements.items())


@dataclass
class ConsoleLogger:
    """Log each event to stdlib ``logging`` at ``level`` (console/file handlers attached by app).

    The bridge from the domain-observation channel to the diagnostics channel: handy for a live
    scroll of per-rotation ``R_obs`` while chasing a residual. Attach a handler (or call
    :func:`logging.basicConfig`) at the app boundary to see it.
    """

    level: int = logging.INFO

    def report(self, event: Event) -> None:
        _log.log(self.level, "%s %s", event.channel, format_measurements(event))
