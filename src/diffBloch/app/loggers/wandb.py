"""Weights & Biases logger backend -- the only module that touches the ``wandb`` SDK.

``wandb`` is imported lazily inside :meth:`WandbLogger.report`, so it is an optional dependency
(``uv sync --extra wandb``): importing ``diffBloch.app.loggers`` never requires it, and the pure
core never touches it. The app owns the run lifecycle (``wandb.init`` / ``wandb.finish``); this
logger only forwards each event's measurements to the active run as ``{channel}/{metric}`` series.
"""

from __future__ import annotations

from dataclasses import dataclass

from diffBloch.observability import Event


@dataclass
class WandbLogger:
    """Log each event's measurements to Weights & Biases as ``{channel}/{metric}`` series."""

    def report(self, event: Event) -> None:
        import wandb  # lazy: optional dependency, never imported by the pure core

        payload = {f"{event.channel}/{name}": value for name, value in event.measurements.items()}
        wandb.log(payload)
