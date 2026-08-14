"""Comet ML logger backend -- confined home for the ``comet_ml`` integration.

Unlike Weights & Biases (a module-level ``wandb.log`` after ``wandb.init``), Comet ML logs to an
``Experiment`` **object**. So the app owns the run lifecycle -- it creates the
``comet_ml.Experiment`` (``uv sync --extra comet``) and passes it in -- and this logger only
forwards each event's measurements to that experiment as ``{channel}/{metric}`` metric series. The
logger itself imports no vendor SDK (it duck-types ``experiment.log_metrics``), so importing
``diffBloch.app.loggers``
never requires ``comet_ml`` and the pure core never touches it.
"""

from __future__ import annotations

from dataclasses import dataclass

from diffBloch.app.loggers import namespaced_measurements
from diffBloch.observability import Event


@dataclass
class CometLogger:
    """Log each event's measurements to a Comet ML experiment as ``{channel}/{metric}`` series.

    ``experiment`` is a ``comet_ml.Experiment`` created and owned by the app (``Experiment.end()``
    at the end of the run); this logger only calls ``experiment.log_metrics`` per event.
    """

    experiment: object

    def report(self, event: Event) -> None:
        self.experiment.log_metrics(  # type: ignore[attr-defined]
            namespaced_measurements(event), step=event.step
        )
