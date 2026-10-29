"""Application entry points: the default experiment runners and the logger backends.

This is the public surface of the ``app`` layer -- the runnable entry points a caller reaches for
(``run_experiment`` / ``refine_experiment`` / ``preprocess_experiment``) and the concrete logger
backends (``ConsoleLogger`` / ``ReportLogger``, and the optional ``WandbLogger`` / ``CometLogger``).
The CLI (:mod:`diffBloch.app.cli`) is the console-script wrapper around these.

The optional backends are safe to re-export here: each imports its SDK lazily (only when logging),
so ``import diffBloch.app`` never requires the ``wandb`` / ``comet`` extras -- only *using* the
logger does.
"""

from diffBloch.app.loggers import (
    ConsoleLogger,
    ReportLogger,
)
from diffBloch.app.loggers.comet import CometLogger
from diffBloch.app.loggers.wandb import WandbLogger
from diffBloch.app.program import (
    converge_experiment,
    preprocess_experiment,
    refine_experiment,
    run_experiment,
)

__all__ = [
    "CometLogger",
    "ConsoleLogger",
    "ReportLogger",
    "WandbLogger",
    "converge_experiment",
    "preprocess_experiment",
    "refine_experiment",
    "run_experiment",
]
