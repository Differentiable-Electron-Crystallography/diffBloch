"""Matplotlib plotting backend (the ``diffBloch[plot]`` extra).

Confines the matplotlib dependency to this module, imported lazily on first use, mirroring
``app.loggers.wandb``/``app.loggers.comet`` -- importing ``diffBloch.app.loggers`` never requires
matplotlib to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from diffBloch.observability import Event, ThicknessOptimized

__all__ = ["ThicknessPlotLogger"]


@dataclass
class ThicknessPlotLogger:
    """Save one ``{rotation_index}.png`` per rotation: wR2 vs. every scored candidate thickness.

    Consumes :class:`~diffBloch.observability.ThicknessOptimized` (every other event is a no-op),
    reading its ``candidate_thicknesses``/``candidate_wr2`` fields -- the full grid
    ``optimize_thickness`` scored, not just the winner. Each point is one grid-search candidate; a
    dashed vertical line marks the winning thickness that got baked onto the plan. ``output_dir`` is
    created on first use. Being a no-op on every other event means this composes with
    :class:`~diffBloch.app.loggers.ConsoleLogger` / :class:`~diffBloch.app.loggers.CSVLogger` via
    :class:`~diffBloch.observability.MultiLogger` without stealing their events.
    """

    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def report(self, event: Event) -> None:
        if not isinstance(event, ThicknessOptimized):
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter(
            event.candidate_thicknesses,
            event.candidate_wr2,
            facecolors="none",
            edgecolors="b",
        )
        ax.axvline(event.thickness, color="r", linestyle="--")
        ax.set_xlabel("Thickness (Å)")
        ax.set_ylabel("wR2")
        ax.set_title(f"wR2 vs thickness — rotation {event.rotation_index}")
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{event.rotation_index}.png")
        plt.close(fig)
