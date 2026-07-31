"""Matplotlib plotting backend (the ``diffBloch[plot]`` extra).

Confines the matplotlib dependency to this module, imported lazily on first use, mirroring
``app.loggers.wandb``/``app.loggers.comet`` -- importing ``diffBloch.app.loggers`` never requires
matplotlib to be installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diffBloch.observability import Event, ThicknessOptimized

__all__ = ["ThicknessPlotLogger", "plot_thickness_nn_shape"]

_BLUE = "#3b4cc0"
_RED = "#b40426"


def _apply_house_style(ax: Any) -> None:
    """Shared look for report plots: Arial, blue/red palette, clean integer-ish y-ticks, no grid."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams["font.family"] = "Arial"
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.label.set_fontsize(13)
    ax.yaxis.label.set_fontsize(13)
    ax.title.set_fontsize(14)
    ax.title.set_fontweight("bold")


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
            edgecolors=_BLUE,
            linewidths=1.5,
        )
        ax.axvline(event.thickness, color=_RED, linestyle="--", linewidth=1.5)
        ax.set_xlabel("Thickness (Å)")
        ax.set_ylabel("wR2")
        ax.set_title(f"wR2 vs thickness — rotation {event.rotation_index}")
        _apply_house_style(ax)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{event.rotation_index}.png")
        plt.close(fig)


def plot_thickness_nn_shape(
    rows: Sequence[tuple[float, float]],
    output_path: str | Path,
    *,
    xlabel: str = "alpha (degrees)",
) -> None:
    """Save one PNG: the fitted thickness NN's predicted thickness vs. rotation angle.

    ``rows`` is ``(angle, thickness)`` pairs, one per rotation, in any order -- plotted sorted by
    angle so the curve reads left to right. This is the final learned shape after refinement
    completes, not a training-progress plot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows)
    angles = [row[0] for row in ordered]
    thicknesses = [row[1] for row in ordered]

    fig, ax = plt.subplots()
    ax.plot(
        angles,
        thicknesses,
        marker="o",
        linestyle="-",
        color=_BLUE,
        markerfacecolor=_BLUE,
        markeredgecolor=_BLUE,
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Predicted thickness (Å)")
    ax.set_title("Thickness NN final shape")
    _apply_house_style(ax)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
