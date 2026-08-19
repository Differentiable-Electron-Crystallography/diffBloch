"""One visual style for every report figure.

Colour is assigned by the job it does, not by taste.

**Categorical** (identity -- a series, a dataset, a pass) draws from :data:`SERIES` in fixed slot
order, never cycled. The order is the colour-vision-deficiency safety mechanism: the first four
slots clear the adjacent-pair CVD gate (worst OKLab ΔE 9.1, target >= 8) and the normal-vision floor
(worst ΔE 22.9, floor 15) against this surface. Two slots sit below 3:1 contrast, which is why every
figure with more than one series carries a legend -- identity is never colour alone. Cycling matters
because matplotlib's default behaviour is to reuse its ten-colour cycle silently, which paints a
hundred single-population curves in ten hues and implies a grouping that does not exist.

**Sequential** (continuous magnitude -- the heatmaps) uses :data:`SEQUENTIAL`, which is ``viridis``.
A hand-rolled ramp interpolated through a UI palette's hex steps is *not* perceptually uniform: equal
steps in value would not read as equal steps in colour, which injects structure into a quantitative
field that the data does not contain. ``viridis`` is monotonic in lightness by construction, so it
survives greyscale printing and all three dichromacies. Crameri's scientific colour maps (``batlow``
and friends) are the other defensible family and are worth adopting if these figures ever go into a
manuscript that cites them; both are correct, and the local UI palette is not.

Chrome is recessive: solid hairline gridlines one shade off the surface, muted axis ink, thin marks.
Dashes are reserved for lines that genuinely mean a threshold, so a dashed rule is never mistaken for
a gridline. Type sizes stay at or above 7pt, the floor Nature and Science set for figure text, so an
exported panel is legible at print scale without re-typesetting.

Applied per figure through :func:`styled` rather than by mutating global ``rcParams``, so importing
this package never reaches into an interactive session's matplotlib settings.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from cycler import cycler

__all__ = ["INK", "MUTED", "REPORT_RC", "SEQUENTIAL", "SERIES", "SURFACE", "styled"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed categorical slot order -- assign by position, never cycle past the end.
SERIES = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

# Perceptually uniform, monotonic in lightness, greyscale- and CVD-safe.
SEQUENTIAL = "viridis"

REPORT_RC: dict[str, Any] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.prop_cycle": cycler(color=SERIES),
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    # Not "medium": matplotlib's bundled faces ship no medium weight, so it warns on every render
    # and silently falls back to regular anyway.
    "axes.titleweight": "regular",
    "axes.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linestyle": "-",  # solid hairline; dashes mean "threshold" in these figures
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 7,  # the Nature/Science figure-text floor
    "ytick.labelsize": 7,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "legend.labelcolor": INK_SECONDARY,
    "image.cmap": SEQUENTIAL,
    "font.size": 8,
    "font.family": "sans-serif",
    "figure.dpi": 110,
    "savefig.dpi": 300,  # the print floor; SVG export ignores it and stays vector
    "savefig.bbox": "tight",
}


def styled[F: Callable[..., Any]](plot: F) -> F:
    """Render ``plot`` under :data:`REPORT_RC` without touching global matplotlib state."""

    @wraps(plot)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        with plt.rc_context(REPORT_RC):
            return plot(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
