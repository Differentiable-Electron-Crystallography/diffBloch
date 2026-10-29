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

Chrome is recessive: no gridlines, muted axis ink, thin marks. Dashes are reserved for lines that
genuinely mean a threshold. Text is Arial (falling back to a metric-compatible sans-serif where Arial
is not installed), with mathtext set in the same face so metric names such as R_obs render with a
real subscript. Figures carry no drawn titles: the notebook shows each figure's name above it.

Applied per figure through :func:`styled` rather than by mutating global ``rcParams``, so importing
this package never reaches into an interactive session's matplotlib settings.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, cast

from cycler import cycler

__all__ = ["INK", "MUTED", "REPORT_RC", "SEQUENTIAL", "SERIES", "SURFACE", "styled"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
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
    "axes.labelsize": 20,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "legend.frameon": False,
    "legend.fontsize": 14,
    "legend.labelcolor": INK_SECONDARY,
    "image.cmap": SEQUENTIAL,
    "font.size": 20,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "figure.dpi": 110,
    "savefig.dpi": 300,  # the print floor; SVG export ignores it and stays vector
    "savefig.bbox": "tight",
}


def styled[F: Callable[..., Any]](plot: F) -> F:
    """Render ``plot`` under :data:`REPORT_RC` without touching global matplotlib state."""

    @wraps(plot)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        with plt.rc_context(cast(Any, REPORT_RC)):  # matplotlib's stub wants its Literal keys
            return plot(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
