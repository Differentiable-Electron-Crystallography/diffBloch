"""Render the quartz-anchor mean R_obs trend from anchor-history.csv.

Reads anchor-history.csv (date,sha,mean_r_obs; empty value = gap), sorts by date
(file order is never trusted -- the backfill and live CI append in different
orders), and renders one SVG per GitHub README theme:

    anchor-trend.svg        light
    anchor-trend-dark.svg   dark

The README embeds both via <picture> with prefers-color-scheme. Gaps break the
line rather than interpolate. The shaded band is the anchor's pinned tolerance
(EXPECTED_COUPLED_MEAN_R_OBS 0.0506 +/- 1e-2 in tests/e2e/test_anchor.py): a
flat line inside the band is the desired outcome (reproducibility), a point
outside it is the event this plot exists to show.

Run:  MPLBACKEND=Agg uv run --python 3.12 --with 'matplotlib==3.*' python plot_anchor_trend.py
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
CSV_PATH = HERE / "anchor-history.csv"

PINNED = 0.0506  # EXPECTED_COUPLED_MEAN_R_OBS
TOL = 1e-2  # COUPLED_MEAN_R_OBS_TOL

# Reference dataviz palette: categorical slot 1 (blue) stepped per surface, with
# the standard chrome inks. Backgrounds stay transparent so the README page
# color shows through.
THEMES = {
    "": {  # light
        "series": "#2a78d6",
        "band": "#cde2fb",
        "ink_secondary": "#52514e",
        "ink_muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
    },
    "-dark": {
        "series": "#3987e5",
        "band": "#184f95",
        "ink_secondary": "#c3c2b7",
        "ink_muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
    },
}


def load_rows() -> list[tuple[datetime, str, float]]:
    rows = []
    with CSV_PATH.open() as fh:
        for row in csv.DictReader(fh):
            value = float(row["mean_r_obs"]) if row["mean_r_obs"] else math.nan
            rows.append((datetime.fromisoformat(row["date"]), row["sha"], value))
    rows.sort(key=lambda r: r[0])
    return rows


def render(rows: list[tuple[datetime, str, float]], suffix: str, theme: dict) -> None:
    xs = range(len(rows))
    ys = [value for _, _, value in rows]

    fig, ax = plt.subplots(figsize=(8.0, 3.2), dpi=100)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    ax.axhspan(
        PINNED - TOL, PINNED + TOL, color=theme["band"],
        alpha=0.35, linewidth=0, zorder=1,
    )
    ax.axhline(PINNED, color=theme["band"], linewidth=1.0, zorder=2)
    # NaN gaps break the line; no per-point markers at this density -- except isolated
    # points (both neighbors are gaps/edges), which a marker-less line cannot show at all.
    ax.plot(xs, ys, color=theme["series"], linewidth=2.0, zorder=3)
    isolated = [
        (i, y) for i, y in zip(xs, ys)
        if not math.isnan(y)
        and (i == 0 or math.isnan(ys[i - 1]))
        and (i == len(ys) - 1 or math.isnan(ys[i + 1]))
    ]
    if isolated:
        ax.scatter(
            [i for i, _ in isolated], [y for _, y in isolated],
            s=12, color=theme["series"], linewidths=0, zorder=3,
        )

    ax.set_ylim(PINNED - 1.6 * TOL, PINNED + 1.6 * TOL)
    ax.set_xlim(-1, len(rows))
    ax.set_ylabel("mean R_obs", color=theme["ink_muted"], fontsize=9)
    ax.set_title(
        "quartz coupled anchor — mean R_obs per merge to main",
        color=theme["ink_secondary"], fontsize=10, loc="left", pad=10,
    )
    ax.annotate(
        f"pinned {PINNED} ± {TOL}",
        xy=(0.995, PINNED + TOL), xycoords=("axes fraction", "data"),
        ha="right", va="bottom", fontsize=8, color=theme["ink_muted"],
    )

    # Sparse date ticks: ~6 across the sequence, deduplicated (a dense tail of same-day
    # merges would otherwise repeat the label). The last point is always ticked -- without
    # it the axis reads as ending days before the newest data.
    if rows:
        step = max(1, len(rows) // 6)
        candidates = list(range(0, len(rows), step))
        if candidates[-1] != len(rows) - 1:
            candidates.append(len(rows) - 1)
        ticks, labels = [], []
        for i in candidates:
            label = rows[i][0].strftime("%b %d")
            if labels and label == labels[-1]:
                ticks[-1] = i  # same-day duplicate: keep the later index (axis ends dated)
                continue
            ticks.append(i)
            labels.append(label)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(colors=theme["ink_muted"], labelsize=8, length=0)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.75)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["baseline"])

    fig.tight_layout()
    out = HERE / f"anchor-trend{suffix}.svg"
    fig.savefig(out, format="svg", transparent=True)
    plt.close(fig)
    print(f"wrote {out.name} ({len(rows)} rows)")


def main() -> None:
    matplotlib.rcParams["svg.fonttype"] = "none"  # text as text, not paths
    matplotlib.rcParams["font.family"] = "sans-serif"
    rows = load_rows()
    for suffix, theme in THEMES.items():
        render(rows, suffix, theme)


if __name__ == "__main__":
    main()
