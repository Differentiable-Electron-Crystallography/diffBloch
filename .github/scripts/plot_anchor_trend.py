"""Render the quartz e2e anchor trend: worst per-rotation R_obs deviation per merge to main.

Usage:  python plot_anchor_trend.py <history.csv> <out_dir>

Reads ``date,sha,max_abs_r_obs_deviation,r_obs_atol`` rows (an empty deviation is a gap: the
merge recorded no measurement) and writes one SVG per GitHub README theme into ``out_dir``:

    anchor-trend.svg        light
    anchor-trend-dark.svg   dark

The README embeds both via <picture> with prefers-color-scheme. Each point is the largest
``|r_obs - reference r_obs|`` over the rotations ``tests/e2e/inference/test_inference.py`` ran;
the dashed line is the ``atol`` that test held each rotation to, read per row from the CSV
(never hardcoded here -- a hardcoded tolerance is how the previous chart went stale). Points on
or under the line passed; a point above it is the event this plot exists to show.

CI (``publish`` job) runs this from main's checkout against the history on the ``badges`` branch.
"""

from __future__ import annotations

import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

THEMES = {
    "": {  # light
        "series": "#2a78d6",
        "ink_secondary": "#52514e",
        "ink_muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
    },
    "-dark": {
        "series": "#3987e5",
        "ink_secondary": "#c3c2b7",
        "ink_muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
    },
}

type Row = tuple[datetime, float, float]


def load_rows(path: Path) -> list[Row]:
    rows = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            deviation = row["max_abs_r_obs_deviation"]
            atol = row["r_obs_atol"]
            rows.append(
                (
                    datetime.fromisoformat(row["date"]),
                    float(deviation) if deviation else math.nan,
                    float(atol) if atol else math.nan,
                )
            )
    rows.sort(key=lambda r: r[0])  # file order is never trusted
    return rows


def render(rows: list[Row], out: Path, theme: dict[str, str]) -> None:
    xs = list(range(len(rows)))
    deviations = [d for _, d, _ in rows]
    atols = [a for _, _, a in rows]

    fig, ax = plt.subplots(figsize=(8.0, 3.2), dpi=100)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    finite_atol = [a for a in atols if not math.isnan(a)]
    finite_dev = [d for d in deviations if not math.isnan(d)]
    # No recorded atol means no tolerance line or label: never draw one no run reported.
    top = max([1.5 * a for a in finite_atol] + [1.1 * d for d in finite_dev], default=0.0)
    if top > 0:
        ax.set_ylim(0.0, top)
    ax.set_xlim(-0.5, max(len(rows) - 0.5, 0.5))

    # Tolerance as a step line so a re-pinned atol shows where it changed.
    ax.stairs(
        atols,
        [x - 0.5 for x in xs] + [len(rows) - 0.5],
        baseline=None,
        color=theme["ink_muted"],
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    if finite_atol:
        ax.annotate(
            f"atol {finite_atol[-1]:.0e}",
            xy=(0.005, finite_atol[-1]),
            xycoords=("axes fraction", "data"),
            xytext=(0, 3),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
            color=theme["ink_muted"],
        )
    # NaN gaps break the line; markers show every merge, including isolated ones.
    ax.plot(
        xs,
        deviations,
        color=theme["series"],
        linewidth=2.0,
        marker="o",
        markersize=4,
        zorder=3,
    )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.1e}"))
    ax.set_ylabel("max |ΔR_obs|", color=theme["ink_muted"], fontsize=9)
    ax.set_title(
        "quartz e2e anchor — worst per-rotation R_obs deviation from reference, per merge",
        color=theme["ink_secondary"],
        fontsize=10,
        loc="left",
        pad=10,
    )

    # Sparse date ticks: ~6 across the sequence, same-day duplicates collapsed to the later
    # index, and the newest point always ticked so the axis ends at the latest data.
    if rows:
        step = max(1, len(rows) // 6)
        candidates = list(range(0, len(rows), step))
        if candidates[-1] != len(rows) - 1:
            candidates.append(len(rows) - 1)
        ticks: list[int] = []
        labels: list[str] = []
        for i in candidates:
            label = rows[i][0].strftime("%b %d")
            if labels and label == labels[-1]:
                ticks[-1] = i
                continue
            ticks.append(i)
            labels.append(label)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
    else:
        ax.set_xticks([])
    ax.tick_params(colors=theme["ink_muted"], labelsize=8, length=0)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.75)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["baseline"])

    fig.tight_layout()
    fig.savefig(out, format="svg", transparent=True)
    plt.close(fig)
    print(f"wrote {out} ({len(rows)} rows)")


def main() -> None:
    csv_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    matplotlib.rcParams["svg.fonttype"] = "none"  # text as text, not paths
    matplotlib.rcParams["font.family"] = "sans-serif"
    rows = load_rows(csv_path)
    for suffix, theme in THEMES.items():
        render(rows, out_dir / f"anchor-trend{suffix}.svg", theme)


if __name__ == "__main__":
    main()
