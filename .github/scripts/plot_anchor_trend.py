"""Render the quartz e2e anchor trend: worst per-rotation R_obs deviation per merge to main.

Usage:  python plot_anchor_trend.py <history.csv> <out_dir>

Reads ``date,sha,max_abs_r_obs_deviation,r_obs_atol`` rows (an empty deviation is a gap: the
merge recorded no measurement) and writes into ``out_dir``:

    anchor-trend.svg        light chart (README, via <picture> with prefers-color-scheme)
    anchor-trend-dark.svg   dark chart
    anchor-trend.html       both charts inline, each point linked to its row in a table

The README's image is proxied as a plain <img>, which disables links inside the SVG, so the
README links to the HTML page for the interactive view. Each point is the largest
``|r_obs - reference r_obs|`` over the rotations ``tests/e2e/inference/test_inference.py`` ran;
the dashed line is the ``atol`` that test held each rotation to, read per row from the CSV
(never hardcoded here -- a hardcoded tolerance is how the previous chart went stale). Points on
or under the line were within tolerance; a point above it is the event this plot exists to show.

CI (``publish`` job) runs this from main's checkout against the history on the ``badges`` branch.
"""

from __future__ import annotations

import csv
import html
import io
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

COMMIT_URL = "https://github.com/Differentiable-Electron-Crystallography/diffBloch/commit/"

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

type Row = tuple[datetime, str, float, float]


def load_rows(path: Path) -> list[Row]:
    rows = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            deviation = row["max_abs_r_obs_deviation"]
            atol = row["r_obs_atol"]
            rows.append(
                (
                    datetime.fromisoformat(row["date"]),
                    row["sha"],
                    float(deviation) if deviation else math.nan,
                    float(atol) if atol else math.nan,
                )
            )
    rows.sort(key=lambda r: r[0])  # file order is never trusted
    return rows


def _hover(row: Row) -> str:
    date, sha, deviation, atol = row
    return f"{date:%Y-%m-%d} {sha[:7]}: max |ΔR_obs| {deviation:.2e} (atol {atol:.0e})"


def render(rows: list[Row], theme: dict[str, str]) -> str:
    """Return the chart as an SVG document; each measured point links to its commit."""
    xs = list(range(len(rows)))
    deviations = [r[2] for r in rows]
    atols = [r[3] for r in rows]

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
    # NaN gaps break the line; one linked marker per measured merge, including isolated ones.
    ax.plot(xs, deviations, color=theme["series"], linewidth=2.0, zorder=3)
    measured = [(x, r) for x, r in zip(xs, rows, strict=True) if not math.isnan(r[2])]
    points = ax.scatter(
        [x for x, _ in measured],
        [r[2] for _, r in measured],
        s=24,
        color=theme["series"],
        linewidths=0,
        zorder=4,
    )
    points.set_urls([COMMIT_URL + r[1] for _, r in measured])

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
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", transparent=True)
    plt.close(fig)
    svg = buffer.getvalue()
    # Hover text: matplotlib writes no <title>, so add one inside each point's link.
    for _, row in measured:
        link = f'<a xlink:href="{COMMIT_URL}{row[1]}" target="_blank">'
        svg = svg.replace(link, f"{link}<title>{html.escape(_hover(row))}</title>", 1)
    return svg


def _status(deviation: float, atol: float) -> str:
    if math.isnan(deviation):
        return "no measurement"
    if math.isnan(atol):
        return "no tolerance recorded"
    return "within tolerance" if deviation <= atol else "over tolerance"


def page(rows: list[Row], light_svg: str, dark_svg: str) -> str:
    """Both charts inline and a newest-first table of every merge, with no script.

    Each point links to its table row (``#<short sha>``); CSS ``:target`` highlights that row,
    which holds the commit link, and the sticky chart stays in view while the page scrolls to it.
    The fragment also makes each merge shareable. Hover still shows the point's ``<title>``.
    """

    def inline(svg: str, prefix: str = "") -> str:
        svg = svg[svg.index("<svg") :]
        # Both charts share one document: prefix ids and their references so they can't clash.
        if prefix:
            svg = re.sub(r'(id="|url\(#|href="#)', rf"\g<1>{prefix}", svg)
        # Commit links (absolute URLs, untouched above) become links to the point's table row.
        point = rf'xlink:href="{re.escape(COMMIT_URL)}([0-9a-f]{{7}})[0-9a-f]*" target="_blank"'
        return re.sub(point, r'xlink:href="#\1"', svg)

    body = "\n".join(
        f'<tr id="{sha[:7]}">'
        f"<td>{date:%Y-%m-%d}</td>"
        f'<td><a href="{COMMIT_URL}{sha}"><code>{sha[:7]}</code></a></td>'
        f"<td class=num>{'—' if math.isnan(dev) else f'{dev:.2e}'}</td>"
        f"<td class=num>{'—' if math.isnan(atol) else f'{atol:.0e}'}</td>"
        f"<td>{_status(dev, atol)}</td>"
        "</tr>"
        for date, sha, dev, atol in reversed(rows)
    )
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>diffBloch anchor trend</title>
<style>
:root {{ --bg: #fff; --ink: #1f2328; --muted: #59636e; --rule: #d1d9e0; --link: #0969da;
  --hl: #ddf4ff; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #0d1117; --ink: #e6edf3; --muted: #9198a1; --rule: #3d444d; --link: #4493f8;
    --hl: #0c2d6b; }}
}}
body {{ margin: 0 auto; max-width: 960px; padding: 16px; background: var(--bg); color: var(--ink);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
a {{ color: var(--link); }}
p {{ color: var(--muted); }}
.chart {{ position: sticky; top: 0; z-index: 1; background: var(--bg); }}
.chart svg {{ width: 100%; height: auto; display: block; }}
.chart.dark {{ display: none; }}
@media (prefers-color-scheme: dark) {{ .chart.light {{ display: none; }} .chart.dark {{ display: block; }} }}
.table {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 6px 12px 6px 0; border-bottom: 1px solid var(--rule); white-space: nowrap; }}
th {{ color: var(--muted); font-weight: 600; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
/* Land a targeted row below the sticky chart: its height is 0.4 x its width (8 x 3.2 in). */
tr {{ scroll-margin-top: calc((min(100vw, 960px) - 32px) * 0.4 + 16px); }}
tr:target td {{ background: var(--hl); }}
</style>
<h1>quartz e2e anchor</h1>
<p>For every merge to <code>main</code>: the largest deviation of any rotation's
<code>R_obs</code> from its committed reference, against the tolerance the test allows. Hover a
point for its values; click it to highlight its row below, which links to the commit's diff.</p>
<div class="chart light">{inline(light_svg)}</div>
<div class="chart dark">{inline(dark_svg, "dark-")}</div>
<div class="table"><table>
<thead><tr><th>Date</th><th>Commit</th><th class=num>max |ΔR_obs|</th><th class=num>atol</th>
<th>Status</th></tr></thead>
<tbody>
{body}
</tbody></table></div>
</html>
"""


def main() -> None:
    csv_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    matplotlib.rcParams["svg.fonttype"] = "none"  # text as text, not paths
    matplotlib.rcParams["font.family"] = "sans-serif"
    rows = load_rows(csv_path)
    svgs = {}
    for suffix, theme in THEMES.items():
        svgs[suffix] = render(rows, theme)
        (out_dir / f"anchor-trend{suffix}.svg").write_text(svgs[suffix])
    (out_dir / "anchor-trend.html").write_text(page(rows, svgs[""], svgs["-dark"]))
    print(f"wrote anchor-trend{{,-dark}}.svg and anchor-trend.html to {out_dir} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
