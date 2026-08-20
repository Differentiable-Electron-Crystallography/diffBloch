"""Matplotlib figures over a diffBloch JSONL event report.

Each ``plot_*`` takes the parsed records and returns a ``Figure``, or ``None`` when the report
carries no events of the kind it draws -- so a preprocess-only report simply yields fewer figures
rather than erroring. :func:`build_figures` runs them all and drops the empty ones.

These live in a module rather than in a notebook cell so they can be imported, diffed, and tested;
``event_report.ipynb`` is a thin driver over this file. Nothing here is imported by
``src/diffBloch``: rendering is a consumer concern, and matplotlib is a dev/tooling dependency.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from diffBloch.observability import EventRecord

from .reader import by_dataset, finite_mean, records_of, sorted_by_rotation
from .style import INK, MUTED, SERIES, styled

__all__ = [
    "build_figures",
    "build_sections",
    "export_figures",
    "plot_convergence_sweeps",
    "plot_coupling_geometry",
    "plot_coupling_segment_heatmap",
    "plot_dataset_summary",
    "plot_epoch_curve",
    "plot_orientation_optimization",
    "plot_orientation_search_trace",
    "plot_refined_rotation_scores",
    "plot_thickness_grids",
    "plot_thickness_heatmap",
    "plot_thickness_model",
]


def _rotation_labels(records: Sequence[EventRecord]) -> list[str]:
    return [f"{record.dataset or ''}:{record.rotation_index}" for record in records]


def _thin_ticks(ax: plt.Axes, labels: Sequence[str], *, keep: int = 12) -> None:
    """Label at most ``keep`` x ticks -- a hundred rotations of text is unreadable overlap."""
    stride = max(1, len(labels) // keep)
    positions = list(range(len(labels)))[::stride]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels[::stride], rotation=45, ha="right")


# Panel order for the convergence sweep; anything else follows, alphabetically.
_CONTROL_ORDER = ("g_max", "sg_max", "tilt_steps")


def _control_rank(control: str) -> tuple[int, str]:
    return (
        (_CONTROL_ORDER.index(control), "")
        if control in _CONTROL_ORDER
        else (len(_CONTROL_ORDER), control)
    )


@styled
def plot_convergence_sweeps(records: Sequence[EventRecord]) -> Figure | None:
    """Each numerical control's convergence ladder, one panel per control.

    The y axis is *not* a loss. A ``ConvergenceTrial`` simulates at ``previous`` and again at
    ``candidate`` and reports the R-factor **between the two**, so the curve answers "has the answer
    stopped changing?" rather than "is this setting good?". The pass's ``r_factor_threshold`` is
    drawn as a rule and the first candidate under it is marked: that crossing *is* the settled
    value, and a curve that never crosses is a sweep that ran out of range rather than one that
    converged.
    """
    trials = records_of(records, "ConvergenceTrial")
    if not trials:
        return None
    thresholds = {
        int(record.payload["pass_index"]): float(record.payload["r_factor_threshold"])
        for record in records_of(records, "ConvergencePassStarted")
    }
    controls = sorted({str(record.payload["control"]) for record in trials}, key=_control_rank)
    fig, axes = plt.subplots(
        len(controls), 1, figsize=(9, 3.2 * len(controls)), squeeze=False, sharey=True
    )
    positive = all(float(record.payload["r_factor"]) > 0.0 for record in trials)
    for ax, control in zip(axes[:, 0], controls, strict=True):
        labelled_settled = False
        for pass_index, group in sorted(_by_pass(trials, control).items()):
            ordered = sorted(group, key=lambda record: int(record.payload["trial_index"]))
            x = [float(record.payload["candidate"]) for record in ordered]
            y = [float(record.payload["r_factor"]) for record in ordered]
            ax.plot(x, y, marker="o", linewidth=1.2, label=f"pass {pass_index}")
            threshold = thresholds.get(pass_index)
            if threshold is None:
                continue
            settled = next(((cx, cy) for cx, cy in zip(x, y, strict=True) if cy < threshold), None)
            if settled is not None:
                # One fixed colour and one legend entry: the star always means "settled", and
                # which pass it belongs to is already shown by the line it sits on.
                ax.scatter(
                    [settled[0]],
                    [settled[1]],
                    marker="*",
                    s=140,
                    zorder=3,
                    color=INK,
                    label=None if labelled_settled else "settled",
                )
                labelled_settled = True
        for threshold in sorted(set(thresholds.values())):
            # The one dashed line in these figures, and it earns it: a dash reads as "threshold",
            # which is exactly what this is. Gridlines stay solid hairlines.
            ax.axhline(threshold, linestyle="--", linewidth=1.0, color=MUTED)
        if positive:
            # The R-factors span orders of magnitude as a control converges; linear hides the tail.
            ax.set_yscale("log")
        ax.set_ylabel("R between steps")
        ax.set_title(control)
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize="small")
    axes[-1, 0].set_xlabel("candidate value")
    fig.tight_layout()
    return fig


def _by_pass(trials: Sequence[EventRecord], control: str) -> dict[int, list[EventRecord]]:
    grouped: dict[int, list[EventRecord]] = {}
    for record in trials:
        if str(record.payload["control"]) != control:
            continue
        grouped.setdefault(int(record.payload["pass_index"]), []).append(record)
    return grouped


@styled
def plot_epoch_curve(records: Sequence[EventRecord]) -> Figure | None:
    """Train/validation wR2 and R_obs against refinement epoch."""
    steps = records_of(records, "RefinementStep")
    if not steps:
        return None
    x = [record.payload["iteration"] + 1 for record in steps]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label in (
        ("wr2", "train wR2"),
        ("r_obs", "train R_obs"),
        ("val_wr2", "validation wR2"),
        ("val_r_obs", "validation R_obs"),
    ):
        y = [record.payload.get(key) for record in steps]
        if any(value is not None for value in y):
            ax.plot(x, y, marker="o", linewidth=1.5, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_title("Epoch curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig


@styled
def plot_orientation_optimization(records: Sequence[EventRecord]) -> Figure | None:
    """Seed vs fitted score per rotation, with the fitted goniometer angle deltas beneath."""
    fits = sorted_by_rotation(records_of(records, "OrientationOptimized"))
    if not fits:
        return None
    x = range(len(fits))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(
        x,
        [record.payload.get("seed_score") for record in fits],
        marker="o",
        linewidth=1,
        label="before",
    )
    axes[0].plot(
        x, [record.payload.get("score") for record in fits], marker="o", linewidth=1, label="after"
    )
    axes[0].set_ylabel("score")
    axes[0].set_title("Orientation optimization")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    for key in ("alpha", "beta", "omega"):
        axes[1].plot(
            x, [record.payload.get(key) for record in fits], marker="o", linewidth=1, label=key
        )
    axes[1].set_ylabel("delta angle (deg)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    _thin_ticks(axes[1], _rotation_labels(fits))
    fig.tight_layout()
    return fig


@styled
def plot_dataset_summary(records: Sequence[EventRecord]) -> Figure | None:
    """Mean final wR2 / R_obs per dataset. Pooled experiments only -- one dataset has nothing to
    compare against."""
    metrics = [record for record in records_of(records, "RefinedRotationMetrics") if record.dataset]
    grouped = by_dataset(metrics)
    datasets = sorted(grouped)
    if len(datasets) <= 1:
        return None
    wr2 = [finite_mean(r.payload.get("wr2") for r in grouped[name]) for name in datasets]
    r_obs = [finite_mean(r.payload.get("r_obs") for r in grouped[name]) for name in datasets]
    x = range(len(datasets))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([value - width / 2 for value in x], wr2, width=width, label="wR2")
    ax.bar([value + width / 2 for value in x], r_obs, width=width, label="R_obs")
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_title("Per-dataset final scores")
    ax.legend()
    fig.tight_layout()
    return fig


@styled
def plot_refined_rotation_scores(records: Sequence[EventRecord]) -> Figure | None:
    """Final refined wR2 and R_obs per rotation, with held-out rotations marked."""
    metrics = records_of(records, "RefinedRotationMetrics")
    if not metrics:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for dataset, group in sorted(by_dataset(metrics).items()):
        ordered = sorted(group, key=lambda record: record.rotation_index or -1)
        x = [record.rotation_index for record in ordered]
        held_out = [record for record in ordered if record.payload.get("is_validation")]
        for ax, key, label in ((axes[0], "wr2", "wR2"), (axes[1], "r_obs", "R_obs")):
            ax.plot(
                x,
                [record.payload.get(key) for record in ordered],
                marker="o",
                linewidth=1,
                label=dataset or "dataset",
            )
            if held_out:
                ax.scatter(
                    [record.rotation_index for record in held_out],
                    [record.payload.get(key) for record in held_out],
                    marker="x",
                    s=50,
                    linewidths=1.5,
                    label=f"{dataset or 'dataset'} validation",
                )
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
    axes[0].set_title("Final refined per-rotation scores")
    axes[1].set_xlabel("rotation")
    axes[0].legend(fontsize="small", ncols=2)
    fig.tight_layout()
    return fig


@styled
def plot_thickness_grids(records: Sequence[EventRecord]) -> Figure | None:
    """Every rotation's scored thickness grid, with the selected thickness marked."""
    fits = records_of(records, "ThicknessOptimized")
    if not fits:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for record in fits:
        x = record.series.get("candidate_thicknesses")
        y = record.series.get("candidate_score")
        if not x or not y:
            continue
        # Every curve is one colour on purpose. Letting the prop cycle run would paint a single
        # population of rotations in eight hues and imply a grouping that does not exist.
        ax.plot(x, y, linewidth=0.8, alpha=0.35, color=SERIES[0])
        selected = record.payload.get("thickness")
        score = record.payload.get("score")
        if selected is not None and score is not None:
            ax.scatter([selected], [score], s=14, color=INK, alpha=0.55, zorder=3)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("thickness")
    ax.set_ylabel("score")
    ax.set_title("Thickness score grids")
    ax.grid(True, alpha=0.25)
    return fig


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile over the finite values, or ``None`` when there are none."""
    kept = sorted(value for value in values if math.isfinite(value))
    if not kept:
        return None
    return kept[min(len(kept) - 1, int(fraction * (len(kept) - 1)))]


@styled
def plot_thickness_heatmap(records: Sequence[EventRecord]) -> Figure | None:
    """The thickness grid as rotation x thickness -> score, with the fitted thickness traced.

    The companion to :func:`plot_thickness_grids`: that one shows the *shape* of each minimum but
    overlays a hundred rotations into a hairball, while this one keeps them apart and answers the
    question the per-rotation fit actually raises -- does the fitted thickness drift smoothly with
    tilt, as an irregular specimen implies, or jump around, which means rotations are landing in
    different minima.

    One panel per dataset, since a pooled experiment's datasets may be gridded over different
    thickness ranges. Rotations whose grid disagrees with the rest of their dataset are skipped
    rather than silently stretched onto the wrong axis.
    """
    fits = records_of(records, "ThicknessOptimized")
    if not fits:
        return None
    panels = []
    for dataset, group in sorted(by_dataset(fits).items()):
        ordered = sorted(group, key=lambda record: record.rotation_index or -1)
        grid = ordered[0].series.get("candidate_thicknesses") or []
        rows = [
            record
            for record in ordered
            if record.series.get("candidate_thicknesses") == grid
            and record.series.get("candidate_score")
        ]
        if len(grid) > 1 and rows:
            panels.append((dataset, grid, rows))
    if not panels:
        return None
    heights = [max(3.0, min(10.0, 0.12 * len(rows) + 2.0)) for _, _, rows in panels]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(9, sum(heights)), squeeze=False, height_ratios=heights
    )
    for ax, (dataset, grid, rows) in zip(axes[:, 0], panels, strict=True):
        matrix = [list(record.series["candidate_score"]) for record in rows]
        image = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=(grid[0], grid[-1], len(rows) - 0.5, -0.5),
            # The far ends of the grid score arbitrarily badly and would otherwise take most of
            # the colour range, flattening the basin -- the part actually being read -- into one
            # tone. Clipping at the 95th percentile keeps the scale in absolute score units.
            vmax=_percentile([value for row in matrix for value in row], 0.95),
        )
        selected = [record.payload.get("thickness") for record in rows]
        # Dark ink, not white: the basin the trace runs through is the *light* end of a
        # lightness-monotonic ramp, so a white line would vanish exactly where it is read.
        ax.plot(
            selected,
            range(len(rows)),
            color=INK,
            linewidth=1.2,
            marker=".",
            markersize=3,
            label="fitted thickness",
        )
        indices = [record.rotation_index for record in rows]
        stride = max(1, len(indices) // 15)
        ax.set_yticks(list(range(len(indices)))[::stride])
        ax.set_yticklabels([str(index) for index in indices[::stride]])
        ax.set_ylabel("rotation")
        ax.set_title(f"Thickness score grid: {dataset or 'dataset'}")
        ax.legend(fontsize="small", loc="upper right")
        fig.colorbar(image, ax=ax, label="score")
    axes[-1, 0].set_xlabel("thickness")
    fig.tight_layout()
    return fig


@styled
def plot_thickness_model(records: Sequence[EventRecord]) -> Figure | None:
    """Each dataset's *learned* apparent-thickness curve against tilt angle.

    Not to be confused with :func:`plot_thickness_grids` / :func:`plot_thickness_heatmap`, which are
    the preprocess stage's per-rotation grid search -- one fitted scalar per rotation, picked by
    argmin. This is the refinement stage's ``ApparentThicknessNN``: a trained *function* of tilt,
    one per dataset, evaluated after the loop. Two different quantities from two different stages,
    which is why they carry different names and sit under separate headings.
    """
    profiles = records_of(records, "ThicknessProfile")
    if not profiles:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for record in profiles:
        alphas = record.series.get("alphas")
        thicknesses = record.series.get("thicknesses")
        if not alphas or not thicknesses:
            continue
        ordered = sorted(zip(alphas, thicknesses, strict=True))
        ax.plot(
            [row[0] for row in ordered],
            [row[1] for row in ordered],
            marker="o",
            linewidth=1.5,
            label=record.payload.get("label") or record.dataset or record.channel,
        )
    ax.set_xlabel("alpha (degrees)")
    ax.set_ylabel("predicted thickness")
    ax.set_title("Learned thickness model")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig


@styled
def plot_coupling_geometry(records: Sequence[EventRecord]) -> Figure | None:
    """Per-rotation coupled-solve shape: beam counts above, segment/tilt counts below."""
    rows = sorted_by_rotation(records_of(records, "RotationCoupling"))
    if not rows:
        return None
    x = range(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for ax, key, label in (
        (axes[0], "n_union_beams", "union beams"),
        (axes[0], "max_beams_per_segment", "max beams/segment"),
        (axes[1], "n_coupling_segments", "segments"),
        (axes[1], "max_tilts_per_segment", "max tilts/segment"),
    ):
        ax.plot(
            x, [record.payload.get(key) for record in rows], marker="o", linewidth=1, label=label
        )
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize="small")
    axes[0].set_title("Coupled solve geometry")
    axes[1].set_xlabel("dataset:rotation")
    _thin_ticks(axes[1], _rotation_labels(rows))
    fig.tight_layout()
    return fig


@styled
def plot_orientation_search_trace(records: Sequence[EventRecord]) -> Figure | None:
    """The scored search path of the longest-running rotation's orientation fit.

    One rotation, not all of them: the traces overlay into noise, and the longest search is the one
    worth inspecting. Row position is the trial index (the event stores no index column).
    """
    traces = records_of(records, "OrientationSearchTrace")
    if not traces:
        return None
    trace = max(traces, key=lambda record: len(record.series.get("score", ())))
    score = trace.series.get("score", [])
    if not score:
        return None
    trial = list(range(len(score)))
    comparable = trace.series.get("comparable_score", [])
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    axes[0].plot(trial, score, linewidth=1.2, label=trace.payload.get("residual", "score"))
    if comparable and comparable != score:
        axes[0].plot(trial, comparable, linewidth=1.0, alpha=0.75, label="comparable")
    for key, marker, label in (("is_seed", "o", "seed"), ("is_final", "x", "final")):
        flags = trace.series.get(key, [])
        marked = [(t, s) for t, s, flag in zip(trial, score, flags, strict=True) if flag]
        if marked:
            axes[0].scatter(
                [point[0] for point in marked],
                [point[1] for point in marked],
                marker=marker,
                s=60,
                label=label,
            )
    axes[0].set_ylabel("score")
    axes[0].set_title(
        f"Orientation search trace: {trace.dataset or 'dataset'}:{trace.rotation_index}"
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize="small")
    for key in ("alpha", "beta", "omega"):
        values = trace.series.get(key, [])
        if values:
            axes[1].plot(trial, values, linewidth=1.0, label=key)
    axes[1].set_xlabel("trial")
    axes[1].set_ylabel("angle delta (deg)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize="small")
    fig.tight_layout()
    return fig


@styled
def plot_coupling_segment_heatmap(records: Sequence[EventRecord]) -> Figure | None:
    """Per-rotation, per-segment beam counts as a heatmap (rows padded with NaN to the widest)."""
    traces = sorted_by_rotation(records_of(records, "RotationCouplingSegments"))
    if not traces:
        return None
    widest = max(len(record.series.get("n_segment_beams", ())) for record in traces)
    if widest == 0:
        return None
    matrix = []
    for record in traces:
        beams = list(record.series.get("n_segment_beams", ()))
        matrix.append(beams + [math.nan] * (widest - len(beams)))
    labels = _rotation_labels(traces)
    fig, ax = plt.subplots(figsize=(10, max(4.0, min(12.0, 0.25 * len(matrix) + 2.0))))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_xlabel("segment")
    ax.set_ylabel("dataset:rotation")
    ax.set_title("Coupling segment beam counts")
    ax.set_xticks(range(widest))
    stride = max(1, len(labels) // 20)
    ax.set_yticks(range(len(labels))[::stride])
    ax.set_yticklabels(labels[::stride])
    fig.colorbar(image, ax=ax, label="segment beams")
    fig.tight_layout()
    return fig


@dataclass(frozen=True)
class Section:
    """One headed group of figures, and the event types that place it in the run."""

    title: str
    builders: tuple[tuple[str, Callable[[Sequence[EventRecord]], Figure | None]], ...]
    event_types: tuple[str, ...]


# Stage groups, each headed separately rather than pooled into one flat run of figures. Their
# *order* is not hardcoded: `preprocess.stage_order` can put the thickness fit before the
# orientation fit, so the sections are sorted by where their events actually appear in the report
# (see `build_sections`). The listing order below is only the fallback for a tie.
SECTIONS = (
    Section(
        "Convergence",
        (("convergence_sweeps", plot_convergence_sweeps),),
        ("ConvergenceTrial",),
    ),
    Section(
        "Preprocess — orientation optimization",
        (
            ("orientation_optimization", plot_orientation_optimization),
            ("orientation_search_trace", plot_orientation_search_trace),
        ),
        ("OrientationOptimized", "OrientationSearchTrace"),
    ),
    Section(
        # "per-rotation" earns its place: the refinement stage has a thickness section too, and
        # that one is a learned function of tilt rather than one fitted scalar per rotation.
        "Preprocess — per-rotation thickness fit",
        (
            ("thickness_grids", plot_thickness_grids),
            ("thickness_heatmap", plot_thickness_heatmap),
        ),
        ("ThicknessOptimized",),
    ),
    Section(
        "Preprocess — coupled solve geometry",
        (
            ("coupling_geometry", plot_coupling_geometry),
            ("coupling_segment_heatmap", plot_coupling_segment_heatmap),
        ),
        ("RotationCoupling", "RotationCouplingSegments"),
    ),
    Section(
        "Refinement — epoch history",
        (("epoch_curve", plot_epoch_curve),),
        ("RefinementStep",),
    ),
    Section(
        "Refinement — per-rotation scores",
        (("refined_rotation_scores", plot_refined_rotation_scores),),
        ("RefinedRotationMetrics",),
    ),
    Section(
        "Refinement — datasets",
        (("per_dataset_summary", plot_dataset_summary),),
        ("RefinedRotationMetrics",),
    ),
    Section(
        "Refinement — learned thickness model",
        (("thickness_model", plot_thickness_model),),
        ("ThicknessProfile",),
    ),
)


def build_sections(records: Sequence[EventRecord]) -> list[tuple[str, dict[str, Figure]]]:
    """The report's figures grouped under stage headings, in the order the run produced them.

    A section is placed by the earliest ``sequence`` among the events it draws from, so a run
    configured ``stage_order: thickness_first`` heads its thickness section before its orientation
    one without this module knowing the recipe. Sections whose events are absent -- and sections
    whose figures all declined to render -- are dropped rather than shown empty.
    """
    first_seen: dict[str, int] = {}
    for record in records:
        first_seen.setdefault(record.event_type, record.sequence)
    ordered = sorted(
        enumerate(SECTIONS),
        key=lambda item: (
            min(
                (first_seen[name] for name in item[1].event_types if name in first_seen),
                default=len(records),
            ),
            item[0],
        ),
    )
    sections = []
    for _, section in ordered:
        built = ((name, builder(records)) for name, builder in section.builders)
        figures = {name: figure for name, figure in built if figure is not None}
        if figures:
            sections.append((section.title, figures))
    return sections


def build_figures(records: Sequence[EventRecord]) -> dict[str, Figure]:
    """Every figure the report has the events for, flattened out of :func:`build_sections`.

    The flat view is what :func:`export_figures` names files from; the notebook uses the sectioned
    one so each stage gets its own heading.
    """
    return {
        name: figure for _, figures in build_sections(records) for name, figure in figures.items()
    }


def export_figures(
    figures: dict[str, Figure], output_dir: Path, formats: Sequence[str] = ("svg",)
) -> list[Path]:
    """Write each figure to ``output_dir``. The only place this package writes image files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, figure in figures.items():
        for suffix in formats:
            path = output_dir / f"{name}.{suffix}"
            figure.savefig(path, bbox_inches="tight", dpi=160)
            written.append(path)
    return written
