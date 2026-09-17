"""Matplotlib figures over a diffBloch JSONL event report.

Each ``plot_*`` takes the parsed records and returns a ``Figure``, or ``None`` when the report
carries no events of the kind it draws -- so a preprocess-only report simply yields fewer figures
rather than erroring. :func:`build_figures` runs them all and drops the empty ones.

Every figure reads its events through :func:`~tools.event_report.reader.events_of`, i.e. as the
library's own event dataclasses, and never through the envelope's payload dict: the field names
below are checked against :mod:`diffBloch.observability` by the type checker and by the golden
report test, and a report that no longer fits fails at the read with the field named.

These live in a module rather than in a notebook cell so they can be imported, diffed, and tested;
``event_report.ipynb`` is a thin driver over this file. Nothing here is imported by
``src/diffBloch``: rendering is a consumer concern, and matplotlib is a dev/tooling dependency.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from diffBloch.observability import (
    ConvergencePassStarted,
    ConvergenceTrial,
    EventRecord,
    OrientationOptimizationStarted,
    OrientationOptimized,
    OrientationSearchTrace,
    RefinedRotationMetrics,
    RefinementCompleted,
    RefinementOrientationStep,
    RefinementStep,
    RotationCoupling,
    RotationCouplingSegments,
    ThicknessOptimized,
    ThicknessProfile,
    event_from_record,
)

from .reader import Positioned, by_dataset, events_of, finite_mean, sorted_by_rotation
from .style import INK, MUTED, SERIES, styled

__all__ = [
    "build_figures",
    "build_sections",
    "export_figures",
    "plot_angle_deltas_vs_tilt",
    "plot_convergence_sweeps",
    "plot_coupling_geometry",
    "plot_coupling_segment_heatmap",
    "plot_dataset_summary",
    "plot_epoch_curve",
    "plot_objective_decomposition",
    "plot_orientation_optimization",
    "plot_orientation_search_headroom",
    "plot_orientation_search_trace",
    "plot_refined_rotation_scores",
    "plot_refinement_gain",
    "plot_rotation_cost",
    "plot_rotation_epoch_heatmap",
    "plot_score_distributions",
    "plot_thickness_grid_vs_model",
    "plot_thickness_grids",
    "plot_thickness_heatmap",
    "plot_thickness_model",
]


def _rotation_labels(events: Sequence[Positioned]) -> list[str]:
    return [f"{event.dataset or ''}:{event.rotation_index}" for event in events]


def _thin_ticks(ax: Axes, labels: Sequence[str], *, keep: int = 12) -> None:
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
    trials = events_of(records, ConvergenceTrial)
    if not trials:
        return None
    thresholds = {
        started.pass_index: started.r_factor_threshold
        for started in events_of(records, ConvergencePassStarted)
    }
    controls = sorted({trial.control for trial in trials}, key=_control_rank)
    fig, axes = plt.subplots(
        len(controls), 1, figsize=(9, 3.2 * len(controls)), squeeze=False, sharey=True
    )
    positive = all(trial.r_factor > 0.0 for trial in trials)
    for ax, control in zip(axes[:, 0], controls, strict=True):
        labelled_settled = False
        for pass_index, group in sorted(_by_pass(trials, control).items()):
            ordered = sorted(group, key=lambda trial: trial.trial_index)
            x = [trial.candidate for trial in ordered]
            y = [trial.r_factor for trial in ordered]
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


def _by_pass(trials: Sequence[ConvergenceTrial], control: str) -> dict[int, list[ConvergenceTrial]]:
    grouped: dict[int, list[ConvergenceTrial]] = {}
    for trial in trials:
        if trial.control == control:
            grouped.setdefault(trial.pass_index, []).append(trial)
    return grouped


@styled
def plot_epoch_curve(records: Sequence[EventRecord]) -> Figure | None:
    """Train/validation wR2 and R_obs against refinement epoch."""
    steps = events_of(records, RefinementStep)
    if not steps:
        return None
    x = [step.iteration + 1 for step in steps]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    series: tuple[tuple[str, Callable[[RefinementStep], float | None]], ...] = (
        ("train wR2", lambda step: step.wr2),
        ("train R_obs", lambda step: step.r_obs),
        ("validation wR2", lambda step: step.val_wr2),
        ("validation R_obs", lambda step: step.val_r_obs),
    )
    for label, read in series:
        y = [read(step) for step in steps]
        if any(value is not None for value in y):
            # An epoch that did not report the metric leaves a gap in the line, not a zero.
            ax.plot(
                x,
                [math.nan if value is None else value for value in y],
                marker="o",
                linewidth=1.5,
                label=label,
            )
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_title("Epoch curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig


@styled
def plot_orientation_optimization(records: Sequence[EventRecord]) -> Figure | None:
    """Seed vs fitted score per rotation, with the fitted goniometer angle deltas beneath."""
    fits = sorted_by_rotation(events_of(records, OrientationOptimized))
    if not fits:
        return None
    x = range(len(fits))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(x, [fit.seed_score for fit in fits], marker="o", linewidth=1, label="before")
    axes[0].plot(x, [fit.score for fit in fits], marker="o", linewidth=1, label="after")
    axes[0].set_ylabel("score")
    axes[0].set_title("Orientation optimization")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    angles: tuple[tuple[str, Callable[[OrientationOptimized], float]], ...] = (
        ("alpha", lambda fit: fit.alpha),
        ("beta", lambda fit: fit.beta),
        ("omega", lambda fit: fit.omega),
    )
    for label, read in angles:
        axes[1].plot(x, [read(fit) for fit in fits], marker="o", linewidth=1, label=label)
    axes[1].set_ylabel("delta angle (deg)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    _thin_ticks(axes[1], _rotation_labels(fits))
    fig.tight_layout()
    return fig


# The two final-score metrics every per-rotation refinement figure draws, by name.
_ROTATION_METRICS: tuple[tuple[str, Callable[[RefinedRotationMetrics], float]], ...] = (
    ("wR2", lambda row: row.wr2),
    ("R_obs", lambda row: row.r_obs),
)


@styled
def plot_dataset_summary(records: Sequence[EventRecord]) -> Figure | None:
    """Mean final wR2 / R_obs per dataset, split into training and held-out validation rotations.

    Pooled experiments only -- one dataset has nothing to compare against. The split matters because
    a dataset whose training mean looks fine can still be the one that generalizes worst; averaging
    the two together hides exactly that. The validation bars appear only when the run held rotations
    out (``refinement.split.train_test``), and each tick states its ``train/validation`` rotation
    counts, since a mean over fewer rotations is a different quantity rather than a better one.
    """
    metrics = [row for row in events_of(records, RefinedRotationMetrics) if row.dataset]
    grouped = by_dataset(metrics)
    datasets = sorted(grouped)
    if len(datasets) <= 1:
        return None
    splits = [("train", False)]
    if any(row.is_validation for row in metrics):
        splits.append(("validation", True))
    width = 0.8 / len(splits)
    x = range(len(datasets))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    for ax, (label, read) in zip(axes, _ROTATION_METRICS, strict=True):
        for slot, (split, is_validation) in enumerate(splits):
            means = [
                finite_mean(
                    read(row) for row in grouped[name] if row.is_validation == is_validation
                )
                for name in datasets
            ]
            offset = (slot - (len(splits) - 1) / 2) * width
            ax.bar(
                [value + offset for value in x],
                [math.nan if mean is None else mean for mean in means],
                width=width,
                label=split,
            )
        ax.set_ylabel(label)
        if len(splits) > 1:
            ax.legend()
    counts = [
        (
            sum(not row.is_validation for row in grouped[name]),
            sum(row.is_validation for row in grouped[name]),
        )
        for name in datasets
    ]
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(
        [
            f"{name}\n({train}/{held_out})"
            for name, (train, held_out) in zip(datasets, counts, strict=True)
        ],
        rotation=30,
        ha="right",
    )
    axes[1].set_xlabel("dataset (train/validation rotations)")
    axes[0].set_title("Per-dataset final scores")
    fig.tight_layout()
    return fig


@styled
def plot_refined_rotation_scores(records: Sequence[EventRecord]) -> Figure | None:
    """Final refined wR2 and R_obs per rotation, with held-out rotations marked."""
    metrics = events_of(records, RefinedRotationMetrics)
    if not metrics:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for dataset, group in sorted(by_dataset(metrics).items()):
        ordered = sorted(group, key=lambda row: row.rotation_index)
        x = [row.rotation_index for row in ordered]
        held_out = [row for row in ordered if row.is_validation]
        for ax, (label, read) in zip(axes, _ROTATION_METRICS, strict=True):
            ax.plot(
                x,
                [read(row) for row in ordered],
                marker="o",
                linewidth=1,
                label=dataset or "dataset",
            )
            if held_out:
                ax.scatter(
                    [row.rotation_index for row in held_out],
                    [read(row) for row in held_out],
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
    fits = events_of(records, ThicknessOptimized)
    if not fits:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for fit in fits:
        if not fit.candidate_thicknesses or not fit.candidate_score:
            continue
        # Every curve is one colour on purpose. Letting the prop cycle run would paint a single
        # population of rotations in eight hues and imply a grouping that does not exist.
        ax.plot(
            fit.candidate_thicknesses,
            fit.candidate_score,
            linewidth=0.8,
            alpha=0.35,
            color=SERIES[0],
        )
        ax.scatter([fit.thickness], [fit.score], s=14, color=INK, alpha=0.55, zorder=3)
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
    fits = events_of(records, ThicknessOptimized)
    if not fits:
        return None
    panels = []
    for dataset, group in sorted(by_dataset(fits).items()):
        ordered = sorted(group, key=lambda fit: fit.rotation_index)
        grid = ordered[0].candidate_thicknesses
        rows = [fit for fit in ordered if fit.candidate_thicknesses == grid and fit.candidate_score]
        if len(grid) > 1 and rows:
            panels.append((dataset, grid, rows))
    if not panels:
        return None
    heights = [max(3.0, min(10.0, 0.12 * len(rows) + 2.0)) for _, _, rows in panels]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(9, sum(heights)), squeeze=False, height_ratios=heights
    )
    for ax, (dataset, grid, rows) in zip(axes[:, 0], panels, strict=True):
        matrix = [list(fit.candidate_score) for fit in rows]
        ax.grid(False)  # the house gridlines would draw over the image
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
        # Dark ink, not white: the basin the trace runs through is the *light* end of a
        # lightness-monotonic ramp, so a white line would vanish exactly where it is read.
        ax.plot(
            [fit.thickness for fit in rows],
            range(len(rows)),
            color=INK,
            linewidth=1.2,
            marker=".",
            markersize=3,
            label="fitted thickness",
        )
        indices = [fit.rotation_index for fit in rows]
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
    profiles = events_of(records, ThicknessProfile)
    if not profiles:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for profile in profiles:
        if not profile.alphas or not profile.thicknesses:
            continue
        ordered = sorted(zip(profile.alphas, profile.thicknesses, strict=True))
        ax.plot(
            [row[0] for row in ordered],
            [row[1] for row in ordered],
            marker="o",
            linewidth=1.5,
            label=profile.label or profile.channel,
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
    rows = sorted_by_rotation(events_of(records, RotationCoupling))
    if not rows:
        return None
    x = range(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    series: tuple[tuple[int, str, Callable[[RotationCoupling], int]], ...] = (
        (0, "union beams", lambda row: row.n_union_beams),
        (0, "max beams/segment", lambda row: row.max_beams_per_segment),
        (1, "segments", lambda row: row.n_coupling_segments),
        (1, "max tilts/segment", lambda row: row.max_tilts_per_segment),
    )
    for panel, label, read in series:
        ax = axes[panel]
        ax.plot(x, [read(row) for row in rows], marker="o", linewidth=1, label=label)
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
    traces = events_of(records, OrientationSearchTrace)
    if not traces:
        return None
    trace = max(traces, key=lambda trace: len(trace.score))
    if not trace.score:
        return None
    trial = list(range(len(trace.score)))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    axes[0].plot(trial, trace.score, linewidth=1.2, label=trace.residual)
    if trace.comparable_score != trace.score:
        axes[0].plot(trial, trace.comparable_score, linewidth=1.0, alpha=0.75, label="comparable")
    for flags, marker, label in ((trace.is_seed, "o", "seed"), (trace.is_final, "x", "final")):
        marked = [(t, s) for t, s, flag in zip(trial, trace.score, flags, strict=True) if flag]
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
    for label, values in (("alpha", trace.alpha), ("beta", trace.beta), ("omega", trace.omega)):
        axes[1].plot(trial, values, linewidth=1.0, label=label)
    axes[1].set_xlabel("trial")
    axes[1].set_ylabel("angle delta (deg)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize="small")
    fig.tight_layout()
    return fig


@styled
def plot_coupling_segment_heatmap(records: Sequence[EventRecord]) -> Figure | None:
    """Per-rotation, per-segment beam counts as a heatmap (rows padded with NaN to the widest)."""
    traces = sorted_by_rotation(events_of(records, RotationCouplingSegments))
    if not traces:
        return None
    widest = max(len(trace.n_segment_beams) for trace in traces)
    if widest == 0:
        return None
    matrix = [
        list(trace.n_segment_beams) + [math.nan] * (widest - len(trace.n_segment_beams))
        for trace in traces
    ]
    labels = _rotation_labels(traces)
    fig, ax = plt.subplots(figsize=(10, max(4.0, min(12.0, 0.25 * len(matrix) + 2.0))))
    ax.grid(False)  # the house gridlines would draw over the image
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


def _pooled_index(
    records: Sequence[EventRecord], local: Iterable[Positioned]
) -> dict[tuple[str, int], int]:
    """Map a preprocess event's ``(dataset, file-local rotation_index)`` to the pooled index.

    ``optimize_orientation``/``optimize_thickness`` run per dataset before pooling renumbers the
    rotations, so their ``rotation_index`` is file-local; every refinement-side event carries the
    *pooled* index. ``report_coupling`` fires on the settled pooled plan on every run, so its
    ``RotationCoupling`` events give each dataset's pooled indices in order, and pooling preserves
    order within a dataset -- pairing the two sorted lists positionally is the mapping. A dataset
    whose two counts disagree is left unmapped rather than guessed at.
    """
    pooled = by_dataset(events_of(records, RotationCoupling))
    mapping: dict[tuple[str, int], int] = {}
    for dataset, group in by_dataset(local).items():
        after = sorted(
            {e.rotation_index for e in pooled.get(dataset, []) if e.rotation_index is not None}
        )
        before = sorted({e.rotation_index for e in group if e.rotation_index is not None})
        if after and len(after) == len(before):
            mapping.update({(dataset, b): a for b, a in zip(before, after, strict=True)})
    return mapping


@styled
def plot_objective_decomposition(records: Sequence[EventRecord]) -> Figure | None:
    """The objective per epoch, and what each composed term contributed to it.

    ``RefinementStep.components`` carries every term's ``contribution`` (raw * weight) each epoch;
    stacked, they show whether the restraints or the diffraction term is doing the work -- the
    question every soft-penalty run raises and the epoch curve cannot answer. The upper panel is the
    objective itself with the epoch the run *selected* marked (``RefinementCompleted.best_step``,
    labelled by what selected it), which is not necessarily the last or the lowest-training-loss
    epoch under a validation split.
    """
    steps = events_of(records, RefinementStep)
    if not steps:
        return None
    completed = events_of(records, RefinementCompleted)
    x = [step.iteration + 1 for step in steps]
    terms: list[str] = []
    for step in steps:
        for term in step.components:
            if term not in terms:
                terms.append(term)
    fig, axes = plt.subplots(
        2 if terms else 1, 1, figsize=(8, 7 if terms else 4), sharex=True, squeeze=False
    )
    top = axes[0, 0]
    top.plot(x, [step.loss for step in steps], marker="o", linewidth=1.5, label="objective")
    if completed:
        best = completed[-1]
        selected = next((step for step in steps if step.iteration == best.best_step), None)
        top.axvline(best.best_step + 1, linestyle="--", linewidth=1.0, color=MUTED)
        if selected is not None:
            # The star sits on the training curve at the selected epoch. Under a validation split
            # ``best_loss`` is the *validation* objective -- a different population -- so it is
            # quoted in the label rather than drawn as a point on this curve.
            label = f"selected on {best.selection}"
            if best.selection == "validation":
                label += f" (validation objective {best.best_loss:.4g})"
            top.scatter(
                [best.best_step + 1],
                [selected.loss],
                marker="*",
                s=140,
                zorder=3,
                color=INK,
                label=label,
            )
    top.set_ylabel("objective")
    top.xaxis.set_major_locator(MaxNLocator(integer=True))
    top.set_title("Objective per epoch")
    top.legend(fontsize="small")
    if terms:
        bottom = axes[1, 0]
        contributions = [
            [step.components.get(term, {}).get("contribution", math.nan) for step in steps]
            for term in terms
        ]
        bottom.stackplot(x, contributions, labels=terms, alpha=0.85)
        bottom.set_ylabel("contribution")
        bottom.set_title("Objective composition")
        bottom.legend(fontsize="small", loc="upper right")
    axes[-1, 0].set_xlabel("epoch")
    fig.tight_layout()
    return fig


@styled
def plot_orientation_search_headroom(records: Sequence[EventRecord]) -> Figure | None:
    """Each rotation's Nelder-Mead pass count against the cap, and its matched-reflection count.

    The calibration figure for ``preprocess.orientation.max_iterations``: a rotation that ran to
    the cap (highlighted) stopped because it was told to, not because it converged, and the value
    it settled on is a budget artefact. The lower panel shows the matched-reflection count the
    search ended on and, where the search trace is in the report, the count it *started* from --
    the ``penalize_fewer_reflections`` guard exists to stop the fit "winning" by matching fewer.
    """
    fits = sorted_by_rotation(events_of(records, OrientationOptimized))
    if not fits:
        return None
    seed_matched: dict[tuple[str, int], int] = {}
    for trace in events_of(records, OrientationSearchTrace):
        seeds = [
            n for n, is_seed in zip(trace.n_matched_hkl, trace.is_seed, strict=True) if is_seed
        ]
        if seeds:
            seed_matched[trace.dataset, trace.rotation_index] = seeds[0]
    x = list(range(len(fits)))
    capped = [fit.n_passes >= fit.pass_cap for fit in fits]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(
        x,
        [fit.n_passes for fit in fits],
        color=[SERIES[7] if flag else SERIES[0] for flag in capped],
        label="passes",
    )
    for cap in sorted({fit.pass_cap for fit in fits}):
        axes[0].axhline(cap, linestyle="--", linewidth=1.0, color=MUTED, label=f"cap {cap}")
    if any(capped):
        axes[0].scatter([], [], color=SERIES[7], marker="s", label=f"ran to cap ({sum(capped)})")
    axes[0].set_ylabel("Nelder-Mead passes")
    axes[0].set_title("Orientation search headroom")
    axes[0].legend(fontsize="small")
    axes[1].plot(x, [fit.n_matched_hkl for fit in fits], marker="o", linewidth=1, label="after")
    before = [seed_matched.get((fit.dataset, fit.rotation_index)) for fit in fits]
    if any(value is not None for value in before):
        axes[1].plot(
            x,
            [math.nan if value is None else value for value in before],
            marker="o",
            linewidth=1,
            label="seed",
        )
    axes[1].set_ylabel("matched reflections")
    axes[1].legend(fontsize="small")
    _thin_ticks(axes[1], _rotation_labels(fits))
    fig.tight_layout()
    return fig


@styled
def plot_thickness_grid_vs_model(records: Sequence[EventRecord]) -> Figure | None:
    """The refinement's learned thickness curve with the preprocess grid-search points on it.

    Two stages estimate the same physical quantity: ``optimize_thickness`` picks one scalar per
    rotation by argmin, and ``ApparentThicknessNN`` learns a function of tilt. Drawn on the same
    alpha axis, the points show what the grid search found and the curve what the network settled
    on -- agreement means the NN learned the specimen shape, systematic offset means the refinement
    moved it, scatter far from the curve names the rotations the grid search got wrong. Needs both
    stages in the report; a report with only one is served by the single-stage figures.
    """
    fits = events_of(records, ThicknessOptimized)
    profiles = events_of(records, ThicknessProfile)
    if not fits or not profiles:
        return None
    pooled = _pooled_index(records, fits)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    drawn = False
    for slot, profile in enumerate(profiles):
        colour = SERIES[slot % len(SERIES)]
        alpha_of = dict(zip(profile.rotation_indices, profile.alphas, strict=True))
        curve = sorted(zip(profile.alphas, profile.thicknesses, strict=True))
        ax.plot(
            [row[0] for row in curve],
            [row[1] for row in curve],
            linewidth=1.5,
            color=colour,
            label=f"{profile.label} learned",
        )
        points = [
            (alpha_of[pooled[fit.dataset, fit.rotation_index]], fit.thickness)
            for fit in fits
            if fit.dataset == profile.label
            and (fit.dataset, fit.rotation_index) in pooled
            and pooled[fit.dataset, fit.rotation_index] in alpha_of
        ]
        if points:
            drawn = True
            ax.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                s=18,
                color=colour,
                alpha=0.7,
                label=f"{profile.label} grid search",
            )
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel("alpha (degrees)")
    ax.set_ylabel("thickness")
    ax.set_title("Grid-search thickness vs learned model")
    ax.legend(fontsize="small")
    return fig


@styled
def plot_rotation_epoch_heatmap(records: Sequence[EventRecord]) -> Figure | None:
    """Per-rotation wR2 across epochs (``--verbose-refinement`` runs only).

    The epoch curve is a mean; this is the matrix under it. A few rotations that degrade or
    oscillate while the mean improves are exactly what a mean hides, and they are the frames to
    look at (or exclude) next.
    """
    steps = events_of(records, RefinementOrientationStep)
    if not steps:
        return None
    rows = sorted({(step.dataset, step.rotation_index) for step in steps})
    epochs = sorted({step.iteration for step in steps})
    position = {key: index for index, key in enumerate(rows)}
    column = {epoch: index for index, epoch in enumerate(epochs)}
    matrix = [[math.nan] * len(epochs) for _ in rows]
    for step in steps:
        value = step.wr2 if step.wr2 is not None else step.r_obs
        matrix[position[step.dataset, step.rotation_index]][column[step.iteration]] = (
            math.nan if value is None else value
        )
    fig, ax = plt.subplots(figsize=(9, max(3.5, min(12.0, 0.2 * len(rows) + 2.0))))
    ax.grid(False)  # the house gridlines would draw over the image
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([str(epoch + 1) for epoch in epochs])
    stride = max(1, len(rows) // 20)
    ax.set_yticks(range(len(rows))[::stride])
    ax.set_yticklabels([f"{dataset}:{index}" for dataset, index in rows][::stride])
    ax.set_xlabel("epoch")
    ax.set_ylabel("dataset:rotation")
    ax.set_title("Per-rotation wR2 across epochs")
    fig.colorbar(image, ax=ax, label="wR2")
    fig.tight_layout()
    return fig


@styled
def plot_refinement_gain(records: Sequence[EventRecord]) -> Figure | None:
    """Each rotation's score after the orientation search vs after refinement.

    The same shape as the seed-vs-fitted panel of the orientation figure, one stage later: what the
    structure refinement bought (or cost) per rotation, on top of the geometry preprocessing
    settled. Compared under the residual the search actually scored, so a ``robs`` search is set
    against the refined ``R_obs``, not ``wR2``.
    """
    fits = sorted_by_rotation(events_of(records, OrientationOptimized))
    refined = events_of(records, RefinedRotationMetrics)
    if not fits or not refined:
        return None
    pooled = _pooled_index(records, fits)
    final = {(row.dataset, row.rotation_index): row for row in refined}
    pairs = [
        (fit, final[fit.dataset, pooled[fit.dataset, fit.rotation_index]])
        for fit in fits
        if (fit.dataset, fit.rotation_index) in pooled
        and (fit.dataset, pooled[fit.dataset, fit.rotation_index]) in final
    ]
    if not pairs:
        return None
    residual = pairs[0][0].residual
    after = [(row.r_obs if residual == "robs" else row.wr2) for _, row in pairs]
    x = list(range(len(pairs)))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        x,
        [fit.score for fit, _ in pairs],
        marker="o",
        linewidth=1,
        label="after orientation search",
    )
    ax.plot(x, after, marker="o", linewidth=1, label="after refinement")
    held_out = [
        (i, value)
        for i, (value, (_, row)) in enumerate(zip(after, pairs, strict=True))
        if row.is_validation
    ]
    if held_out:
        ax.scatter(
            [i for i, _ in held_out],
            [value for _, value in held_out],
            marker="x",
            s=50,
            linewidths=1.5,
            color=INK,
            label="validation",
        )
    ax.set_ylabel("R_obs" if residual == "robs" else "wR2")
    ax.set_title("What refinement bought, per rotation")
    ax.legend(fontsize="small")
    _thin_ticks(ax, _rotation_labels([fit for fit, _ in pairs]))
    fig.tight_layout()
    return fig


@styled
def plot_score_distributions(records: Sequence[EventRecord]) -> Figure | None:
    """Empirical CDFs of the final per-rotation wR2 and R_obs, train vs validation per dataset.

    At a hundred rotations the per-index line plots are a hairball; a distribution reads at any
    count, and putting the held-out rotations on the same axes makes the generalization gap the
    first thing seen rather than something to infer from marker positions.
    """
    metrics = events_of(records, RefinedRotationMetrics)
    if not metrics:
        return None
    groups: dict[tuple[str, str], list[RefinedRotationMetrics]] = {}
    for row in metrics:
        groups.setdefault((row.dataset, "validation" if row.is_validation else "train"), []).append(
            row
        )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (label, read) in zip(axes, _ROTATION_METRICS, strict=True):
        for slot, ((dataset, split), rows) in enumerate(sorted(groups.items())):
            values = sorted(v for v in (read(row) for row in rows) if math.isfinite(v))
            if not values:
                continue
            # Markers as well as the step: a group of one rotation has no step to draw, and a
            # held-out set of two or three is common enough that it must still be visible.
            ax.step(
                values,
                [(i + 1) / len(values) for i in range(len(values))],
                where="post",
                linewidth=1.5,
                marker="o",
                markersize=4,
                color=SERIES[slot % len(SERIES)],
                linestyle="-" if split == "train" else ":",
                label=f"{dataset or 'dataset'} {split} (n={len(values)})",
            )
        ax.set_xlabel(label)
        ax.set_ylabel("fraction of rotations")
        ax.set_ylim(0, 1)
        ax.legend(fontsize="small")
    axes[0].set_title("Final per-rotation score distributions")
    fig.tight_layout()
    return fig


@styled
def plot_angle_deltas_vs_tilt(records: Sequence[EventRecord]) -> Figure | None:
    """The fitted goniometer corrections against the rotation's tilt angle.

    Against rotation *index* (the orientation figure) the deltas are a sequence; against alpha they
    are a function. A smooth trend with tilt is a systematic misalignment the search is correcting
    the same way every time -- worth fixing at the source -- while scatter is per-frame noise. The
    tilt angles come from ``ThicknessProfile`` (the one event that carries alpha per rotation), so
    this needs a thickness-NN refinement in the report.
    """
    fits = events_of(records, OrientationOptimized)
    profiles = events_of(records, ThicknessProfile)
    if not fits or not profiles:
        return None
    pooled = _pooled_index(records, fits)
    alpha_of: dict[tuple[str, int], float] = {}
    for profile in profiles:
        for index, alpha in zip(profile.rotation_indices, profile.alphas, strict=True):
            alpha_of[profile.label, index] = alpha
    located = [
        (alpha_of[fit.dataset, pooled[fit.dataset, fit.rotation_index]], fit)
        for fit in fits
        if (fit.dataset, fit.rotation_index) in pooled
        and (fit.dataset, pooled[fit.dataset, fit.rotation_index]) in alpha_of
    ]
    if not located:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    deltas: tuple[tuple[str, Callable[[OrientationOptimized], float]], ...] = (
        ("alpha", lambda fit: fit.alpha),
        ("beta", lambda fit: fit.beta),
        ("omega", lambda fit: fit.omega),
    )
    datasets = sorted({fit.dataset for _, fit in located})
    for ax, (label, read) in zip(axes, deltas, strict=True):
        for slot, dataset in enumerate(datasets):
            points = [(tilt, read(fit)) for tilt, fit in located if fit.dataset == dataset]
            ax.scatter(
                [p[0] for p in points],
                [p[1] for p in points],
                s=16,
                color=SERIES[slot % len(SERIES)],
                label=dataset or "dataset",
            )
        ax.axhline(0.0, linewidth=0.8, color=MUTED)
        ax.set_ylabel(f"delta {label} (deg)")
    axes[0].set_title("Goniometer corrections vs tilt")
    if len(datasets) > 1:
        axes[0].legend(fontsize="small")
    axes[-1].set_xlabel("tilt alpha (degrees)")
    fig.tight_layout()
    return fig


@styled
def plot_rotation_cost(records: Sequence[EventRecord]) -> Figure | None:
    """Wall time per orientation search, and against the size of the solve it required.

    Derived from the envelope alone: consecutive ``OrientationOptimized`` records' timestamps (the
    first measured from ``OrientationOptimizationStarted``). With ``workers > 1`` events arrive in
    completion order and the deltas are inter-arrival gaps, not per-rotation durations -- the bars
    then read as throughput, not cost. The right panel puts each rotation's time against its
    coupled-solve basis size (``RotationCoupling.n_union_beams``), which is the cost model the
    memory and budget knobs are set against.
    """
    ordered = [
        record
        for record in records
        if record.event_type
        in {OrientationOptimized.__name__, OrientationOptimizationStarted.__name__}
    ]
    stamps = [datetime.fromisoformat(record.timestamp_utc) for record in ordered]
    costs: list[tuple[OrientationOptimized, float]] = []
    for record, t0, t1 in zip(ordered[1:], stamps, stamps[1:], strict=False):
        if record.event_type != OrientationOptimized.__name__:
            continue
        fit = event_from_record(record)
        assert isinstance(fit, OrientationOptimized)
        costs.append((fit, (t1 - t0).total_seconds()))
    if not costs:
        return None
    pooled = _pooled_index(records, [fit for fit, _ in costs])
    beams = {
        (row.dataset, row.rotation_index): row.n_union_beams
        for row in events_of(records, RotationCoupling)
    }
    sized = [
        (beams[fit.dataset, pooled[fit.dataset, fit.rotation_index]], seconds)
        for fit, seconds in costs
        if (fit.dataset, fit.rotation_index) in pooled
        and (fit.dataset, pooled[fit.dataset, fit.rotation_index]) in beams
    ]
    fig, axes = plt.subplots(1, 2 if sized else 1, figsize=(12 if sized else 8, 4.2), squeeze=False)
    bars = axes[0, 0]
    bars.bar(range(len(costs)), [seconds for _, seconds in costs], color=SERIES[0])
    bars.set_ylabel("seconds")
    bars.set_title(f"Orientation search time per rotation (total {sum(s for _, s in costs):.0f} s)")
    _thin_ticks(bars, _rotation_labels([fit for fit, _ in costs]))
    if sized:
        scatter = axes[0, 1]
        scatter.scatter([b for b, _ in sized], [s for _, s in sized], s=18, color=SERIES[0])
        scatter.set_xlabel("union beams in the solve")
        scatter.set_ylabel("seconds")
        scatter.set_title("Time vs solve size")
    fig.tight_layout()
    return fig


@dataclass(frozen=True)
class Section:
    """One headed group of figures, and the event types that place it in the run."""

    title: str
    builders: tuple[tuple[str, Callable[[Sequence[EventRecord]], Figure | None]], ...]
    event_types: tuple[type, ...]


# Stage groups, each headed separately rather than pooled into one flat run of figures. Their
# *order* is not hardcoded: `preprocess.stage_order` can put the thickness fit before the
# orientation fit, so the sections are sorted by where their events actually appear in the report
# (see `build_sections`). The listing order below is only the fallback for a tie.
SECTIONS = (
    Section(
        "Convergence",
        (("convergence_sweeps", plot_convergence_sweeps),),
        (ConvergenceTrial,),
    ),
    Section(
        "Preprocess — orientation optimization",
        (
            ("orientation_optimization", plot_orientation_optimization),
            ("orientation_search_headroom", plot_orientation_search_headroom),
            ("orientation_search_trace", plot_orientation_search_trace),
            ("angle_deltas_vs_tilt", plot_angle_deltas_vs_tilt),
            ("rotation_cost", plot_rotation_cost),
        ),
        (OrientationOptimized, OrientationSearchTrace),
    ),
    Section(
        # "per-rotation" earns its place: the refinement stage has a thickness section too, and
        # that one is a learned function of tilt rather than one fitted scalar per rotation.
        "Preprocess — per-rotation thickness fit",
        (
            ("thickness_grids", plot_thickness_grids),
            ("thickness_heatmap", plot_thickness_heatmap),
        ),
        (ThicknessOptimized,),
    ),
    Section(
        "Preprocess — coupled solve geometry",
        (
            ("coupling_geometry", plot_coupling_geometry),
            ("coupling_segment_heatmap", plot_coupling_segment_heatmap),
        ),
        (RotationCoupling, RotationCouplingSegments),
    ),
    Section(
        "Refinement — epoch history",
        (
            ("epoch_curve", plot_epoch_curve),
            ("objective_decomposition", plot_objective_decomposition),
            ("rotation_epoch_heatmap", plot_rotation_epoch_heatmap),
        ),
        (RefinementStep, RefinementOrientationStep),
    ),
    Section(
        "Refinement — per-rotation scores",
        (
            ("refined_rotation_scores", plot_refined_rotation_scores),
            ("refinement_gain", plot_refinement_gain),
            ("score_distributions", plot_score_distributions),
        ),
        (RefinedRotationMetrics,),
    ),
    Section(
        "Refinement — datasets",
        (("per_dataset_summary", plot_dataset_summary),),
        (RefinedRotationMetrics,),
    ),
    Section(
        "Refinement — learned thickness model",
        (
            ("thickness_model", plot_thickness_model),
            ("thickness_grid_vs_model", plot_thickness_grid_vs_model),
        ),
        (ThicknessProfile,),
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
                (
                    first_seen[cls.__name__]
                    for cls in item[1].event_types
                    if cls.__name__ in first_seen
                ),
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
