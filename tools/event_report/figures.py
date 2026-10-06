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

import base64
import html
import io
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from diffBloch.observability import (
    ConvergencePassStarted,
    ConvergenceTrial,
    EventRecord,
    OrientationOptimized,
    OrientationSearchTrace,
    RefinedRotationMetrics,
    RefinementOrientationStep,
    RefinementStep,
    ThicknessOptimized,
    ThicknessProfile,
)

from .reader import Positioned, by_dataset, events_of, finite_mean, sorted_by_rotation
from .style import INK, MUTED, REPORT_RC, SERIES, styled

__all__ = [
    "build_figures",
    "build_sections",
    "export_figures",
    "figure_dropdown_html",
    "plot_convergence_sweeps",
    "plot_dataset_summary",
    "plot_epoch_curve",
    "plot_orientation_optimization",
    "plot_matched_reflections",
    "plot_refined_rotation_scores",
    "plot_rotation_epoch_heatmap",
    "plot_structure_drift",
    "plot_thickness_grids",
    "plot_thickness_model",
]


def _rotation_labels(events: Sequence[Positioned]) -> list[str]:
    """One label per rotation: its exact index, prefixed by the dataset only when there are several.

    A single-dataset report (the common case) reads ``0, 1, 2, ...`` -- the PETS frame numbers
    themselves -- rather than repeating one file name a hundred times.
    """
    datasets = {event.dataset or "" for event in events}
    if len(datasets) <= 1:
        return [str(event.rotation_index) for event in events]
    return [f"{event.dataset or ''}:{event.rotation_index}" for event in events]


def _rotation_axis_title(events: Sequence[Positioned]) -> str:
    """The axis title matching :func:`_rotation_labels`: plain ``rotation`` for one dataset."""
    return (
        "Orientation"
        if len({event.dataset or "" for event in events}) <= 1
        else "Dataset:orientation"
    )


# Every figure is drawn at one width. The notebook scales each image to the page width, so a
# shared figure width is what makes the text render at the same size in every plot.
FIGURE_WIDTH = 12.0

# Orientation and epoch axes carry a tick at every index but a number only at every fifth.
_LABEL_EVERY = 5

# Inches per row of a vertical orientation axis (the per-orientation epoch heatmap).
_INCHES_PER_ROTATION = 0.3


def _rotation_extent(n_rotations: int, *, minimum: float) -> float:
    """The figure height that fits one label per orientation row."""
    return max(minimum, _INCHES_PER_ROTATION * n_rotations + 2.0)


def _index_ticks(
    axis: Axis,
    positions: Sequence[float],
    values: Sequence[int],
    labels: Sequence[str] | None = None,
) -> None:
    """Tick every index on ``axis``; number only those whose value is a multiple of five.

    ``values`` are the indices themselves (orientation or epoch numbers) and decide which ticks
    get a label; ``labels`` (default: the values) is the text shown. An axis with fewer than five
    entries, or none on a multiple of five, labels every tick.
    """
    shown = list(labels) if labels is not None else [str(value) for value in values]
    keep = [i for i, value in enumerate(values) if value % _LABEL_EVERY == 0]
    if len(values) < _LABEL_EVERY or not keep:
        keep = list(range(len(values)))
    axis.set_ticks([positions[i] for i in keep])
    axis.set_ticklabels([shown[i] for i in keep])
    axis.set_ticks(list(positions), minor=True)


def _label_every_rotation(ax: Axes, events: Sequence[Positioned]) -> None:
    """Tick the orientation x axis at positions ``0..n-1``, numbered by orientation index."""
    _index_ticks(
        ax.xaxis,
        range(len(events)),
        [event.rotation_index or 0 for event in events],
        _rotation_labels(events),
    )
    ax.set_xlim(-0.75, len(events) - 0.25)


# Axis-label forms of the metric and control names, set as mathtext so subscripts render as
# subscripts rather than as a literal underscore.
_WR2 = r"$wR_2$"
_R_OBS = r"$R_{\mathrm{obs}}$"
# Each metric keeps one colour in every figure: wR2 red, R_obs blue.
_WR2_COLOR = SERIES[7]
_R_OBS_COLOR = SERIES[0]
_ALPHA = r"$\alpha$ (°)"


def _residual_axis(residual: str) -> tuple[str, str]:
    """The axis label and colour for a preprocessing search's configured residual."""
    return (_R_OBS, _R_OBS_COLOR) if residual == "robs" else (_WR2, _WR2_COLOR)


_ANGLE_DELTA = r"$\Delta$ angle (°)"
_GONIOMETER_ANGLES = (r"$\alpha$", r"$\beta$", r"$\omega$")
_CONTROL_LABELS = {
    "g_max": r"$g_{\mathrm{max}}$",
    "sg_max": r"$s_{g,\mathrm{max}}$",
    "tilt_steps": "tilt steps",
}

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
        len(controls), 1, figsize=(FIGURE_WIDTH, 3.6 * len(controls)), squeeze=False, sharey=True
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
        # R1 between the simulations at the previous and candidate settings, not against data.
        ax.set_ylabel(r"$R_1$ vs previous setting")
        ax.set_xlabel(f"Candidate {_CONTROL_LABELS.get(control, control)}")
        ax.legend(fontsize="small")
    fig.set_label("Convergence sweeps")
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
    """wR2 (upper axes) and R_obs (lower axes) against refinement epoch.

    The training curve is solid; a validation curve, when the run held orientations out, is dotted.
    """
    steps = events_of(records, RefinementStep)
    if not steps:
        return None
    x = [step.iteration + 1 for step in steps]
    fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 8), sharex=True)
    metrics: tuple[
        tuple[
            str,
            Callable[[RefinementStep], float | None],
            Callable[[RefinementStep], float | None],
        ],
        ...,
    ] = (
        (_WR2, lambda step: step.wr2, lambda step: step.val_wr2),
        (_R_OBS, lambda step: step.r_obs, lambda step: step.val_r_obs),
    )
    colors = (_WR2_COLOR, _R_OBS_COLOR)
    has_validation = any(step.val_wr2 is not None or step.val_r_obs is not None for step in steps)
    for ax, color, (label, train, validation) in zip(axes, colors, metrics, strict=True):
        for read, linestyle, split in ((train, "-", "train"), (validation, ":", "validation")):
            y = [read(step) for step in steps]
            if any(value is not None for value in y):
                # An epoch that did not report the metric leaves a gap in the line, not a zero.
                ax.plot(
                    x,
                    [math.nan if value is None else value for value in y],
                    marker="o",
                    linewidth=1.5,
                    linestyle=linestyle,
                    color=color,
                    label=split,
                )
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        if has_validation:
            ax.legend()
    axes[-1].set_xlabel("Epoch")
    _index_ticks(axes[-1].xaxis, x, x)
    axes[-1].set_xlim(min(x) - 0.5, max(x) + 0.5)
    fig.set_label("Epoch curve")
    fig.tight_layout()
    return fig


@styled
def plot_orientation_optimization(records: Sequence[EventRecord]) -> Figure | None:
    """Seed vs fitted score per rotation, with the fitted goniometer angle deltas beneath."""
    fits = sorted_by_rotation(events_of(records, OrientationOptimized))
    if not fits:
        return None
    x = range(len(fits))
    fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 8), sharex=True)
    metric, color = _residual_axis(fits[0].residual)
    axes[0].plot(
        x, [fit.seed_score for fit in fits], marker="o", linewidth=1, color=MUTED, label="initial"
    )
    axes[0].plot(
        x, [fit.score for fit in fits], marker="o", linewidth=1, color=color, label="final"
    )
    axes[0].set_ylabel(metric)
    axes[0].set_ylim(bottom=0)
    axes[0].legend()
    angles: tuple[tuple[str, Callable[[OrientationOptimized], float]], ...] = (
        (_GONIOMETER_ANGLES[0], lambda fit: fit.alpha),
        (_GONIOMETER_ANGLES[1], lambda fit: fit.beta),
        (_GONIOMETER_ANGLES[2], lambda fit: fit.omega),
    )
    for label, read in angles:
        axes[1].plot(x, [read(fit) for fit in fits], marker="o", linewidth=1, label=label)
    axes[1].set_ylabel(_ANGLE_DELTA)
    axes[1].set_xlabel(_rotation_axis_title(fits))
    axes[1].legend()
    _label_every_rotation(axes[1], fits)
    fig.set_label("Orientation optimization")
    fig.tight_layout()
    return fig


@styled
def plot_structure_drift(records: Sequence[EventRecord]) -> Figure | None:
    """How far the structure has moved from the starting model, per epoch, side by side.

    Left: Cartesian RMSD of the asymmetric-unit atom positions (Angstrom). Right: RMSD of each
    atom's equivalent isotropic ADP (Angstrom^2). Each point is the structure after that epoch's
    update, measured against the starting model.
    """
    steps = [
        step
        for step in events_of(records, RefinementStep)
        if step.position_rmsd is not None and step.ueq_rmsd is not None
    ]
    if not steps:
        return None
    x = [step.iteration + 1 for step in steps]
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 5))
    panels: tuple[tuple[str, list[float | None]], ...] = (
        ("Coordinate RMSD (Å)", [step.position_rmsd for step in steps]),
        (r"$U_{\mathrm{eq}}$ RMSD (Å$^2$)", [step.ueq_rmsd for step in steps]),
    )
    for ax, (label, values) in zip(axes, panels, strict=True):
        ax.plot(x, values, marker="o", linewidth=1.5, color=INK)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        _index_ticks(ax.xaxis, x, x)
        ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
    fig.set_label("Structure change from starting model")
    fig.tight_layout()
    return fig


# The two final-score metrics every per-rotation refinement figure draws, by name.
_ROTATION_METRICS: tuple[tuple[str, str, Callable[[RefinedRotationMetrics], float]], ...] = (
    (_WR2, _WR2_COLOR, lambda row: row.wr2),
    (_R_OBS, _R_OBS_COLOR, lambda row: row.r_obs),
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
    fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 8), sharex=True)
    for ax, (label, color, read) in zip(axes, _ROTATION_METRICS, strict=True):
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
                color=color,
                # Held-out bars are the same colour, lighter, beside the training bar.
                alpha=0.45 if is_validation else 1.0,
                label=split,
            )
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
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
        ha="center",
    )
    axes[1].set_xlabel("Dataset (train/validation orientations)")
    fig.set_label("Per-dataset final scores")
    fig.tight_layout()
    return fig


@styled
def plot_refined_rotation_scores(records: Sequence[EventRecord]) -> Figure | None:
    """Final refined wR2 (upper axes) and R_obs (lower axes) per orientation.

    Training orientations are circles; held-out validation orientations are crosses, with a legend
    whenever the run held any out.
    """
    metrics = events_of(records, RefinedRotationMetrics)
    if not metrics:
        return None
    has_validation = any(row.is_validation for row in metrics)
    fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 8), sharex=True)
    for slot, (_, group) in enumerate(sorted(by_dataset(metrics).items())):
        ordered = sorted(group, key=lambda row: row.rotation_index)
        x = [row.rotation_index for row in ordered]
        train = [row for row in ordered if not row.is_validation]
        held_out = [row for row in ordered if row.is_validation]
        for ax, (label, color, read) in zip(axes, _ROTATION_METRICS, strict=True):
            ax.plot(x, [read(row) for row in ordered], linewidth=1, color=color)
            ax.scatter(
                [row.rotation_index for row in train],
                [read(row) for row in train],
                marker="o",
                s=25,
                color=color,
                zorder=3,
                label="train" if slot == 0 else None,
            )
            if held_out:
                ax.scatter(
                    [row.rotation_index for row in held_out],
                    [read(row) for row in held_out],
                    marker="x",
                    s=70,
                    linewidths=2,
                    color=INK,
                    zorder=4,
                    label="validation" if slot == 0 else None,
                )
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
    if has_validation:
        for ax in axes:
            ax.legend()
    axes[1].set_xlabel("Orientation")
    # The x axis is the orientation index itself here, so tick the indices that exist rather than
    # matplotlib's round numbers.
    indices = sorted({row.rotation_index for row in metrics})
    _index_ticks(axes[1].xaxis, indices, indices)
    fig.set_label("Final refined per-orientation scores")
    fig.tight_layout()
    return fig


@styled
def plot_thickness_grids(records: Sequence[EventRecord]) -> Figure | None:
    """One small panel per orientation: the thickness grid search's residual against thickness.

    Every panel shares both axes so the orientations compare directly, the y axis starts at 0, and
    the selected thickness is marked. The orientation is named inside its panel.
    """
    fits = sorted_by_rotation(
        fit for fit in events_of(records, ThicknessOptimized) if fit.candidate_score
    )
    if not fits:
        return None
    metric, color = _residual_axis(fits[0].residual)
    labels = _rotation_labels(fits)
    n_cols = min(6, len(fits))
    n_rows = math.ceil(len(fits) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(FIGURE_WIDTH, 2.0 * n_rows + 1.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, fit, label in zip(axes.flat, fits, labels, strict=False):
        ax.plot(fit.candidate_thicknesses, fit.candidate_score, linewidth=1.5, color=color)
        ax.scatter([fit.thickness], [fit.score], s=30, color=INK, zorder=3)
        ax.text(0.04, 0.92, f"Orientation {label}", transform=ax.transAxes, fontsize=14, va="top")
        ax.xaxis.set_major_locator(MaxNLocator(3))
    for ax in list(axes.flat)[len(fits) :]:
        ax.set_visible(False)
    # Headroom above the highest curve keeps the in-panel orientation label clear of the data.
    highest = max(max(fit.candidate_score) for fit in fits)
    axes[0, 0].set_ylim(0, 1.3 * highest)
    fig.supxlabel("Thickness (Å)", fontsize=20)
    fig.supylabel(metric, fontsize=20)
    fig.set_label("Thickness fit per orientation")
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
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 5))
    for profile in profiles:
        if not profile.alphas or not profile.thicknesses:
            continue
        ordered = sorted(zip(profile.alphas, profile.thicknesses, strict=True))
        ax.plot(
            [row[0] for row in ordered],
            [row[1] for row in ordered],
            marker="o",
            linewidth=1.5,
        )
    ax.set_xlabel(_ALPHA)
    ax.set_ylabel("Learned Thickness (Å)")
    ax.set_ylim(bottom=0)
    fig.set_label("Learned thickness model")
    fig.tight_layout()
    return fig


@styled
def plot_matched_reflections(records: Sequence[EventRecord]) -> Figure | None:
    """Each orientation's matched-reflection count at the start and end of the orientation search.

    The ``penalize_fewer_reflections`` guard exists to stop the fit "winning" by matching fewer
    reflections; this shows whether the final orientation matches fewer than the initial one. The
    initial counts come from the search trace and are drawn only when it is in the report.
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
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 5))
    initial = [seed_matched.get((fit.dataset, fit.rotation_index)) for fit in fits]
    if any(value is not None for value in initial):
        ax.plot(
            x,
            [math.nan if value is None else value for value in initial],
            marker="o",
            linewidth=1,
            color=MUTED,
            label="initial",
        )
    ax.plot(
        x,
        [fit.n_matched_hkl for fit in fits],
        marker="o",
        linewidth=1,
        color=SERIES[0],
        label="final",
    )
    ax.set_ylabel("Matched reflections")
    ax.set_xlabel(_rotation_axis_title(fits))
    ax.legend(fontsize="small")
    _label_every_rotation(ax, fits)
    fig.set_label("Matched reflections")
    fig.tight_layout()
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
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, _rotation_extent(len(rows), minimum=3.5)))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    _index_ticks(ax.xaxis, range(len(epochs)), [epoch + 1 for epoch in epochs])
    single = len({dataset for dataset, _ in rows}) <= 1
    _index_ticks(
        ax.yaxis,
        range(len(rows)),
        [index for _, index in rows],
        [str(index) if single else f"{dataset}:{index}" for dataset, index in rows],
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Orientation" if single else "Dataset:orientation")
    fig.colorbar(image, ax=ax, label=_WR2)
    fig.set_label("Per-orientation wR2 across epochs")
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
            ("matched_reflections", plot_matched_reflections),
        ),
        (OrientationOptimized, OrientationSearchTrace),
    ),
    Section(
        # "per-orientation" earns its place: the refinement stage has a thickness section too, and
        # that one is a learned function of tilt rather than one fitted scalar per rotation.
        "Preprocess — per-orientation thickness fit",
        (("thickness_grids", plot_thickness_grids),),
        (ThicknessOptimized,),
    ),
    Section(
        "Refinement — epoch history",
        (
            ("epoch_curve", plot_epoch_curve),
            ("structure_drift", plot_structure_drift),
            ("rotation_epoch_heatmap", plot_rotation_epoch_heatmap),
        ),
        (RefinementStep, RefinementOrientationStep),
    ),
    Section(
        "Refinement — per-orientation scores",
        (("refined_rotation_scores", plot_refined_rotation_scores),),
        (RefinedRotationMetrics,),
    ),
    Section(
        "Refinement — datasets",
        (("per_dataset_summary", plot_dataset_summary),),
        (RefinedRotationMetrics,),
    ),
    Section(
        "Refinement — learned thickness model",
        (("thickness_model", plot_thickness_model),),
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


def figure_dropdown_html(figure: Figure) -> str:
    """The figure as a collapsible HTML block: its title (the figure's label) is the toggle.

    Plots carry no drawn title; each ``plot_*`` names its figure with ``Figure.set_label`` and the
    name is shown here, above the image, as the header that opens and closes it.
    """
    buffer = io.BytesIO()
    # Fonts and mathtext resolve at draw time, so the style has to be active here too.
    with plt.rc_context(cast(Any, REPORT_RC)):
        figure.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    title = html.escape(str(figure.get_label() or "Figure"))
    return (
        "<details open>"
        '<summary style="font-family: Arial, sans-serif; font-size: 20px; cursor: pointer;">'
        f"{title}</summary>"
        f'<img src="data:image/png;base64,{encoded}" style="max-width: 100%;">'
        "</details>"
    )


def export_figures(
    figures: dict[str, Figure], output_dir: Path, formats: Sequence[str] = ("svg",)
) -> list[Path]:
    """Write each figure to ``output_dir``. The only place this package writes image files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, figure in figures.items():
        for suffix in formats:
            path = output_dir / f"{name}.{suffix}"
            with plt.rc_context(cast(Any, REPORT_RC)):
                figure.savefig(path, bbox_inches="tight", dpi=160)
            written.append(path)
    return written
