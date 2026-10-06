"""Summary tables over a diffBloch JSONL event report.

The run-level facts a figure cannot carry: which preprocess stages ran and with what settings, and
the refined run's headline numbers at the epoch it selected. Each ``*_table`` takes the parsed
records and returns ``(label, value)`` rows, or ``None`` when the report carries no events of that
kind -- the same contract as :mod:`tools.event_report.figures`. :func:`build_tables` renders the
non-empty ones to Markdown for the notebook.

Values are formatted here, not in the notebook, so the rendering is imported and tested rather than
living in a cell nothing in CI runs. Like the figures, the tables read typed events
(:func:`~tools.event_report.reader.events_of`), so a field this module names is one the event
actually has.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from diffBloch.observability import (
    EventRecord,
    ExperimentDeclared,
    PreprocessCompleted,
    RefinementCompleted,
    RefinementStep,
)

from .reader import events_of, records_of

__all__ = ["build_tables", "markdown_table", "preprocess_table", "refinement_table"]

Rows = list[tuple[str, str]]

# The fitting stages a run may or may not include, each with the key of its *own* spec. The other
# params a step records (``coupling``, ``absorption``) are shared composition context that other
# steps are configured with too, not a setting of this step, so they are not repeated per stage.
_FIT_STAGES = (
    ("Orientation optimization", "optimize_orientation", "search"),
    ("Thickness optimization", "optimize_thickness", "grid"),
)


def _general(value: float | None) -> str:
    """A finite number to six significant figures, or ``n/a``."""
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.6g}"


def _leaf(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _flatten(prefix: str, value: object) -> Rows:
    """``key=value`` rows from a serialized spec, dropping the ``__type__`` provenance tags."""
    if isinstance(value, Mapping):
        inner = {key: entry for key, entry in value.items() if key != "__type__"}
        if not inner:
            return [(prefix, "none")]
        return [
            row
            for key, entry in inner.items()
            for row in _flatten(f"{prefix}.{key}" if prefix else str(key), entry)
        ]
    return [(prefix, _leaf(value))]


def preprocess_table(records: Sequence[EventRecord]) -> Rows | None:
    """The settled plan's counts, and whether each fitting stage ran with which settings.

    A stage that did not run says so rather than being absent: "not in the recipe" and "missing from
    the report" must not look the same. For the same reason a report written before
    ``PreprocessCompleted`` carried ``steps`` gets one "not recorded" row, never a false "not run".
    """
    completed = events_of(records, PreprocessCompleted)
    if not completed:
        return None
    event = completed[-1]
    rows: Rows = [
        ("Rotations", str(event.n_rotations)),
        ("Stages", str(event.n_stages)),
        ("Total HKLs", str(event.total_hkl)),
        ("Matched HKLs", str(event.matched_hkl)),
    ]
    # ``steps`` defaults to () on the event, so once typed an absent key is indistinguishable
    # from "no stage ran". The one place the envelope is consulted: was it written at all?
    if "steps" not in records_of(records, "PreprocessCompleted")[-1].payload:
        rows.append(("Stage settings", "not recorded in this report"))
        return rows
    steps = dict(event.steps)
    for label, step_name, own_key in _FIT_STAGES:
        if step_name not in steps:
            rows.append((label, "not run"))
            continue
        rows.append((label, "ran"))
        params = steps[step_name]
        own = params.get(own_key) if isinstance(params, Mapping) else None
        rows.extend(_flatten(own_key, own) if own is not None else [(own_key, "none")])
    return rows


def _mean_percent(value: float | None, evaluated: int | None, total: int | None) -> str:
    """A mean as a percentage with the rotation count it was taken over, e.g. ``4.21 [97/99]``.

    Every mean states its denominator: a mean over fewer rotations is a different quantity, not a
    better one. A non-finite mean means nothing was evaluated, which the count already says.
    """
    if value is None:
        return "n/a"
    rendered = f"{100.0 * value:.2f}" if math.isfinite(value) else "n/a"
    if evaluated is None or total is None:
        return rendered
    return f"{rendered} [{evaluated}/{total}]"


def refinement_table(records: Sequence[EventRecord]) -> Rows | None:
    """The refined run's headline numbers at the epoch it selected.

    With a held-out validation set the training and validation means are reported side by side;
    without one there is a single population, so the rows carry no train/val prefix.
    """
    completed_events = events_of(records, RefinementCompleted)
    if not completed_events:
        return None
    completed = completed_events[-1]
    best = next(
        (
            step
            for step in events_of(records, RefinementStep)
            if step.iteration == completed.best_step
        ),
        None,
    )
    declared = events_of(records, ExperimentDeclared)
    experiment = declared[-1] if declared else None

    rows: Rows = []
    if experiment is not None:
        rows.append(("Experiment", experiment.name))
    rows += [
        ("Best epoch", f"{completed.best_step + 1} / {completed.n_steps}"),
        ("Selected on", completed.selection),
        ("Objective", _general(completed.best_loss)),
    ]
    if experiment is not None:
        rows += [
            ("Optimizer", experiment.optimizer),
            ("Learning rate", _leaf(experiment.learning_rate)),
        ]

    has_validation = best is not None and best.val_wr2 is not None
    prefix = "Train " if has_validation else ""
    if best is None:
        rows += [(f"{prefix}wR2 (%)", "n/a"), (f"{prefix}R_obs (%)", "n/a")]
    else:
        rows += [
            (
                f"{prefix}wR2 (%)",
                _mean_percent(best.wr2, best.n_wr2_evaluated, best.n_rotations),
            ),
            (
                f"{prefix}R_obs (%)",
                _mean_percent(best.r_obs, best.n_r_obs_evaluated, best.n_rotations),
            ),
        ]
        if has_validation:
            rows += [
                (
                    "Val wR2 (%)",
                    _mean_percent(best.val_wr2, best.val_n_wr2_evaluated, best.val_n_rotations),
                ),
                (
                    "Val R_obs (%)",
                    _mean_percent(best.val_r_obs, best.val_n_r_obs_evaluated, best.val_n_rotations),
                ),
            ]
    rows.append(("Diffraction loss", _general(None if best is None else best.diff_loss)))

    counts = completed.reflection_counts
    matched = (
        f"{counts['matched_i_gt_3sigma']} / {counts['matched']}"
        if "matched_i_gt_3sigma" in counts and "matched" in counts
        else "n/a"
    )
    rows.append(("Matched HKLs (I>3σ/total)", matched))
    return rows


def markdown_table(rows: Rows, headers: tuple[str, str] = ("", "")) -> str:
    """Two-column Markdown table. Pipes in cells are escaped so a value cannot split a row."""

    def cell(text: str) -> str:
        return text.replace("|", "\\|")

    lines = [
        f"| {cell(headers[0])} | {cell(headers[1])} |",
        "| --- | --- |",
        *(f"| {cell(label)} | {cell(value)} |" for label, value in rows),
    ]
    return "\n".join(lines)


def build_tables(records: Sequence[EventRecord]) -> list[tuple[str, str]]:
    """``(title, markdown)`` for every table the report has the events for, in run order."""
    tables = (
        ("Preprocess", preprocess_table(records)),
        ("Refinement summary", refinement_table(records)),
    )
    return [
        (title, markdown_table(rows, ("Parameter", "Value")))
        for title, rows in tables
        if rows is not None
    ]
