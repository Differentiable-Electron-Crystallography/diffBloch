"""Summary tables over a diffBloch JSONL event report.

The run-level facts a figure cannot carry: which preprocess stages ran and with what settings, and
the refined run's headline numbers at the epoch it selected. Each ``*_table`` takes the parsed
records and returns ``(label, value)`` rows, or ``None`` when the report carries no events of that
kind -- the same contract as :mod:`tools.event_report.figures`. :func:`build_tables` renders the
non-empty ones to Markdown for the notebook.

Values are formatted here, not in the notebook, so the rendering is imported and tested rather than
living in a cell nothing in CI runs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from diffBloch.observability import EventRecord

from .reader import records_of

__all__ = ["build_tables", "markdown_table", "preprocess_table", "refinement_table"]

Rows = list[tuple[str, str]]

# The fitting stages a run may or may not include, each with the key of its *own* spec. The other
# params a step records (``coupling``, ``absorption``) are shared composition context that other
# steps are configured with too, not a setting of this step, so they are not repeated per stage.
_FIT_STAGES = (
    ("Orientation optimization", "optimize_orientation", "search"),
    ("Thickness optimization", "optimize_thickness", "grid"),
)


def _number(value: object) -> float | None:
    """A payload number as a float. ``EventRecord`` writes non-finite floats as strings."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)  # "NaN" / "Infinity" / "-Infinity"
        except ValueError:
            return None
    return None


def _general(value: object) -> str:
    """A finite number to six significant figures, or ``n/a``."""
    number = _number(value)
    return "n/a" if number is None or not math.isfinite(number) else f"{number:.6g}"


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
    completed = records_of(records, "PreprocessCompleted")
    if not completed:
        return None
    payload = completed[-1].payload
    # A count an older report never recorded is unknown ("n/a"), not a spec set to ``None``.
    rows: Rows = [
        (label, "n/a" if payload.get(key) is None else _leaf(payload[key]))
        for label, key in (
            ("Rotations", "n_rotations"),
            ("Stages", "n_stages"),
            ("Total HKLs", "total_hkl"),
            ("Matched HKLs", "matched_hkl"),
        )
    ]
    if "steps" not in payload:
        rows.append(("Stage settings", "not recorded in this report"))
        return rows
    steps: dict[str, Any] = {str(name): params for name, params in payload["steps"]}
    for label, step_name, own_key in _FIT_STAGES:
        if step_name not in steps:
            rows.append((label, "not run"))
            continue
        rows.append((label, "ran"))
        params = steps[step_name]
        own = params.get(own_key) if isinstance(params, Mapping) else None
        rows.extend(_flatten(own_key, own) if own is not None else [(own_key, "none")])
    return rows


def _mean_percent(value: object, evaluated: object, total: object) -> str:
    """A mean as a percentage with the rotation count it was taken over, e.g. ``4.21 [97/99]``.

    Every mean states its denominator: a mean over fewer rotations is a different quantity, not a
    better one. A non-finite mean means nothing was evaluated, which the count already says.
    """
    number = _number(value)
    if number is None:
        return "n/a"
    rendered = f"{100.0 * number:.2f}" if math.isfinite(number) else "n/a"
    if evaluated is None or total is None:
        return rendered
    return f"{rendered} [{_leaf(evaluated)}/{_leaf(total)}]"


def refinement_table(records: Sequence[EventRecord]) -> Rows | None:
    """The refined run's headline numbers at the epoch it selected.

    With a held-out validation set the training and validation means are reported side by side;
    without one there is a single population, so the rows carry no train/val prefix.
    """
    completed_records = records_of(records, "RefinementCompleted")
    if not completed_records:
        return None
    completed = completed_records[-1].payload
    best_step = int(completed["best_step"])
    best = next(
        (
            record.payload
            for record in records_of(records, "RefinementStep")
            if record.payload.get("iteration") == best_step
        ),
        {},
    )
    declared = records_of(records, "ExperimentDeclared")
    experiment = declared[-1].payload if declared else {}

    rows: Rows = []
    if experiment:
        rows.append(("Experiment", _leaf(experiment.get("name"))))
    rows += [
        ("Best epoch", f"{best_step + 1} / {_leaf(completed.get('n_steps'))}"),
        ("Selected on", _leaf(completed.get("selection", "training"))),
        ("Objective", _general(completed.get("best_loss"))),
    ]
    if experiment:
        rows += [
            ("Optimizer", _leaf(experiment.get("optimizer"))),
            ("Learning rate", _leaf(experiment.get("learning_rate"))),
        ]

    train = (
        ("wR2 (%)", "wr2", "n_wr2_evaluated"),
        ("R_obs (%)", "r_obs", "n_r_obs_evaluated"),
    )
    has_validation = best.get("val_wr2") is not None
    prefix = "Train " if has_validation else ""
    for label, key, evaluated in train:
        rows.append(
            (
                f"{prefix}{label}",
                _mean_percent(best.get(key), best.get(evaluated), best.get("n_rotations")),
            )
        )
    if has_validation:
        for label, key, evaluated in train:
            rows.append(
                (
                    f"Val {label}",
                    _mean_percent(
                        best.get(f"val_{key}"),
                        best.get(f"val_{evaluated}"),
                        best.get("val_n_rotations"),
                    ),
                )
            )
    rows.append(("Diffraction loss", _general(best.get("diff_loss"))))

    counts = completed.get("reflection_counts") or {}
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
