"""Load and index a diffBloch JSONL event report.

The one place that knows how to find, parse, and slice a report; :mod:`tools.event_report.figures`,
:mod:`tools.event_report.tables` and the notebook all read through it. Consumers take *events*
(:func:`events_of`), rebuilt as the library's own dataclasses, not the envelope's payload dict: a
renamed field then fails at the read with the event and field named
(:class:`~diffBloch.observability.ReportSchemaError`), instead of ``payload.get`` quietly yielding
``None`` and a figure going blank.

This lives outside ``src/diffBloch`` on purpose: it consumes the event contract, it is not part of
the refinement library.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import fmean
from typing import Protocol, cast

from diffBloch.observability import Event, EventRecord, event_from_record

__all__ = [
    "by_dataset",
    "default_event_log",
    "events_of",
    "finite",
    "finite_mean",
    "read_records",
    "read_records_text",
    "records_of",
    "repository_root",
    "resolve_event_log_path",
    "sorted_by_rotation",
]


def repository_root(start: Path | None = None) -> Path:
    """The checkout root, found by walking up from ``start`` (default: the working directory)."""
    current = (Path.cwd() if start is None else start).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "diffBloch").is_dir():
            return candidate
    return Path.cwd().resolve()


def resolve_event_log_path(path: Path | str) -> Path:
    """Resolve a report path against the working directory and then the repository root.

    A notebook is commonly launched from either place, and example-relative paths should work from
    both. A relative path that matches neither raises with the locations actually tried.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Event log does not exist: {candidate}")

    attempts: list[Path] = []
    for root in (Path.cwd().resolve(), repository_root()):
        resolved = root / candidate
        if resolved not in attempts:
            attempts.append(resolved)
        if resolved.is_file():
            return resolved
    tried = "\n  ".join(str(attempt) for attempt in attempts)
    raise FileNotFoundError(f"Event log not found. Tried:\n  {tried}")


def default_event_log() -> Path:
    """The newest report to open when the caller named none.

    ``DIFFBLOCH_EVENT_LOG`` wins; otherwise the most recently modified report under the working
    directory's ``reproducibility/`` or any bundled example's.
    """
    from_environment = os.environ.get("DIFFBLOCH_EVENT_LOG")
    if from_environment:
        return Path(from_environment)
    root = repository_root()
    reports = sorted(
        (
            *Path("reproducibility/reports").glob("report-*.jsonl"),
            *Path("reproducibility").glob("report-*.jsonl"),
            *root.glob("examples/*/data/*/reproducibility/reports/report-*.jsonl"),
            *root.glob("examples/*/data/*/reproducibility/report-*.jsonl"),
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if reports:
        return reports[0]
    return Path("reproducibility/reports")


def read_records(path: Path | str) -> list[EventRecord]:
    """Parse a JSONL report, resolving ``path`` the way :func:`resolve_event_log_path` does."""
    return read_records_text(resolve_event_log_path(path).read_text())


def read_records_text(text: str) -> list[EventRecord]:
    """Parse JSONL report *content* -- the upload path, where there is no file on disk."""
    return [EventRecord.model_validate_json(line) for line in text.splitlines() if line.strip()]


def records_of(records: Iterable[EventRecord], event_type: str) -> list[EventRecord]:
    """Every record of one event type, in emission order -- the envelope, for callers that need
    ``sequence`` or the raw payload. Rendering code wants :func:`events_of` instead."""
    return [record for record in records if record.event_type == event_type]


def events_of[E: Event](records: Iterable[EventRecord], cls: type[E]) -> list[E]:
    """Every ``cls`` event in the report, rebuilt as the dataclass itself, in emission order."""
    return [
        cast(E, event_from_record(record))
        for record in records
        if record.event_type == cls.__name__
    ]


class Positioned(Protocol):
    """What the per-rotation helpers below need: a dataset label and a rotation index.

    Satisfied by the envelope (``EventRecord``) and by every per-rotation event, so the same
    grouping/sorting serves both.
    """

    @property
    def dataset(self) -> str | None: ...
    @property
    def rotation_index(self) -> int | None: ...


def by_dataset[P: Positioned](items: Iterable[P]) -> dict[str, list[P]]:
    """Group under the dataset label (unlabeled items under ``""``)."""
    grouped: dict[str, list[P]] = defaultdict(list)
    for item in items:
        grouped[item.dataset or ""].append(item)
    return dict(grouped)


def sorted_by_rotation[P: Positioned](items: Iterable[P]) -> list[P]:
    """In ``(dataset, rotation_index)`` order -- the per-rotation plotting/table order."""
    return sorted(items, key=lambda item: (item.dataset or "", item.rotation_index or -1))


def finite(values: Iterable[object]) -> list[float]:
    """The finite numbers in ``values``, dropping ``None`` / NaN / infinity."""
    out: list[float] = []
    for value in values:
        if isinstance(value, int | float) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def finite_mean(values: Iterable[object]) -> float | None:
    """Mean over the finite entries, or ``None`` when nothing was finite."""
    kept = finite(values)
    return fmean(kept) if kept else None
