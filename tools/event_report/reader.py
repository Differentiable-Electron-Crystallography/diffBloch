"""Load and index a diffBloch JSONL event report.

The one place that knows how to find, parse, and slice a report; :mod:`tools.event_report.figures`
and the notebook both read through it.

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

from diffBloch.observability import EventRecord

__all__ = [
    "by_dataset",
    "default_event_log",
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
    """Every record of one event type, in emission order."""
    return [record for record in records if record.event_type == event_type]


def by_dataset(records: Iterable[EventRecord]) -> dict[str, list[EventRecord]]:
    """Group records under their dataset label (unlabeled records under ``""``)."""
    grouped: dict[str, list[EventRecord]] = defaultdict(list)
    for record in records:
        grouped[record.dataset or ""].append(record)
    return dict(grouped)


def sorted_by_rotation(records: Iterable[EventRecord]) -> list[EventRecord]:
    """Records in ``(dataset, rotation_index)`` order -- the per-rotation plotting/table order."""
    return sorted(records, key=lambda record: (record.dataset or "", record.rotation_index or -1))


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
