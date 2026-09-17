"""The golden report: the event contract pinned to disk, read back by the current code.

``tests/fixtures/reports/golden-v1.jsonl`` is a small, plausible run carrying every event class the
library defines (see ``build_golden.py`` beside it). These tests are the reader-side half of the
report contract: they fail when the current checkout can no longer read a report an earlier one
wrote, when a table or figure that used to render from it no longer does, when the committed file
has drifted from the builder, or when a new event exists that the golden run never emits.

The failure messages say what to do, because each is a deliberate choice: regenerate the fixture and
commit the diff (a reviewed schema change), or add the new event to the builder.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from tests.fixtures.reports import build_golden  # noqa: E402
from tools.event_report import reader  # noqa: E402
from tools.event_report.figures import SECTIONS, build_sections  # noqa: E402
from tools.event_report.tables import build_tables  # noqa: E402

from diffBloch.observability import (  # noqa: E402
    EVENT_TYPES,
    EventRecord,
    RefinementOutputsWritten,
    event_from_record,
)

GOLDEN = build_golden.GOLDEN
REGENERATE = (
    "regenerate with `uv run python tests/fixtures/reports/build_golden.py` and commit the diff"
)


@pytest.fixture(scope="module")
def golden() -> list[EventRecord]:
    return reader.read_records(GOLDEN)


@pytest.fixture(autouse=True)
def _close_figures():  # type: ignore[no-untyped-def]
    yield
    matplotlib.pyplot.close("all")


def test_golden_report_covers_every_event_the_library_defines(golden: list[EventRecord]) -> None:
    """A new event must be added to the golden run, or nothing downstream ever exercises it."""
    emitted = {record.event_type for record in golden}
    missing = set(EVENT_TYPES) - emitted
    assert not missing, (
        f"events never emitted by the golden run: {sorted(missing)}; add them to build_golden.build_events()"
    )
    assert not emitted - set(EVENT_TYPES)


def test_golden_report_rebuilds_into_this_checkouts_events(golden: list[EventRecord]) -> None:
    """Every committed record still fits the event it was written from.

    This is backward compatibility with a report on disk: a required field renamed or removed since
    the fixture was written fails here, naming the record and field, before any figure is drawn.
    """
    events = [event_from_record(record) for record in golden]  # raises ReportSchemaError on drift

    assert [type(event).__name__ for event in events] == [r.event_type for r in golden]
    # The live surface is reproduced too, not just the fields.
    assert all(
        event.measurements == record.measurements
        for event, record in zip(events, golden, strict=True)
    )


def test_golden_report_renders_every_table_and_figure(golden: list[EventRecord]) -> None:
    """The golden run has the events for *all* of them, so none may silently decline."""
    every_figure = {name for section in SECTIONS for name, _ in section.builders}
    sections = build_sections(golden)

    assert {name for _, figures in sections for name in figures} == every_figure
    assert [title for title, _ in sections] == [
        "Convergence",
        "Preprocess — orientation optimization",
        "Preprocess — per-rotation thickness fit",
        "Preprocess — coupled solve geometry",
        "Refinement — epoch history",
        "Refinement — per-rotation scores",
        "Refinement — datasets",
        "Refinement — reflections",
        "Refinement — learned thickness model",
    ]
    tables = dict(build_tables(golden))
    assert list(tables) == ["Preprocess", "Refinement summary"]
    assert "| Orientation optimization | ran |" in tables["Preprocess"]
    assert "| search.max_iterations | 60 |" in tables["Preprocess"]
    assert "| Val wR2 (%) | 4.60 [1/1] |" in tables["Refinement summary"]
    assert "| Matched HKLs (I>3σ/total) | 52 / 70 |" in tables["Refinement summary"]


def test_golden_report_matches_its_builder() -> None:
    """The committed file is what the builder produces from today's events.

    A producer change that alters the written form (a new field, a renamed one, a different
    serialization) shows up here as a fixture diff to regenerate and review -- the schema change
    becomes visible in the PR instead of only in a user's notebook later.
    """
    expected = build_golden.render(build_golden.build_records())

    assert GOLDEN.read_text() == expected, f"golden fixture is stale: {REGENERATE}"


def test_golden_report_is_portable(golden: list[EventRecord]) -> None:
    """The artifact manifest names files relative to the experiment directory it also carries."""
    (outputs,) = [
        event_from_record(r) for r in golden if r.event_type == "RefinementOutputsWritten"
    ]
    assert isinstance(outputs, RefinementOutputsWritten)

    assert not any(Path(path).is_absolute() for path in outputs.artifacts.values())
    assert Path(outputs.experiment_directory).is_absolute()
