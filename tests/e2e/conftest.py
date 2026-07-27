"""pytest configuration for the e2e characterization tests.

Mirrors the research repo's ``MATERIAL`` deselection so a single material can be run in isolation
(``MATERIAL=quartz uv run pytest -m e2e``) without cluttering output with the others.

Also hosts the anchor-metrics recorder: anchor tests deposit their *measured* values (not the
pinned expectations) into the ``anchor_metrics`` fixture, and the session dumps them as JSON to
the path in ``DIFFBLOCH_ANCHOR_METRICS`` -- CI publishes the series as a trend plot on the
``badges`` branch. With the env var unset (the default everywhere but CI main pushes) nothing
is written.
"""

import json
import os
from typing import Any

import pytest

# Module-level store: a session fixture's value is not reachable from pytest_sessionfinish,
# so the fixture hands tests this dict and the hook reads it directly.
_ANCHOR_METRICS: dict[str, float] = {}


@pytest.fixture(scope="session")
def anchor_metrics() -> dict[str, float]:
    """Measured anchor values, written to ``DIFFBLOCH_ANCHOR_METRICS`` at session end."""
    return _ANCHOR_METRICS


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    out = os.environ.get("DIFFBLOCH_ANCHOR_METRICS")
    if out and _ANCHOR_METRICS:
        with open(out, "w") as fh:
            json.dump(_ANCHOR_METRICS, fh, indent=2, sort_keys=True)


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Deselect materials not matching the ``MATERIAL`` env var (keeps output focused)."""
    material = os.environ.get("MATERIAL")
    if not material:
        return

    selected, deselected = [], []
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is not None and callspec.params.get("material") not in (None, material):
            deselected.append(item)
        else:
            selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
