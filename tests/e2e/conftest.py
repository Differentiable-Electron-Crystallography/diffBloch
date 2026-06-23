"""pytest configuration for the e2e characterization tests.

Mirrors the research repo's ``MATERIAL`` deselection so a single material can be run in isolation
(``MATERIAL=quartz uv run pytest -m e2e``) without cluttering output with the others.
"""

import os
from typing import Any


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
