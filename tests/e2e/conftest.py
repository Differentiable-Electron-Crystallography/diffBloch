"""pytest configuration for end-to-end tests."""

from __future__ import annotations

import os
from typing import Any


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Deselect parametrized materials not matching ``MATERIAL``."""
    material = os.environ.get("MATERIAL")
    if not material:
        return

    selected = []
    deselected = []
    for item in items:
        callspec = getattr(item, "callspec", None)
        if (
            item.get_closest_marker("e2e")
            and callspec is not None
            and callspec.params.get("material") != material
        ):
            deselected.append(item)
            continue
        selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
