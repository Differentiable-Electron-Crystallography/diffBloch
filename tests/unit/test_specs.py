"""The validated parameter value-types (parse, don't validate).

Each frozen dataclass is the single home of its sweep's bounds rules; the pydantic config blocks
delegate to these (see ``test_config.py``), and the pure ``fit_*`` steps trust them. Pins the
defaults against the private and the construction-time validation.
"""

from __future__ import annotations

import pytest

from diffBloch.specs import HexagonalSearch, ThicknessGrid


def test_thickness_grid_defaults_match_the_private() -> None:
    grid = ThicknessGrid()
    assert grid.min_thickness == 5.0
    assert grid.max_thickness == 2000.0
    assert grid.n_steps == 100


def test_hexagonal_search_defaults_match_the_private() -> None:
    search = HexagonalSearch()
    assert search.max_search_angle == 0.4
    assert search.min_search_angle == 0.001
    assert search.n_steps == 6
    assert search.max_iterations == 200


def test_thickness_grid_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="thickness bounds must be positive"):
        ThicknessGrid(min_thickness=-1.0)
    with pytest.raises(ValueError, match="max_thickness must exceed min_thickness"):
        ThicknessGrid(min_thickness=400.0, max_thickness=400.0)
    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        ThicknessGrid(n_steps=0)


def test_hexagonal_search_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="search angles must be positive"):
        HexagonalSearch(min_search_angle=0.0)
    with pytest.raises(ValueError, match="max_search_angle must exceed min_search_angle"):
        HexagonalSearch(max_search_angle=0.001)
    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        HexagonalSearch(n_steps=0)
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        HexagonalSearch(max_iterations=0)


def test_value_types_are_frozen() -> None:
    grid = ThicknessGrid()
    with pytest.raises((AttributeError, TypeError)):
        grid.min_thickness = 1.0  # type: ignore[misc]
