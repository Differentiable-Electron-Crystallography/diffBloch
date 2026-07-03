"""The validated parameter value-types (parse, don't validate).

Each frozen dataclass is the single home of its sweep's bounds rules; the pydantic config blocks
delegate to these (see ``test_config.py``), and the pure ``fit_*`` steps trust them. Pins the
defaults against the private and the construction-time validation.
"""

from __future__ import annotations

import pytest

from diffBloch.specs import (
    BeamSelection,
    ConvergenceTest,
    ConvergenceTolerance,
    HexagonalSearch,
    Mosaicity,
    RockingCurve,
    ThicknessGrid,
)


def test_convergence_tolerance_defaults_match_the_private() -> None:
    tol = ConvergenceTolerance()
    assert tol.r_factor_threshold == 0.005
    assert tol.max_iterations == 100


def test_convergence_tolerance_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="r_factor_threshold must be positive"):
        ConvergenceTolerance(r_factor_threshold=0.0)
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        ConvergenceTolerance(max_iterations=0)


def test_convergence_test_defaults() -> None:
    test = ConvergenceTest()
    assert test.operation == "both"  # the private's full operation
    assert test.num_passes == 2  # the e2e's fixed pass count
    assert test.start_g_max_refine == 0.5
    assert (test.pool_step, test.window_step, test.tilt_step) == (0.1, 0.2, 2.0)


def test_convergence_test_rejects_invalid_operation_and_bounds() -> None:
    with pytest.raises(ValueError, match="operation must be"):
        ConvergenceTest(operation="coverage_and_stability")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="start_g_max_refine must be positive"):
        ConvergenceTest(start_g_max_refine=0.0)
    with pytest.raises(ValueError, match="pool_step, window_step and tilt_step must be positive"):
        ConvergenceTest(window_step=0.0)
    with pytest.raises(ValueError, match="num_passes must be >= 1"):
        ConvergenceTest(num_passes=0)


def test_beam_selection_defaults_match_the_private() -> None:
    selection = BeamSelection()
    assert selection.rsg == 0.9
    assert selection.dsg == 0.0015
    assert selection.integration_semiangle == 1.0


def test_beam_selection_rejects_nonpositive_cutoffs() -> None:
    with pytest.raises(ValueError, match="rsg must be positive"):
        BeamSelection(rsg=0.0)
    with pytest.raises(ValueError, match="integration_semiangle must be positive"):
        BeamSelection(integration_semiangle=0.0)


def test_beam_selection_allows_a_loosening_negative_margin() -> None:
    # dsg carries no positivity invariant: a negative margin legitimately loosens the cone.
    assert BeamSelection(dsg=-0.01).dsg == -0.01


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
    assert search.max_iterations == 2000


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


def test_rocking_curve_defaults_and_double_role() -> None:
    rocking = RockingCurve()
    assert rocking.semiangle == 1.0  # shares BeamSelection.integration_semiangle
    assert rocking.sampling == 42  # the quartz reference tilt count
    assert rocking.geometry == "continuous_rotation"


def test_rocking_curve_rejects_invalid_geometry_and_bounds() -> None:
    with pytest.raises(ValueError, match="semiangle must be positive"):
        RockingCurve(semiangle=0.0)
    with pytest.raises(ValueError, match="sampling must be >= 1"):
        RockingCurve(sampling=0)
    with pytest.raises(ValueError, match="geometry must be"):
        RockingCurve(geometry="spiral")  # type: ignore[arg-type]


def test_mosaicity_defaults_to_the_faithful_window() -> None:
    # Faithful private default (the private hardcodes window_size = 5); 2.0 exposes it as tunable.
    assert Mosaicity().window == 5
    assert Mosaicity(window=3).window == 3


def test_mosaicity_rejects_a_nonpositive_window() -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        Mosaicity(window=0)
