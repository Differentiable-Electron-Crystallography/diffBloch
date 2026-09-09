"""The validated parameter value-types (parse, don't validate).

Each frozen dataclass is the single home of its sweep's bounds rules; the pydantic config blocks
delegate to these (see ``test_config.py``), and the pure ``fit_*`` steps trust them. Pins the
defaults and the construction-time validation.
"""

from __future__ import annotations

import pytest

from diffBloch.specs import (
    BeamSelection,
    ConvergenceTest,
    ConvergenceTolerance,
    IntegrationGeometry,
    RockingCurve,
    ThicknessGrid,
)


def test_convergence_tolerance_defaults() -> None:
    tol = ConvergenceTolerance()
    assert tol.r_factor_threshold == 0.01
    assert tol.max_iterations == 100


def test_convergence_tolerance_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="r_factor_threshold must be positive"):
        ConvergenceTolerance(r_factor_threshold=0.0)
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        ConvergenceTolerance(max_iterations=0)


def test_convergence_test_defaults() -> None:
    test = ConvergenceTest()
    assert test.num_passes == 2
    assert (test.g_max_step, test.sg_max_step, test.tilt_steps_step) == (0.1, 0.005, 2)


def test_convergence_test_rejects_invalid_bounds() -> None:
    with pytest.raises(
        ValueError,
        match="g_max_step, sg_max_step, and tilt_steps_step must be positive",
    ):
        ConvergenceTest(g_max_step=0.0)
    with pytest.raises(ValueError, match="g_max_step, sg_max_step, and tilt_steps_step"):
        ConvergenceTest(sg_max_step=0.0)
    with pytest.raises(ValueError, match="g_max_step, sg_max_step, and tilt_steps_step"):
        ConvergenceTest(tilt_steps_step=0)
    with pytest.raises(ValueError, match="num_passes must be >= 1"):
        ConvergenceTest(num_passes=0)


def test_integration_geometry_defaults_and_validation() -> None:
    geometry = IntegrationGeometry()
    assert geometry.semiangle == 1.0  # the shared tilt half-width / Klar window
    assert geometry.geometry == "continuous_rotation"
    with pytest.raises(ValueError, match="semiangle must be positive"):
        IntegrationGeometry(semiangle=0.0)
    with pytest.raises(ValueError, match="geometry must be"):
        IntegrationGeometry(geometry="spiral")  # type: ignore[arg-type]


def test_beam_selection_defaults() -> None:
    selection = BeamSelection()
    assert selection.rsg == 0.66
    assert selection.dsg == 0.0015
    assert selection.integration == IntegrationGeometry()  # the shared integration geometry


def test_beam_selection_rejects_nonpositive_cutoffs() -> None:
    with pytest.raises(ValueError, match="rsg must be positive"):
        BeamSelection(rsg=0.0)
    # The semiangle now lives on IntegrationGeometry, which validates it.
    with pytest.raises(ValueError, match="semiangle must be positive"):
        BeamSelection(integration=IntegrationGeometry(semiangle=0.0))


def test_beam_selection_allows_a_loosening_negative_margin() -> None:
    # dsg carries no positivity invariant: a negative margin legitimately loosens the cone.
    assert BeamSelection(dsg=-0.01).dsg == -0.01


def test_thickness_grid_defaults() -> None:
    grid = ThicknessGrid()
    assert grid.min_thickness == 5.0
    assert grid.max_thickness == 2000.0
    assert grid.n_steps == 100


def test_thickness_grid_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="thickness bounds must be positive"):
        ThicknessGrid(min_thickness=-1.0)
    with pytest.raises(ValueError, match="max_thickness must exceed min_thickness"):
        ThicknessGrid(min_thickness=400.0, max_thickness=400.0)
    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        ThicknessGrid(n_steps=0)


def test_value_types_are_frozen() -> None:
    grid = ThicknessGrid()
    with pytest.raises((AttributeError, TypeError)):
        grid.min_thickness = 1.0  # type: ignore[misc]


def test_rocking_curve_defaults_and_double_role() -> None:
    rocking = RockingCurve()
    assert rocking.integration.semiangle == 1.0  # the shared integration half-width
    assert rocking.sampling == 42  # the quartz reference tilt count
    assert rocking.integration.geometry == "continuous_rotation"


def test_rocking_curve_rejects_invalid_geometry_and_bounds() -> None:
    with pytest.raises(ValueError, match="semiangle must be positive"):
        RockingCurve(integration=IntegrationGeometry(semiangle=0.0))
    with pytest.raises(ValueError, match="sampling must be >= 1"):
        RockingCurve(sampling=0)
    with pytest.raises(ValueError, match="geometry must be"):
        RockingCurve(integration=IntegrationGeometry(geometry="spiral"))  # type: ignore[arg-type]
