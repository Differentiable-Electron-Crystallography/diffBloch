"""``ThicknessPlotLogger`` and the report's thickness-NN shape plot (matplotlib backend)."""

from __future__ import annotations

from pathlib import Path

import pytest

from diffBloch.app.loggers.plotting import (
    ThicknessPlotLogger,
    _apply_house_style,
    plot_thickness_nn_shape,
)
from diffBloch.observability import OrientationOptimized, ThicknessOptimized

_EVENT = ThicknessOptimized(
    rotation_index=3,
    score=0.031,
    residual="wr2",
    thickness=460.0,
    candidate_thicknesses=(400.0, 430.0, 460.0, 490.0, 520.0),
    candidate_score=(0.08, 0.05, 0.031, 0.045, 0.07),
)


def test_report_writes_one_png_named_after_the_rotation_index(tmp_path: Path) -> None:
    output_dir = tmp_path / "thickness_optim"
    logger = ThicknessPlotLogger(output_dir)

    logger.report(_EVENT)

    png = output_dir / "3.png"
    assert png.is_file()
    assert png.stat().st_size > 0


def test_post_init_creates_the_output_directory_eagerly(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "thickness_optim"
    assert not output_dir.exists()

    ThicknessPlotLogger(output_dir)

    assert output_dir.is_dir()


def test_report_ignores_events_that_are_not_thickness_optimized(tmp_path: Path) -> None:
    output_dir = tmp_path / "thickness_optim"
    logger = ThicknessPlotLogger(output_dir)

    logger.report(
        OrientationOptimized(
            rotation_index=3,
            score=0.05,
            residual="wr2",
            n_matched_hkl=100,
            n_trials=40,
            n_passes=12,
            pass_cap=60,
        )
    )

    assert list(output_dir.iterdir()) == []


def test_report_writes_a_separate_png_per_rotation(tmp_path: Path) -> None:
    output_dir = tmp_path / "thickness_optim"
    logger = ThicknessPlotLogger(output_dir)

    logger.report(_EVENT)
    logger.report(
        ThicknessOptimized(
            rotation_index=4,
            score=0.02,
            residual="wr2",
            thickness=500.0,
            candidate_thicknesses=(440.0, 470.0, 500.0),
            candidate_score=(0.03, 0.025, 0.02),
        )
    )

    assert {p.name for p in output_dir.iterdir()} == {"3.png", "4.png"}


def test_report_floors_the_y_axis_at_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """wR2/R_obs never go negative, so the plot must not auto-zoom into a narrow non-zero band."""
    import matplotlib.axes

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original = matplotlib.axes.Axes.set_ylim

    def recording_set_ylim(self: matplotlib.axes.Axes, *args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylim", recording_set_ylim)

    ThicknessPlotLogger(tmp_path / "thickness_optim").report(_EVENT)

    assert any(kwargs.get("bottom") == 0 for _, kwargs in calls)


def test_thickness_nn_shape_plot_carries_its_dataset_title(tmp_path: Path) -> None:
    png = tmp_path / "thickness_nn_shape_a.png"

    plot_thickness_nn_shape(
        [(0.0, 400.0), (10.0, 450.0)], png, title="Thickness NN final shape -- a.cif_pets"
    )

    assert png.is_file()
    assert png.stat().st_size > 0


def test_house_style_prefers_arial_via_the_sans_serif_fallback_list() -> None:
    """Regression for #140: font.family must be the generic 'sans-serif' family with Arial as the
    preferred face in font.sans-serif, not a direct font.family = "Arial" -- the direct form makes
    every glyph search Arial by itself and log its own findfont warning when it's missing."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        _apply_house_style(ax)
        assert plt.rcParams["font.family"] == ["sans-serif"]
        assert plt.rcParams["font.sans-serif"][:2] == ["Arial", "DejaVu Sans"]
    finally:
        plt.close(fig)


def test_house_style_logs_no_findfont_warning_when_the_preferred_font_is_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Deterministic regardless of whether Arial is actually installed on the host: even with a
    guaranteed-missing font first in the preference list, the sans-serif family fallback must not
    log findfont -- proving the mechanism the #140 fix relies on, not just today's environment.

    Sets rcParams directly (the same shape ``_apply_house_style`` sets, with a font guaranteed
    absent standing in for "Arial happens to be missing") rather than monkeypatching the function
    itself, so this exercises the real fallback mechanism, not a stand-in.
    """
    import logging

    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["NotARealFontXYZ", "DejaVu Sans"]
    fig, ax = plt.subplots()
    try:
        with caplog.at_level(logging.WARNING, logger="matplotlib.font_manager"):
            ax.set_title("t")
            fig.canvas.draw()
        assert not any("findfont" in record.getMessage() for record in caplog.records)
    finally:
        plt.close(fig)
