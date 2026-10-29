"""Pattern visualization stays an application-shell consumer of forward values."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import matplotlib
import numpy as np
import pytest
import torch

from diffBloch.app.cli import main
from diffBloch.app.pattern_visualization import (
    PatternVisualization,
    PatternVisualizationOutput,
    _closest_rotation,
    render_patterns,
)
from diffBloch.core.products import BlochSolution
from diffBloch.preprocess import Plan

matplotlib.use("Agg")


def test_closest_rotation_removes_scale_and_shear() -> None:
    matrix = np.array([[1.8, 0.2, 0.0], [0.0, 0.7, 0.1], [0.0, 0.0, 1.2]])

    rotation = _closest_rotation(matrix)

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_render_patterns_writes_frames(tmp_path: Path) -> None:
    pattern = SimpleNamespace(rotation_index=7, dataset="quartz.cif_pets")
    orientation = SimpleNamespace(pattern=pattern, orientation=torch.eye(3))
    plan = SimpleNamespace(
        structure_factor_grid=SimpleNamespace(cell=torch.eye(3)),
        orientations=(orientation,),
    )
    solution = BlochSolution.from_propagation(
        torch.tensor([[1 + 0j, 0.5 + 0j, 0.25 + 0j]]),
        torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        torch.tensor([100.0]),
    )

    output = render_patterns(
        cast(Plan, plan),
        (solution,),
        tmp_path,
        name="quartz",
        style=PatternVisualization(image_size=64, n_label_hkls=1),
    )

    assert output.frames == (tmp_path / "frames" / "quartz_quartz_0007.png",)
    assert output.frames[0].is_file()


def test_style_rejects_invalid_display_values() -> None:
    with pytest.raises(ValueError, match="image_size"):
        PatternVisualization(image_size=32)


def test_pattern_vis_cli_delegates_to_app_renderer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = tmp_path / "frames" / "sample_0000.png"
    frame.parent.mkdir()
    frame.touch()
    called: dict[str, object] = {}

    def fake_visualize(experiment_directory: str, **kwargs: object) -> PatternVisualizationOutput:
        called.update(experiment_directory=experiment_directory, **kwargs)
        return PatternVisualizationOutput(frames=(frame,))

    monkeypatch.setattr("diffBloch.app.cli.visualize_experiment", fake_visualize)

    assert (
        main(
            [
                "pattern-vis",
                "experiment",
                "--device",
                "cpu",
                "--theme",
                "light",
                "--source",
                "experimental",
                "--side-by-side",
            ]
        )
        == 0
    )
    assert called["experiment_directory"] == "experiment"
    assert called["device"] == "cpu"
    style = cast(PatternVisualization, called["style"])
    assert style.theme == "light"
    assert style.source == "experimental"
    assert style.side_by_side is True
    assert str(frame.parent.resolve()) in capsys.readouterr().out
