"""The thin CLI validates an experiment file and reports success."""

from pathlib import Path

import pytest

from diffBloch.app.cli import main

FIXTURE = Path(__file__).parent.parent / "fixtures" / "quartz_min" / "experiment.yaml"


def test_validate_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["validate", str(FIXTURE)])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_no_command_prints_help_returns_zero() -> None:
    assert main([]) == 0
