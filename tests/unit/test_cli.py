"""The thin CLI validates an experiment file and reports success."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from diffBloch.app.cli import main

FIXTURE = Path(__file__).parent.parent / "fixtures" / "quartz_min" / "experiment.yaml"


def test_validate_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["validate", str(FIXTURE)])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_no_command_prints_help_returns_zero() -> None:
    assert main([]) == 0


def test_missing_file_reports_concise_error(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["validate", "/no/such/experiment.yaml"])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_invalid_config_reports_concise_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: only-a-name\n")  # missing required `inputs`
    rc = main(["validate", str(bad)])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_debug_flag_reraises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: only-a-name\n")
    with pytest.raises(ValidationError):
        main(["--debug", "validate", str(bad)])


def test_run_pack_exports_run_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = tmp_path / "run_001"
    run.mkdir()
    (run / "run_manifest.json").write_text("{}\n")
    (run / "history.jsonl").write_text("{}\n")

    rc = main(["run", "pack", str(run), "--format", "zip"])
    assert rc == 0
    output = Path(capsys.readouterr().out.strip())
    assert output.is_file()
    assert output.suffix == ".zip"


def test_run_pack_missing_manifest_reports_concise_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run_001"
    run.mkdir()
    rc = main(["run", "pack", str(run)])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err
