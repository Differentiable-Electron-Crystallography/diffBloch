"""The thin CLI validates an experiment file and reports success."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from diffBloch.app.cli import main
from diffBloch.observability import MultiLogger, NullLogger
from diffBloch.preprocess.inference import InferenceResult, RotationInference

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


def test_run_infer_delegates_to_run_experiment_and_reports(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_run_experiment(
        experiment_dir: str,
        *,
        logger: object,
        checkpoint: bool = True,
        refresh: bool = False,
        device: object = None,
        workers: int = 1,
        max_batch: object = None,
    ) -> InferenceResult:
        captured["dir"] = experiment_dir
        captured["logger"] = logger
        captured["checkpoint"] = checkpoint
        captured["refresh"] = refresh
        captured["workers"] = workers
        rotation = RotationInference(r_obs=0.05, n_observed=9, n_beams=20)
        return InferenceResult(per_rotation=(rotation,))

    monkeypatch.setattr("diffBloch.app.cli.run_experiment", fake_run_experiment)
    rc = main(["run", "infer", "/some/experiment"])

    assert rc == 0
    assert captured["dir"] == "/some/experiment"
    assert isinstance(captured["logger"], NullLogger)  # no --console/--csv => null sink
    assert captured["checkpoint"] is True  # checkpoint on by default
    assert captured["refresh"] is False
    assert captured["workers"] == 1  # sequential by default
    out = capsys.readouterr().out
    assert "evaluated 1 rotations" in out
    assert "mean R_obs = 0.0500" in out


def test_run_infer_checkpoint_flags_thread_through(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run_experiment(
        experiment_dir: str,
        *,
        logger: object,
        checkpoint: bool = True,
        refresh: bool = False,
        device: object = None,
        workers: int = 1,
        max_batch: object = None,
    ) -> InferenceResult:
        seen["checkpoint"] = checkpoint
        seen["refresh"] = refresh
        seen["workers"] = workers
        return InferenceResult(per_rotation=())

    monkeypatch.setattr("diffBloch.app.cli.run_experiment", fake_run_experiment)
    assert main(["run", "infer", "x", "--no-checkpoint", "--refresh", "--workers", "4"]) == 0
    assert seen == {"checkpoint": False, "refresh": True, "workers": 4}


def test_run_infer_builds_console_and_csv_sinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def fake_run_experiment(
        experiment_dir: str,
        *,
        logger: object,
        checkpoint: bool = True,
        refresh: bool = False,
        device: object = None,
        workers: int = 1,
        max_batch: object = None,
    ) -> InferenceResult:
        seen["logger"] = logger
        return InferenceResult(per_rotation=())

    monkeypatch.setattr("diffBloch.app.cli.run_experiment", fake_run_experiment)
    csv_path = tmp_path / "observations.csv"
    rc = main(["run", "infer", "x", "--console", "--csv", str(csv_path)])

    assert rc == 0
    logger = seen["logger"]
    assert isinstance(logger, MultiLogger)
    assert len(logger.loggers) == 2  # console + csv fanned out
    assert csv_path.is_file()  # CSVLogger writes its header at construction


def test_run_infer_missing_experiment_reports_concise_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["run", "infer", "/no/such/experiment"])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


class _FakePlan:
    """Minimal stand-in for a settled Plan: enough for the CLI's summary line."""

    def __init__(self) -> None:
        self.orientations = (object(), object())
        self.provenance = (
            SimpleNamespace(name="fit_orientation"),
            SimpleNamespace(name="fit_thickness"),
        )


def test_run_preprocess_delegates_and_reports_without_scoring(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_preprocess_experiment(
        experiment_dir: str,
        *,
        logger: object,
        checkpoint: bool = True,
        refresh: bool = False,
        device: object = None,
        workers: int = 1,
        max_batch: object = None,
    ) -> _FakePlan:
        captured["dir"] = experiment_dir
        captured["logger"] = logger
        captured["checkpoint"] = checkpoint
        captured["refresh"] = refresh
        captured["device"] = device
        captured["workers"] = workers
        return _FakePlan()

    monkeypatch.setattr("diffBloch.app.cli.preprocess_experiment", fake_preprocess_experiment)
    rc = main(["run", "preprocess", "/some/experiment"])

    assert rc == 0
    assert captured["dir"] == "/some/experiment"
    assert isinstance(captured["logger"], NullLogger)  # no --console/--csv => null sink
    assert captured["checkpoint"] is True and captured["refresh"] is False
    assert captured["workers"] == 1 and captured["device"] is None
    out = capsys.readouterr().out
    assert "preprocessed 2 rotations" in out
    assert "fit_orientation, fit_thickness" in out  # recipe echoed; no R_obs (no scoring)
    assert "R_obs" not in out


def test_run_preprocess_flags_thread_through(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_preprocess_experiment(
        experiment_dir: str,
        *,
        logger: object,
        checkpoint: bool = True,
        refresh: bool = False,
        device: object = None,
        workers: int = 1,
        max_batch: object = None,
    ) -> _FakePlan:
        seen["checkpoint"] = checkpoint
        seen["refresh"] = refresh
        seen["device"] = device
        seen["workers"] = workers
        seen["max_batch"] = max_batch
        return _FakePlan()

    monkeypatch.setattr("diffBloch.app.cli.preprocess_experiment", fake_preprocess_experiment)
    rc = main(
        ["run", "preprocess", "x"]
        + [
            "--no-checkpoint",
            "--refresh",
            "--device",
            "cuda",
            "--workers",
            "4",
            "--max-batch",
            "1024",
        ]
    )
    assert rc == 0
    assert seen == {
        "checkpoint": False,
        "refresh": True,
        "device": "cuda",
        "workers": 4,
        "max_batch": 1024,
    }


def test_run_preprocess_missing_experiment_reports_concise_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["run", "preprocess", "/no/such/experiment"])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


def _fake_refinement_result() -> SimpleNamespace:
    """Minimal stand-in for a RefinementResult: enough for the CLI's summary line."""
    return SimpleNamespace(losses=torch.tensor([2.0, 1.0]), best_loss=1.0, best_step=1)


def test_run_refine_delegates_and_reports(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_refine_experiment(
        experiment_dir: str,
        *,
        logger: object,
        checkpoint: bool = True,
        refresh: bool = False,
        device: object = None,
        workers: int = 1,
        max_batch: object = None,
    ) -> SimpleNamespace:
        captured["dir"] = experiment_dir
        captured["logger"] = logger
        captured["checkpoint"] = checkpoint
        return _fake_refinement_result()

    monkeypatch.setattr("diffBloch.app.cli.refine_experiment", fake_refine_experiment)
    rc = main(["run", "refine", "/some/experiment"])

    assert rc == 0
    assert captured["dir"] == "/some/experiment"
    assert isinstance(captured["logger"], NullLogger)
    out = capsys.readouterr().out
    assert "refined 2 steps" in out
    assert "objective 2.000000 -> 1.000000 (best at step 1)" in out


def test_run_refine_flags_thread_through(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_refine_experiment(
        experiment_dir: str,
        *,
        logger: object,
        checkpoint: bool = True,
        refresh: bool = False,
        device: object = None,
        workers: int = 1,
        max_batch: object = None,
    ) -> SimpleNamespace:
        seen["checkpoint"] = checkpoint
        seen["refresh"] = refresh
        seen["device"] = device
        seen["workers"] = workers
        return _fake_refinement_result()

    monkeypatch.setattr("diffBloch.app.cli.refine_experiment", fake_refine_experiment)
    rc = main(
        ["run", "refine", "x"]
        + ["--no-checkpoint", "--refresh", "--device", "cuda", "--workers", "4"]
    )
    assert rc == 0
    assert seen == {"checkpoint": False, "refresh": True, "device": "cuda", "workers": 4}


def test_run_refine_missing_experiment_reports_concise_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["run", "refine", "/no/such/experiment"])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err
