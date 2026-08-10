"""The thin CLI validates an experiment file and reports success."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from diffBloch.app.cli import _release_sinks, main
from diffBloch.app.loggers import ConsoleLogger
from diffBloch.observability import MultiLogger, NullLogger, OrientationOptimized
from diffBloch.preprocess.inference import InferenceResult, RotationInference

FIXTURE = Path(__file__).parent.parent / "fixtures" / "quartz_min" / "experiment.yaml"


def _summary_row(out: str, label: str, value: str) -> bool:
    """Whether the summary box shows ``value`` on ``label``'s row, ignoring the column padding.

    The box pads labels to a fixed width; asserting on that spacing would make every test here fail
    on a label-width change that broke nothing, so match the pair rather than the layout.
    """
    return re.search(rf"{re.escape(label)}\s+{re.escape(value)}(\s|$)", out) is not None


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
        **_kwargs: object,
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
    assert isinstance(captured["logger"], ConsoleLogger)  # console on by default (no --quiet)
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
        **_kwargs: object,
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
        **_kwargs: object,
    ) -> InferenceResult:
        seen["logger"] = logger
        return InferenceResult(per_rotation=())

    monkeypatch.setattr("diffBloch.app.cli.run_experiment", fake_run_experiment)
    csv_path = tmp_path / "experimental_data.csv"
    rc = main(["run", "infer", "x", "--csv", str(csv_path)])

    assert rc == 0
    logger = seen["logger"]
    assert isinstance(logger, MultiLogger)
    assert len(logger.loggers) == 2  # console + csv fanned out
    assert csv_path.is_file()  # CSVLogger writes its header at construction


def test_run_infer_quiet_silences_the_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--quiet`` opts out of the default console stream -> the null sink (no experimental_data)."""
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
        **_kwargs: object,
    ) -> InferenceResult:
        seen["logger"] = logger
        return InferenceResult(per_rotation=())

    monkeypatch.setattr("diffBloch.app.cli.run_experiment", fake_run_experiment)
    assert main(["run", "infer", "x", "--quiet"]) == 0
    assert isinstance(seen["logger"], NullLogger)


def test_run_infer_missing_experiment_reports_concise_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["run", "infer", "/no/such/experiment"])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_run_converge_delegates_and_reports(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_converge_experiment(
        experiment_dir: str,
        *,
        logger: object,
        device: object,
        n_orientations: int,
    ) -> SimpleNamespace:
        assert experiment_dir == "/some/experiment"
        assert isinstance(logger, ConsoleLogger)
        assert device == "cuda"
        assert n_orientations == 1
        return SimpleNamespace(g_max=2.5, sg_max=0.02, tilt_steps=46)

    monkeypatch.setattr("diffBloch.app.cli.converge_experiment", fake_converge_experiment)

    assert main(["run", "converge", "/some/experiment"]) == 0
    assert capsys.readouterr().out == (
        "========================================\n"
        "HYPERPARAMETER OPTIMIZATION RESULT\n"
        "gmax: 2.5\n"
        "sgmax: 0.02\n"
        "tilt_steps: 46\n"
        "========================================\n"
        "optimized_hyperparams gmax=2.5 sgmax=0.02 tilt_steps=46\n"
    )


class _FakePlan:
    """Minimal stand-in for a settled Plan: enough for the CLI's summary line."""

    def __init__(self) -> None:
        self.orientations = (
            SimpleNamespace(
                pattern=SimpleNamespace(hkl=torch.empty((3, 3))),
                alignment=SimpleNamespace(hkl=torch.empty((2, 3))),
                beam_hkl=torch.empty((9, 3)),
            ),
            SimpleNamespace(
                pattern=SimpleNamespace(hkl=torch.empty((4, 3))),
                alignment=SimpleNamespace(hkl=torch.empty((3, 3))),
                beam_hkl=torch.empty((7, 3)),
            ),
        )
        self.provenance = (
            SimpleNamespace(name="optimize_orientation"),
            SimpleNamespace(name="optimize_thickness"),
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
        **_kwargs: object,
    ) -> _FakePlan:
        captured["dir"] = experiment_dir
        captured["logger"] = logger
        captured["checkpoint"] = checkpoint
        captured["refresh"] = refresh
        captured["device"] = device
        captured["workers"] = workers
        logger.report(
            OrientationOptimized(
                rotation_index=3,
                score=0.25,
                residual="wr2",
                n_matched_hkl=2,
                n_trials=10,
                n_passes=3,
                pass_cap=2000,
            )
        )
        logger.report(
            OrientationOptimized(
                rotation_index=8,
                score=0.5,
                residual="wr2",
                n_matched_hkl=3,
                n_trials=10,
                n_passes=3,
                pass_cap=2000,
            )
        )
        return _FakePlan()

    monkeypatch.setattr("diffBloch.app.cli.preprocess_experiment", fake_preprocess_experiment)
    rc = main(["run", "preprocess", "/some/experiment"])

    assert rc == 0
    assert captured["dir"] == "/some/experiment"
    assert isinstance(captured["logger"], MultiLogger)
    assert captured["checkpoint"] is True and captured["refresh"] is False
    assert captured["workers"] == 1 and captured["device"] == "cuda"
    out = capsys.readouterr().out
    assert "PREPROCESS COMPLETE" in out
    assert _summary_row(out, "Rotations", "2")
    assert _summary_row(out, "Total HKLs", "7")
    assert _summary_row(out, "Matched HKLs", "5")
    assert _summary_row(out, "Mean wR2", "0.375")
    assert "Optimize Orientation" in out
    assert "Optimize Thickness" in out
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
        **_kwargs: object,
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
    history = [
        SimpleNamespace(wr2=0.2, r_obs=0.3, diff_loss=2.0),
        SimpleNamespace(wr2=0.1, r_obs=0.2, diff_loss=1.0),
    ]
    return SimpleNamespace(
        losses=torch.tensor([2.0, 1.0]),
        best_loss=1.0,
        best_step=1,
        history=history,
        reflection_counts={
            "matched": 12,
            "matched_i_gt_3sigma": 8,
            "matched_i_le_3sigma": 4,
            "unmatched_observed": 3,
        },
        artifacts={"refined_structure": "/tmp/refined_structure.cif"},
    )


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
        verbose: bool = False,
        profile: bool = False,
        checkpoint_activations: bool = True,
        **_kwargs: object,
    ) -> SimpleNamespace:
        captured["dir"] = experiment_dir
        captured["logger"] = logger
        captured["checkpoint"] = checkpoint
        return _fake_refinement_result()

    monkeypatch.setattr("diffBloch.app.cli.refine_experiment", fake_refine_experiment)
    rc = main(["run", "refine", "/some/experiment"])

    assert rc == 0
    assert captured["dir"] == "/some/experiment"
    # The refine path fans out to the console and to the summary sink that writes
    # refinement_report.txt -- composed here, not inside refine_experiment.
    logger = captured["logger"]
    assert isinstance(logger, MultiLogger)
    assert [type(s).__name__ for s in logger.loggers] == ["ConsoleLogger", "SummaryLogger"]
    out = capsys.readouterr().out
    assert "REFINEMENT COMPLETE" in out
    assert _summary_row(out, "Best epoch", "2")
    assert _summary_row(out, "wR2", "0.1")
    assert _summary_row(out, "R_obs", "0.2")
    assert _summary_row(out, "Diffraction loss", "1")
    assert _summary_row(out, "HKLs (Observed/total)", "8 / 12")
    assert "Refined Structure" in out
    assert "/tmp/refined_structure.cif" in out


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
        verbose: bool = False,
        profile: bool = False,
        checkpoint_activations: bool = True,
        **_kwargs: object,
    ) -> SimpleNamespace:
        seen["checkpoint"] = checkpoint
        seen["refresh"] = refresh
        seen["device"] = device
        seen["workers"] = workers
        seen["verbose"] = verbose
        seen["profile"] = profile
        seen["checkpoint_activations"] = checkpoint_activations
        return _fake_refinement_result()

    monkeypatch.setattr("diffBloch.app.cli.refine_experiment", fake_refine_experiment)
    rc = main(
        ["run", "refine", "x"]
        + ["--no-checkpoint", "--refresh", "--device", "cuda", "--workers", "4"]
        + ["--verbose-refinement", "--profile", "--no-checkpoint-activations"]
    )
    assert rc == 0
    assert seen == {
        "checkpoint": False,
        "refresh": True,
        "device": "cuda",
        "workers": 4,
        "verbose": True,
        "profile": True,
        "checkpoint_activations": False,
    }


def test_run_refine_missing_experiment_reports_concise_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["run", "refine", "/no/such/experiment"])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_run_converge_accepts_the_sink_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """converge emits the sweep stream, so it takes the same sinks as every other subcommand."""
    seen: dict[str, object] = {}

    def fake_converge_experiment(
        experiment_dir: str, *, logger: object, device: object, n_orientations: int
    ) -> SimpleNamespace:
        seen["logger"] = logger
        return SimpleNamespace(g_max=2.5, sg_max=0.02, tilt_steps=46)

    monkeypatch.setattr("diffBloch.app.cli.converge_experiment", fake_converge_experiment)
    assert main(["run", "converge", "/some/experiment", "--quiet"]) == 0
    assert isinstance(seen["logger"], NullLogger)  # --quiet reaches it


def test_released_sinks_give_back_their_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live display owns the terminal; a run with no terminal event must still release it."""
    closed: list[str] = []

    class _Holder:
        def report(self, event: object) -> None:
            return None

        def close(self) -> None:
            closed.append("released")

    _release_sinks(MultiLogger((_Holder(), NullLogger())))  # type: ignore[arg-type]
    assert closed == ["released"]
    _release_sinks(NullLogger())  # a sink with no close() is simply skipped

    # Fan-outs nest: `run refine --tui --csv` puts _build_logger's own MultiLogger inside a second
    # one alongside the summary sink, so a single-level walk would miss the display entirely.
    closed.clear()
    nested = MultiLogger((MultiLogger((_Holder(), NullLogger())), NullLogger()))  # type: ignore[arg-type]
    _release_sinks(nested)
    assert closed == ["released"]


@pytest.mark.parametrize("command", ["infer", "preprocess", "refine", "converge"])
def test_tui_without_the_extra_reports_concisely(
    command: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing optional extra is a user mistake: a message and exit 1, never a traceback."""
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "rich" else real(name, *a, **k),
    )

    assert main(["run", command, "/some/experiment", "--tui"]) == 1
    captured = capsys.readouterr()
    assert "uv sync --extra tui" in captured.err
    assert "Traceback" not in captured.err
