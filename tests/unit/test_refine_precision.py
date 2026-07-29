"""Refinement precision wiring for the app-level refine path."""

from pathlib import Path
from types import SimpleNamespace

import torch

from diffBloch.app.program import refine_experiment
from diffBloch.config.schema import ExperimentConfig


def test_refine_experiment_threads_config_precision_to_engine(monkeypatch) -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "name": "q",
            "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"},
            "refinement": {"precision": "fp32"},
        }
    )
    refinement = SimpleNamespace(params=SimpleNamespace(to=lambda device: "moved"))
    prepared = object()
    seen: dict[str, object] = {}

    monkeypatch.setattr("diffBloch.app.program.load_experiment", lambda root: (cfg, object()))
    monkeypatch.setattr(
        "diffBloch.app.program._preprocess",
        lambda *args, **kwargs: (refinement, prepared),
    )

    def fake_build_engine(plan, setup, *, loss, method, precision, max_batch):
        seen["plan"] = plan
        seen["setup"] = setup
        seen["method"] = method
        seen["precision"] = precision
        seen["max_batch"] = max_batch
        return "engine"

    monkeypatch.setattr("diffBloch.app.program.build_engine", fake_build_engine)
    monkeypatch.setattr(
        "diffBloch.app.program.build_refinement_model",
        lambda *, initial: SimpleNamespace(initial=initial),
    )
    monkeypatch.setattr("diffBloch.app.program.build_refinement_problem", lambda: "problem")

    def fake_run_refinement_model(engine, model, problem, **kwargs):
        seen["engine"] = engine
        seen["initial"] = model.initial
        return SimpleNamespace(losses=torch.tensor([1.0]), best_loss=1.0, best_step=0)

    monkeypatch.setattr("diffBloch.app.program.run_refinement_model", fake_run_refinement_model)
    monkeypatch.setattr(
        "diffBloch.app.program._write_refinement_outputs",
        lambda root, cfg, refinement, result: result,
    )

    result = refine_experiment(Path("/experiment"), device="cuda", max_batch=7)

    assert result.best_loss == 1.0
    assert seen["plan"] is prepared
    assert seen["setup"] is refinement
    assert seen["method"] == "matrix_exp"
    assert seen["precision"] == "fp32"
    assert seen["max_batch"] == 7
    assert seen["engine"] == "engine"
    assert seen["initial"] == "moved"
