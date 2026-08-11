"""The default recipe's large-cell fork (``_recipe_steps``): routing + checkpoint transparency.

The orientation fit forks on unit-cell volume -- a large cell skips the per-trial gather
integrity checks; a small cell takes the exact, fully-validated path. Two properties matter and are
pinned here: (1) the fork routes at ``_LARGE_CELL_THRESHOLD_A3``, and (2) it is *transparent to the
recipe identity* -- both branches record the same step names/params, so resolving it (as
``_prepare`` does) never changes the lock. Property (2) keeps the committed quartz checkpoint valid.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from diffBloch.app.program import (
    _LARGE_CELL_THRESHOLD_A3,
    _preprocess,
    _recipe_steps,
    _select_device,
    preprocess_experiment,
)
from diffBloch.config import load_experiment
from diffBloch.io import read_experimental_data, read_structure
from diffBloch.observability import NULL_LOGGER, DeviceSelected, RecordingLogger
from diffBloch.preprocess import from_experiment, resolve_recipe, step_records
from diffBloch.preprocess.pipeline import Fork

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _steps(material: str = "quartz_anchor") -> list:
    root = FIXTURES / material
    cfg, _ = load_experiment(root)
    structure = read_structure(root / cfg.inputs.structure)
    experimental_data = read_experimental_data(root / cfg.inputs.exp_data)
    setup = from_experiment(structure, experimental_data, cfg)
    return _recipe_steps(cfg, setup.refinement, setup.integration, NULL_LOGGER)


def _the_fork() -> Fork:
    forks = [s for s in _steps() if isinstance(s, Fork)]
    assert len(forks) == 1, "the fit tail should be exactly one fork"
    return forks[0]


def test_select_device_falls_back_from_cuda_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = RecordingLogger()
    monkeypatch.setattr("diffBloch.app.program.torch.cuda.is_available", lambda: False)

    selected = _select_device("cuda", logger=logger)

    assert selected == "cpu"
    assert logger.events == [DeviceSelected(requested="cuda", selected="cpu", cuda_available=False)]


def test_select_device_keeps_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = RecordingLogger()
    monkeypatch.setattr("diffBloch.app.program.torch.cuda.is_available", lambda: True)

    selected = _select_device("cuda", logger=logger)

    assert selected == "cuda"
    assert logger.events == [DeviceSelected(requested="cuda", selected="cuda", cuda_available=True)]


def test_preprocess_experiment_default_device_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    plan = SimpleNamespace()

    def fake_preprocess(
        *args: object, **kwargs: object
    ) -> tuple[object, object, object, object, object]:
        seen["device"] = kwargs["device"]
        return object(), object(), plan, object(), None

    monkeypatch.setattr("diffBloch.app.program.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("diffBloch.app.program.load_experiment", lambda _root: (object(), object()))
    monkeypatch.setattr("diffBloch.app.program._preprocess", fake_preprocess)

    assert preprocess_experiment("experiment-dir") is plan
    assert seen["device"] == "cpu"


def test_fork_predicate_routes_at_the_threshold() -> None:
    """Just above the volume threshold -> large branch; just below -> small branch."""
    fork = _the_fork()
    assert fork.predicate(SimpleNamespace(cell_volume=_LARGE_CELL_THRESHOLD_A3 + 1.0)) is True
    assert fork.predicate(SimpleNamespace(cell_volume=_LARGE_CELL_THRESHOLD_A3 - 1.0)) is False


def test_coupling_config_flows_into_the_fit_orientation_record() -> None:
    """A ``blochwave`` override reaches the fit's recorded per-trial policy.

    The recipe reads its coupling from config (not a hardcoded ``UnionCoupling()``), so a
    config that overrides the SOLVE-union bounds re-keys the ``optimize_orientation`` step -- the
    mechanism that lets abiraterone's ``fixed_n_segments=4`` coupling be expressed through the
    standard config path.
    """
    root = FIXTURES / "quartz_anchor"
    cfg, _ = load_experiment(root)
    cfg = cfg.model_copy(
        update={"blochwave": cfg.blochwave.model_copy(update={"fixed_n_segments": 4})}
    )
    structure = read_structure(root / cfg.inputs.structure)
    experimental_data = read_experimental_data(root / cfg.inputs.exp_data)
    setup = from_experiment(structure, experimental_data, cfg)
    steps = _recipe_steps(cfg, setup.refinement, setup.integration, NULL_LOGGER)
    records = step_records(resolve_recipe(steps, SimpleNamespace(cell_volume=100.0)))
    fit = next(r for r in records if r.name == "optimize_orientation")
    assert fit.params["coupling"]["policy"]["fixed_n_segments"] == 4


def test_fork_is_transparent_to_recipe_identity() -> None:
    """For one config, the small-cell and large-cell branches resolve to the *same* records.

    Resolving the *same* recipe against a below-threshold vs above-threshold grid takes the
    validated vs fast-path branch, yet both record identical step names *and* params: ``validate``
    lives only in the branch closures (execution-only), never in the step identity. This is the
    property that keeps the committed quartz checkpoint reusable after the fork lands (the fork is
    invisible to the lock).
    """
    steps = _steps("quartz_anchor")
    small = step_records(resolve_recipe(steps, SimpleNamespace(cell_volume=100.0)))
    large = step_records(resolve_recipe(steps, SimpleNamespace(cell_volume=5000.0)))
    names = [
        "build_orientation_plans",
        "optimize_orientation",
        "optimize_thickness",
    ]
    assert [r.name for r in small] == names == [r.name for r in large]
    # identical params too (optimize_orientation records only {search, coupling}; no validate)
    assert [r.params for r in small] == [r.params for r in large]


def test_orientations_csv_prepends_an_import_orientations_step(tmp_path: Path) -> None:
    root = FIXTURES / "quartz_anchor"
    cfg, _ = load_experiment(root)
    structure = read_structure(root / cfg.inputs.structure)
    experimental_data = read_experimental_data(root / cfg.inputs.exp_data)
    setup = from_experiment(structure, experimental_data, cfg)
    csv_path = tmp_path / "orientations.csv"
    csv_path.write_text("Rotation Index,Orientation Matrix\n")

    records = step_records(
        resolve_recipe(
            _recipe_steps(
                cfg,
                setup.refinement,
                setup.integration,
                NULL_LOGGER,
                orientations_csv=csv_path,
            ),
            SimpleNamespace(cell_volume=100.0),
        )
    )

    assert records[0].name == "import_orientations"
    assert records[0].params == {"csv_path": str(csv_path)}
    assert [r.name for r in records[1:]] == [
        "build_orientation_plans",
        "optimize_orientation",
        "optimize_thickness",
    ]


def test_stage_order_thickness_first_runs_thickness_before_orientation() -> None:
    root = FIXTURES / "quartz_anchor"
    cfg, _ = load_experiment(root)
    cfg = cfg.model_copy(
        update={"preprocess": cfg.preprocess.model_copy(update={"stage_order": "thickness_first"})}
    )
    structure = read_structure(root / cfg.inputs.structure)
    experimental_data = read_experimental_data(root / cfg.inputs.exp_data)
    setup = from_experiment(structure, experimental_data, cfg)

    records = step_records(
        resolve_recipe(
            _recipe_steps(cfg, setup.refinement, setup.integration, NULL_LOGGER),
            SimpleNamespace(cell_volume=100.0),
        )
    )

    assert [r.name for r in records] == [
        "build_orientation_plans",
        "optimize_thickness",
        "optimize_orientation",
    ]


def test_preprocess_wraps_the_logger_with_a_thickness_plot_logger(tmp_path: Path) -> None:
    """``plot_thickness`` (API/CLI) composes a ``ThicknessPlotLogger`` into the recipe's logger.

    Both fitting stages are disabled so only the unconditional ``build_orientation_plans`` geometry
    build runs -- fast, and enough to exercise the composition branch without paying for a search.
    """
    root = FIXTURES / "quartz_anchor"
    cfg, _ = load_experiment(root)
    cfg = cfg.model_copy(
        update={
            "preprocess": cfg.preprocess.model_copy(
                update={"optimize_orientation": False, "optimize_thickness": False}
            )
        }
    )
    plot_dir = tmp_path / "thickness_optim"

    refinement, _integrations, plan, _validation_rotation_indices, _plan_lock_sha256s = _preprocess(
        root,
        cfg,
        logger=NULL_LOGGER,
        checkpoint=False,
        refresh=False,
        device=None,
        workers=1,
        max_batch=None,
        plot_thickness=True,
        plot_thickness_dir=plot_dir,
    )

    assert plan.orientations  # the geometry build actually ran
    assert refinement is not None
    # no thickness fit ran, so no PNG was written -- but the branch itself (directory resolution +
    # MultiLogger composition) executed without error, which is what this test pins.
    assert plot_dir.is_dir()


def test_fit_stages_can_be_enabled_independently() -> None:
    root = FIXTURES / "quartz_anchor"
    cfg, _ = load_experiment(root)
    cfg = cfg.model_copy(
        update={
            "preprocess": cfg.preprocess.model_copy(
                update={
                    "optimize_orientation": True,
                    "optimize_thickness": False,
                }
            )
        }
    )
    structure = read_structure(root / cfg.inputs.structure)
    experimental_data = read_experimental_data(root / cfg.inputs.exp_data)
    setup = from_experiment(structure, experimental_data, cfg)
    records = step_records(
        resolve_recipe(
            _recipe_steps(cfg, setup.refinement, setup.integration, NULL_LOGGER),
            SimpleNamespace(cell_volume=100.0),
        )
    )
    assert [record.name for record in records] == [
        "build_orientation_plans",
        "optimize_orientation",
    ]
