"""Experiment locks and run manifests keep input identity separate from generated artifacts."""

import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest

from diffBloch.config import (
    PreprocessLock,
    RecipeStep,
    RunManifest,
    artifact_hash_for,
    code_version,
    config_digest,
    load_config,
    load_experiment,
    pack_run,
    preprocess_lock_status,
    read_preprocess_lock,
    sha256_file,
    write_preprocess_lock,
    write_run_manifest,
)

LOCKED = Path(__file__).parent.parent / "fixtures" / "locked_min"


def _lock_and_recipe(tmp_path: Path):
    """A written plan.npz + its fresh PreprocessLock + the recipe/config it was built against."""
    cfg = load_config(LOCKED / "experiment.yaml")
    recipe = [RecipeStep(name="select_beams", params={"__type__": "BeamSelection", "rsg": 0.9})]
    npz = tmp_path / "plan.npz"
    npz.write_bytes(b"fake-checkpoint-bytes")
    lock = PreprocessLock(
        experiment_lock_sha256=sha256_file(LOCKED / "experiment.lock"),
        config_digest=config_digest(cfg),
        code_version=code_version(),
        recipe=recipe,
        plan=artifact_hash_for(npz, root=tmp_path),
    )
    return cfg, recipe, npz, lock


def test_load_experiment_verifies_locked_input_bytes() -> None:
    cfg, lock = load_experiment(LOCKED)
    assert cfg.name == "locked-min"
    assert lock.structure.ref == "enantiomer_1.cif"
    assert lock.structure.sha256 == sha256_file(LOCKED / "enantiomer_1.cif")


def test_load_experiment_detects_input_drift(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    for name in ["experiment.yaml", "experiment.lock", "enantiomer_1.cif", "exp_data.cif_pets"]:
        (experiment / name).write_bytes((LOCKED / name).read_bytes())
    (experiment / "enantiomer_1.cif").write_text("changed\n")

    with pytest.raises(ValueError, match="input drift"):
        load_experiment(experiment)


def test_run_manifest_hashes_generated_artifacts(tmp_path: Path) -> None:
    run = tmp_path / "run_001"
    run.mkdir()
    for name in [
        "resolved_config.yaml",
        "config.diff.yaml",
        "data_used.parquet",
        "history.jsonl",
        "parameter_table.parquet",
    ]:
        (run / name).write_text(f"{name}\n")

    manifest = RunManifest(
        experiment_lock_sha256="input-lock-hash",
        resolved_config=artifact_hash_for(run / "resolved_config.yaml", root=run),
        config_diff=artifact_hash_for(run / "config.diff.yaml", root=run),
        data_used=artifact_hash_for(run / "data_used.parquet", root=run),
        history=artifact_hash_for(run / "history.jsonl", root=run),
        parameter_table=artifact_hash_for(run / "parameter_table.parquet", root=run),
        code_version="test",
        environment={"python": "test"},
    )
    write_run_manifest(run / "run_manifest.json", manifest)

    loaded = RunManifest.model_validate_json((run / "run_manifest.json").read_text())
    assert loaded.resolved_config.path == "resolved_config.yaml"
    assert loaded.experiment_lock_sha256 == "input-lock-hash"


def test_pack_run_exports_zip_preserving_full_run_name(tmp_path: Path) -> None:
    run = tmp_path / "run.v2"
    run.mkdir()
    (run / "run_manifest.json").write_text("{}\n")
    (run / "history.jsonl").write_text("{}\n")

    package = pack_run(run, package_format="zip")
    assert package.is_file()
    assert package.name == "run.v2.zip"
    with ZipFile(package) as archive:
        assert "run.v2/run_manifest.json" in archive.namelist()


@pytest.mark.parametrize(
    ("package_format", "expected_name"),
    [
        ("tar", "run_001.tar"),
        ("bagit", "run_001.bagit.zip"),
        ("ro-crate", "run_001.ro-crate.zip"),
    ],
)
def test_pack_run_export_formats(tmp_path: Path, package_format: str, expected_name: str) -> None:
    run = tmp_path / "run_001"
    run.mkdir()
    (run / "run_manifest.json").write_text("{}\n")
    (run / "history.jsonl").write_text("{}\n")

    package = pack_run(run, package_format=package_format)
    assert package.name == expected_name
    if package_format == "tar":
        with tarfile.open(package) as archive:
            assert "run_001/run_manifest.json" in archive.getnames()
    else:
        with ZipFile(package) as archive:
            names = archive.namelist()
        assert any(name.endswith("run_manifest.json") for name in names)


def test_pack_run_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pack_run(tmp_path / "missing_manifest")


# --- preprocess checkpoint lock: the four-axis freshness check ---


def _other_method(method: str) -> str:
    return "bloch_eigen" if method == "matrix_exp" else "matrix_exp"


def test_config_digest_is_stable_and_value_sensitive() -> None:
    cfg = load_config(LOCKED / "experiment.yaml")
    assert config_digest(cfg) == config_digest(load_config(LOCKED / "experiment.yaml"))
    # sensitive to a Plan-determining value (a numerics knob), not to the experiment label
    bumped = cfg.model_copy(
        update={
            "numerics": cfg.numerics.model_copy(
                update={"g_max_refine": cfg.numerics.g_max_refine + 1.0}
            )
        }
    )
    assert config_digest(bumped) != config_digest(cfg)


def test_config_digest_scopes_to_preprocess_determining_config() -> None:
    """The digest keys only on what determines the settled Plan, so unrelated config edits reuse."""
    cfg = load_config(LOCKED / "experiment.yaml")
    base = config_digest(cfg)

    def with_solver(**update: object) -> object:
        return cfg.model_copy(update={"solver": cfg.solver.model_copy(update=update)})

    def with_refinement(**update: object) -> object:
        return cfg.model_copy(update={"refinement": cfg.refinement.model_copy(update=update)})

    # excluded -- cannot alter the preprocess Plan, so must not restale the checkpoint
    assert config_digest(cfg.model_copy(update={"name": "different"})) == base
    assert config_digest(with_solver(inference=_other_method(cfg.solver.inference))) == base
    assert (
        config_digest(
            with_refinement(
                objective=cfg.refinement.objective.model_copy(update={"data_term": "least_squares"})
            )
        )
        == base
    )
    assert (
        config_digest(
            with_refinement(optimizer=cfg.refinement.optimizer.model_copy(update={"name": "adam"}))
        )
        == base
    )
    # included -- determine the settled Plan, so a change must restale
    assert config_digest(with_solver(refine=_other_method(cfg.solver.refine))) != base
    assert (
        config_digest(
            with_refinement(
                split=cfg.refinement.split.model_copy(update={"validation": "every_5th_rotation"})
            )
        )
        != base
    )


def test_code_version_carries_the_package_version() -> None:
    from diffBloch import __version__

    assert code_version().startswith(__version__)  # bare, or "<version>+g<sha>[.dirty]"


def test_preprocess_lock_round_trips(tmp_path: Path) -> None:
    _cfg, _recipe, _npz, lock = _lock_and_recipe(tmp_path)
    path = tmp_path / "plan.lock"
    write_preprocess_lock(path, lock)
    assert read_preprocess_lock(path) == lock


def _args(cfg, recipe, npz, tmp_path):
    return dict(
        experiment_lock_sha256=sha256_file(LOCKED / "experiment.lock"),
        config_digest=config_digest(cfg),
        code_version=code_version(),
        recipe=recipe,
        plan_path=npz,
        root=tmp_path,
    )


def test_identical_recipe_reuses(tmp_path: Path) -> None:
    cfg, recipe, npz, lock = _lock_and_recipe(tmp_path)
    assert preprocess_lock_status(lock, **_args(cfg, recipe, npz, tmp_path)) == "reuse"


def test_appended_step_resumes(tmp_path: Path) -> None:
    cfg, recipe, npz, lock = _lock_and_recipe(tmp_path)
    args = _args(cfg, recipe, npz, tmp_path)
    args["recipe"] = [*recipe, RecipeStep(name="fit_thickness", params=None)]  # tail extension
    assert preprocess_lock_status(lock, **args) == "resume"


def test_changed_middle_step_is_stale(tmp_path: Path) -> None:
    cfg, recipe, npz, lock = _lock_and_recipe(tmp_path)
    args = _args(cfg, recipe, npz, tmp_path)
    changed = {"__type__": "BeamSelection", "rsg": 0.5}
    args["recipe"] = [RecipeStep(name="select_beams", params=changed)]
    assert preprocess_lock_status(lock, **args) == "stale"


def test_shorter_recipe_is_stale(tmp_path: Path) -> None:
    cfg, recipe, npz, lock = _lock_and_recipe(tmp_path)
    args = _args(cfg, recipe, npz, tmp_path)
    args["recipe"] = []  # lock's recipe is longer than intended -> not a prefix of it
    assert preprocess_lock_status(lock, **args) == "stale"


def test_config_change_is_stale(tmp_path: Path) -> None:
    cfg, recipe, npz, lock = _lock_and_recipe(tmp_path)
    args = _args(cfg, recipe, npz, tmp_path)
    args["config_digest"] = config_digest(
        cfg.model_copy(
            update={
                "numerics": cfg.numerics.model_copy(
                    update={"g_max_refine": cfg.numerics.g_max_refine + 1.0}
                )
            }
        )
    )
    assert preprocess_lock_status(lock, **args) == "stale"


def test_code_version_change_is_stale(tmp_path: Path) -> None:
    cfg, recipe, npz, lock = _lock_and_recipe(tmp_path)
    args = _args(cfg, recipe, npz, tmp_path)
    args["code_version"] = "9.9.9+deadbeef"  # a different *release* -> stale
    assert preprocess_lock_status(lock, **args) == "stale"


def test_same_release_different_sha_reuses(tmp_path: Path) -> None:
    """The reuse gate keys on the release version, so a differing git SHA still reuses.

    This is what makes a *committed* checkpoint usable: the lock records the SHA of whatever commit
    generated it, but any later commit of the same release must still reuse it (else a shipped
    checkpoint would go stale on the very next commit).
    """
    from diffBloch import __version__

    cfg, recipe, npz, lock = _lock_and_recipe(tmp_path)
    # Lock stamped by one commit; current run is a different commit (and dirty) of the SAME release.
    stamped = lock.model_copy(update={"code_version": f"{__version__}+gdeadbeef"})
    args = _args(cfg, recipe, npz, tmp_path)
    args["code_version"] = f"{__version__}+gfeedface.dirty"
    assert preprocess_lock_status(stamped, **args) == "reuse"


def test_input_drift_is_stale(tmp_path: Path) -> None:
    cfg, recipe, npz, lock = _lock_and_recipe(tmp_path)
    args = _args(cfg, recipe, npz, tmp_path)
    args["experiment_lock_sha256"] = "0" * 64
    assert preprocess_lock_status(lock, **args) == "stale"


def test_tampered_checkpoint_is_stale(tmp_path: Path) -> None:
    cfg, recipe, npz, lock = _lock_and_recipe(tmp_path)
    npz.write_bytes(b"tampered-different-bytes")  # same path, changed bytes
    assert preprocess_lock_status(lock, **_args(cfg, recipe, npz, tmp_path)) == "stale"


def test_missing_checkpoint_is_stale(tmp_path: Path) -> None:
    cfg, recipe, npz, lock = _lock_and_recipe(tmp_path)
    npz.unlink()
    assert preprocess_lock_status(lock, **_args(cfg, recipe, npz, tmp_path)) == "stale"
