"""Experiment locks keep input identity separate from generated artifacts."""

from pathlib import Path

import pytest

from diffBloch.config import (
    ExperimentLock,
    InputLock,
    PreprocessLock,
    RecipeStep,
    RefinementLock,
    artifact_hash_for,
    code_version,
    config_digest,
    input_lock_for,
    load_config,
    load_experiment,
    preprocess_lock_status,
    read_preprocess_lock,
    read_refinement_lock,
    refinement_config_digest,
    sha256_file,
    write_preprocess_lock,
    write_refinement_lock,
)
from diffBloch.config.manifest import _verify_experimental_data

LOCKED = Path(__file__).parent.parent / "fixtures" / "locked_min"


def _lock_and_recipe(tmp_path: Path):
    """A written plan.npz + its fresh PreprocessLock + the recipe/config it was built against."""
    cfg = load_config(LOCKED / "experiment.yaml")
    recipe = [RecipeStep(name="select_beams", params={"__type__": "BeamSelection", "rsg": 0.9})]
    npz = tmp_path / "plan.npz"
    npz.write_bytes(b"fake-checkpoint-bytes")
    lock = PreprocessLock(
        experiment_lock_sha256=sha256_file(LOCKED / "reproducibility" / "experiment.lock"),
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
    (experiment / "reproducibility").mkdir()
    for name in ["experiment.yaml", "enantiomer_1.cif", "exp_data.cif_pets"]:
        (experiment / name).write_bytes((LOCKED / name).read_bytes())
    (experiment / "reproducibility" / "experiment.lock").write_bytes(
        (LOCKED / "reproducibility" / "experiment.lock").read_bytes()
    )
    (experiment / "enantiomer_1.cif").write_text("changed\n")

    with pytest.raises(ValueError, match="input drift"):
        load_experiment(experiment)


def test_experiment_lock_accepts_a_list_of_input_locks_for_pooled_datasets() -> None:
    a = InputLock(ref="a.cif_pets", sha256="a" * 64, bytes=1)
    b = InputLock(ref="b.cif_pets", sha256="b" * 64, bytes=2)
    lock = ExperimentLock(
        structure=InputLock(ref="s.cif", sha256="c" * 64, bytes=3),
        experimental_data=[a, b],
    )
    assert lock.experimental_data == [a, b]


def test_verify_experimental_data_checks_every_pooled_file(tmp_path: Path) -> None:
    a = tmp_path / "a.cif_pets"
    b = tmp_path / "b.cif_pets"
    a.write_bytes(b"dataset a")
    b.write_bytes(b"dataset b")
    locks = [input_lock_for(a, ref="a.cif_pets"), input_lock_for(b, ref="b.cif_pets")]

    _verify_experimental_data(tmp_path, ["a.cif_pets", "b.cif_pets"], locks)

    b.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="input drift"):
        _verify_experimental_data(tmp_path, ["a.cif_pets", "b.cif_pets"], locks)


def test_verify_experimental_data_rejects_a_pooled_count_mismatch(tmp_path: Path) -> None:
    a = tmp_path / "a.cif_pets"
    a.write_bytes(b"dataset a")
    locks = [input_lock_for(a, ref="a.cif_pets")]

    with pytest.raises(ValueError, match="experimental_data entries"):
        _verify_experimental_data(tmp_path, ["a.cif_pets", "b.cif_pets"], locks)


def test_verify_experimental_data_rejects_a_list_vs_single_shape_mismatch(tmp_path: Path) -> None:
    a = tmp_path / "a.cif_pets"
    a.write_bytes(b"dataset a")
    single_lock = input_lock_for(a, ref="a.cif_pets")

    with pytest.raises(ValueError, match="shape does not match"):
        _verify_experimental_data(tmp_path, ["a.cif_pets"], single_lock)


# --- preprocess checkpoint lock: the four-axis freshness check ---


def _other_method(method: str) -> str:
    return "bloch_eigen" if method == "matrix_exp" else "matrix_exp"


def test_config_digest_is_stable_and_value_sensitive() -> None:
    cfg = load_config(LOCKED / "experiment.yaml")
    assert config_digest(cfg) == config_digest(load_config(LOCKED / "experiment.yaml"))
    # sensitive to a Plan-determining value (a numerics knob), not to the experiment label
    bumped = cfg.model_copy(
        update={"blochwave": cfg.blochwave.model_copy(update={"g_max": cfg.blochwave.g_max + 1.0})}
    )
    assert config_digest(bumped) != config_digest(cfg)


def test_config_digest_scopes_to_preprocess_determining_config() -> None:
    """The digest keys only on what determines the settled Plan, so unrelated config edits reuse."""
    cfg = load_config(LOCKED / "experiment.yaml")
    base = config_digest(cfg)

    def with_solver(**update: object) -> object:
        return cfg.model_copy(
            update={
                "blochwave": cfg.blochwave.model_copy(
                    update={"solver": cfg.blochwave.solver.model_copy(update=update)}
                )
            }
        )

    def with_refinement(**update: object) -> object:
        return cfg.model_copy(update={"refinement": cfg.refinement.model_copy(update=update)})

    # excluded -- cannot alter the preprocess Plan, so must not restale the checkpoint
    assert config_digest(cfg.model_copy(update={"name": "different"})) == base
    assert (
        config_digest(with_solver(inference=_other_method(cfg.blochwave.solver.inference))) == base
    )
    assert (
        config_digest(
            with_refinement(optimizer=cfg.refinement.optimizer.model_copy(update={"name": "adam"}))
        )
        == base
    )
    # included -- determine the settled Plan, so a change must restale
    assert config_digest(with_solver(refine=_other_method(cfg.blochwave.solver.refine))) != base
    assert (
        config_digest(
            with_refinement(split=cfg.refinement.split.model_copy(update={"val_frac": 0.3}))
        )
        != base
    )
    # included -- objective drives optimize_orientation/optimize_thickness's search too, not just
    # the gradient refinement stage, so it must restale the preprocess checkpoint like split does.
    assert (
        config_digest(
            cfg.model_copy(
                update={"loss_metrics": cfg.loss_metrics.model_copy(update={"residual": "robs"})}
            )
        )
        != base
    )


def test_config_digest_excludes_orientation_thickness_when_their_step_is_off() -> None:
    """orientation/thickness sub-config only counts when the matching optimize_* flag runs it."""
    cfg = load_config(LOCKED / "experiment.yaml")
    assert cfg.preprocess.optimize_orientation is True
    assert cfg.preprocess.optimize_thickness is True

    off = cfg.model_copy(
        update={
            "preprocess": cfg.preprocess.model_copy(
                update={"optimize_orientation": False, "optimize_thickness": False}
            )
        }
    )
    base = config_digest(off)

    bumped_orientation = off.model_copy(
        update={
            "preprocess": off.preprocess.model_copy(
                update={
                    "orientation": off.preprocess.orientation.model_copy(
                        update={
                            "nelder_mead": off.preprocess.orientation.nelder_mead.model_copy(
                                update={
                                    "step_size": off.preprocess.orientation.nelder_mead.step_size
                                    + 1.0
                                }
                            )
                        }
                    )
                }
            )
        }
    )
    assert config_digest(bumped_orientation) == base  # step never runs -> can't have mattered

    bumped_thickness = off.model_copy(
        update={
            "preprocess": off.preprocess.model_copy(
                update={
                    "thickness": off.preprocess.thickness.model_copy(
                        update={"n_steps": off.preprocess.thickness.n_steps + 1}
                    )
                }
            )
        }
    )
    assert config_digest(bumped_thickness) == base  # step never runs -> can't have mattered

    # flip the flags back on: now the same edits DO restale the digest
    on = off.model_copy(
        update={
            "preprocess": off.preprocess.model_copy(
                update={"optimize_orientation": True, "optimize_thickness": True}
            )
        }
    )
    on_base = config_digest(on)
    on_bumped = on.model_copy(
        update={
            "preprocess": on.preprocess.model_copy(
                update={
                    "thickness": on.preprocess.thickness.model_copy(
                        update={"n_steps": on.preprocess.thickness.n_steps + 1}
                    )
                }
            )
        }
    )
    assert config_digest(on_bumped) != on_base


def test_config_digest_excludes_thickness_plot() -> None:
    """thickness.plot only selects PNG output; it never touches the fitted Plan."""
    cfg = load_config(LOCKED / "experiment.yaml")
    assert cfg.preprocess.optimize_thickness is True
    base = config_digest(cfg)

    plotted = cfg.model_copy(
        update={
            "preprocess": cfg.preprocess.model_copy(
                update={"thickness": cfg.preprocess.thickness.model_copy(update={"plot": True})}
            )
        }
    )
    assert config_digest(plotted) == base


def test_refinement_config_digest_is_the_complement_of_config_digest() -> None:
    """refinement_config_digest tracks exactly what config_digest excludes, and nothing else."""
    cfg = load_config(LOCKED / "experiment.yaml")
    base = refinement_config_digest(cfg)

    # included -- these determine the gradient-refined result on top of a settled Plan
    bumped_steps = cfg.model_copy(
        update={"refinement": cfg.refinement.model_copy(update={"steps": cfg.refinement.steps + 1})}
    )
    assert refinement_config_digest(bumped_steps) != base
    bumped_optimizer = cfg.model_copy(
        update={
            "refinement": cfg.refinement.model_copy(
                update={"optimizer": cfg.refinement.optimizer.model_copy(update={"name": "adam"})}
            )
        }
    )
    assert refinement_config_digest(bumped_optimizer) != base

    # excluded -- split shapes the Plan itself and is already covered by config_digest
    bumped_split = cfg.model_copy(
        update={
            "refinement": cfg.refinement.model_copy(
                update={"split": cfg.refinement.split.model_copy(update={"val_frac": 0.3})}
            )
        }
    )
    assert refinement_config_digest(bumped_split) == base

    # excluded -- objective is top-level (drives preprocess search too) and already covered by
    # config_digest, exactly like split
    bumped_objective = cfg.model_copy(
        update={"loss_metrics": cfg.loss_metrics.model_copy(update={"residual": "robs"})}
    )
    assert refinement_config_digest(bumped_objective) == base

    # excluded -- preprocess-only config never enters the refinement-stage digest
    bumped_blochwave = cfg.model_copy(
        update={"blochwave": cfg.blochwave.model_copy(update={"g_max": cfg.blochwave.g_max + 1.0})}
    )
    assert refinement_config_digest(bumped_blochwave) == base


def test_refinement_lock_round_trips(tmp_path: Path) -> None:
    lock = RefinementLock(
        plan_lock_sha256="ab" * 32,
        refinement_config_digest="cd" * 32,
        code_version=code_version(),
        refined_structure=artifact_hash_for(
            _write(tmp_path / "refined_structure.cif", "data"), root=tmp_path
        ),
        refined_parameters=artifact_hash_for(
            _write(tmp_path / "refined_parameters.npz", "data"), root=tmp_path
        ),
    )
    path = tmp_path / "refinement.lock"
    write_refinement_lock(path, lock)
    assert read_refinement_lock(path) == lock


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


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
        experiment_lock_sha256=sha256_file(LOCKED / "reproducibility" / "experiment.lock"),
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
                "blochwave": cfg.blochwave.model_copy(update={"g_max": cfg.blochwave.g_max + 1.0})
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
