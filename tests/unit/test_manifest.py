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
    dataset_config_digest,
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
EXP_REF = "exp_data.cif_pets"  # locked_min's single inputs.exp_data ref


def _input_locks(cfg) -> tuple[InputLock, InputLock]:
    """The locked_min fixture's structure + (single) dataset InputLocks."""
    assert isinstance(cfg.inputs.exp_data, str)
    return (
        input_lock_for(LOCKED / cfg.inputs.structure, ref=cfg.inputs.structure),
        input_lock_for(LOCKED / cfg.inputs.exp_data, ref=cfg.inputs.exp_data),
    )


def _lock_and_recipe(tmp_path: Path):
    """A written plan npz + its fresh per-dataset PreprocessLock + the recipe/config behind it."""
    cfg = load_config(LOCKED / "experiment.yaml")
    structure_lock, dataset_lock = _input_locks(cfg)
    recipe = [RecipeStep(name="select_beams", params={"__type__": "BeamSelection", "rsg": 0.9})]
    npz = tmp_path / "plan.exp_data.npz"
    npz.write_bytes(b"fake-checkpoint-bytes")
    lock = PreprocessLock(
        structure=structure_lock,
        experimental_data=dataset_lock,
        ignored_rotations=(),
        config_digest=dataset_config_digest(cfg, exp_data=cfg.inputs.exp_data),
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


def test_dataset_config_digest_is_stable_and_value_sensitive() -> None:
    cfg = load_config(LOCKED / "experiment.yaml")
    reloaded = load_config(LOCKED / "experiment.yaml")
    assert dataset_config_digest(cfg, exp_data=EXP_REF) == dataset_config_digest(
        reloaded, exp_data=EXP_REF
    )
    # sensitive to a Plan-determining value (a numerics knob), not to the experiment label
    bumped = cfg.model_copy(
        update={"blochwave": cfg.blochwave.model_copy(update={"g_max": cfg.blochwave.g_max + 1.0})}
    )
    assert dataset_config_digest(bumped, exp_data=EXP_REF) != dataset_config_digest(
        cfg, exp_data=EXP_REF
    )


def test_dataset_config_digest_scopes_to_plan_determining_config() -> None:
    """The digest keys only on what determines the settled per-dataset Plan."""
    cfg = load_config(LOCKED / "experiment.yaml")
    base = dataset_config_digest(cfg, exp_data=EXP_REF)

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

    # excluded -- cannot alter the per-dataset preprocess Plan, so must not restale the checkpoint
    assert (
        dataset_config_digest(cfg.model_copy(update={"name": "different"}), exp_data=EXP_REF)
        == base
    )
    assert (
        dataset_config_digest(
            with_solver(inference=_other_method(cfg.blochwave.solver.inference)),
            exp_data=EXP_REF,
        )
        == base
    )
    assert (
        dataset_config_digest(
            with_refinement(optimizer=cfg.refinement.optimizer.model_copy(update={"name": "adam"})),
            exp_data=EXP_REF,
        )
        == base
    )
    # excluded -- split partitions rotations at refinement time and no longer shapes the
    # checkpointed plan; it rides in refinement_config_digest instead.
    assert (
        dataset_config_digest(
            with_refinement(split=cfg.refinement.split.model_copy(update={"val_frac": 0.3})),
            exp_data=EXP_REF,
        )
        == base
    )
    # included -- determine the settled Plan, so a change must restale
    assert (
        dataset_config_digest(
            with_solver(refine=_other_method(cfg.blochwave.solver.refine)), exp_data=EXP_REF
        )
        != base
    )
    # included -- objective drives optimize_orientation/optimize_thickness's search
    assert (
        dataset_config_digest(
            cfg.model_copy(
                update={"loss_metrics": cfg.loss_metrics.model_copy(update={"residual": "robs"})}
            ),
            exp_data=EXP_REF,
        )
        != base
    )


def test_dataset_config_digest_is_independent_of_other_datasets() -> None:
    """The core per-dataset property: nothing about the *pool* enters one dataset's digest.

    Changing exp_data list membership/order, the multi_dataset flag, the split, or the pooled
    ignore_orientations leaves a given file's digest unchanged -- so adding, removing, or
    reordering other datasets (or re-slicing the pool) never restales this one's checkpoint.
    """
    cfg = load_config(LOCKED / "experiment.yaml")
    base = dataset_config_digest(cfg, exp_data=EXP_REF)

    pooled = cfg.model_copy(
        update={
            "inputs": cfg.inputs.model_copy(
                update={"exp_data": [EXP_REF, "other.cif_pets"], "multi_dataset": True}
            )
        }
    )
    assert dataset_config_digest(pooled, exp_data=EXP_REF) == base

    reordered = cfg.model_copy(
        update={
            "inputs": cfg.inputs.model_copy(
                update={"exp_data": ["other.cif_pets", EXP_REF], "multi_dataset": True}
            )
        }
    )
    assert dataset_config_digest(reordered, exp_data=EXP_REF) == base

    ignored = cfg.model_copy(
        update={"blochwave": cfg.blochwave.model_copy(update={"ignore_orientations": (3, 7)})}
    )
    assert dataset_config_digest(ignored, exp_data=EXP_REF) == base

    # ...while the digest still distinguishes WHICH dataset it describes
    assert dataset_config_digest(pooled, exp_data="other.cif_pets") != base


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
    base = dataset_config_digest(off, exp_data=EXP_REF)

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
    assert (
        dataset_config_digest(bumped_orientation, exp_data=EXP_REF) == base
    )  # step never runs -> can't have mattered

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
    assert (
        dataset_config_digest(bumped_thickness, exp_data=EXP_REF) == base
    )  # step never runs -> can't have mattered

    # flip the flags back on: now the same edits DO restale the digest
    on = off.model_copy(
        update={
            "preprocess": off.preprocess.model_copy(
                update={"optimize_orientation": True, "optimize_thickness": True}
            )
        }
    )
    on_base = dataset_config_digest(on, exp_data=EXP_REF)
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
    assert dataset_config_digest(on_bumped, exp_data=EXP_REF) != on_base


def test_config_digest_excludes_thickness_plot() -> None:
    """thickness.plot only selects PNG output; it never touches the fitted Plan."""
    cfg = load_config(LOCKED / "experiment.yaml")
    assert cfg.preprocess.optimize_thickness is True
    base = dataset_config_digest(cfg, exp_data=EXP_REF)

    plotted = cfg.model_copy(
        update={
            "preprocess": cfg.preprocess.model_copy(
                update={"thickness": cfg.preprocess.thickness.model_copy(update={"plot": True})}
            )
        }
    )
    assert dataset_config_digest(plotted, exp_data=EXP_REF) == base


def test_refinement_config_digest_is_the_complement_of_the_dataset_digest() -> None:
    """refinement_config_digest tracks exactly what dataset_config_digest excludes under
    refinement, and nothing else."""
    cfg = load_config(LOCKED / "experiment.yaml")
    base = refinement_config_digest(cfg)

    # included -- these determine the gradient-refined result on top of settled Plans
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

    # included -- split partitions rotations at refinement time (it left the preprocess digest)
    bumped_split = cfg.model_copy(
        update={
            "refinement": cfg.refinement.model_copy(
                update={"split": cfg.refinement.split.model_copy(update={"val_frac": 0.3})}
            )
        }
    )
    assert refinement_config_digest(bumped_split) != base

    # excluded -- objective is top-level (drives the preprocess search) and covered by
    # dataset_config_digest
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
        plan_lock_sha256s=["ab" * 32, "ef" * 32],
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
    structure_lock, dataset_lock = _input_locks(cfg)
    return dict(
        structure=structure_lock,
        experimental_data=dataset_lock,
        ignored_rotations=(),
        config_digest=dataset_config_digest(cfg, exp_data=EXP_REF),
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
    args["config_digest"] = dataset_config_digest(
        cfg.model_copy(
            update={
                "blochwave": cfg.blochwave.model_copy(update={"g_max": cfg.blochwave.g_max + 1.0})
            }
        ),
        exp_data=EXP_REF,
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


def test_dataset_byte_drift_is_stale(tmp_path: Path) -> None:
    cfg, recipe, npz, lock = _lock_and_recipe(tmp_path)
    args = _args(cfg, recipe, npz, tmp_path)
    args["experimental_data"] = InputLock(ref=EXP_REF, sha256="0" * 64, bytes=1)
    assert preprocess_lock_status(lock, **args) == "stale"


def test_structure_drift_is_stale(tmp_path: Path) -> None:
    cfg, recipe, npz, lock = _lock_and_recipe(tmp_path)
    args = _args(cfg, recipe, npz, tmp_path)
    args["structure"] = InputLock(ref=cfg.inputs.structure, sha256="0" * 64, bytes=1)
    assert preprocess_lock_status(lock, **args) == "stale"


def test_changed_file_local_ignore_set_is_stale(tmp_path: Path) -> None:
    cfg, recipe, npz, lock = _lock_and_recipe(tmp_path)
    args = _args(cfg, recipe, npz, tmp_path)
    args["ignored_rotations"] = (2,)
    assert preprocess_lock_status(lock, **args) == "stale"


def test_renamed_dataset_with_identical_bytes_still_reuses(tmp_path: Path) -> None:
    """Input identity is sha+bytes, never ref: a byte-identical rename is the same measurement."""
    cfg, recipe, npz, lock = _lock_and_recipe(tmp_path)
    args = _args(cfg, recipe, npz, tmp_path)
    renamed = args["experimental_data"].model_copy(update={"ref": "renamed.cif_pets"})
    args["experimental_data"] = renamed
    assert preprocess_lock_status(lock, **args) == "reuse"


def test_tampered_checkpoint_is_stale(tmp_path: Path) -> None:
    cfg, recipe, npz, lock = _lock_and_recipe(tmp_path)
    npz.write_bytes(b"tampered-different-bytes")  # same path, changed bytes
    assert preprocess_lock_status(lock, **_args(cfg, recipe, npz, tmp_path)) == "stale"


def test_missing_checkpoint_is_stale(tmp_path: Path) -> None:
    cfg, recipe, npz, lock = _lock_and_recipe(tmp_path)
    npz.unlink()
    assert preprocess_lock_status(lock, **_args(cfg, recipe, npz, tmp_path)) == "stale"
