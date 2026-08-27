"""``SummaryLogger``: the run summary built from the event stream, not from orchestration state.

Every test here drives the sink the way a real run does -- by reporting events at it -- so these
also pin the contract that the events carry everything the summary needs. Reading the refined CIF
back off disk is the one deliberate exception, and ``RefinementOutputsWritten`` is what points at it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from diffBloch.app.loggers.summary import SummaryLogger, ascii_table
from diffBloch.observability import (
    ExperimentDeclared,
    IsotropicMosaicityRefined,
    ObjectiveManifest,
    ObjectiveTerm,
    PreprocessCompleted,
    RefinedRotationMetrics,
    RefinementCompleted,
    RefinementOutputsWritten,
    RefinementStep,
    RotationScored,
    ThicknessProfile,
)

_CIF = """data_q
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'P 1'
_space_group_IT_number 1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
C1 C 0.10 0.20 0.30 0.5 0.01
O1 O 0.40 0.50 0.60 0.5 0.02
"""


def _structure(tmp_path: Path) -> Path:
    path = tmp_path / "refined_structure.cif"
    path.write_text(_CIF)
    return path


def _experiment() -> ExperimentDeclared:
    return ExperimentDeclared(
        name="q",
        structure="q.cif",
        experimental_data="q.cif_pets",
        optimizer="lbfgs",
        seed_thicknesses_by_dataset=(("q.cif_pets", (820.0,)),),
        integration_semiangles=(1.0,),
        rocking_curve_sampling=42,
        dsg=0.0015,
        rsg=0.9,
        solve_g_max=2.25,
        sg_max=0.01,
        absorption=False,
        incoherent_mosaicity=False,
        steps=3,
        learning_rate=0.001,
    )


def _step(**overrides: object) -> RefinementStep:
    fields: dict[str, object] = {
        "iteration": 0,
        "loss": 0.2,
        "wr2": 0.1,
        "r_obs": 0.05,
        "diff_loss": 0.2,
        "objective_total": 0.35,
        "n_rotations": 4,
        "n_wr2_evaluated": 3,
        "n_r_obs_evaluated": 2,
        "components": {
            "diffraction": {"raw": 0.2, "weight": 1.0, "contribution": 0.2},
            "bond_length": {"raw": 0.05, "weight": 3.0, "contribution": 0.15},
        },
    }
    fields.update(overrides)
    return RefinementStep(**fields)  # type: ignore[arg-type]


def _run(
    tmp_path: Path,
    *,
    experiment: ExperimentDeclared | None = None,
    steps: tuple[RefinementStep, ...] = (),
    rotations: tuple[RefinedRotationMetrics, ...] = (),
    preprocess: PreprocessCompleted | None = None,
    profiles: tuple[ThicknessProfile, ...] = (),
    mosaicities: tuple[IsotropicMosaicityRefined, ...] = (),
    manifest: ObjectiveManifest | None = None,
    completed: RefinementCompleted | None = None,
) -> str:
    """Drive a SummaryLogger through one run's worth of events and return the written text."""
    structure = _structure(tmp_path)
    logger = SummaryLogger(tmp_path / "refinement_report.txt")
    logger.report(experiment if experiment is not None else _experiment())
    logger.report(
        manifest
        if manifest is not None
        else ObjectiveManifest(penalties=(ObjectiveTerm(name="bond_length", weight=3.0),))
    )
    if preprocess is not None:
        logger.report(preprocess)
    for step in steps or (_step(),):
        logger.report(step)
    logger.report(
        completed
        if completed is not None
        else RefinementCompleted(
            n_steps=1,
            best_step=0,
            best_loss=0.2,
            reflection_counts={"matched": 12, "matched_i_gt_3sigma": 8},
        )
    )
    for rotation in rotations:
        logger.report(rotation)
    for profile in profiles:
        logger.report(profile)
    for mosaicity in mosaicities:
        logger.report(mosaicity)
    logger.report(RefinementOutputsWritten(structure=str(structure)))
    return (tmp_path / "refinement_report.txt").read_text()


def _rotation(
    index: int,
    *,
    wr2: float = 0.2,
    is_validation: bool = False,
    dataset: str = "q.cif_pets",
) -> RefinedRotationMetrics:
    return RefinedRotationMetrics(
        rotation_index=index,
        wr2=wr2,
        r_obs=0.15,
        n_matched=5,
        is_validation=is_validation,
        dataset=dataset,
    )


def test_summary_is_written_only_on_the_terminal_event(tmp_path: Path) -> None:
    """No committed structure means no summary -- the sink does not write a half-finished run."""
    logger = SummaryLogger(tmp_path / "refinement_report.txt")
    logger.report(_experiment())
    logger.report(_step())
    assert not (tmp_path / "refinement_report.txt").exists()

    logger.report(RefinementOutputsWritten(structure=str(_structure(tmp_path))))
    assert (tmp_path / "refinement_report.txt").is_file()


def test_summary_renders_every_section_from_events_alone(tmp_path: Path) -> None:
    text = _run(tmp_path, rotations=(_rotation(0),))

    assert "Simulation / refinement parameters" in text
    assert "Crystallographic parameters" in text
    assert "Per-rotation wR2 / R_obs" in text
    assert "Thickness NN" in text
    assert "enabled: no" in text
    assert "Isotropic mosaicity" in text
    assert "enabled: no (refinement.trainable.mosaicity_sigma is off)" in text
    assert "C1" in text and "O1" in text  # atom labels read back out of the committed CIF
    # The declared composition is stated up front, including the categories that are empty.
    assert "penalties  : bond_length (weight 3)" in text
    assert "constraints: none" in text
    # Every composed term reports raw, weight and contribution separately.
    assert "objective total = 0.35" in text
    # Every mean states the rotation count it was averaged over.
    assert "10.00 [3/4]" in text  # best-epoch wR2 (%), 3 of 4 rotations finite
    assert "5.00 [2/4]" in text  # best-epoch R_obs (%), 2 of 4
    assert "mean wR2   = 0.200000 [1/1]" in text
    assert "Matched HKLs (I>3σ/total)" in text and "8 / 12" in text


def test_summary_renders_dataset_labeled_seed_thicknesses(tmp_path: Path) -> None:
    experiment = _experiment()
    text = _run(
        tmp_path,
        experiment=ExperimentDeclared(
            name=experiment.name,
            structure=experiment.structure,
            experimental_data="dataset_1.cif_pets, dataset_2.cif_pets",
            optimizer=experiment.optimizer,
            seed_thicknesses_by_dataset=(
                ("dataset_1.cif_pets", (400.0,)),
                ("dataset_2.cif_pets", (800.0,)),
            ),
            integration_semiangles=(1.0, 2.0),
            rocking_curve_sampling=experiment.rocking_curve_sampling,
            dsg=experiment.dsg,
            rsg=experiment.rsg,
            solve_g_max=experiment.solve_g_max,
            sg_max=experiment.sg_max,
            absorption=experiment.absorption,
            incoherent_mosaicity=experiment.incoherent_mosaicity,
            steps=experiment.steps,
            learning_rate=experiment.learning_rate,
        ),
    )

    assert "dataset_1.cif_pets: 400" in text
    assert "dataset_2.cif_pets: 800" in text


def test_summary_reports_the_refined_mosaicity_sigma_when_one_was_trained(tmp_path: Path) -> None:
    text = _run(
        tmp_path,
        rotations=(_rotation(0),),
        mosaicities=(
            IsotropicMosaicityRefined(
                label="a.cif_pets", sigma_degrees=0.0241, pets_sigma_degrees=0.0220
            ),
        ),
    )

    assert "Isotropic mosaicity" in text
    assert "a.cif_pets" in text
    assert "0.0220" in text  # PETS sigma
    assert "0.0241" in text  # refined sigma


def test_summary_selects_the_best_epoch_not_the_last(tmp_path: Path) -> None:
    """The reported epoch is the one RefinementCompleted names, wherever it sits in the stream."""
    text = _run(
        tmp_path,
        steps=(
            _step(iteration=0, wr2=0.9, objective_total=9.0),
            _step(iteration=1, wr2=0.1, objective_total=1.0),
            _step(iteration=2, wr2=0.5, objective_total=5.0),
        ),
        completed=RefinementCompleted(n_steps=3, best_step=1, best_loss=1.0),
        rotations=(_rotation(0),),
    )

    assert "objective total = 1" in text
    assert "10.00 [3/4]" in text  # the best epoch's wR2, not epoch 2's


def test_summary_epoch_curve_lists_every_recorded_step(tmp_path: Path) -> None:
    """The epoch curve is the whole trajectory, not just the best epoch."""
    text = _run(
        tmp_path,
        steps=(
            _step(iteration=0, wr2=0.9, r_obs=0.8),
            _step(iteration=1, wr2=0.1, r_obs=0.05),
        ),
        completed=RefinementCompleted(n_steps=2, best_step=1, best_loss=1.0),
        rotations=(_rotation(0),),
    )

    assert "Epoch curve" in text
    epoch_curve = text.split("Epoch curve")[1].split("\n--- ")[0]
    assert "1" in epoch_curve and "0.900000" in epoch_curve and "0.800000" in epoch_curve
    assert "2" in epoch_curve and "0.100000" in epoch_curve and "0.050000" in epoch_curve


def test_summary_epoch_curve_renders_na_for_unevaluated_epochs(tmp_path: Path) -> None:
    text = _run(
        tmp_path,
        steps=(_step(wr2=None, r_obs=None),),
        rotations=(_rotation(0),),
    )

    epoch_curve = text.split("Epoch curve")[1].split("\n--- ")[0]
    assert "n/a" in epoch_curve


def test_summary_epoch_curve_adds_validation_columns_when_a_selection_engine_ran(
    tmp_path: Path,
) -> None:
    text = _run(
        tmp_path,
        steps=(
            _step(iteration=0, wr2=0.9, r_obs=0.8, val_wr2=0.95, val_r_obs=0.85),
            _step(iteration=1, wr2=0.1, r_obs=0.05, val_wr2=0.2, val_r_obs=0.15),
        ),
        completed=RefinementCompleted(n_steps=2, best_step=1, best_loss=1.0),
        rotations=(_rotation(0),),
    )

    epoch_curve = text.split("Epoch curve")[1].split("\n--- ")[0]
    assert "Val wR2" in epoch_curve and "Val R_obs" in epoch_curve
    assert "0.950000" in epoch_curve and "0.850000" in epoch_curve
    assert "0.200000" in epoch_curve and "0.150000" in epoch_curve


def test_summary_epoch_curve_omits_validation_columns_without_a_selection_engine(
    tmp_path: Path,
) -> None:
    text = _run(tmp_path, rotations=(_rotation(0),))

    epoch_curve = text.split("Epoch curve")[1].split("\n--- ")[0]
    assert "Val wR2" not in epoch_curve


def test_summary_preprocess_defaults_to_not_done(tmp_path: Path) -> None:
    text = _run(tmp_path, rotations=(_rotation(0),))

    preprocess_section = text.split("Preprocess")[1].split("\n--- ")[0]
    assert "Orientation optimization: false" in preprocess_section
    assert "Thickness optimization: false" in preprocess_section


def test_summary_preprocess_reports_orientation_optimization_params_when_done(
    tmp_path: Path,
) -> None:
    text = _run(
        tmp_path,
        preprocess=PreprocessCompleted(
            n_rotations=1,
            n_stages=1,
            total_hkl=10,
            matched_hkl=8,
            steps=(
                (
                    "optimize_orientation",
                    {
                        "search": {
                            "__type__": "NelderMeadSearch",
                            "step_size": 0.05,
                            "max_iterations": 60,
                        },
                        "coupling": None,
                    },
                ),
            ),
        ),
        rotations=(_rotation(0),),
    )

    preprocess_section = text.split("Preprocess")[1].split("\n--- ")[0]
    assert "Orientation optimization: true" in preprocess_section
    assert "step_size=0.05" in preprocess_section
    assert "max_iterations=60" in preprocess_section
    # Composition-site kwargs (coupling, absorption) are shared context, not this step's own
    # setting, so they must not show up here.
    assert "coupling" not in preprocess_section
    assert "Thickness optimization: false" in preprocess_section


def test_summary_per_dataset_summary_is_omitted_for_a_single_dataset(tmp_path: Path) -> None:
    text = _run(tmp_path, rotations=(_rotation(0), _rotation(1)))

    assert "Per-dataset summary" not in text


def test_summary_per_dataset_summary_breaks_out_train_validation_and_total(
    tmp_path: Path,
) -> None:
    text = _run(
        tmp_path,
        rotations=(
            _rotation(0, wr2=0.1, dataset="a.cif_pets"),
            _rotation(1, wr2=0.3, is_validation=True, dataset="a.cif_pets"),
            _rotation(2, wr2=0.2, dataset="b.cif_pets"),
        ),
    )

    assert "Per-dataset summary" in text
    per_dataset = text.split("Per-dataset summary")[1].split("\n--- ")[0]
    assert "a.cif_pets" in per_dataset and "b.cif_pets" in per_dataset
    assert "All datasets" in per_dataset
    # a.cif_pets: 1 train (wr2=0.1), 1 validation (wr2=0.3), 2 total.
    assert "0.100000 [1/1]" in per_dataset  # a's train wR2
    assert "0.300000 [1/1]" in per_dataset  # a's validation wR2
    # b.cif_pets has no validation rotations -- the cell must not blow up on an empty split.
    assert "n/a [0/0]" in per_dataset
    # All datasets aggregates every rotation regardless of which dataset it came from.
    assert "0.200000 [3/3]" in per_dataset  # mean wR2 over all three rotations


def test_summary_renders_an_unevaluated_mean_as_na(tmp_path: Path) -> None:
    """One spelling for "nothing was evaluated" across every mean the summary prints."""
    text = _run(
        tmp_path,
        steps=(
            _step(wr2=float("nan"), r_obs=float("nan"), n_wr2_evaluated=0, n_r_obs_evaluated=0),
        ),
        rotations=(_rotation(0, wr2=float("nan")),),
    )

    assert "n/a [0/4]" in text
    assert "nan [" not in text


def test_summary_adds_a_validation_section_only_when_rotations_are_held_out(
    tmp_path: Path,
) -> None:
    held_out = _run(
        tmp_path,
        rotations=(_rotation(0), _rotation(1, is_validation=True)),
    )
    assert "Validation set (held out from the refinement objective)" in held_out
    assert "n_rotations = 1" in held_out
    assert "mean wR2   = 0.200000 [2/2]" in held_out  # the full table still covers both

    none_held_out = _run(tmp_path, rotations=(_rotation(0), _rotation(1)))
    assert "Validation set" not in none_held_out


def _profile(
    label: str,
    *,
    rotation_indices: tuple[int, ...] = (0,),
    alphas: tuple[float, ...] = (12.5,),
    thicknesses: tuple[float, ...] = (430.25,),
) -> ThicknessProfile:
    return ThicknessProfile(
        form="quadratic",
        min_thickness=100.0,
        max_thickness=1000.0,
        rotation_indices=rotation_indices,
        alphas=alphas,
        thicknesses=thicknesses,
        label=label,
    )


def test_summary_reports_the_thickness_profile_when_one_was_trained(tmp_path: Path) -> None:
    text = _run(tmp_path, rotations=(_rotation(0),), profiles=(_profile("q.cif_pets"),))

    assert "Thickness NN -- q.cif_pets" in text
    assert "enabled: yes (quadratic, bounds [100, 1000] A)" in text
    assert "alpha (degrees)" in text
    assert "12.5000" in text and "430.25" in text

    assert (tmp_path / "thickness_nn_shape_q.png").is_file()
    assert "plot: thickness_nn_shape_q.png" in text


def test_summary_reports_one_thickness_section_per_dataset(tmp_path: Path) -> None:
    text = _run(
        tmp_path,
        rotations=(_rotation(0),),
        profiles=(
            _profile("a.cif_pets"),
            _profile("sub/b.cif_pets", rotation_indices=(1,), alphas=(-3.5,), thicknesses=(210.0,)),
        ),
    )

    assert "Thickness NN -- a.cif_pets" in text
    assert "Thickness NN -- sub/b.cif_pets" in text
    assert "430.25" in text and "210.00" in text

    # Filenames use the dataset checkpoint stem: path separators fold to "__", suffix drops.
    assert (tmp_path / "thickness_nn_shape_a.png").is_file()
    assert (tmp_path / "thickness_nn_shape_sub__b.png").is_file()


def test_summary_keeps_the_last_thickness_profile_reported_per_dataset(tmp_path: Path) -> None:
    text = _run(
        tmp_path,
        rotations=(_rotation(0),),
        profiles=(
            _profile("a.cif_pets"),
            _profile("a.cif_pets", thicknesses=(555.5,)),
        ),
    )

    assert text.count("Thickness NN -- a.cif_pets") == 1
    assert "555.50" in text
    assert "430.25" not in text


def test_summary_still_writes_when_the_plot_renderer_cannot_be_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """matplotlib is a core dependency, so this is broken-install insurance, not a supported mode.

    The report is written on the run's terminal event, potentially hours in; a rendering import
    that fails must cost the plot, not the whole report.
    """

    def unimportable(*args: object, **kwargs: object) -> None:
        raise ModuleNotFoundError("No module named 'matplotlib'")

    monkeypatch.setattr("diffBloch.app.loggers.plotting.plot_thickness_nn_shape", unimportable)

    text = _run(tmp_path, rotations=(_rotation(0),), profiles=(_profile("q.cif_pets"),))

    assert "plot: skipped (matplotlib is not importable in this environment)" in text
    assert not (tmp_path / "thickness_nn_shape_q.png").exists()
    # The section itself survives: the table is rendered from the event, not from the renderer.
    assert "Thickness NN -- q.cif_pets" in text
    assert "430.25" in text


def test_thickness_profile_channel_names_its_dataset() -> None:
    assert _profile("sub/b.cif_pets").channel == "thickness_profile[sub/b.cif_pets]"


def test_summary_degrades_without_the_declaration_events(tmp_path: Path) -> None:
    """A sink attached mid-run says so rather than inventing values or raising."""
    logger = SummaryLogger(tmp_path / "refinement_report.txt")
    logger.report(RotationScored(index=0, r_obs=0.1, wr2=0.1, n_matched=5))
    logger.report(RefinementOutputsWritten(structure=str(_structure(tmp_path))))

    text = (tmp_path / "refinement_report.txt").read_text()
    assert "n/a (no recorded experiment declaration)" in text
    assert "n/a (no recorded objective manifest)" in text
    assert "n/a (no recorded objective components)" in text


def test_thickness_profile_rejects_ragged_columns() -> None:
    with pytest.raises(ValueError, match="equal length"):
        ThicknessProfile(
            form="quadratic",
            min_thickness=100.0,
            max_thickness=1000.0,
            rotation_indices=(0, 1),
            alphas=(12.5,),
            thicknesses=(430.25,),
            label="q.cif_pets",
        )
    with pytest.raises(ValueError, match="label"):
        _profile("")


def test_ascii_table_widens_a_column_to_its_content() -> None:
    """Widths are measured, so a long cell never overflows into the next column."""
    table = ascii_table(["A", "B"], [["a-very-long-cell", "1"]])
    header, rule, row = table.splitlines()
    assert len(header) == len(rule) == len(row)
