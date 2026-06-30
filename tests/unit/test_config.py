"""The Pydantic config boundary validates and supplies sensible defaults."""

import pytest
from pydantic import ValidationError

from diffBloch.config.schema import BeamDamageConfig, ExperimentConfig


def test_minimal_config_validates_with_defaults() -> None:
    cfg = ExperimentConfig.model_validate(
        {"name": "quartz", "inputs": {"structure": "q.cif", "observations": "q.cif_pets"}}
    )
    assert cfg.name == "quartz"
    # defaults-as-code: the experiment file only needs inputs + overrides
    assert cfg.solver.refine == "matrix_exp"
    assert cfg.solver.inference == "bloch_eigen"
    assert cfg.sample.thicknesses == (820.0,)
    assert cfg.numerics.sg_max == 0.01
    assert cfg.observation.beam_damage.activate is False
    assert cfg.refinement.optimizer.name == "lbfgs"
    assert cfg.refinement.objective.data_term == "weighted_r"
    assert cfg.refinement.split.validation == "every_10th_rotation"


def test_missing_required_input_fails_fast() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"name": "quartz"})  # no inputs


def test_beam_damage_off_by_default() -> None:
    assert BeamDamageConfig().activate is False


def test_sample_thicknesses_are_positive_and_nonempty() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {"structure": "q.cif", "observations": "q.cif_pets"},
                "sample": {"thicknesses": []},
            }
        )
    with pytest.raises(ValidationError, match="positive"):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {"structure": "q.cif", "observations": "q.cif_pets"},
                "sample": {"thicknesses": [0.0]},
            }
        )


def test_input_refs_must_stay_inside_experiment_directory() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {"structure": "/tmp/q.cif", "observations": "q.cif_pets"},
            }
        )
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {"structure": "../q.cif", "observations": "q.cif_pets"},
            }
        )


def test_optimizer_and_objective_values_are_enumerated() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {"structure": "q.cif", "observations": "q.cif_pets"},
                "refinement": {"optimizer": {"name": "made_up"}},
            }
        )
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {"structure": "q.cif", "observations": "q.cif_pets"},
                "refinement": {"objective": {"data_term": "made_up"}},
            }
        )


def test_preprocess_orientation_defaults_match_the_private() -> None:
    cfg = ExperimentConfig.model_validate(
        {"name": "quartz", "inputs": {"structure": "q.cif", "observations": "q.cif_pets"}}
    )
    orientation = cfg.preprocess.orientation
    assert orientation.max_search_angle == 0.4
    assert orientation.min_search_angle == 0.001
    assert orientation.n_steps == 6
    assert orientation.max_iterations == 200


def test_orientation_search_bounds_are_validated() -> None:
    base = {"name": "bad", "inputs": {"structure": "q.cif", "observations": "q.cif_pets"}}
    with pytest.raises(ValidationError, match="must be positive"):
        ExperimentConfig.model_validate(
            {**base, "preprocess": {"orientation": {"min_search_angle": 0.0}}}
        )
    with pytest.raises(ValidationError, match="must exceed"):
        ExperimentConfig.model_validate(
            {**base, "preprocess": {"orientation": {"max_search_angle": 0.001}}}
        )
    with pytest.raises(ValidationError, match="n_steps must be >= 1"):
        ExperimentConfig.model_validate({**base, "preprocess": {"orientation": {"n_steps": 0}}})
    with pytest.raises(ValidationError, match="max_iterations must be >= 1"):
        ExperimentConfig.model_validate(
            {**base, "preprocess": {"orientation": {"max_iterations": 0}}}
        )


def test_preprocess_thickness_defaults_match_the_private() -> None:
    cfg = ExperimentConfig.model_validate(
        {"name": "quartz", "inputs": {"structure": "q.cif", "observations": "q.cif_pets"}}
    )
    thickness = cfg.preprocess.thickness
    assert thickness.min_thickness == 5.0
    assert thickness.max_thickness == 2000.0
    assert thickness.n_steps == 100


def test_thickness_grid_bounds_are_validated() -> None:
    base = {"name": "bad", "inputs": {"structure": "q.cif", "observations": "q.cif_pets"}}
    with pytest.raises(ValidationError, match="thickness bounds must be positive"):
        ExperimentConfig.model_validate(
            {**base, "preprocess": {"thickness": {"min_thickness": 0.0}}}
        )
    with pytest.raises(ValidationError, match="max_thickness must exceed min_thickness"):
        ExperimentConfig.model_validate(
            {**base, "preprocess": {"thickness": {"min_thickness": 100.0, "max_thickness": 100.0}}}
        )
    with pytest.raises(ValidationError, match="n_steps must be >= 1"):
        ExperimentConfig.model_validate({**base, "preprocess": {"thickness": {"n_steps": 0}}})


def test_numerics_beam_selection_cutoffs_are_validated() -> None:
    # NumericsConfig delegates its beam-selection subset to BeamSelection (fail-fast at load).
    base = {"name": "bad", "inputs": {"structure": "q.cif", "observations": "q.cif_pets"}}
    with pytest.raises(ValidationError, match="rsg must be positive"):
        ExperimentConfig.model_validate({**base, "numerics": {"rsg": 0.0}})
    with pytest.raises(ValidationError, match="integration_semiangle must be positive"):
        ExperimentConfig.model_validate({**base, "numerics": {"integration_semiangle": 0.0}})
