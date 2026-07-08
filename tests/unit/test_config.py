"""The Pydantic config boundary validates and supplies sensible defaults."""

import pytest
from pydantic import ValidationError

from diffBloch.config.schema import ExperimentConfig
from diffBloch.specs import (
    HexagonalSearch,
    RockingCurve,
    ThicknessGrid,
)


def test_minimal_config_validates_with_defaults() -> None:
    cfg = ExperimentConfig.model_validate(
        {"name": "quartz", "inputs": {"structure": "q.cif", "observations": "q.cif_pets"}}
    )
    assert cfg.name == "quartz"
    # defaults-as-code: the experiment file only needs inputs + overrides
    assert cfg.solver.refine == "matrix_exp"
    assert cfg.solver.inference == "bloch_eigen"
    assert cfg.sample.thicknesses == (820.0,)
    assert cfg.numerics.g_max == 4.5
    assert cfg.refinement.optimizer.name == "lbfgs"
    assert cfg.refinement.objective.data_term == "weighted_r"
    assert cfg.refinement.split.validation == "every_10th_rotation"


def test_solver_method_must_be_a_known_method() -> None:
    # The solver fields are typed as the core Method literal, so an unknown method fails fast at
    # config load rather than deep in the forward model.
    base = {"name": "bad", "inputs": {"structure": "q.cif", "observations": "q.cif_pets"}}
    with pytest.raises(ValidationError, match="Input should be"):
        ExperimentConfig.model_validate({**base, "solver": {"refine": "nope"}})


def test_missing_required_input_fails_fast() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"name": "quartz"})  # no inputs


def test_unknown_key_is_rejected_not_ignored() -> None:
    # The allowlist guard (extra="forbid"): a stale/misspelled key is a load-time error, not a
    # silent drop -- so config can only carry fields a consumer reads. Guards against a regression
    # to pydantic's default "ignore", which is what let the dead g_max_sf / sg_max keys linger.
    base = {"name": "q", "inputs": {"structure": "q.cif", "observations": "q.cif_pets"}}
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        ExperimentConfig.model_validate({**base, "numerics": {"sg_max": 0.01}})  # removed field
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        ExperimentConfig.model_validate({**base, "nonsense": True})  # unknown top-level key


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
    # The config is a 1:1 edge over HexagonalSearch: its defaults derive from the value-type, so a
    # default config round-trips to the value-type's own defaults. The concrete values (the private
    # numbers, incl. the quartz-calibrated max_iterations) are pinned once, in test_specs.
    assert orientation.to_search() == HexagonalSearch()


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
    # 1:1 edge over ThicknessGrid: a default config round-trips to the value-type's defaults; the
    # concrete values are pinned once, in test_specs.
    assert thickness.to_grid() == ThicknessGrid()


def test_numerics_to_rocking_curve_shares_the_integration_geometry() -> None:
    cfg = ExperimentConfig.model_validate(
        {"name": "quartz", "inputs": {"structure": "q.cif", "observations": "q.cif_pets"}}
    )
    numerics = cfg.numerics
    # The tilt span/geometry come from the SAME IntegrationGeometry as the beam window, so the two
    # cannot disagree; rocking_curve_sampling is the tilt count.
    assert numerics.to_rocking_curve() == RockingCurve(
        sampling=numerics.rocking_curve_sampling, integration=numerics.integration
    )
    assert numerics.to_beam_selection().integration is numerics.integration


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
    with pytest.raises(ValidationError, match="semiangle must be positive"):
        ExperimentConfig.model_validate({**base, "numerics": {"integration": {"semiangle": 0.0}}})
