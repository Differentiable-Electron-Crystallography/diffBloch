"""The Pydantic config boundary validates and supplies sensible defaults."""

from dataclasses import fields

import pytest
from pydantic import ValidationError

from diffBloch.config.schema import (
    ExperimentConfig,
    ObjectiveConfig,
    OptimizerConfig,
    TrainableConfig,
)
from diffBloch.engine import AtomSelection, TrainableSpec
from diffBloch.engine.losses import scaled_w_rbragg_loss, weighted_mse_loss
from diffBloch.engine.refine import _TRAINABLE_FIELDS
from diffBloch.specs import (
    HexagonalSearch,
    RockingCurve,
    ThicknessGrid,
    TiltSegmentUnion,
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
    assert cfg.numerics.g_max_refine == 1.6
    assert cfg.refinement.trainable.positions == "all"
    assert cfg.refinement.trainable.adp == "all"
    assert cfg.refinement.trainable.occupancy == "none"
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


def test_trainable_config_to_spec_maps_groups_to_selections() -> None:
    spec = TrainableConfig(positions="all", adp="none", occupancy="all").to_spec()
    assert isinstance(spec, TrainableSpec)
    assert spec.positions == AtomSelection.all()
    assert spec.adp == AtomSelection.none()
    assert spec.occupancy == AtomSelection.all()


def test_objective_data_term_parses_to_loss() -> None:
    assert ObjectiveConfig(data_term="weighted_r").to_loss() is scaled_w_rbragg_loss
    assert ObjectiveConfig(data_term="least_squares").to_loss() is weighted_mse_loss


def test_objective_rejects_unimplemented_data_term() -> None:
    with pytest.raises(ValidationError):
        ObjectiveConfig(data_term="poisson_nll")  # deferred: no LossFn


def test_optimizer_rejects_deferred_backend() -> None:
    with pytest.raises(ValidationError):
        OptimizerConfig(name="least_squares")  # deferred: not in OptimizerName


def test_optimizer_rejects_unconsumed_line_search_field() -> None:
    # max_line_search_steps was accepted but never wired to any torch backend; removed so the config
    # cannot promise a knob it does not honour (strict configs reject the now-unknown key).
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        OptimizerConfig(max_line_search_steps=20)


def test_trainable_group_keysets_do_not_drift() -> None:
    assert set(_TRAINABLE_FIELDS) == {field.name for field in fields(TrainableSpec)}
    assert set(_TRAINABLE_FIELDS) == set(TrainableConfig.model_fields)


def test_refinement_trainable_replaces_string_targets() -> None:
    base = {"name": "bad", "inputs": {"structure": "q.cif", "observations": "q.cif_pets"}}
    cfg = ExperimentConfig.model_validate(
        {**base, "refinement": {"trainable": {"positions": "none", "occupancy": "all"}}}
    )
    assert cfg.refinement.trainable.positions == "none"
    assert cfg.refinement.trainable.occupancy == "all"
    with pytest.raises(ValidationError, match="Input should be"):
        ExperimentConfig.model_validate(
            {**base, "refinement": {"trainable": {"positions": "heavy_only"}}}
        )
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        ExperimentConfig.model_validate({**base, "refinement": {"targets": ["positions"]}})
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        ExperimentConfig.model_validate({**base, "refinement": {"trainable": {"thickness": "all"}}})


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


def test_coupling_has_no_default_and_is_absent_unless_declared() -> None:
    # Unlike the numerical preprocess blocks, coupling has NO faithful default -- omitting it leaves
    # preprocess.coupling None (the recipe build, not config load, rejects a missing policy).
    cfg = ExperimentConfig.model_validate(
        {"name": "quartz", "inputs": {"structure": "q.cif", "observations": "q.cif_pets"}}
    )
    assert cfg.preprocess.coupling is None


def test_coupling_policy_requires_all_fields_when_declared() -> None:
    # The block is all-or-nothing explicit: a partial coupling block is a load-time error, not a
    # silent per-field fill from the value-type.
    base = {"name": "abi", "inputs": {"structure": "a.cif", "observations": "a.cif_pets"}}
    with pytest.raises(ValidationError, match="[Ff]ield required"):
        ExperimentConfig.model_validate(
            {**base, "preprocess": {"coupling": {"n_splits": 4, "g_max": 1.5}}}  # missing fields
        )


def test_coupling_policy_override_parses() -> None:
    base = {"name": "abi", "inputs": {"structure": "a.cif", "observations": "a.cif_pets"}}
    cfg = ExperimentConfig.model_validate(
        {
            **base,
            "preprocess": {"coupling": {"n_splits": 4, "g_max": 1.5, "sg_max": 0.02}},
        }
    )
    assert cfg.preprocess.coupling is not None
    assert cfg.preprocess.coupling.to_policy() == TiltSegmentUnion(
        n_splits=4, g_max=1.5, sg_max=0.02
    )


def test_coupling_adaptive_fields_default_off_and_thread_through() -> None:
    base = {"name": "abi", "inputs": {"structure": "a.cif", "observations": "a.cif_pets"}}
    fixed = {"n_splits": 4, "g_max": 1.5, "sg_max": 0.02}
    # Omitted -> the faithful fixed even-split (adaptive off), unchanged from the current behaviour.
    default = ExperimentConfig.model_validate({**base, "preprocess": {"coupling": fixed}})
    assert default.preprocess.coupling is not None
    assert default.preprocess.coupling.to_policy().union_adaptive is False
    # Declared -> threaded into the value-type the coupled fit consumes.
    adaptive = ExperimentConfig.model_validate(
        {
            **base,
            "preprocess": {
                "coupling": {**fixed, "union_adaptive": True, "union_max_new_beams_pct": 0.02}
            },
        }
    )
    assert adaptive.preprocess.coupling is not None
    policy = adaptive.preprocess.coupling.to_policy()
    assert policy.union_adaptive is True
    assert policy.union_max_new_beams_pct == 0.02


def test_coupling_policy_bounds_are_validated() -> None:
    base = {"name": "bad", "inputs": {"structure": "q.cif", "observations": "q.cif_pets"}}

    def coupling(**overrides: float) -> dict:
        policy = {"n_splits": 12, "g_max": 2.25, "sg_max": 0.01, **overrides}
        return {**base, "preprocess": {"coupling": policy}}

    with pytest.raises(ValidationError, match="n_splits must be >= 1"):
        ExperimentConfig.model_validate(coupling(n_splits=0))
    with pytest.raises(ValidationError, match="g_max and sg_max must be positive"):
        ExperimentConfig.model_validate(coupling(sg_max=0.0))
    with pytest.raises(ValidationError, match="g_max and sg_max must be positive"):
        ExperimentConfig.model_validate(coupling(g_max=0.0))


def test_load_hydrogens_defaults_off_and_parses() -> None:
    base = {"name": "abi", "inputs": {"structure": "a.cif", "observations": "a.cif_pets"}}
    assert ExperimentConfig.model_validate(base).inputs.load_hydrogens is False
    withH = ExperimentConfig.model_validate(
        {
            "name": "abi",
            "inputs": {"structure": "a.cif", "observations": "a.cif_pets", "load_hydrogens": True},
        }
    )
    assert withH.inputs.load_hydrogens is True


def test_refinement_rejects_hydrogen_mode_as_unknown_key() -> None:
    # H handling is scientific composition (Python/API via with_hydrogen_riding), not a config mode;
    # a strict config rejects the removed key rather than silently accepting a dead knob.
    base = {"name": "abi", "inputs": {"structure": "a.cif", "observations": "a.cif_pets"}}
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        ExperimentConfig.model_validate({**base, "refinement": {"hydrogen_mode": "riding"}})


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
