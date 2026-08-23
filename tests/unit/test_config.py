"""The Pydantic config boundary validates and supplies sensible defaults."""

from dataclasses import fields

import pytest
from pydantic import ValidationError

from diffBloch.config.schema import (
    DataSplitConfig,
    ExperimentConfig,
    LossMetricsConfig,
    OptimizerConfig,
    TrainableConfig,
    dataset_checkpoint_stem,
)
from diffBloch.engine import AtomSelection, TrainableSpec
from diffBloch.engine.losses import (
    rbragg_loss,
    robs_scores,
    wr2_loss,
    wr2_scores,
)
from diffBloch.engine.refine import _TRAINABLE_FIELDS
from diffBloch.specs import (
    Absorption,
    ApparentThicknessNetwork,
    IntegrationGeometry,
    NelderMeadSearch,
    OrientationSelection,
    PerTiltCoupling,
    RockingCurve,
    ThicknessGrid,
    UnionCoupling,
)


def test_minimal_config_validates_with_defaults() -> None:
    cfg = ExperimentConfig.model_validate(
        {"name": "quartz", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    )
    assert cfg.name == "quartz"
    # defaults-as-code: the experiment file only needs inputs + overrides
    assert cfg.blochwave.solver == "matrix_exp"
    assert cfg.blochwave.scattering_factors == "lobato2026"
    assert cfg.sample.thicknesses == (820.0,)
    assert cfg.refinement.trainable.positions == "all"
    assert cfg.refinement.trainable.adp == "all"
    assert cfg.refinement.trainable.occupancy == "none"
    assert cfg.refinement.optimizer.name == "adam"
    assert cfg.loss_metrics.residual == "wr2"
    assert cfg.refinement.thickness_nn.to_spec() == ApparentThicknessNetwork()
    assert cfg.refinement.split.train_test is False
    assert cfg.refinement.split.val_frac == 0.2
    assert cfg.blochwave.ignore_orientations == ()
    assert cfg.blochwave.rsg == 0.66
    assert cfg.blochwave.to_absorption() == Absorption()
    assert cfg.preprocess.optimize_orientation is True
    assert cfg.preprocess.optimize_thickness is True
    assert cfg.preprocess.stage_order == "thickness_first"


def test_solver_method_must_be_a_known_method() -> None:
    # solver is typed as the core SolverMethod literal, so an unknown method fails fast at
    # config load rather than deep in the forward model.
    base = {"name": "bad", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    with pytest.raises(ValidationError, match="Input should be"):
        ExperimentConfig.model_validate({**base, "blochwave": {"solver": "nope"}})


def test_removed_nested_solver_config_is_rejected() -> None:
    base = {"name": "bad", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    with pytest.raises(ValidationError, match="Input should be|Extra"):
        ExperimentConfig.model_validate(
            {
                **base,
                "blochwave": {"solver": {"refine": "matrix_exp", "inference": "matrix_exp"}},
            }
        )


def test_absorption_config_is_typed_and_requires_matrix_exp() -> None:
    base = {"name": "q", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    cfg = ExperimentConfig.model_validate({**base, "blochwave": {"absorption": True}})
    assert cfg.blochwave.to_absorption() == Absorption(enabled=True)
    with pytest.raises(ValidationError, match="absorption requires"):
        ExperimentConfig.model_validate(
            {
                **base,
                "blochwave": {
                    "absorption": True,
                    "solver": "bloch_eigen",
                },
            }
        )


def test_multi_dataset_mean_thicknesses_are_keyed_by_exp_data() -> None:
    base = {
        "name": "pooled",
        "inputs": {
            "structure": "q.cif",
            "exp_data": ["a.cif_pets", "b.cif_pets"],
            "multi_dataset": True,
        },
    }
    cfg = ExperimentConfig.model_validate(
        {
            **base,
            "sample": {
                "mean_thickness_by_dataset": {
                    "a.cif_pets": 400.0,
                    "b.cif_pets": 800.0,
                }
            },
        }
    )
    assert cfg.sample.seed_thicknesses_for("a.cif_pets") == (400.0,)
    assert cfg.sample.seed_thicknesses_for("b.cif_pets") == (800.0,)
    declared = cfg.to_declaration(
        (IntegrationGeometry(semiangle=1.0), IntegrationGeometry(semiangle=2.0))
    )
    assert declared.seed_thicknesses_by_dataset == (
        ("a.cif_pets", (400.0,)),
        ("b.cif_pets", (800.0,)),
    )

    with pytest.raises(ValidationError, match="keys must exactly match"):
        ExperimentConfig.model_validate(
            {
                **base,
                "sample": {"mean_thickness_by_dataset": {"a.cif_pets": 400.0}},
            }
        )


def test_sample_thicknesses_must_be_finite() -> None:
    base = {"name": "q", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}

    with pytest.raises(ValidationError, match="thicknesses must be finite and positive"):
        ExperimentConfig.model_validate({**base, "sample": {"thicknesses": [float("nan")]}})

    with pytest.raises(ValidationError, match="thicknesses must be finite and positive"):
        ExperimentConfig.model_validate({**base, "sample": {"thicknesses": [float("inf")]}})


def test_multi_dataset_mean_thicknesses_must_be_finite() -> None:
    base = {
        "name": "pooled",
        "inputs": {
            "structure": "q.cif",
            "exp_data": ["a.cif_pets", "b.cif_pets"],
            "multi_dataset": True,
        },
    }

    with pytest.raises(ValidationError, match="finite and positive"):
        ExperimentConfig.model_validate(
            {
                **base,
                "sample": {
                    "mean_thickness_by_dataset": {
                        "a.cif_pets": 400.0,
                        "b.cif_pets": float("nan"),
                    }
                },
            }
        )


def test_refinement_thickness_nn_config_is_typed_and_validated() -> None:
    base = {"name": "q", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    cfg = ExperimentConfig.model_validate(
        {
            **base,
            "refinement": {
                "thickness_nn": {
                    "enabled": True,
                    "num_samples": 12,
                    "sample_thickness": True,
                    "min_thickness": 1000.0,
                    "max_thickness": 3000.0,
                }
            },
        }
    )
    assert cfg.refinement.thickness_nn.to_spec() == ApparentThicknessNetwork(
        enabled=True,
        num_samples=12,
        sample_thickness=True,
        min_thickness=1000.0,
        max_thickness=3000.0,
    )
    with pytest.raises(ValidationError, match="max_thickness must exceed"):
        ExperimentConfig.model_validate(
            {
                **base,
                "refinement": {
                    "thickness_nn": {
                        "enabled": True,
                        "min_thickness": 1000.0,
                        "max_thickness": 1000.0,
                    }
                },
            }
        )


def test_missing_required_input_fails_fast() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"name": "quartz"})  # no inputs


def test_unknown_key_is_rejected_not_ignored() -> None:
    # The allowlist guard (extra="forbid"): a stale/misspelled key is a load-time error, not a
    # silent drop -- so config can only carry fields a consumer reads. Guards against a regression
    # to pydantic's default "ignore", which is what let the dead g_max_sf key linger.
    base = {"name": "q", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        ExperimentConfig.model_validate({**base, "blochwave": {"g_max_sf": 5.0}})
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        ExperimentConfig.model_validate({**base, "nonsense": True})  # unknown top-level key


def test_ignore_orientations_parses_to_validated_source_indices() -> None:
    base = {"name": "q", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    cfg = ExperimentConfig.model_validate(
        {**base, "blochwave": {"ignore_orientations": [0, 18, 56]}}
    )
    assert cfg.blochwave.to_orientation_selection() == OrientationSelection((0, 18, 56))

    with pytest.raises(ValidationError, match="non-negative"):
        ExperimentConfig.model_validate({**base, "blochwave": {"ignore_orientations": [-1]}})
    with pytest.raises(ValidationError, match="duplicate"):
        ExperimentConfig.model_validate({**base, "blochwave": {"ignore_orientations": [2, 2]}})


def test_sample_thicknesses_are_positive_and_nonempty() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"},
                "sample": {"thicknesses": []},
            }
        )
    with pytest.raises(ValidationError, match="positive"):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"},
                "sample": {"thicknesses": [0.0]},
            }
        )


def test_input_refs_must_stay_inside_experiment_directory() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {"structure": "/tmp/q.cif", "exp_data": "q.cif_pets"},
            }
        )
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {"structure": "../q.cif", "exp_data": "q.cif_pets"},
            }
        )


def test_multi_dataset_defaults_off_with_a_single_exp_data_path() -> None:
    cfg = ExperimentConfig.model_validate(
        {"name": "q", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    )
    assert cfg.inputs.multi_dataset is False
    assert cfg.inputs.exp_data == "q.cif_pets"


def test_multi_dataset_true_accepts_a_list_of_two_or_more_paths() -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "name": "q",
            "inputs": {
                "structure": "q.cif",
                "exp_data": ["a.cif_pets", "b.cif_pets"],
                "multi_dataset": True,
            },
        }
    )
    assert cfg.inputs.exp_data == ["a.cif_pets", "b.cif_pets"]


def test_multi_dataset_true_rejects_a_single_path() -> None:
    with pytest.raises(ValidationError, match="multi_dataset=true requires"):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets", "multi_dataset": True},
            }
        )


def test_multi_dataset_false_rejects_a_list_of_paths() -> None:
    with pytest.raises(ValidationError, match="multi_dataset is false"):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {"structure": "q.cif", "exp_data": ["a.cif_pets", "b.cif_pets"]},
            }
        )


def test_multi_dataset_exp_data_paths_are_each_validated_relative() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {
                    "structure": "q.cif",
                    "exp_data": ["a.cif_pets", "/tmp/b.cif_pets"],
                    "multi_dataset": True,
                },
            }
        )


def test_multi_dataset_rejects_duplicate_exp_data_paths() -> None:
    with pytest.raises(ValidationError, match="more than once"):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {
                    "structure": "q.cif",
                    "exp_data": ["a.cif_pets", "a.cif_pets"],
                    "multi_dataset": True,
                },
            }
        )


def test_multi_dataset_rejects_exp_data_paths_with_colliding_checkpoint_stems() -> None:
    # `a/frame.cif_pets` and `a__frame.cif_pets` both sanitize to the stem `a__frame`.
    with pytest.raises(ValidationError, match="same .*checkpoint name"):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {
                    "structure": "q.cif",
                    "exp_data": ["a/frame.cif_pets", "a__frame.cif_pets"],
                    "multi_dataset": True,
                },
            }
        )


def test_multi_dataset_accepts_default_thickness_nn() -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "name": "q",
            "inputs": {
                "structure": "q.cif",
                "exp_data": ["a.cif_pets", "b.cif_pets"],
                "multi_dataset": True,
            },
        }
    )

    assert cfg.refinement.thickness_nn.enabled is True


def test_dataset_checkpoint_stem_flattens_paths_and_drops_the_pets_suffix() -> None:
    assert dataset_checkpoint_stem("undamaged/frame_1.cif_pets") == "undamaged__frame_1"
    assert dataset_checkpoint_stem("exp_data.cif_pets") == "exp_data"
    assert dataset_checkpoint_stem("other.xyz") == "other.xyz"


def test_trainable_config_to_spec_maps_groups_to_selections() -> None:
    spec = TrainableConfig(positions="all", adp="none", occupancy="all").to_spec()
    assert isinstance(spec, TrainableSpec)
    assert spec.positions == AtomSelection.all()
    assert spec.adp == AtomSelection.none()
    assert spec.occupancy == AtomSelection.all()


def test_loss_metrics_residual_parses_to_loss() -> None:
    assert LossMetricsConfig(residual="wr2").to_loss() is wr2_loss
    assert LossMetricsConfig(residual="robs").to_loss() is rbragg_loss


def test_loss_metrics_residual_parses_to_scores() -> None:
    """to_scores is the per-thickness counterpart to_loss sums -- same residual, matching metric."""
    assert LossMetricsConfig(residual="wr2").to_scores() is wr2_scores
    assert LossMetricsConfig(residual="robs").to_scores() is robs_scores


def test_experiment_config_loss_metrics_is_top_level_not_under_refinement() -> None:
    """loss_metrics governs preprocess search + refinement, so it's on ExperimentConfig directly."""
    cfg = ExperimentConfig.model_validate(
        {
            "name": "quartz",
            "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"},
            "loss_metrics": {"residual": "robs"},
        }
    )
    assert cfg.loss_metrics.residual == "robs"
    assert not hasattr(cfg.refinement, "loss_metrics")


@pytest.mark.parametrize("val_frac", [0.0, 1.0, -0.1, 1.5])
def test_data_split_config_rejects_val_frac_outside_zero_one(val_frac: float) -> None:
    with pytest.raises(ValidationError):
        DataSplitConfig(val_frac=val_frac)


def test_loss_metrics_rejects_unimplemented_residual() -> None:
    with pytest.raises(ValidationError):
        LossMetricsConfig(residual="poisson_nll")  # deferred: no LossFn
    with pytest.raises(ValidationError):
        LossMetricsConfig(residual="least_squares")  # removed residual, not optimizer backend


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
    base = {"name": "bad", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
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


def test_refinement_precision_is_not_a_config_field() -> None:
    # precision was removed as a selectable knob: the solve always runs at fp32/complex64, so
    # a stray "precision" key must be rejected by the allowlist guard, not silently accepted.
    base = {"name": "q", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        ExperimentConfig.model_validate({**base, "refinement": {"precision": "fp32"}})


def test_optimizer_and_loss_metrics_values_are_enumerated() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"},
                "refinement": {"optimizer": {"name": "made_up"}},
            }
        )
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"},
                "loss_metrics": {"residual": "made_up"},
            }
        )


def test_preprocess_orientation_defaults_match_specs() -> None:
    cfg = ExperimentConfig.model_validate(
        {"name": "quartz", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    )
    orientation = cfg.preprocess.orientation
    # The config is a 1:1 edge over NelderMeadSearch: its defaults derive from the value-type, so a
    # default config round-trips to the value-type's own defaults. The concrete values are pinned
    # once, in test_specs.
    assert orientation.to_search() == NelderMeadSearch()


def test_orientation_search_bounds_are_validated() -> None:
    base = {"name": "bad", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    with pytest.raises(ValidationError, match="must be positive"):
        ExperimentConfig.model_validate(
            {**base, "preprocess": {"orientation": {"nelder_mead": {"step_size": 0.0}}}}
        )
    with pytest.raises(ValidationError, match="max_iterations must be >= 1"):
        ExperimentConfig.model_validate(
            {**base, "preprocess": {"orientation": {"nelder_mead": {"max_iterations": 0}}}}
        )
    with pytest.raises(ValidationError, match="must be positive"):
        ExperimentConfig.model_validate(
            {**base, "preprocess": {"orientation": {"nelder_mead": {"x_tolerance": 0.0}}}}
        )


def test_preprocess_thickness_defaults_match_specs() -> None:
    cfg = ExperimentConfig.model_validate(
        {"name": "quartz", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    )
    thickness = cfg.preprocess.thickness
    # 1:1 edge over ThicknessGrid: a default config round-trips to the value-type's defaults; the
    # concrete values are pinned once, in test_specs.
    assert thickness.to_grid() == ThicknessGrid()


def test_blochwave_has_one_complete_default() -> None:
    cfg = ExperimentConfig.model_validate(
        {"name": "quartz", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    )
    assert cfg.blochwave.to_policy() == UnionCoupling()


def test_scattering_factors_override_parses() -> None:
    base = {"name": "abi", "inputs": {"structure": "a.cif", "exp_data": "a.cif_pets"}}
    cfg = ExperimentConfig.model_validate(
        {**base, "blochwave": {"scattering_factors": "lobato2014"}}
    )
    assert cfg.blochwave.scattering_factors == "lobato2014"


def test_scattering_factors_rejects_unknown_model() -> None:
    base = {"name": "abi", "inputs": {"structure": "a.cif", "exp_data": "a.cif_pets"}}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({**base, "blochwave": {"scattering_factors": "lobato2099"}})


def test_coupling_policy_override_parses() -> None:
    base = {"name": "abi", "inputs": {"structure": "a.cif", "exp_data": "a.cif_pets"}}
    cfg = ExperimentConfig.model_validate(
        {
            **base,
            "blochwave": {"fixed_n_segments": 4, "g_max": 1.5, "sg_max": 0.02},
        }
    )
    assert cfg.blochwave.to_policy() == UnionCoupling(fixed_n_segments=4, g_max=1.5, sg_max=0.02)


def test_per_tilt_coupling_policy_parses_without_union_settings() -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "name": "paper",
            "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"},
            "blochwave": {"coupling_mode": "per_tilt", "g_max": 2.5, "sg_max": 0.01},
        }
    )
    assert cfg.blochwave.to_policy() == PerTiltCoupling(g_max=2.5, sg_max=0.01)


def test_coupling_adaptive_fields_default_off_and_thread_through() -> None:
    base = {"name": "abi", "inputs": {"structure": "a.cif", "exp_data": "a.cif_pets"}}
    fixed = {"fixed_n_segments": 4, "g_max": 1.5, "sg_max": 0.02}
    # Omitted -> the faithful fixed even-split (adaptive off), unchanged from the current behaviour.
    default = ExperimentConfig.model_validate({**base, "blochwave": fixed})
    assert default.blochwave.to_policy().union_adaptive is True
    # Declared -> threaded into the value-type the coupled fit consumes.
    adaptive = ExperimentConfig.model_validate(
        {
            **base,
            "blochwave": {
                **fixed,
                "union_adaptive": True,
                "union_max_new_beams_pct": 0.02,
            },
        }
    )
    policy = adaptive.blochwave.to_policy()
    assert policy.union_adaptive is True
    assert policy.union_max_new_beams_pct == 0.02


def test_coupling_policy_bounds_are_validated() -> None:
    base = {"name": "bad", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}

    def coupling(**overrides: float) -> dict:
        policy = {"fixed_n_segments": 12, "g_max": 2.25, "sg_max": 0.01, **overrides}
        return {**base, "blochwave": policy}

    with pytest.raises(ValidationError, match="fixed_n_segments must be >= 1"):
        ExperimentConfig.model_validate(coupling(fixed_n_segments=0))
    with pytest.raises(ValidationError, match="g_max and sg_max must be positive"):
        ExperimentConfig.model_validate(coupling(sg_max=0.0))
    with pytest.raises(ValidationError, match="g_max and sg_max must be positive"):
        ExperimentConfig.model_validate(coupling(g_max=0.0))


def test_load_hydrogens_defaults_off_and_parses() -> None:
    base = {"name": "abi", "inputs": {"structure": "a.cif", "exp_data": "a.cif_pets"}}
    assert ExperimentConfig.model_validate(base).inputs.load_hydrogens is False
    withH = ExperimentConfig.model_validate(
        {
            "name": "abi",
            "inputs": {"structure": "a.cif", "exp_data": "a.cif_pets", "load_hydrogens": True},
        }
    )
    assert withH.inputs.load_hydrogens is True


def test_isotropic_displacements_only_defaults_off_and_parses() -> None:
    base = {"name": "abi", "inputs": {"structure": "a.cif", "exp_data": "a.cif_pets"}}
    assert ExperimentConfig.model_validate(base).inputs.isotropic_displacements_only is False
    forced = ExperimentConfig.model_validate(
        {
            "name": "abi",
            "inputs": {
                "structure": "a.cif",
                "exp_data": "a.cif_pets",
                "isotropic_displacements_only": True,
            },
        }
    )
    assert forced.inputs.isotropic_displacements_only is True


def test_refinement_rejects_hydrogen_mode_as_unknown_key() -> None:
    # H handling is scientific composition (Python/API via with_hydrogen_riding), not a config mode;
    # a strict config rejects the removed key rather than silently accepting a dead knob.
    base = {"name": "abi", "inputs": {"structure": "a.cif", "exp_data": "a.cif_pets"}}
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        ExperimentConfig.model_validate({**base, "refinement": {"hydrogen_mode": "riding"}})


def test_numerics_to_rocking_curve_shares_the_integration_geometry() -> None:
    cfg = ExperimentConfig.model_validate(
        {"name": "quartz", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    )
    blochwave = cfg.blochwave
    integration = IntegrationGeometry(semiangle=1.25)
    assert blochwave.to_rocking_curve(integration) == RockingCurve(
        sampling=blochwave.rocking_curve_sampling, integration=integration
    )
    assert blochwave.to_beam_selection(integration).integration is integration


def test_numerics_rejects_removed_integration_config() -> None:
    base = {"name": "bad", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        ExperimentConfig.model_validate({**base, "blochwave": {"integration": {"semiangle": 1.0}}})


def test_preprocess_rejects_removed_orientations_csv_config() -> None:
    base = {"name": "bad", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        ExperimentConfig.model_validate(
            {**base, "preprocess": {"orientations_csv": "optim_orientation.csv"}}
        )


def test_thickness_grid_bounds_are_validated() -> None:
    base = {"name": "bad", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
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
    base = {"name": "bad", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    with pytest.raises(ValidationError, match="rsg must be positive"):
        ExperimentConfig.model_validate({**base, "blochwave": {"rsg": 0.0}})
