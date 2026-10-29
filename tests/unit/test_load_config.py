"""`load_config` reads and validates an experiment.yaml from disk (the YAML boundary path)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from diffBloch.config import load_config
from diffBloch.config.schema import BlochwaveConfig, TrainableConfig

FIXTURE = Path(__file__).parent.parent / "fixtures" / "quartz_min" / "experiment.yaml"


def test_load_config_reads_yaml_and_applies_defaults() -> None:
    cfg = load_config(FIXTURE)
    # values from the file
    assert cfg.name == "quartz-min"
    assert cfg.inputs.structure == "enantiomer_1.cif"
    assert cfg.inputs.exp_data == "exp_data.cif_pets"
    # defaults-as-code applied for everything the file omits
    assert cfg.blochwave.solver == "matrix_exp"
    assert cfg.blochwave.incoherent_mosaicity is False


def test_incoherent_mosaicity_config_accepts_only_booleans() -> None:
    assert (
        BlochwaveConfig.model_validate({"incoherent_mosaicity": True}).incoherent_mosaicity is True
    )
    assert (
        BlochwaveConfig.model_validate({"incoherent_mosaicity": False}).incoherent_mosaicity
        is False
    )
    with pytest.raises(ValidationError):
        BlochwaveConfig.model_validate({"incoherent_mosaicity": {"samples": 3}})


def test_mosaicity_samples_defaults_to_five_and_rejects_nonpositive_values() -> None:
    assert BlochwaveConfig.model_validate({}).mosaicity_samples == 5
    assert BlochwaveConfig.model_validate({"mosaicity_samples": 3}).mosaicity_samples == 3
    with pytest.raises(ValidationError, match="mosaicity_samples must be >= 1"):
        BlochwaveConfig.model_validate({"mosaicity_samples": 0})


def test_mosaicity_sigma_trainable_flag_defaults_to_false() -> None:
    assert TrainableConfig.model_validate({}).mosaicity_sigma is False
    assert TrainableConfig.model_validate({"mosaicity_sigma": True}).mosaicity_sigma is True
    with pytest.raises(ValidationError):
        TrainableConfig.model_validate({"mosaicity_sigma": "not-a-bool"})
