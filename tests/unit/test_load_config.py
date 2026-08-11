"""`load_config` reads and validates an experiment.yaml from disk (the YAML boundary path)."""

from pathlib import Path

from diffBloch.config import load_config
from diffBloch.config.schema import BlochwaveConfig
from diffBloch.specs import Mosaicity

FIXTURE = Path(__file__).parent.parent / "fixtures" / "quartz_min" / "experiment.yaml"


def test_load_config_reads_yaml_and_applies_defaults() -> None:
    cfg = load_config(FIXTURE)
    # values from the file
    assert cfg.name == "quartz-min"
    assert cfg.inputs.structure == "enantiomer_1.cif"
    assert cfg.inputs.exp_data == "exp_data.cif_pets"
    # defaults-as-code applied for everything the file omits
    assert cfg.blochwave.solver == "matrix_exp"
    assert cfg.blochwave.mosaicity is False


def test_mosaicity_config_accepts_boolean_and_legacy_window_forms() -> None:
    assert BlochwaveConfig.model_validate({"mosaicity": True}).mosaicity is True
    assert BlochwaveConfig.model_validate({"mosaicity": False}).mosaicity is False
    legacy = BlochwaveConfig.model_validate({"mosaicity": {"window": 3}}).mosaicity
    assert legacy == Mosaicity(window=3)
