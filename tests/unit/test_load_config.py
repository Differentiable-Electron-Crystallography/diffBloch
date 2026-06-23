"""`load_config` reads and validates an experiment.yaml from disk (the YAML boundary path)."""

from pathlib import Path

from diffBloch.config import load_config

FIXTURE = Path(__file__).parent.parent / "fixtures" / "quartz_min" / "experiment.yaml"


def test_load_config_reads_yaml_and_applies_defaults() -> None:
    cfg = load_config(FIXTURE)
    # values from the file
    assert cfg.name == "quartz-min"
    assert cfg.inputs.structure == "enantiomer_1.cif"
    assert cfg.inputs.observations == "exp_data.cif_pets"
    assert cfg.numerics.g_max_refine == 1.6
    # defaults-as-code applied for everything the file omits
    assert cfg.solver.refine == "matrix_exp"
    assert cfg.observation.beam_damage.activate is False
