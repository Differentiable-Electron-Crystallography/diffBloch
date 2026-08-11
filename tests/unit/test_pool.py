"""``pool``: merging settled per-dataset plans onto the pooled rotation-index space."""

from pathlib import Path

import pytest

from diffBloch.config import load_config
from diffBloch.io import read_experimental_data, read_structure
from diffBloch.preprocess import pool, setup_datasets
from diffBloch.preprocess.pipeline import StepRecord

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures"
QUARTZ = FIXTURE_ROOT / "quartz_anchor"


def _quartz_inputs():
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    record = read_experimental_data(QUARTZ / "exp_data.cif_pets")
    config = load_config(QUARTZ / "experiment.yaml")
    return structure, record, config


def test_pool_renumbers_with_file_offsets_preserving_ignore_gaps() -> None:
    structure, record, config = _quartz_inputs()
    n = record.n_rotations
    # Global ignore 1 lands on dataset 0 (local 1); global n lands on dataset 1 (local 0).
    ignoring = config.model_copy(
        update={"blochwave": config.blochwave.model_copy(update={"ignore_orientations": (1, n)})}
    )
    _, datasets = setup_datasets(structure, (record, record), ignoring)
    assert datasets[0].ignored_rotations == (1,)
    assert datasets[1].ignored_rotations == (0,)

    pooled = pool([d.plan for d in datasets], offsets=[0, n])

    indices = [op.pattern.rotation_index for op in pooled.orientations]
    assert indices == sorted(set(range(2 * n)) - {1, n})


def test_pool_single_plan_is_identity_on_indices() -> None:
    structure, record, config = _quartz_inputs()
    _, datasets = setup_datasets(structure, (record,), config)
    pooled = pool([datasets[0].plan], offsets=[0])
    assert [op.pattern.rotation_index for op in pooled.orientations] == list(
        range(record.n_rotations)
    )
    assert pooled.structure_factor_grid is datasets[0].plan.structure_factor_grid


def test_pool_rejects_mismatched_energies() -> None:
    structure, record, config = _quartz_inputs()
    other_voltage = record.model_copy(update={"wavelength": 0.0335})  # ~120 kV vs quartz's 200 kV
    _, datasets = setup_datasets(structure, (record, other_voltage), config)
    with pytest.raises(ValueError, match="different beam energies"):
        pool([d.plan for d in datasets], offsets=[0, record.n_rotations])


def test_pool_rejects_mismatched_grids() -> None:
    structure, record, config = _quartz_inputs()
    _, datasets = setup_datasets(structure, (record,), config)
    coarser = config.model_copy(
        update={
            "blochwave": config.blochwave.model_copy(update={"g_max": config.blochwave.g_max + 0.5})
        }
    )
    _, other = setup_datasets(structure, (record,), coarser)
    with pytest.raises(ValueError, match="structure-factor grid"):
        pool([datasets[0].plan, other[0].plan], offsets=[0, record.n_rotations])


def test_pool_rejects_empty_and_mismatched_offsets() -> None:
    structure, record, config = _quartz_inputs()
    _, datasets = setup_datasets(structure, (record,), config)
    with pytest.raises(ValueError, match="at least one plan"):
        pool([], offsets=[])
    with pytest.raises(ValueError, match="offsets"):
        pool([datasets[0].plan], offsets=[0, 5])


def test_pool_rejects_colliding_offsets() -> None:
    structure, record, config = _quartz_inputs()
    _, datasets = setup_datasets(structure, (record, record), config)
    with pytest.raises(ValueError, match="collide"):
        pool([d.plan for d in datasets], offsets=[0, 0])


def test_pool_carries_the_first_plans_provenance() -> None:
    structure, record, config = _quartz_inputs()
    _, datasets = setup_datasets(structure, (record, record), config)
    stamped = datasets[0].plan.__class__(
        structure_factor_grid=datasets[0].plan.structure_factor_grid,
        orientations=datasets[0].plan.orientations,
        provenance=(StepRecord(name="select_beams", params=None),),
    )
    pooled = pool([stamped, datasets[1].plan], offsets=[0, record.n_rotations])
    assert [r.name for r in pooled.provenance] == ["select_beams"]
