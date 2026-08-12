"""Multi-dataset seams: always-tuple reads, per-dataset construction, per-file converge sweeps.

The pooling mechanics themselves are covered in ``test_pool.py``; the checkpoint independence in
``test_program_checkpoint.py``; the config validation in ``test_config.py``.
"""

import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from diffBloch.app.program import _read_experimental_data, converge_experiment
from diffBloch.config import ExperimentConfig, input_lock_for, load_config
from diffBloch.core.crystal import cell_matrix_from_parameters
from diffBloch.io import read_experimental_data, read_structure
from diffBloch.preprocess import setup_datasets
from diffBloch.preprocess.driver import ConvergenceState
from diffBloch.preprocess.experiment import _mean_inner_potential

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures"
QUARTZ = FIXTURE_ROOT / "quartz_anchor"


def _quartz_inputs():
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    record = read_experimental_data(QUARTZ / "exp_data.cif_pets")
    config = load_config(QUARTZ / "experiment.yaml")
    return structure, record, config


# --- _read_experimental_data: always a tuple, in exp_data order ---


def test_read_experimental_data_returns_a_one_tuple_for_a_single_path() -> None:
    cfg = load_config(QUARTZ / "experiment.yaml")
    records = _read_experimental_data(QUARTZ, cfg)
    assert len(records) == 1
    direct = read_experimental_data(QUARTZ / cfg.inputs.exp_data)
    assert records[0].wavelength == direct.wavelength
    assert records[0].n_rotations == direct.n_rotations


def test_read_experimental_data_reads_every_pooled_file_in_order(tmp_path: Path) -> None:
    for name in ("a.cif_pets", "b.cif_pets"):
        (tmp_path / name).write_bytes((QUARTZ / "exp_data.cif_pets").read_bytes())
    (tmp_path / "q.cif").write_bytes((QUARTZ / "enantiomer_1.cif").read_bytes())
    cfg = load_config(QUARTZ / "experiment.yaml").model_copy(
        update={
            "inputs": load_config(QUARTZ / "experiment.yaml").inputs.model_copy(
                update={
                    "structure": "q.cif",
                    "exp_data": ["a.cif_pets", "b.cif_pets"],
                    "multi_dataset": True,
                }
            )
        }
    )
    records = _read_experimental_data(tmp_path, cfg)
    assert len(records) == 2
    assert all(r.n_rotations == records[0].n_rotations for r in records)


# --- setup_datasets: per-dataset construction ---


def test_setup_datasets_rejects_empty_records() -> None:
    structure, _record, config = _quartz_inputs()
    with pytest.raises(ValueError, match="no experimental data"):
        setup_datasets(structure, (), config)


def test_setup_datasets_keeps_file_local_indices_and_shares_the_grid() -> None:
    structure, record, config = _quartz_inputs()
    _, datasets = setup_datasets(structure, (record, record), config)
    for dataset in datasets:
        assert [op.pattern.rotation_index for op in dataset.plan.orientations] == list(
            range(record.n_rotations)
        )
        # the same grid OBJECT rides on every per-dataset plan
        assert dataset.plan.structure_factor_grid is datasets[0].plan.structure_factor_grid
        assert dataset.n_rotations == record.n_rotations


def test_setup_datasets_gives_each_dataset_its_own_integration_geometry() -> None:
    """The old shared-semiangle restriction is gone: pooled precession angles may differ."""
    structure, record, config = _quartz_inputs()
    other_angle = record.model_copy(
        update={"precession_angles": np.full_like(np.asarray(record.precession_angles), 2.0)}
    )
    _, datasets = setup_datasets(structure, (record, other_angle), config)
    assert datasets[0].integration.semiangle == record.integration_semiangle
    assert datasets[1].integration.semiangle == 2.0


def test_setup_datasets_preserves_each_dataset_collection_geometry() -> None:
    structure, record, config = _quartz_inputs()
    precession = record.model_copy(update={"data_collection_geometry": "precession"})
    _, datasets = setup_datasets(structure, (record, precession), config)
    assert datasets[0].integration.geometry == "continuous_rotation"
    assert datasets[1].integration.geometry == "precession"


def test_setup_datasets_seeds_each_dataset_with_its_configured_mean_thickness() -> None:
    structure, record, config = _quartz_inputs()
    raw = config.model_dump(mode="python")
    raw["inputs"] = {
        **raw["inputs"],
        "exp_data": ["a.cif_pets", "b.cif_pets"],
        "multi_dataset": True,
    }
    raw["sample"] = {
        **raw["sample"],
        "mean_thickness_by_dataset": {
            "a.cif_pets": 400.0,
            "b.cif_pets": 800.0,
        },
    }
    multi_config = ExperimentConfig.model_validate(raw)

    _, datasets = setup_datasets(structure, (record, record), multi_config)

    assert datasets[0].plan.orientations[0].thickness.tolist() == [400.0]
    assert datasets[1].plan.orientations[0].thickness.tolist() == [800.0]


def test_setup_datasets_translates_global_ignore_to_file_local_slices() -> None:
    structure, record, config = _quartz_inputs()
    n = record.n_rotations
    ignoring = config.model_copy(
        update={
            "blochwave": config.blochwave.model_copy(update={"ignore_orientations": (0, 2, n + 3)})
        }
    )
    _, datasets = setup_datasets(structure, (record, record), ignoring)
    assert datasets[0].ignored_rotations == (0, 2)
    assert datasets[1].ignored_rotations == (3,)
    assert len(datasets[0].plan.orientations) == n - 2
    assert len(datasets[1].plan.orientations) == n - 1
    # gaps preserved: the ignored local indices are absent, later frames keep their numbers
    first_indices = [op.pattern.rotation_index for op in datasets[0].plan.orientations]
    assert first_indices[:3] == [1, 3, 4]


def test_setup_datasets_rejects_out_of_range_and_fully_ignored() -> None:
    structure, record, config = _quartz_inputs()
    n = record.n_rotations
    out_of_range = config.model_copy(
        update={"blochwave": config.blochwave.model_copy(update={"ignore_orientations": (2 * n,)})}
    )
    with pytest.raises(ValueError, match="outside the PETS rotation range"):
        setup_datasets(structure, (record, record), out_of_range)

    all_of_second = config.model_copy(
        update={
            "blochwave": config.blochwave.model_copy(
                update={"ignore_orientations": tuple(range(n, 2 * n))}
            )
        }
    )
    with pytest.raises(ValueError, match="excludes every PETS rotation of dataset 1"):
        setup_datasets(structure, (record, record), all_of_second)


def test_setup_datasets_computes_u0_once_per_distinct_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure, record, config = _quartz_inputs()
    calls: list[float] = []

    def counting(*args, **kwargs):
        calls.append(kwargs["energy"])
        return _mean_inner_potential(*args, **kwargs)

    monkeypatch.setattr("diffBloch.preprocess.experiment._mean_inner_potential", counting)
    setup_datasets(structure, (record, record, record), config)
    assert len(calls) == 1  # three identical-wavelength datasets -> one u0 computation


def test_multi_dataset_second_file_checked_against_first_not_structure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    structure, record, config = _quartz_inputs()
    cell = np.asarray(record.cell_parameters, dtype=np.float64)
    second = record.model_copy(
        update={
            "cell_parameters": cell * np.array([1.02, 1.0, 1.0, 1.0, 1.0, 1.0]),
            "source_path": Path("second.cif_pets"),
        }
    )

    with caplog.at_level(logging.WARNING, logger="diffBloch.preprocess.experiment"):
        _, datasets = setup_datasets(structure, (record, second), config)

    [log_record] = caplog.records
    message = log_record.getMessage()
    assert "second.cif_pets" in message
    assert str(record.source_path) in message
    assert "overrides" in message
    np.testing.assert_allclose(
        datasets[0].plan.structure_factor_grid.cell.numpy(),
        cell_matrix_from_parameters(record.cell_parameters),
    )


def test_multi_dataset_over_5pct_between_combined_files_raises() -> None:
    structure, record, config = _quartz_inputs()
    second = record.model_copy(
        update={
            "cell_parameters": record.cell_parameters * np.array([1.08, 1.0, 1.0, 1.0, 1.0, 1.0]),
            "source_path": Path("second.cif_pets"),
        }
    )

    with pytest.raises(ValueError, match="more than 5%") as excinfo:
        setup_datasets(structure, (record, second), config)

    message = str(excinfo.value)
    assert "second.cif_pets" in message
    assert str(record.source_path) in message
    assert "a:" in message


# --- converge_experiment: per-dataset sweeps ---


def _multi_experiment(tmp_path: Path) -> Path:
    """A valid two-dataset experiment dir built from quartz_anchor's inputs."""
    exp = tmp_path / "experiment"
    (exp / "reproducibility").mkdir(parents=True)
    (exp / "enantiomer_1.cif").write_bytes((QUARTZ / "enantiomer_1.cif").read_bytes())
    for name in ("a.cif_pets", "b.cif_pets"):
        (exp / name).write_bytes((QUARTZ / "exp_data.cif_pets").read_bytes())
    base = yaml.safe_load((QUARTZ / "experiment.yaml").read_text())
    base["inputs"] = {
        "structure": "enantiomer_1.cif",
        "exp_data": ["a.cif_pets", "b.cif_pets"],
        "multi_dataset": True,
    }
    # Disabled to keep these preprocess-focused tests off the (supported) per-dataset networks.
    base.setdefault("refinement", {})["thickness_nn"] = {"enabled": False}
    (exp / "experiment.yaml").write_text(yaml.safe_dump(base))
    lock = {
        "structure": input_lock_for(exp / "enantiomer_1.cif", ref="enantiomer_1.cif").model_dump(),
        "experimental_data": [
            input_lock_for(exp / name, ref=name).model_dump()
            for name in ("a.cif_pets", "b.cif_pets")
        ],
    }
    (exp / "reproducibility" / "experiment.lock").write_text(yaml.safe_dump(lock))
    return exp


def test_converge_experiment_sweeps_per_dataset_and_returns_the_elementwise_max(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp = _multi_experiment(tmp_path)
    settled = iter(
        [
            ConvergenceState(g_max=1.0, sg_max=0.02, tilt_steps=40),
            ConvergenceState(g_max=1.4, sg_max=0.01, tilt_steps=60),
        ]
    )
    starting: list[ConvergenceState] = []

    def fake_run_convergence(plan, state, *args, **kwargs):
        starting.append(state)
        return plan, next(settled)

    monkeypatch.setattr("diffBloch.app.program.run_convergence", fake_run_convergence)
    result = converge_experiment(exp, device="cpu")

    assert len(starting) == 2  # one sweep per dataset
    assert result == ConvergenceState(g_max=1.4, sg_max=0.02, tilt_steps=60)


def test_converge_starting_g_max_falls_back_per_dataset_when_dstar_max_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file without a dstar_max tag anchors at the configured g_max -- never at another file's."""
    exp = _multi_experiment(tmp_path)
    cfg = load_config(exp / "experiment.yaml")

    real_read = read_experimental_data

    def read_with_mixed_dstar_max(path):
        record = real_read(path)
        dstar = 1.4 if Path(path).name == "a.cif_pets" else None
        return record.model_copy(update={"dstar_max": dstar})

    monkeypatch.setattr("diffBloch.app.program.read_experimental_data", read_with_mixed_dstar_max)

    starting: list[ConvergenceState] = []

    def fake_run_convergence(plan, state, *args, **kwargs):
        starting.append(state)
        return plan, state

    monkeypatch.setattr("diffBloch.app.program.run_convergence", fake_run_convergence)
    converge_experiment(exp, device="cpu")

    assert starting[0].g_max == 1.4  # dataset a: its own dstar_max
    assert starting[1].g_max == cfg.blochwave.g_max  # dataset b: configured fallback
