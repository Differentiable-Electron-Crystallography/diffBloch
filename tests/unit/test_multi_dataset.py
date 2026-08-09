"""``inputs.multi_dataset``: pooling several PETS files into one experiment.

Default (``multi_dataset=False``) must stay byte-for-byte the existing single-dataset behavior --
these tests exercise only the dispatch/pooling seam (:func:`_read_experimental_data`,
:func:`_as_records`) app/program.py adds on top of it. The pooling mechanics themselves
(rotation_index offsetting, per-file energy/u0, the shared-semiangle guard) are covered in
test_from_experiment.py.

``ExperimentalRecord`` carries numpy array fields, so pydantic's default ``==`` raises (array truth
value is ambiguous) -- these tests compare scalar fields or object identity, never the record as a
whole.
"""

from __future__ import annotations

from pathlib import Path

from diffBloch.app.program import _as_records, _read_experimental_data
from diffBloch.config import load_experiment
from diffBloch.io import ExperimentalRecord, read_experimental_data

FIXTURES = Path(__file__).parent.parent / "fixtures"
QUARTZ = FIXTURES / "quartz_anchor"


def test_read_experimental_data_returns_one_record_by_default() -> None:
    cfg, _lock = load_experiment(QUARTZ)
    direct = read_experimental_data(QUARTZ / cfg.inputs.exp_data)

    result = _read_experimental_data(QUARTZ, cfg)

    assert isinstance(result, ExperimentalRecord)
    assert result.wavelength == direct.wavelength
    assert result.n_rotations == direct.n_rotations


def test_read_experimental_data_pools_every_file_when_multi_dataset() -> None:
    cfg, _lock = load_experiment(QUARTZ)
    direct = read_experimental_data(QUARTZ / cfg.inputs.exp_data)
    pooled_cfg = cfg.model_copy(
        update={
            "inputs": cfg.inputs.model_copy(
                update={
                    "exp_data": [cfg.inputs.exp_data, cfg.inputs.exp_data],
                    "multi_dataset": True,
                }
            )
        }
    )

    result = _read_experimental_data(QUARTZ, pooled_cfg)

    assert isinstance(result, tuple)
    assert len(result) == 2
    for record in result:
        assert record.wavelength == direct.wavelength
        assert record.n_rotations == direct.n_rotations


def test_as_records_normalizes_a_single_record_to_a_one_tuple() -> None:
    direct = read_experimental_data(QUARTZ / "exp_data.cif_pets")

    normalized = _as_records(direct)

    assert normalized == (direct,)  # a 1-tuple of the same object: identity, not deep equality
    assert normalized[0] is direct


def test_as_records_passes_a_pooled_tuple_through_unchanged() -> None:
    direct = read_experimental_data(QUARTZ / "exp_data.cif_pets")
    pooled = (direct, direct)

    assert _as_records(pooled) is pooled
