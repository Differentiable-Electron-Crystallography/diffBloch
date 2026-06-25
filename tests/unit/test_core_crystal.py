from pathlib import Path

import numpy as np
import pytest

from diffBloch.core import (
    cell_matrix_from_parameters,
    cell_volume,
    g_vector_lengths,
    g_vectors,
    gmax_mask,
    make_hkl_grid,
    ravel_hkl,
    reciprocal_cell,
    reciprocal_space_gpts,
    reflection_condition,
)
from diffBloch.io import read_structure

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"


def test_cell_matrix_from_parameters_matches_quartz_fixture() -> None:
    record = read_structure(FIXTURE_ROOT / "enantiomer_1.cif")

    assert cell_matrix_from_parameters(record.cell_parameters) == pytest.approx(record.unit_cell)


def test_reciprocal_cell_matches_private_convention() -> None:
    cell = np.asarray(
        [
            [5.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 7.0],
        ],
        dtype=np.float64,
    )

    reciprocal = reciprocal_cell(cell)

    assert reciprocal == pytest.approx(
        np.asarray(
            [
                [1 / 5, 0.0, 0.0],
                [0.0, 1 / 6, 0.0],
                [0.0, 0.0, 1 / 7],
            ],
            dtype=np.float64,
        )
    )
    assert cell @ reciprocal.T == pytest.approx(np.eye(3))
    assert cell_volume(cell) == pytest.approx(210.0)


def test_g_vectors_and_lengths() -> None:
    reciprocal = np.diag([0.2, 0.2, 0.2])
    hkl = np.asarray([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.int64)

    assert g_vectors(hkl, reciprocal) == pytest.approx(
        np.asarray([[0.2, 0.0, 0.0], [0.2, 0.2, 0.0], [0.2, 0.2, 0.2]])
    )
    assert g_vector_lengths(hkl, reciprocal) == pytest.approx(
        np.asarray([0.2, 0.2 * np.sqrt(2), 0.2 * np.sqrt(3)])
    )


def test_reciprocal_space_grid_covers_gmax_and_stays_symmetric() -> None:
    cell = np.eye(3) * 5.0

    assert reciprocal_space_gpts(cell, 0.5) == (7, 7, 7)

    hkl = make_hkl_grid(cell, 0.5)
    lengths = g_vector_lengths(hkl, reciprocal_cell(cell))
    assert np.all(lengths <= 0.5 + 1e-12)
    assert np.any(np.all(hkl == 0, axis=1))
    for reflection in hkl:
        assert np.any(np.all(hkl == -reflection, axis=1))


def test_reciprocal_space_grid_accepts_zero_gmax() -> None:
    cell = np.eye(3) * 5.0

    assert reciprocal_space_gpts(cell, 0.0) == (1, 1, 1)
    assert make_hkl_grid(cell, 0.0).tolist() == [[0, 0, 0]]


def test_make_hkl_grid_can_restrict_axes() -> None:
    cell = np.eye(3) * 5.0
    hk = make_hkl_grid(cell, 0.5, axes=(0, 1))

    assert hk.ndim == 2
    assert hk.shape[1] == 2
    full = np.column_stack([hk, np.zeros(hk.shape[0], dtype=np.int64)])
    assert np.all(g_vector_lengths(full, reciprocal_cell(cell)) <= 0.5 + 1e-12)


def test_reflection_condition_for_centered_lattices() -> None:
    hkl = np.asarray(
        [
            [1, 1, 1],
            [2, 2, 2],
            [1, 1, 0],
            [2, 1, 0],
        ],
        dtype=np.int64,
    )

    assert reflection_condition(hkl, "P").tolist() == [True, True, True, True]
    assert reflection_condition(hkl, "I").tolist() == [False, True, True, False]
    assert reflection_condition(hkl, "F").tolist() == [True, True, False, False]

    abc_hkl = np.asarray(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
        ],
        dtype=np.int64,
    )
    assert reflection_condition(abc_hkl, "A").tolist() == [False, False, True, True]
    assert reflection_condition(abc_hkl, "B").tolist() == [False, True, False, False]
    assert reflection_condition(abc_hkl, "C").tolist() == [True, False, False, False]


def test_gmax_mask_and_ravel_hkl() -> None:
    reciprocal = np.diag([0.2, 0.2, 0.2])
    hkl = np.asarray([[0, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=np.int64)

    assert gmax_mask(hkl, reciprocal, 0.5).tolist() == [True, True, False]
    assert gmax_mask(hkl, reciprocal, 0.0).tolist() == [True, False, False]
    assert ravel_hkl(np.asarray([[0, 0, 0]], dtype=np.int64), (5, 5, 5))[0] == np.ravel_multi_index(
        (2, 2, 2), (5, 5, 5)
    )


def test_core_helpers_validate_shapes() -> None:
    with pytest.raises(ValueError, match="cell must have shape"):
        reciprocal_cell(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="hkl must have shape"):
        g_vectors(np.zeros((3,), dtype=np.int64), np.eye(3))
    with pytest.raises(ValueError, match="unsupported lattice centering"):
        reflection_condition(np.zeros((1, 3), dtype=np.int64), "X")
