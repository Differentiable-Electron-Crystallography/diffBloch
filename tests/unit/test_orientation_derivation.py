"""Native orientation derivation pinned against the private as-collected golden.

The golden (``orientation_derivation_golden.npz``) is produced out-of-repo by the *private*
diffBloch ``process_file`` (``generate_u_matrix`` + ``generate_crystal_orientations``) on the quartz
PETS anchor (see ``notebooks/iain/gen_orientation_derivation_golden.py``). Oracle independence lives
in the golden; the native path under test re-derives the same orientations from the UB/cell/angles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diffBloch.core.crystal import cell_matrix_from_parameters
from diffBloch.preprocess.orientation import (
    busing_levy_matrix,
    goniometer_rotation,
    orientation_basis,
    orientation_matrices,
    u_matrix,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "quartz_anchor"


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    return dict(np.load(FIXTURES / "orientation_derivation_golden.npz"))


def test_orientation_matrices_match_private_golden(golden: dict[str, np.ndarray]) -> None:
    native = orientation_matrices(
        golden["ub_matrix"],
        golden["cell_params"],
        golden["alphas"],
        golden["betas"],
        golden["omegas"],
    )
    assert native.shape == golden["orientation_matrices"].shape == (99, 3, 3)
    # ~8.6e-7 residual: we compute the exact cell volume; the private impl reads the PETS file's
    # rounded ``_cell_volume`` (113.32800 vs 113.32810). Negligible, and intentional (we do not
    # reproduce the rounding). The convention itself -- order, B form, U = UB B^-1 -- is exact.
    assert np.allclose(native, golden["orientation_matrices"], atol=1e-5)


def test_u_matrix_carries_the_cell_correction(golden: dict[str, np.ndarray]) -> None:
    # det(U) ~ 1.033: U = UB B^-1 folds the ~1% measured-vs-ideal cell correction, so U is NOT a
    # pure rotation. fit_orientation must preserve this (no re-orthonormalisation).
    u = u_matrix(golden["ub_matrix"], golden["cell_params"])
    assert np.linalg.det(u) == pytest.approx(1.03302, abs=1e-4)
    assert not np.allclose(u @ u.T, np.eye(3), atol=1e-2)


def test_goniometer_rotation_is_a_proper_rotation() -> None:
    assert np.allclose(goniometer_rotation(0.0, 0.0, 0.0), np.eye(3))
    r = goniometer_rotation(-49.197, 3.5, 12.0)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)


def test_busing_levy_determinant_is_inverse_volume(golden: dict[str, np.ndarray]) -> None:
    a, b, c, alpha, beta, gamma = golden["cell_params"]
    ca, cb, cg = np.cos(np.deg2rad([alpha, beta, gamma]))
    volume = a * b * c * np.sqrt(1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg)
    assert np.linalg.det(busing_levy_matrix(golden["cell_params"])) == pytest.approx(1.0 / volume)


def test_orientation_basis_matches_oracle() -> None:
    # orientation_basis is the convention home; pin it against the (independently generated)
    # orientation oracle, which stored rotated reciprocal bases for known orientation matrices.
    oracle = np.load(FIXTURES / "orientation_oracle.npz")
    cell = cell_matrix_from_parameters(oracle["cellpar"])
    for k in range(oracle["orientation"].shape[0]):
        basis = orientation_basis(cell, oracle["orientation"][k])
        assert np.allclose(basis, oracle["rotated_reciprocal_basis"][k], atol=1e-12)


def test_invalid_shapes_raise(golden: dict[str, np.ndarray]) -> None:
    with pytest.raises(ValueError, match="cell_parameters must have shape"):
        busing_levy_matrix(np.ones(5))
    with pytest.raises(ValueError, match="ub_matrix must have shape"):
        u_matrix(np.ones((2, 2)), golden["cell_params"])
    with pytest.raises(ValueError, match="equal length"):
        orientation_matrices(
            golden["ub_matrix"], golden["cell_params"], np.zeros(3), np.zeros(2), np.zeros(3)
        )
    with pytest.raises(ValueError, match="shape"):
        orientation_basis(np.eye(3), np.ones((3, 2)))
