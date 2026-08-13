"""Pure unit-cell and lattice-centering helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]


def cell_matrix_from_parameters(parameters: FloatArray) -> FloatArray:
    """Return the fractional-to-Cartesian cell matrix from six cell parameters.

    ``parameters`` are ordered as ``(a, b, c, alpha, beta, gamma)``, with lengths in Angstrom
    and angles in degrees. Rows of the returned matrix are the real-space basis vectors.
    """
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError("cell parameters must have shape (6,)")

    a, b, c, alpha_deg, beta_deg, gamma_deg = values
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    gamma = np.deg2rad(gamma_deg)

    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    volume_factor = np.sqrt(
        1.0 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2.0 * cos_alpha * cos_beta * cos_gamma
    )
    return np.asarray(
        [
            [a, 0.0, 0.0],
            [b * cos_gamma, b * sin_gamma, 0.0],
            [
                c * cos_beta,
                c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
                c * volume_factor / sin_gamma,
            ],
        ],
        dtype=np.float64,
    )


def reciprocal_cell(cell: FloatArray) -> FloatArray:
    """Return reciprocal lattice vectors as rows in Angstrom^-1.

    Uses the ASE-compatible convention ``reciprocal_cell = pinv(cell).T``.
    """
    matrix = _cell_array(cell)
    return np.linalg.pinv(matrix).T


def cell_volume(cell: FloatArray) -> float:
    """Return the positive unit-cell volume in Angstrom^3."""
    matrix = _cell_array(cell)
    return float(abs(np.linalg.det(matrix)))


def orientation_basis(cell: FloatArray, orientation: FloatArray) -> FloatArray:
    """Lab-frame reciprocal basis for one orientation: ``reciprocal_cell(cell @ orientation.T)``.

    The single home for the orientation convention. ``orientation`` is not guaranteed exactly
    orthonormal -- it carries PETS's own UB-vs-cell-parameters fit residual (see
    ``preprocess.orientation``) -- so this is NOT ``reciprocal_basis @ orientation.T``. ``cell`` rows
    are the real-space basis; returns rows ``a*, b*, c*`` in inverse Angstrom. ``orientation = I``
    reproduces ``reciprocal_cell(cell)``.
    """
    matrix = _cell_array(cell)
    rotation = np.asarray(orientation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("orientation must have shape (3, 3)")
    return reciprocal_cell(matrix @ rotation.T)


def reflection_condition(hkl: IntArray, centering: str) -> NDArray[np.bool_]:
    """Return the mask of reflections allowed by a lattice-centering rule."""
    miller = np.asarray(hkl, dtype=np.int64)
    if miller.ndim != 2 or miller.shape[1] != 3:
        raise ValueError("hkl must have shape (N, 3)")

    match centering.upper():
        case "P":
            return np.ones(miller.shape[0], dtype=np.bool_)
        case "I":
            return np.asarray(miller.sum(axis=1) % 2 == 0, dtype=np.bool_)
        case "F":
            even = (miller % 2 == 0).all(axis=1)
            odd = (miller % 2 != 0).all(axis=1)
            return np.asarray(even | odd, dtype=np.bool_)
        case "A":
            return np.asarray((miller[:, 1] + miller[:, 2]) % 2 == 0, dtype=np.bool_)
        case "B":
            return np.asarray((miller[:, 0] + miller[:, 2]) % 2 == 0, dtype=np.bool_)
        case "C":
            return np.asarray((miller[:, 0] + miller[:, 1]) % 2 == 0, dtype=np.bool_)
        case _:
            raise ValueError(f"unsupported lattice centering: {centering!r}")


def _cell_array(cell: FloatArray) -> FloatArray:
    matrix = np.asarray(cell, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("cell must have shape (3, 3)")
    return matrix
