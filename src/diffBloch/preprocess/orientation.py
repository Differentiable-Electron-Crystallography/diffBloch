"""Native crystal-orientation derivation for the preprocess pipeline.

Reconstructs per-rotation crystal orientation matrices from the experiment's goniometer geometry --
the UB matrix and per-rotation tilt angles recorded in the PETS data -- with no side-car
orientation file. The orientations are first-class inputs to the ``Plan``; ``fit_orientation``
refines them in-Plan -- it must not re-orthonormalise them: they fold a ~1% measured-vs-ideal cell
correction, so a polar/SVD projection would silently drop it (see KNOWN_ISSUES.md).

Convention, pinned against ``diffBloch_private/diffBloch/rotation_dataset.py``::

    orientation = R_z(omega) . R_x(alpha) . R_y(beta) @ U,    U = UB @ B^-1

where ``B`` is the Busing-Levy reciprocal matrix built from the cell parameters and the goniometer
rotations are active, in degrees. Geometry then uses :func:`orientation_basis` =
``reciprocal_cell(cell @ orientation.T)`` (NOT ``reciprocal_basis @ orientation.T``), because the
orientation matrices are generally non-orthonormal -- see ``tests/unit/test_orientation_oracle.py``.

References: W. R. Busing & H. A. Levy, *Acta Cryst.* **22**, 457 (1967) (the UB-matrix formalism);
diffBloch_private's ``rotation_dataset.py`` for the specific rotation ordering and B convention.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from diffBloch.core.crystal import orientation_basis

type FloatArray = NDArray[np.float64]

__all__ = [
    "busing_levy_matrix",
    "goniometer_rotation",
    "orientation_basis",
    "orientation_matrices",
    "u_matrix",
]


def busing_levy_matrix(cell_parameters: FloatArray) -> FloatArray:
    """Busing-Levy reciprocal B matrix from ``(a, b, c, alpha, beta, gamma)``, angles in degrees.

    Rows follow the standard ``a*``-along-x setting. The cell volume is computed exactly from the
    parameters; we deliberately do not consume a rounded ``_cell_volume`` field if the source file
    carries one (for the quartz anchor that rounding shifts orientations by ~1e-6, negligible).
    """
    params = np.asarray(cell_parameters, dtype=np.float64)
    if params.shape != (6,):
        raise ValueError("cell_parameters must have shape (6,)")
    if not np.all(np.isfinite(params)):
        raise ValueError("cell_parameters must be finite")
    a, b, c = params[:3]
    if a <= 0.0 or b <= 0.0 or c <= 0.0:
        raise ValueError("cell lengths must be positive")
    alpha, beta, gamma = np.deg2rad(params[3:])
    ca, cb, cg = np.cos([alpha, beta, gamma])
    sg = np.sin(gamma)
    if abs(sg) < 1e-12:
        raise ValueError("gamma must not be a multiple of 180 degrees (sin(gamma) ~ 0)")
    radicand = 1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg
    if radicand <= 0.0:
        raise ValueError("cell angles are geometrically inconsistent (non-positive cell volume)")
    volume = a * b * c * np.sqrt(radicand)
    return np.array(
        [
            [1.0 / a, 0.0, 0.0],
            [-cg / (a * sg), 1.0 / (b * sg), 0.0],
            [
                b * c / volume * (cg * (ca - cb * cg) / sg - cb * sg),
                a * c / (volume * sg) * (ca - cb * cg),
                a * b * sg / volume,
            ],
        ]
    )


def goniometer_rotation(alpha: float, beta: float, omega: float) -> FloatArray:
    """Active goniometer rotation ``R_z(omega) . R_x(alpha) . R_y(beta)``, angles in degrees."""
    a, b, o = np.deg2rad([alpha, beta, omega])
    rz = np.array([[np.cos(o), -np.sin(o), 0.0], [np.sin(o), np.cos(o), 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0.0, np.sin(b)], [0.0, 1.0, 0.0], [-np.sin(b), 0.0, np.cos(b)]])
    rotation: FloatArray = rz @ rx @ ry
    return rotation


def u_matrix(ub_matrix: FloatArray, cell_parameters: FloatArray) -> FloatArray:
    """Crystal ``U`` matrix ``U = UB @ B^-1``; generally non-orthonormal (folds cell correction)."""
    ub = np.asarray(ub_matrix, dtype=np.float64)
    if ub.shape != (3, 3):
        raise ValueError("ub_matrix must have shape (3, 3)")
    inverse: FloatArray = np.linalg.inv(busing_levy_matrix(cell_parameters))
    product: FloatArray = ub @ inverse
    return product


def orientation_matrices(
    ub_matrix: FloatArray,
    cell_parameters: FloatArray,
    alphas: FloatArray,
    betas: FloatArray,
    omegas: FloatArray,
) -> FloatArray:
    """Per-rotation orientation matrices ``R_goni @ U``, shape ``(R, 3, 3)``.

    ``alphas``/``betas``/``omegas`` are the per-rotation goniometer angles (degrees), one entry per
    PETS zone axis, in the same order as the record's ``zone_axis_ids``.
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    omegas = np.asarray(omegas, dtype=np.float64)
    if alphas.ndim != 1 or not (alphas.shape == betas.shape == omegas.shape):
        raise ValueError("alphas, betas, omegas must be 1-D arrays of equal length")
    u = u_matrix(ub_matrix, cell_parameters)
    return np.stack(
        [goniometer_rotation(a, b, o) @ u for a, b, o in zip(alphas, betas, omegas, strict=True)]
    )
