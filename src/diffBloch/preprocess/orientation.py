"""Native crystal-orientation derivation for the preprocess pipeline.

Reconstructs per-rotation crystal orientation matrices from the experiment's goniometer geometry --
the UB matrix and per-rotation tilt angles recorded in the PETS data -- with no side-car
orientation file. The orientations are first-class inputs to the ``Plan``; ``optimize_orientation``
refines them in-Plan -- it must not re-orthonormalise them: ``U`` carries PETS's own small
UB-vs-cell-parameters fit residual, so a polar/SVD projection would silently drop it.

Convention::

    orientation = R_z(omega) . R_x(alpha) . R_y(beta) @ U,    U = UB @ B^-1

where ``B`` is the Busing-Levy reciprocal matrix built from *this dataset's own* PETS cell
parameters (the same cell ``UB`` was fit against, so ``U`` is close to a pure rotation -- see
``diffBloch.preprocess.experiment._resolve_authoritative_cell`` for how a combined experiment's
*shared* cell is chosen and cross-checked) and the goniometer rotations are active, in degrees.
Geometry then uses :func:`orientation_basis` = ``reciprocal_cell(cell @ orientation.T)`` (NOT
``reciprocal_basis @ orientation.T``), because ``orientation`` is not guaranteed exactly orthonormal
even so.

Reference: W. R. Busing & H. A. Levy, *Acta Cryst.* **22**, 457 (1967) (the UB-matrix formalism).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from diffBloch.core.crystal import orientation_basis

type FloatArray = NDArray[np.float64]

__all__ = [
    "busing_levy_matrix",
    "goniometer_rotation",
    "hexagonal_tilt",
    "orientation_basis",
    "orientation_matrices",
    "mosaic_rocking_curve_tilts",
    "rocking_curve_tilts",
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


def hexagonal_tilt(azimuth: float, polar: float) -> FloatArray:
    """Palatinus hexagonal-search tilt ``R_z(azimuth) . R_x(polar) . R_z(-azimuth)``, in degrees.

    A tilt of magnitude ``polar`` about the in-plane axis at ``azimuth`` -- the delta rotation
    ``optimize_orientation`` right-multiplies onto an orientation (``orientation @ tilt``). Being a
    true rotation (``det = 1``) it preserves whatever small fit residual ``U`` already carries
    exactly, so the re-orthonormalisation trap is dodged by construction.

    Reference: L. Palatinus et al., *Acta Cryst.* **A69**, 171-188 (2013), the hexagonal
    modified-simplex search.
    """
    phi, theta = np.deg2rad([azimuth, polar])
    rz = np.array(
        [[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]]
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]]
    )
    tilt: FloatArray = rz @ rx @ rz.T  # rz.T = R_z(-azimuth)
    return tilt


def rocking_curve_tilts(
    semiangle: float, sampling: int, *, geometry: str = "continuous_rotation"
) -> FloatArray:
    """Rocking-curve integration tilts as ``(N, 3, 3)`` rotation matrices, ``N = sampling``.

    ``sampling`` tilts at angles ``linspace(-semiangle, +semiangle, sampling)`` degrees, each a
    rotation about **x** -- the goniometer axis in the PETS coordinate frame, where these matrices
    left-multiply the already-PETS-rotated orientation (``R_tilt @ orientation``). ``sampling = 1``
    is special-cased to a single tilt at angle 0 (the identity), so composing the integration with a
    unit sampling is a no-op -- ``np.linspace`` would otherwise return the *start* ``-semiangle``
    for ``num = 1``, off-centre from the nominal orientation.

    Continuous-rotation geometry only; ``precession`` (a cone) is a later discriminated mode and
    raises ``NotImplementedError`` here. Callers unpack a validated
    :class:`~diffBloch.specs.RockingCurve` into these raw arguments (the value-type owns the
    invariants), matching :func:`hexagonal_tilt`'s raw-float style.
    """
    if geometry != "continuous_rotation":
        raise NotImplementedError(f"rocking-curve geometry {geometry!r} is not implemented")
    if sampling == 1:
        angles = np.zeros(1)  # single sample sits at the nominal orientation -> identity tilt
    else:
        angles = np.deg2rad(np.linspace(-semiangle, semiangle, sampling))
    cos, sin = np.cos(angles), np.sin(angles)
    tilts = np.zeros((sampling, 3, 3), dtype=np.float64)
    tilts[:, 0, 0] = 1.0
    tilts[:, 1, 1] = cos
    tilts[:, 1, 2] = -sin
    tilts[:, 2, 1] = sin
    tilts[:, 2, 2] = cos
    return tilts


def mosaic_rocking_curve_tilts(
    semiangle: float,
    sampling: int,
    sigma_degrees: float,
    *,
    geometry: str = "continuous_rotation",
) -> tuple[FloatArray, FloatArray]:
    """Rocking tilts expanded by a normalized three-point Gaussian mosaic distribution.

    PETS records apparent mosaicity as the Gaussian sigma in degrees. Each nominal tilt is evaluated
    at offsets ``(-sqrt(3), 0, +sqrt(3)) * sigma`` with Gauss-Hermite weights
    ``(1/6, 2/3, 1/6)``. The returned weights sum to ``sampling`` so mosaicity preserves the
    ordinary rocking-curve integration scale.
    """
    if sigma_degrees < 0.0:
        raise ValueError("mosaicity sigma must be non-negative")
    if sigma_degrees == 0.0:
        return rocking_curve_tilts(semiangle, sampling, geometry=geometry), np.ones(sampling)
    if geometry != "continuous_rotation":
        raise NotImplementedError(f"rocking-curve geometry {geometry!r} is not implemented")
    nominal = np.zeros(1) if sampling == 1 else np.linspace(-semiangle, semiangle, sampling)
    offsets = np.sqrt(3.0) * sigma_degrees * np.asarray([-1.0, 0.0, 1.0])
    angles = (nominal[:, None] + offsets[None, :]).reshape(-1)
    weights = np.broadcast_to(np.asarray([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]), (sampling, 3)).reshape(
        -1
    )
    order = np.argsort(angles, kind="stable")
    radians = np.deg2rad(angles[order])
    cos, sin = np.cos(radians), np.sin(radians)
    tilts = np.zeros((len(radians), 3, 3), dtype=np.float64)
    tilts[:, 0, 0] = 1.0
    tilts[:, 1, 1] = cos
    tilts[:, 1, 2] = -sin
    tilts[:, 2, 1] = sin
    tilts[:, 2, 2] = cos
    return tilts, weights[order].copy()


def u_matrix(ub_matrix: FloatArray, cell_parameters: FloatArray) -> FloatArray:
    """Crystal ``U`` matrix ``U = UB @ B^-1``, from this dataset's own PETS UB and cell parameters.

    Close to a pure rotation (``B`` is built from the same cell PETS fit ``UB`` against), but not
    guaranteed to be exactly orthonormal -- PETS's own UB-vs-cell-parameters fit carries a small
    residual. Not re-orthonormalised (see the module docstring).
    """
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
