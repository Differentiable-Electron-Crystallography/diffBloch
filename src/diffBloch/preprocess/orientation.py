"""Native crystal-orientation derivation for the preprocess pipeline.

Reconstructs per-rotation crystal orientation matrices from the experiment's goniometer geometry --
the UB matrix and per-rotation tilt angles recorded in the PETS data -- with no side-car
orientation file. The orientations are first-class inputs to the ``Plan``; ``optimize_orientation``
refines them in-Plan -- it must not re-orthonormalise them: ``U`` carries PETS's own small
UB-vs-cell-parameters fit residual, so a polar/SVD projection would silently drop it.

Convention::

    orientation = R_z(omega) . R_x(alpha) . R_y(beta) @ U,    U = UB @ B^-1

This is the *as-collected* convention -- what PETS recorded. The goniometer axis is additionally
brought onto x by ``R_z(-rotation_axis_position)`` (:func:`rotation_axis_correction`), composed at
the dataset boundary in :func:`~diffBloch.preprocess.experiment.resolve_dataset_orientations` rather
than here, so this module's derivation stays a pure function of the PETS-recorded geometry.

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
    "compose_mosaic_tilts",
    "goniometer_rotation",
    "hexagonal_tilt",
    "isotropic_mosaic_tilts",
    "orientation_basis",
    "orientation_matrices",
    "rocking_curve_tilts",
    "rotation_axis_correction",
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


def rotation_axis_correction(rotation_axis_position: float) -> FloatArray:
    """``R_z(-rotation_axis_position)``, degrees: brings the true goniometer axis onto x.

    Two independent parts of the pipeline assume the goniometer/rotation axis lies along x in PETS's
    own coordinate frame: :func:`rocking_curve_tilts`, whose tilts are left-multiplied onto an
    orientation (``R_tilt @ orientation``), and
    :func:`~diffBloch.preprocess.steps.beams.klar_beam_mask`, whose lever arm is the lab-frame
    ``(g_y, g_z)`` -- the distance from the x rock axis. Both hold only when PETS's
    ``rotation axis position`` is zero.

    A nonzero value means the real axis sits at that azimuth instead, so every per-rotation
    orientation must be pre-rotated back by its negative before any x-axis tilt is composed onto it,
    or the whole rocking-curve integration runs about the wrong axis. This is the pure matrix; the
    composition and the record-vs-config resolution live in
    :func:`~diffBloch.preprocess.experiment.resolve_dataset_orientations`.
    """
    if not np.isfinite(rotation_axis_position):
        # PETS's free-text float grammar admits `nan`/`inf`; without this the trig below yields an
        # all-NaN matrix that silently poisons every orientation rather than failing here.
        raise ValueError(f"rotation_axis_position must be finite; got {rotation_axis_position!r}")
    theta = np.deg2rad(-rotation_axis_position)
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def hexagonal_tilt(azimuth: float, polar: float) -> FloatArray:
    """Palatinus hexagonal-search tilt ``R_z(azimuth) . R_x(polar) . R_z(-azimuth)``, in degrees.

    A tilt of magnitude ``polar`` about the in-plane axis at ``azimuth`` -- the delta rotation
    ``optimize_orientation`` right-multiplies onto an orientation (``orientation @ tilt``). Being a true
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


def isotropic_mosaic_tilts(
    sigma_degrees: float, samples: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Isotropic Gaussian mosaic-spread tilts, weights, and polar angles: ``(samples, 3, 3)``,
    ``(samples,)``, ``(samples,)``.

    Unlike :func:`rocking_curve_tilts` (tilts confined to one axis, x), this places ``samples``
    small rotations isotropically over *every* azimuth around the identity -- the physical picture
    of mosaic-block orientation spread (:class:`~diffBloch.specs.IsotropicMosaicity`), not a 1-D
    approximation of it. ``sigma_degrees`` is the standard deviation of a 2-D isotropic Gaussian
    orientation distribution (the usual crystallographic mosaicity convention), not a hard cutoff:
    directions are placed by a Fibonacci spherical spiral (golden-angle azimuth step) with the polar
    angle at each point set by the *inverse Rayleigh CDF* -- the polar-angle magnitude of a 2-D
    Gaussian with that standard deviation is Rayleigh-distributed, so this is quantile (not uniform)
    sampling of the true distribution, reaching well past ``sigma_degrees`` with shrinking density,
    exactly like the distribution it represents. Each direction becomes a :func:`hexagonal_tilt` (the
    tilt taking the pole to that direction).

    The returned ``weights`` are each sample's relative Gaussian density
    (``exp(-polar**2 / (2 * sigma_degrees**2))``), for :class:`~diffBloch.core.products.WeightedSum`:
    quantile placement alone assumes every bin carries equal probability mass, which the golden-angle
    azimuth spiral only approximates, so weighting by density on top corrects the residual bias
    quantile sampling leaves at small ``samples``. ``samples = 1`` or ``sigma_degrees = 0`` is the
    identity with weight 1, so a unit/zero-spread mosaicity composes off.

    The returned ``polar`` angles (degrees) are exposed separately from ``weights`` so a caller can
    recompute weights at a *different* sigma without regenerating the tilt geometry itself -- e.g.
    :class:`~diffBloch.engine.components.TrainableIsotropicMosaicity` refines sigma by reweighting
    this same fixed set of directions on every forward pass, rather than moving the directions
    themselves (which would require rebuilding the structure-factor gather every step).
    """
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if sigma_degrees < 0.0:
        raise ValueError("sigma_degrees must be non-negative")
    if samples == 1 or sigma_degrees == 0.0:
        return np.eye(3)[None], np.ones(1), np.zeros(1)
    golden_angle_deg = np.rad2deg(np.pi * (3.0 - np.sqrt(5.0)))
    k = np.arange(samples, dtype=np.float64)
    # Inverse Rayleigh CDF at quantiles (k + 0.5) / samples: the polar-angle magnitude of an
    # isotropic 2-D Gaussian with std sigma_degrees is Rayleigh(sigma_degrees)-distributed.
    polar = sigma_degrees * np.sqrt(-2.0 * np.log(1.0 - (k + 0.5) / samples))
    azimuth = (golden_angle_deg * k) % 360.0
    tilts = np.stack([hexagonal_tilt(az, po) for az, po in zip(azimuth, polar, strict=True)])
    weights = np.exp(-(polar**2) / (2.0 * sigma_degrees**2))
    return tilts, weights, polar


def compose_mosaic_tilts(rocking_tilts: FloatArray, mosaic_tilts: FloatArray) -> FloatArray:
    """Every rocking-curve tilt composed with every mosaic tilt: ``(R * M, 3, 3)``.

    ``rocking_tilts`` (``R`` of them, from :func:`rocking_curve_tilts`) sweep the goniometer/beam
    through the Ewald sphere; ``mosaic_tilts`` (``M``, from :func:`isotropic_mosaic_tilts`) perturb
    which way an individual mosaic block happens to point. Every block sees the same rocking sweep,
    so each mosaic tilt is applied to the *base* orientation first and the rocking sweep composed on
    top (``rocking_tilts[i] @ mosaic_tilts[j]``, matching the ``R_tilt @ orientation``
    left-multiply convention both already use) -- physically a block-orientation offset, then the
    beam sweeping across it, not the other way round. Flattened as ``index = r * M + m``, so
    consecutive entries share a rocking tilt (``m`` fastest) -- convenient for anyone reshaping back
    to ``(R, M, 3, 3)``.
    """
    return np.einsum("rij,mjk->rmik", rocking_tilts, mosaic_tilts).reshape(-1, 3, 3)


def rocking_curve_tilts(
    semiangle: float, sampling: int, *, geometry: str = "continuous_rotation"
) -> FloatArray:
    """Rocking-curve integration tilts as ``(N, 3, 3)`` rotation matrices, ``N = sampling``.

    For ``continuous_rotation``, ``sampling`` tilts span
    ``linspace(-semiangle, +semiangle, sampling)`` degrees about **x**, the goniometer axis in the
    PETS coordinate frame. ``sampling = 1`` is the identity so that unit-sample rocking integration
    composes off. For ``precession``, samples lie at the fixed cone semi-angle and uniformly spaced
    azimuths over ``[0, 360)`` using ``R_z(phi) R_x(semiangle) R_z(-phi)``. This is the convention
    used by the legacy preprocessing path.

    In both modes these matrices left-multiply the already-PETS-rotated nominal orientation
    (``R_tilt @ orientation``). Callers unpack a validated
    :class:`~diffBloch.specs.RockingCurve` into these raw arguments (the value-type owns the
    invariants), matching :func:`hexagonal_tilt`'s raw-float style.
    """
    if geometry == "precession":
        azimuths = np.deg2rad(np.linspace(0.0, 360.0, sampling, endpoint=False))
        polar = np.deg2rad(semiangle)
        cos_phi, sin_phi = np.cos(azimuths), np.sin(azimuths)
        cos_polar, sin_polar = np.cos(polar), np.sin(polar)
        tilts = np.empty((sampling, 3, 3), dtype=np.float64)
        # Expanded R_z(phi) @ R_x(polar) @ R_z(-phi), vectorized over cone azimuth.
        tilts[:, 0, 0] = cos_phi**2 + cos_polar * sin_phi**2
        tilts[:, 0, 1] = (1.0 - cos_polar) * cos_phi * sin_phi
        tilts[:, 0, 2] = sin_polar * sin_phi
        tilts[:, 1, 0] = tilts[:, 0, 1]
        tilts[:, 1, 1] = sin_phi**2 + cos_polar * cos_phi**2
        tilts[:, 1, 2] = -sin_polar * cos_phi
        tilts[:, 2, 0] = -sin_polar * sin_phi
        tilts[:, 2, 1] = sin_polar * cos_phi
        tilts[:, 2, 2] = cos_polar
        return tilts
    if geometry != "continuous_rotation":
        raise ValueError("geometry must be 'continuous_rotation' or 'precession'")
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
    """Per-rotation *as-collected* orientation matrices ``R_goni @ U``, shape ``(R, 3, 3)``.

    ``alphas``/``betas``/``omegas`` are the per-rotation goniometer angles (degrees), one entry per
    PETS zone axis, in the same order as the record's ``zone_axis_ids``.

    As-collected means exactly what PETS recorded: this deliberately applies no goniometer-axis
    correction, so the result is a pure function of ``UB``, the cell, and the angles. Bringing the
    rotation axis onto x is a separate concern with a separate input, and lives in
    :func:`~diffBloch.preprocess.experiment.resolve_dataset_orientations`.
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
