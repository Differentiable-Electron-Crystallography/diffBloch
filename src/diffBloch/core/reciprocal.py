"""Pure reciprocal-space helpers for Miller-index grids."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from diffBloch.core.crystal import reciprocal_cell

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]


def g_vectors(hkl: IntArray, reciprocal_basis: FloatArray) -> FloatArray:
    """Return reciprocal-space vectors for Miller indices and a reciprocal basis."""
    miller = _hkl_array(hkl)
    basis = np.asarray(reciprocal_basis, dtype=np.float64)
    if basis.shape != (3, 3):
        raise ValueError("reciprocal_basis must have shape (3, 3)")
    return miller @ basis


def g_vector_lengths(hkl: IntArray, reciprocal_basis: FloatArray) -> FloatArray:
    """Return ``|g|`` for Miller indices and a reciprocal basis."""
    return np.linalg.norm(g_vectors(hkl, reciprocal_basis), axis=1)


def reciprocal_space_gpts(cell: FloatArray, g_max: float) -> tuple[int, int, int]:
    """Return symmetric reciprocal-grid dimensions needed to cover ``g_max``."""
    if g_max < 0.0:
        raise ValueError("g_max must be non-negative")
    reciprocal_lengths = np.linalg.norm(reciprocal_cell(cell), axis=1)
    return (
        int(np.ceil(g_max / reciprocal_lengths[0])) * 2 + 1,
        int(np.ceil(g_max / reciprocal_lengths[1])) * 2 + 1,
        int(np.ceil(g_max / reciprocal_lengths[2])) * 2 + 1,
    )


def make_hkl_grid(
    cell: FloatArray,
    g_max: float,
    axes: Sequence[int] = (0, 1, 2),
) -> IntArray:
    """Return Miller indices whose reciprocal vectors satisfy ``|g| <= g_max``.

    ``axes`` may restrict the generated Miller dimensions while still filtering in full 3D with
    omitted axes fixed at zero.
    """
    selected_axes = tuple(axes)
    if not selected_axes:
        raise ValueError("axes must not be empty")
    if any(axis not in (0, 1, 2) for axis in selected_axes):
        raise ValueError("axes must contain only 0, 1, and 2")
    if len(set(selected_axes)) != len(selected_axes):
        raise ValueError("axes must not contain duplicates")

    gpts = reciprocal_space_gpts(cell, g_max)
    freqs = tuple(np.fft.fftfreq(n, d=1 / n).astype(np.int64) for n in gpts)
    axis_freqs = tuple(freqs[axis] for axis in selected_axes)
    grids = np.meshgrid(*axis_freqs, indexing="ij")
    partial_hkl = np.stack(grids, axis=-1).reshape((-1, len(selected_axes)))

    full_hkl = np.zeros((partial_hkl.shape[0], 3), dtype=np.int64)
    for source_axis, target_axis in enumerate(selected_axes):
        full_hkl[:, target_axis] = partial_hkl[:, source_axis]

    mask = g_vector_lengths(full_hkl, reciprocal_cell(cell)) <= g_max
    return partial_hkl[mask]


def gmax_mask(hkl: IntArray, reciprocal_basis: FloatArray, g_max: float) -> NDArray[np.bool_]:
    """Return a boolean mask selecting reflections with ``|g| <= g_max``."""
    if g_max < 0.0:
        raise ValueError("g_max must be non-negative")
    return g_vector_lengths(hkl, reciprocal_basis) <= g_max


def ravel_hkl(hkl: IntArray, gpts: tuple[int, int, int]) -> NDArray[np.intp]:
    """Map signed Miller indices to flat indices for a centered reciprocal grid."""
    miller = _hkl_array(hkl)
    if len(gpts) != 3 or any(point <= 0 for point in gpts):
        raise ValueError("gpts must contain three positive grid sizes")
    shift = np.asarray((gpts[0] // 2, gpts[1] // 2, gpts[2] // 2), dtype=np.int64)
    shifted = miller + shift
    return np.ravel_multi_index((shifted[:, 0], shifted[:, 1], shifted[:, 2]), gpts)


def _hkl_array(hkl: IntArray) -> IntArray:
    miller = np.asarray(hkl, dtype=np.int64)
    if miller.ndim != 2 or miller.shape[1] != 3:
        raise ValueError("hkl must have shape (N, 3)")
    return miller
