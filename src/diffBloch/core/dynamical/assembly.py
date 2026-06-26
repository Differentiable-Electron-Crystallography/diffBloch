"""Differentiable structure-matrix assembly (the Bloch ``A`` path).

Combines a geometry-only plan (precomputed, NumPy) with the refined structure factors ``Fgb``
(torch, differentiable). Stage 8 builds this bottom-up — the first piece is the gather that maps
``Fgb`` onto the ``(N, N)`` off-diagonal positions ``F(g_j - g_i)``; the scaling
(``prefactor * Mii_i * Mii_j``), the diagonal, and the full ``build_bloch_system`` follow, drawing
their constants from the sibling ``core.dynamical.primitives`` module.

This is the torch half of ``core.dynamical``; ``primitives`` is the NumPy half. The split mirrors
the codebase's ``core.reciprocal`` (geometry) vs ``core.scattering`` (differentiable) seam.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from diffBloch.core.reciprocal import ravel_hkl

type IntArray = NDArray[np.int64]

# The off-diagonal of the Bloch structure matrix is ``A[i,j] = scale * F(g_j - g_i)`` — a gather of
# the structure factors ``Fgb`` onto every pair of beams. ``F`` is the only refined (differentiable)
# input; the gather *indices* are pure geometry, so they are precomputed once into a frozen plan.
# Ports the gather half of ``diffBloch_private`` ``calculate_structure_matrix`` /
# ``raveled_hkl_to_hkl_torch``, preserving its ``gmh = hkl[None] - hkl[:, None]`` ordering
# (so ``gmh[i,j] = hkl_j - hkl_i`` and ``A[i,j] = F(g_j - g_i)``).


@dataclass(frozen=True)
class StructureFactorGather:
    """Precomputed indices mapping structure factors onto the ``(N, N)`` off-diagonal grid.

    Geometry-only plan: ``source_indices`` ravel the ``Fgb`` support grid and
    ``destination_indices`` ravel the pairwise beam differences ``hkl_j - hkl_i``, both with the
    same ``gpts`` box (the shared-grid contract — one ``gpts`` keeps the two from drifting).
    Consumed by :func:`gather_structure_factors`, which scatters ``Fgb`` into a flat buffer and
    indexes it, preserving gradients.
    """

    source_indices: Tensor
    destination_indices: Tensor
    n_beams: int
    buffer_size: int
    gpts: tuple[int, int, int]


def build_structure_factor_gather(
    grid_hkl: IntArray,
    beam_hkl: IntArray,
    gpts: tuple[int, int, int],
) -> StructureFactorGather:
    """Precompute the structure-factor gather for a beam set against an ``Fgb`` support grid.

    ``grid_hkl`` ``(G, 3)`` are the Miller indices the structure factors are tabulated on;
    ``beam_hkl`` ``(N, 3)`` are the selected beams. The pairwise differences ``hkl_j - hkl_i`` range
    to ~2x the beam ``g_max``, so ``grid_hkl`` must cover them (the difference-support constraint) —
    validated here rather than silently gathering zeros. Both sets ravel through the same ``gpts``
    box (:func:`diffBloch.core.reciprocal.ravel_hkl`, which rejects indices outside the box).
    """
    grid = _beam_index_array(grid_hkl, name="grid_hkl")
    beams = _beam_index_array(beam_hkl, name="beam_hkl")

    # Private ordering: gmh[i, j] = beam_j - beam_i, so A[i, j] = F(beam_j - beam_i).
    gmh = (beams[None] - beams[:, None]).reshape(-1, 3)

    source = ravel_hkl(grid, gpts)  # also validates gpts (len 3, positive) and the grid box
    if np.unique(source).size != source.size:
        raise ValueError("grid_hkl must not contain duplicate Miller indices")

    # ravel_hkl centres the box at gpts // 2; a difference outside it means gpts is too small to
    # span the difference support (the realistic failure: gpts sized to the beam g_max, not 2x).
    # Catch it here with a clear message rather than letting ravel_hkl(gmh, ...) raise numpy's
    # cryptic "invalid entry in coordinates array".
    gpts_box = np.asarray(gpts, dtype=np.int64)
    shifted = gmh + gpts_box // 2
    out_of_box = np.any((shifted < 0) | (shifted >= gpts_box), axis=1)
    if out_of_box.any():
        missing = gmh[out_of_box][0]
        raise ValueError(
            "gpts is too small to contain the beam differences hkl_j - hkl_i; "
            f"first out-of-box difference {tuple(int(component) for component in missing)} "
            "(size gpts to span the difference support, ~2x the beam g_max)"
        )

    destination = ravel_hkl(gmh, gpts)
    uncovered = np.isin(destination, source, invert=True)
    if uncovered.any():
        missing = gmh[uncovered][0]
        raise ValueError(
            "grid_hkl must cover every beam difference hkl_j - hkl_i; "
            f"missing {tuple(int(component) for component in missing)} "
            "(the grid must span the difference support, ~2x the beam g_max)"
        )

    return StructureFactorGather(
        source_indices=torch.tensor(source, dtype=torch.long),
        destination_indices=torch.tensor(destination, dtype=torch.long),
        n_beams=int(beams.shape[0]),
        buffer_size=int(np.prod(gpts)),
        gpts=(int(gpts[0]), int(gpts[1]), int(gpts[2])),
    )


def gather_structure_factors(
    gather: StructureFactorGather,
    structure_factors: Tensor,
) -> Tensor:
    """Gather structure factors onto the ``(N, N)`` off-diagonal grid, preserving gradients.

    ``structure_factors`` ``(G,)`` is the ``Fgb`` tensor aligned with the plan's ``grid_hkl`` order.
    Scatters it into a flat reciprocal buffer (out-of-place ``index_add``) and indexes the buffer at
    the beam differences, so ``out[i, j] = F(beam_j - beam_i)``. Differentiable in
    ``structure_factors``.
    """
    if structure_factors.ndim != 1 or structure_factors.shape[0] != gather.source_indices.shape[0]:
        raise ValueError("structure_factors must have shape (G,) matching the gather grid")

    source = gather.source_indices.to(device=structure_factors.device)
    destination = gather.destination_indices.to(device=structure_factors.device)
    buffer = torch.zeros(
        gather.buffer_size, dtype=structure_factors.dtype, device=structure_factors.device
    )
    buffer = buffer.index_add(0, source, structure_factors)
    return buffer[destination].reshape(gather.n_beams, gather.n_beams)


def _beam_index_array(hkl: IntArray, *, name: str) -> IntArray:
    miller = np.asarray(hkl)
    if miller.ndim != 2 or miller.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    # Reject genuinely fractional / non-finite input rather than silently truncating (0.5 -> 0);
    # integer-valued floats (1.0) are accepted, matching the explicit-contract style elsewhere.
    if not np.issubdtype(miller.dtype, np.integer):
        is_integral = (
            np.issubdtype(miller.dtype, np.floating)
            and bool(np.all(np.isfinite(miller)))
            and bool(np.all(miller == np.rint(miller)))
        )
        if not is_integral:
            raise ValueError(f"{name} must contain integer Miller indices")
    return miller.astype(np.int64)
