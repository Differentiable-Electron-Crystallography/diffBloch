"""Differentiable structure-matrix assembly (the Bloch ``A`` path).

Combines a geometry-only plan (precomputed, NumPy) with the refined structure factors ``Fgb``
(torch, differentiable). Stage 8 builds this bottom-up: the gather maps ``Fgb`` onto the ``(N, N)``
off-diagonal positions ``F(g_j - g_i)``, then :func:`structure_matrix` scales it
(``prefactor * Mii_i * Mii_j``) and fills the diagonal (``2 * k_n * Sg * Mii``). The full
``build_bloch_system`` (psi0, mask, propagators) follows, drawing its constants from the sibling
``core.dynamical.primitives`` module.

This is the torch half of ``core.dynamical``; ``primitives`` is the NumPy half. The split mirrors
the codebase's ``core.reciprocal`` (geometry) vs ``core.scattering`` (differentiable) seam.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from diffBloch.core.dynamical.primitives import (
    excitation_errors,
    m_factors,
    structure_matrix_prefactor,
    wavevector_magnitude,
)
from diffBloch.core.reciprocal import g_vectors, ravel_hkl

type IntArray = NDArray[np.int64]
type FloatArray = NDArray[np.float64]

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


# ---------------------------------------------------------------------------
# Structure-matrix assembly: A = scale(geometry) ⊙ gather(F), diagonal replaced
# ---------------------------------------------------------------------------
# Adds the geometry-only scale and diagonal to the slice-1 gather, completing the Bloch structure
# matrix A. Ports the no-absorption path of ``diffBloch_private`` ``calculate_structure_matrix``:
#   off-diagonal  A[i,j] = prefactor * Mii_i * Mii_j * F(g_j - g_i)
#   diagonal      A[i,i] = 2 * k_n * Sg_i * Mii_i      (replaces, not adds)
# Every constant is a native primitive (structure_matrix_prefactor, m_factors, excitation_errors,
# wavevector_magnitude); only F is refined, so all of it precomputes into a frozen plan.


@dataclass(frozen=True)
class BeamPlan:
    """Geometry/numerics-only plan for a beam set (immutable across refinement).

    Everything fixed by geometry and beam energy, with no dependence on the refined ``Fgb``: the
    slice-1 ``gather`` of structure factors onto beam pairs, the symmetrisation factors ``mii``
    (``(N,)``), the scalar off-diagonal ``prefactor``, the precomputed real structure-matrix
    ``diagonal`` (``(N,)`` = ``2 * k_n * Sg * Mii``), the propagation constant ``k_n``, the incident
    wavefunction ``psi0`` (``(N,)``, 1 at the 000 beam), and the active-beam ``mask`` (``(N,)``).
    :func:`structure_matrix` consumes it with ``Fgb`` to produce ``A``; :func:`build_bloch_system`
    wraps that into a :class:`BlochSystem` for the propagators.
    """

    gather: StructureFactorGather
    mii: Tensor
    prefactor: float
    diagonal: Tensor
    k_n: float
    psi0: Tensor
    mask: Tensor


def build_beam_plan(
    beam_hkl: IntArray,
    grid_hkl: IntArray,
    reciprocal_basis: FloatArray,
    *,
    energy: float,
    gpts: tuple[int, int, int],
    u0: float = 0.0,
) -> BeamPlan:
    """Precompute the geometry/numerics for a beam set.

    ``beam_hkl`` ``(N, 3)`` selects the beams; ``grid_hkl`` ``(G, 3)`` / ``gpts`` define the ``Fgb``
    support grid (slice-1 gather); ``reciprocal_basis`` ``(3, 3)`` gives ``g = beam_hkl @
    reciprocal_basis``. ``energy`` (eV) and ``u0`` (mean-inner-potential) set the wavevector.
    Composes the native primitives into the off-diagonal scale, the structure-matrix diagonal, and
    the propagation pieces (``k_n``, ``psi0``, ``mask``) -- mirroring ``diffBloch_private``
    ``calculate_structure_matrix`` (no-absorption path) and the ``psi0 = (hkl == 000)``
    convention of its dynamical-scattering propagator. ``mask`` is all-True here: the beams are the
    pre-selected active set (per-orientation ``sg_max`` selection is deferred).
    """
    gather = build_structure_factor_gather(grid_hkl, beam_hkl, gpts)
    beams = _beam_index_array(beam_hkl, name="beam_hkl")
    g = g_vectors(beams, reciprocal_basis)
    mii = m_factors(g, energy, u0=u0)
    sg = excitation_errors(g, energy, u0=u0)
    k_n = wavevector_magnitude(energy, u0=u0)
    prefactor = structure_matrix_prefactor(energy)
    diagonal = 2.0 * k_n * sg * mii
    psi0 = np.all(beams == 0, axis=1)  # incident beam: amplitude 1 at the 000 reflection
    return BeamPlan(
        gather=gather,
        mii=torch.tensor(mii, dtype=torch.float64),
        prefactor=prefactor,
        diagonal=torch.tensor(diagonal, dtype=torch.float64),
        k_n=k_n,
        psi0=torch.tensor(psi0, dtype=torch.complex128),
        mask=torch.ones(beams.shape[0], dtype=torch.bool),
    )


def structure_matrix(plan: BeamPlan, structure_factors: Tensor) -> Tensor:
    """Assemble the Bloch structure matrix ``A`` from a plan and structure factors ``Fgb``.

    Off-diagonal ``A[i,j] = prefactor * Mii_i * Mii_j * F(g_j - g_i)`` (slice-1 gather, then
    broadcast-scaled); the diagonal is replaced by the precomputed ``2 * k_n * Sg_i * Mii_i``.
    Differentiable in ``structure_factors`` (the diagonal is a geometry constant). Returns a complex
    ``(N, N)`` tensor in the dtype of ``structure_factors``.
    """
    off = gather_structure_factors(plan.gather, structure_factors)
    mii = plan.mii.to(device=off.device, dtype=off.real.dtype)
    off = off * (plan.prefactor * mii[None] * mii[:, None])
    return _fill_diagonal(off, plan.diagonal.to(device=off.device, dtype=off.dtype))


def _fill_diagonal(matrix: Tensor, diagonal: Tensor) -> Tensor:
    """Return a copy of square ``matrix`` with its diagonal replaced by ``diagonal`` (out-of-place).

    Gradient flows through the off-diagonal copy; the diagonal positions are overwritten. Mirrors
    ``diffBloch_private`` ``utils.py::fill_diagonal_torch``.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square (N, N)")
    if diagonal.shape != (matrix.shape[0],):
        raise ValueError("diagonal must have shape (N,) matching the matrix")
    filled = matrix.clone()
    index = torch.arange(matrix.shape[0], device=matrix.device)
    filled[index, index] = diagonal
    return filled


# ---------------------------------------------------------------------------
# BlochSystem: the closed, solver-agnostic dynamical-diffraction problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlochSystem:
    """A fully specified, propagator-agnostic Bloch-wave system.

    Reifies the coupled dynamical-diffraction equations ``dpsi/dz = (i pi / k_n) A psi`` with
    ``psi(0) = psi0``: the structure-matrix operator ``A`` (``(N, N)`` complex), its symmetrisation
    companion ``mii`` (``(N,)`` -- ``A`` is stored Hermitian-symmetrised so ``eigh`` applies;
    ``mii`` un-does that to recover physical amplitudes), the incident wavefunction ``psi0``
    (``(N,)``), the propagation constant ``k_n``, and the active-beam ``mask`` (``(N,)``).

    Defining invariant: a propagator consumes *only* a ``BlochSystem`` -- no geometry, energy, or
    hkl -- which is what makes it a closed system rather than a field bag. See ``core.solver``.
    """

    a: Tensor
    mii: Tensor
    psi0: Tensor
    k_n: float
    mask: Tensor


def build_bloch_system(plan: BeamPlan, structure_factors: Tensor) -> BlochSystem:
    """Assemble the closed Bloch system for a beam plan and structure factors ``Fgb``.

    ``A`` is built from the differentiable ``Fgb`` (so the system is differentiable in ``Fgb``); the
    symmetrisation factors, incident wavefunction, propagation constant, and active mask are carried
    straight from the geometry plan. The result is solver-agnostic -- see
    :func:`core.solver.propagate`.
    """
    return BlochSystem(
        a=structure_matrix(plan, structure_factors),
        mii=plan.mii,
        psi0=plan.psi0,
        k_n=plan.k_n,
        mask=plan.mask,
    )
