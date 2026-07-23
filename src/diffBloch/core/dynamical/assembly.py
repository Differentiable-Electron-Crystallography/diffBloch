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

from collections.abc import Sequence
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

    Geometry-only plan: ``structure_factor_indices`` ravel the ``Fgb`` support grid and
    ``beam_difference_indices`` ravel the pairwise beam differences ``hkl_j - hkl_i``, both with the
    same ``gpts`` box (the shared-grid contract — one ``gpts`` keeps the two from drifting).
    Consumed by :func:`gather_structure_factors`, which scatters ``Fgb`` into a flat buffer and
    indexes it, preserving gradients.
    """

    structure_factor_indices: Tensor
    beam_difference_indices: Tensor
    n_beams: int
    buffer_size: int
    gpts: tuple[int, int, int]


def grid_source_indices(structure_factor_hkl: IntArray, gpts: tuple[int, int, int]) -> IntArray:
    """The raveled ``structure_factor_hkl`` source offsets -- the grid-constant half of every gather index map.

    ``build_structure_factor_gather`` ravels the support grid to these offsets on every call, but
    they depend only on ``(structure_factor_hkl, gpts)`` -- invariant across beam sets, orientations, and
    trials. Precompute once and pass via ``structure_factor_indices=`` to skip re-raveling the
    (potentially
    large) support grid on every per-trial coupled rebuild. Same values
    ``build_structure_factor_gather`` would compute internally.
    """
    return ravel_hkl(_beam_index_array(structure_factor_hkl, name="structure_factor_hkl"), gpts)


def build_structure_factor_gather(
    structure_factor_hkl: IntArray,
    beam_hkl: IntArray,
    gpts: tuple[int, int, int],
    *,
    validate: bool = True,
    structure_factor_indices: IntArray | None = None,
) -> StructureFactorGather:
    """Precompute the structure-factor gather for a beam set against an ``Fgb`` support grid.

    ``structure_factor_hkl`` ``(G, 3)`` are the Miller indices the structure factors are tabulated on;
    ``beam_hkl`` ``(N, 3)`` are the selected beams. The pairwise differences ``hkl_j - hkl_i`` range
    to ~2x the beam ``g_max``, so ``structure_factor_hkl`` must cover them (the difference-support constraint) —
    validated here rather than silently gathering zeros. Both sets ravel through the same ``gpts``
    box (:func:`diffBloch.core.reciprocal.ravel_hkl`, which rejects indices outside the box).

    ``validate`` (default ``True``) runs three O(N^2 / N^2 log G) integrity checks: no duplicate
    ``structure_factor_hkl``, every beam difference in-box, and ``structure_factor_hkl`` covers every difference. They are
    the dominant cost when rebuilding a gather per trial over a large beam union, and are *pure
    checks* -- when they pass, the returned indices are identical to skipping them.

    ``validate=False`` skips them for a hot loop whose grid coverage is guaranteed **upstream** (a
    ``g_max`` guard). It is *not* self-guarding: the two skipped classes fail differently. A
    genuinely out-of-box difference still raises (``ravel_hkl``, numpy's terser message). But
    ``structure_factor_hkl`` is the ``|g| <= g_max`` sphere -- a subset of the rectangular box -- so an in-box
    difference *outside that sphere* is absent from it yet ravels to a valid box index the scatter
    never wrote: :func:`gather_structure_factors` reads a **silent zero**, with no runtime backstop.
    So ``validate=False`` is sound *only* under the upstream coverage guarantee; that guard (not the
    ``ravel_hkl`` check) is what keeps a mis-sized grid from silently gathering zeros.

    ``structure_factor_indices`` optionally supplies the precomputed grid ravel
    (:func:`grid_source_indices`)
    so a hot rebuild loop skips re-raveling the (large) support grid every call -- it is
    grid-constant, so one precompute serves every beam set. When ``None`` it is raveled here (also
    validating ``gpts`` + the grid box, which a supplied one is trusted to have satisfied).
    """
    beams = _beam_index_array(beam_hkl, name="beam_hkl")

    # Private ordering: gmh[i, j] = beam_j - beam_i, so A[i, j] = F(beam_j - beam_i).
    gmh = (beams[None] - beams[:, None]).reshape(-1, 3)

    if structure_factor_indices is None:
        grid = _beam_index_array(structure_factor_hkl, name="structure_factor_hkl")
        source = ravel_hkl(grid, gpts)  # also validates gpts (len 3, positive) and the grid box
        if validate and np.unique(source).size != source.size:
            raise ValueError("structure_factor_hkl must not contain duplicate Miller indices")
    else:
        source = np.asarray(structure_factor_indices, dtype=np.int64)

    if validate:
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

    destination = ravel_hkl(gmh, gpts)  # validate=False backstop: raises here on an out-of-box gmh
    if validate:
        uncovered = np.isin(destination, source, invert=True)
        if uncovered.any():
            missing = gmh[uncovered][0]
            raise ValueError(
                "structure_factor_hkl must cover every beam difference hkl_j - hkl_i; "
                f"missing {tuple(int(component) for component in missing)} "
                "(the grid must span the difference support, ~2x the beam g_max)"
            )

    return StructureFactorGather(
        structure_factor_indices=torch.tensor(source, dtype=torch.long),
        beam_difference_indices=torch.tensor(destination, dtype=torch.long),
        n_beams=int(beams.shape[0]),
        buffer_size=int(np.prod(gpts)),
        gpts=(int(gpts[0]), int(gpts[1]), int(gpts[2])),
    )


def gather_structure_factors(
    gather: StructureFactorGather,
    structure_factors: Tensor,
) -> Tensor:
    """Gather structure factors onto the ``(N, N)`` off-diagonal grid, preserving gradients.

    ``structure_factors`` ``(G,)`` is the ``Fgb`` tensor aligned with the plan's ``structure_factor_hkl`` order.
    Scatters it into a flat reciprocal buffer (out-of-place ``index_add``) and indexes the buffer at
    the beam differences, so ``out[i, j] = F(beam_j - beam_i)``. Differentiable in
    ``structure_factors``.
    """
    if (
        structure_factors.ndim != 1
        or structure_factors.shape[0] != gather.structure_factor_indices.shape[0]
    ):
        raise ValueError("structure_factors must have shape (G,) matching the gather grid")

    source = gather.structure_factor_indices.to(device=structure_factors.device)
    destination = gather.beam_difference_indices.to(device=structure_factors.device)
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
    structure_factor_hkl: IntArray,
    reciprocal_basis: FloatArray,
    *,
    energy: float,
    gpts: tuple[int, int, int],
    u0: float = 0.0,
    gather: StructureFactorGather | None = None,
    validate: bool = True,
) -> BeamPlan:
    """Precompute the geometry/numerics for a beam set.

    ``beam_hkl`` ``(N, 3)`` selects the beams; ``structure_factor_hkl`` ``(G, 3)`` / ``gpts`` define the ``Fgb``
    support grid (slice-1 gather); ``reciprocal_basis`` ``(3, 3)`` gives ``g = beam_hkl @
    reciprocal_basis``. ``energy`` (eV) and ``u0`` (mean-inner-potential) set the wavevector.
    Composes the native primitives into the off-diagonal scale, the structure-matrix diagonal, and
    the propagation pieces (``k_n``, ``psi0``, ``mask``) -- mirroring ``diffBloch_private``
    ``calculate_structure_matrix`` (no-absorption path) and the ``psi0 = (hkl == 000)``
    convention of its dynamical-scattering propagator. ``mask`` is all-True here: the beams are the
    pre-selected active set (per-orientation ``sg_max`` selection is deferred).

    ``gather`` may be a precomputed :class:`StructureFactorGather` for this exact ``(structure_factor_hkl,
    beam_hkl, gpts)`` -- the F-gather is basis-independent, so callers rebuilding many plans over a
    single beam set (rocking-curve tilts, orientation-search trials) build it once and pass it in,
    skipping the per-call index-map construction and its validation (the dominant cost; ports the
    private's precomputed HKL->index map, ``diffBloch_private`` 6bb3031). When ``None`` it is built
    here. A cheap shape guard rejects a gather that does not match this beam set / box.

    ``validate`` is forwarded to :func:`build_structure_factor_gather` when building the gather here
    (default ``True``); pass ``False`` on a hot rebuild loop whose grid coverage is guaranteed by an
    upstream ``g_max`` guard. Ignored when a precomputed ``gather`` is supplied.
    """
    if gather is None:
        gather = build_structure_factor_gather(
            structure_factor_hkl, beam_hkl, gpts, validate=validate
        )
    elif gather.n_beams != len(beam_hkl) or gather.gpts != tuple(int(p) for p in gpts):
        raise ValueError("precomputed gather does not match this beam_hkl / gpts")
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

    Gradient flows through the off-diagonal copy; the diagonal positions are overwritten. Rank-
    polymorphic: ``matrix`` is ``(..., N, N)`` with any leading batch dims and ``diagonal`` is
    ``(..., N)`` -- for a bare ``(N, N)`` this is exactly the single-matrix fill (byte-identical).
    Mirrors ``diffBloch_private`` ``utils.py::fill_diagonal_torch``.
    """
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must be square (..., N, N)")
    if diagonal.shape != matrix.shape[:-1]:
        raise ValueError("diagonal must have shape (..., N) matching the matrix")
    filled = matrix.clone()
    index = torch.arange(matrix.shape[-1], device=matrix.device)
    filled[..., index, index] = diagonal
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


# ---------------------------------------------------------------------------
# Batched assembly: one gather + one batched operator over a shared beam set
# ---------------------------------------------------------------------------
# The rocking-curve tilts of one orientation share a single beam set (same hkl, energy), so their
# structure matrices differ only in the geometry-per-tilt ``mii`` / ``diagonal`` (the excitation
# errors move with the tilt). Stacking them lets the solver run ONE batched eigendecomposition /
# matrix-exponential over ``(B, N, N)`` instead of a Python loop of B single solves -- the tilt-
# batching perf path. The shared F-gather is done once and broadcast across the batch.


@dataclass(frozen=True)
class BeamPlanBatch:
    """A stack of :class:`BeamPlan`\\ s over one shared beam set (an orientation's rocking tilts).

    Invariant (enforced by :func:`stack_beam_plans`): every plan shares the F-``gather``,
    ``prefactor``, ``k_n``, ``psi0``, and ``mask`` -- only the geometry-per-tilt ``mii`` ``(B, N)``
    and ``diagonal`` ``(B, N)`` are stacked. :func:`build_bloch_systems` consumes it with ``Fgb``
    to produce a batched :class:`BlochSystem` (operator ``(B, N, N)``).
    """

    gather: StructureFactorGather
    mii: Tensor
    prefactor: float
    diagonal: Tensor
    k_n: float
    psi0: Tensor
    mask: Tensor


def stack_beam_plans(plans: Sequence[BeamPlan]) -> BeamPlanBatch:
    """Stack beam plans sharing one beam set into a :class:`BeamPlanBatch`.

    Validates the shared-beam-set invariant -- identical gather indices, ``prefactor``, ``k_n``,
    ``psi0``, and ``mask`` -- then stacks the per-tilt ``mii`` / ``diagonal`` along a new leading
    batch axis. Raises if the plans do not share a beam set (the caller passed unrelated plans, not
    rocking-curve tilts of one orientation). Pure geometry: no dependence on ``Fgb``.
    """
    if not plans:
        raise ValueError("stack_beam_plans requires at least one beam plan")
    first = plans[0]
    for plan in plans[1:]:
        shares_beams = torch.equal(
            plan.gather.beam_difference_indices, first.gather.beam_difference_indices
        ) and torch.equal(
            plan.gather.structure_factor_indices, first.gather.structure_factor_indices
        )
        if not shares_beams:
            raise ValueError("stack_beam_plans requires plans sharing one beam set (gather)")
        if plan.prefactor != first.prefactor or plan.k_n != first.k_n:
            raise ValueError("stack_beam_plans requires plans sharing energy (prefactor, k_n)")
        if not torch.equal(plan.psi0, first.psi0) or not torch.equal(plan.mask, first.mask):
            raise ValueError("stack_beam_plans requires plans sharing psi0 and mask")
    return BeamPlanBatch(
        gather=first.gather,
        mii=torch.stack([plan.mii for plan in plans]),
        prefactor=first.prefactor,
        diagonal=torch.stack([plan.diagonal for plan in plans]),
        k_n=first.k_n,
        psi0=first.psi0,
        mask=first.mask,
    )


def build_bloch_systems(batch: BeamPlanBatch, structure_factors: Tensor) -> BlochSystem:
    """Assemble the batched Bloch system for a beam-plan batch and structure factors ``Fgb``.

    Gathers ``Fgb`` **once** onto the shared ``(N, N)`` off-diagonal grid, then scales it per tilt
    with the stacked ``mii`` and fills the per-tilt ``diagonal`` -- yielding a batched operator
    ``a`` ``(B, N, N)``. The result is a :class:`BlochSystem` whose fields carry the batch dim
    (``a`` ``(B, N, N)``, ``mii`` ``(B, N)``); :func:`core.solver.propagate` is rank-polymorphic
    over it. Differentiable in ``Fgb`` (the shared gather feeds every tilt).
    """
    off = gather_structure_factors(batch.gather, structure_factors)  # (N, N), shared across tilts
    mii = batch.mii.to(device=off.device, dtype=off.real.dtype)  # (B, N)
    off = off[None] * (batch.prefactor * mii[:, None, :] * mii[:, :, None])  # (B, N, N)
    a = _fill_diagonal(off, batch.diagonal.to(device=off.device, dtype=off.dtype))
    return BlochSystem(a=a, mii=batch.mii, psi0=batch.psi0, k_n=batch.k_n, mask=batch.mask)
