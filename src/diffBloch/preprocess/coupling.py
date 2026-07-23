"""Tilt-segment-union beam coupling: the per-tilt-chunk beam sets of the private rocking curve.

Rocking-curve integration solves the crystal at many slightly-tilted sub-orientations. A single
beam set for the whole curve either over-couples (every beam any tilt needs, slow) or drops beams a
tilt needs (a reflection drifts through the Ewald sphere as the crystal rocks). The
``diffBloch_private`` policy instead partitions the tilts into contiguous chunks and couples, within
each chunk, the **union** of the excited-beam sets at the chunk's two boundary tilts.

This module is the pure geometry of that partition: given a
:class:`~diffBloch.specs.SegmentedUnionCoupling`
policy and the rotation's tilt geometry, it returns the ordered :class:`Segment` list (each a beam
set + the disjoint tilt indices it covers). It computes the same excitation mask
(``|Sg| < sg_max`` and ``|g| < g_max``) the private's ``BlochNet.forward`` builds its
union sets from, reusing :func:`~diffBloch.core.dynamical.excitation_errors` and
:func:`~diffBloch.core.crystal.orientation_basis`. It does not build or solve anything -- an engine
step turns the segments into per-chunk plans and reassembles their curves before reduction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from diffBloch.core.crystal import orientation_basis
from diffBloch.core.dynamical import excitation_errors
from diffBloch.specs import SegmentedUnionCoupling

__all__ = [
    "Segment",
    "build_coupling_segments",
]


@dataclass(frozen=True)
class Segment:
    """One tilt chunk's coupling: the beam set it solves and the tilt indices it covers.

    ``union_hkl`` ``(n_beams, 3)`` is the union of the excited beams at the chunk's boundary tilts
    (includes ``(0, 0, 0)``, always excited). ``covered_tilt_indices`` are the global rocking-curve
    tilt indices this segment is responsible for -- contiguous and, across a rotation's segments,
    disjoint and covering every tilt exactly once. The segment solves its beam set at each covered
    tilt; the per-tilt intensities are later scattered back onto each reflection's full rocking
    curve.
    """

    union_hkl: NDArray[np.int64]
    covered_tilt_indices: tuple[int, ...]


def _fixed_segment_ranges(n_tilts: int, fixed_n_segments: int) -> NDArray[np.int64]:
    """The private's contiguous split boundaries into ``fixed_n_segments`` chunks over ``n_tilts``.

    Ported verbatim from ``BlochNet.forward`` (``union_adaptive = False`` path): evenly sized chunks
    with the remainder front-loaded, boundaries clamped so the last index is ``n_tilts - 1``, then
    de-duplicated. Returns the strictly increasing boundary indices (length ``fixed_n_segments + 1``
    before de-duplication), so ``fixed_n_segments`` segments span ``boundaries[i] .. boundaries[i +
    1]``.
    """
    fixed_n_segments = max(1, fixed_n_segments)
    if fixed_n_segments >= n_tilts:
        boundaries = np.arange(n_tilts, dtype=np.int64)
    else:
        base, extra = divmod(n_tilts, fixed_n_segments)
        sizes = [base + 1] * extra + [base] * (fixed_n_segments - extra)
        acc = [0]
        for size in sizes:
            acc.append(acc[-1] + size)
        boundaries = np.asarray(acc, dtype=np.int64)
        boundaries[-1] = n_tilts - 1
    boundaries = np.unique(boundaries)
    if len(boundaries) < 2:
        boundaries = np.asarray([0, n_tilts - 1], dtype=np.int64)
    return boundaries


def _adaptive_segment_ranges(
    n_tilts: int,
    excited_mask: Callable[[int], NDArray[np.bool_]],
    max_new_pct: float,
) -> list[tuple[int, int]]:
    """Adaptive chunk boundaries by recursive bisection (the private's ``union_adaptive`` path).

    Ported from ``BlochNet.forward``: a range ``(a, b)`` is split at its midpoint only while the
    midpoint's excited set adds more than ``max_new_pct`` *new* beams beyond the boundary union
    ``mask(a) | mask(b)`` (else the range freezes as one chunk). Returns the ordered, inclusive
    ``(a, b)`` segment ranges partitioning ``0 .. n_tilts - 1`` (disjoint, covering each tilt once);
    each chunk's beam set is later the boundary union of *its own* endpoints, as in the fixed path.
    """
    if n_tilts <= 1:
        return [(0, max(0, n_tilts - 1))]
    stack: list[tuple[int, int]] = [(0, n_tilts - 1)]
    final: list[tuple[int, int]] = []
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            final.append((a, b))
            continue
        mid = (a + b) // 2
        union_end = excited_mask(a) | excited_mask(b)
        if not bool(union_end.any()):  # defensive: (0,0,0) is always excited, so normally non-empty
            union_end = union_end | excited_mask(mid)
        new_mid = excited_mask(mid) & ~union_end
        new_pct = int(new_mid.sum()) / max(int(union_end.sum()), 1)
        if new_pct > max_new_pct:
            stack.append((mid + 1, b))
            stack.append((a, mid))
        else:
            final.append((a, b))
    final.sort(key=lambda ab: ab[0])
    return final


def build_coupling_segments(
    policy: SegmentedUnionCoupling,
    candidate_hkl: NDArray[np.int64],
    *,
    cell: NDArray[np.float64],
    orientation: NDArray[np.float64],
    tilts: NDArray[np.float64],
    energy: float,
    u0: float,
) -> tuple[Segment, ...]:
    """Partition the rocking curve into boundary-union coupled segments (pure geometry).

    v1 analog: the union-split block in ``BlochNet.forward``.

    ``candidate_hkl`` ``(G, 3)`` is the beam candidate pool to select from (the shared
    :class:`~diffBloch.engine.plan.StructureFactorGrid` ``structure_factor_hkl`` -- radius
    ``2 * g_max``, so the
    ``|g| < g_max`` mask filters it to each tilt's excited set). ``cell`` ``(3, 3)`` is the
    real-space basis, ``orientation`` ``(3, 3)`` the rotation's crystal orientation, ``tilts``
    ``(B, 3, 3)``
    the
    rocking-curve tilt matrices (each left-multiplying ``orientation``). ``energy`` (eV) and ``u0``
    (mean-inner-potential correction) set the Ewald geometry.

    Each boundary tilt's excited mask is ``|Sg| < sg_max`` and ``|g| < g_max`` with
    ``g = candidate_hkl @ orientation_basis(cell, tilt @ orientation)`` (identical to the private's
    ``hkl @ reciprocal_cell(unit_cell @ (R_tilt @ orientation).T)``). Segment ``i`` couples the
    union of the masks at boundary tilts ``i`` and ``i + 1`` and covers the half-open tilt range
    between them; the final segment includes the end, so the covers tile ``0 .. B - 1`` exactly.
    """
    candidate_hkl = np.asarray(candidate_hkl, dtype=np.int64)
    cell = np.asarray(cell, dtype=np.float64)
    orientation = np.asarray(orientation, dtype=np.float64)
    tilts = np.asarray(tilts, dtype=np.float64)
    if tilts.ndim != 3 or tilts.shape[1:] != (3, 3):
        raise ValueError(f"tilts must have shape (B, 3, 3), got {tilts.shape}")
    n_tilts = tilts.shape[0]

    # The per-tilt matrices are pure rotations (norm-preserving) and the orientation is constant
    # across the curve, so ``|g|`` is identical at every tilt -- the ``|g| < g_max`` cut selects the
    # SAME candidate subset at every tilt (verified: spread across tilts ~1e-15). Apply it once up
    # front to shrink the pool each per-tilt excitation mask scans: ``candidate_hkl`` is the full
    # structure-factor grid, but only the ``|g| < g_max`` core can ever couple. Scanning the whole
    # grid at every boundary tilt was the dominant per-trial cost; one pass replaces it.
    #
    # NOTE the ``|g|`` here is in the *orientation* metric -- ``orientation`` folds the experimental
    # cell correction (``u_matrix``: ``U = UB @ B^-1``, deliberately non-orthonormal), so this |g|
    # differs from the ideal-cell ``reciprocal_cell`` metric the grid is tabulated on by the
    # cell-correction magnitude (~1% on quartz). The grid's ``2 * g_max + _SUPPORT_MARGIN`` shell
    # (:meth:`StructureFactorGrid.from_cell_for_beam_cutoff`) covers that difference; it is NOT a
    # per-tilt variation and NOT an orthonormality bug -- coupling in the experimental-cell metric
    # is the physically correct cut.
    g_nominal = candidate_hkl @ orientation_basis(cell, orientation)  # constant across tilts
    pool = candidate_hkl[np.linalg.norm(g_nominal, axis=1) < policy.g_max]

    _mask_cache: dict[int, NDArray[np.bool_]] = {}

    def excited_mask(tilt_index: int) -> NDArray[np.bool_]:
        cached = _mask_cache.get(tilt_index)
        if cached is not None:
            return cached
        basis = orientation_basis(cell, tilts[tilt_index] @ orientation)
        sg = excitation_errors(pool @ basis, energy, u0=u0)
        mask = np.abs(sg) < policy.sg_max  # |g| < g_max already guaranteed by the pool
        _mask_cache[tilt_index] = mask
        return mask

    if policy.union_adaptive:
        # Adaptive boundaries by recursive bisection (fixed_n_segments ignored). Each segment's beam
        # set is the union of its own inclusive endpoints; the covers tile every tilt exactly once.
        ranges = _adaptive_segment_ranges(n_tilts, excited_mask, policy.union_max_new_beams_pct)
        return tuple(
            Segment(
                union_hkl=pool[excited_mask(a) | excited_mask(b)].copy(),
                covered_tilt_indices=tuple(range(a, b + 1)),
            )
            for a, b in ranges
        )

    boundaries = _fixed_segment_ranges(n_tilts, policy.fixed_n_segments)
    segments = []
    for i in range(len(boundaries) - 1):
        a, b = int(boundaries[i]), int(boundaries[i + 1])
        union = excited_mask(a) | excited_mask(b)
        cover = tuple(range(a, n_tilts if i == len(boundaries) - 2 else b))
        segments.append(Segment(union_hkl=pool[union].copy(), covered_tilt_indices=cover))
    return tuple(segments)
