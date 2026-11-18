"""Tilt-segment-union beam coupling: per-tilt-chunk beam sets across a rocking curve.

Rocking-curve integration solves the crystal at many slightly-tilted sub-orientations. A single
beam set for the whole curve either over-couples (every beam any tilt needs, slow) or drops beams a
tilt needs (a reflection drifts through the Ewald sphere as the crystal rocks). This coupling policy
instead partitions the tilts into contiguous chunks and couples, within each chunk, the **union** of
the excited-beam sets at the chunk's two boundary tilts.

This module is the pure geometry of that partition: given a
:class:`~diffBloch.specs.UnionCoupling`
policy and the rotation's tilt geometry, it returns the ordered :class:`Segment` list (each a beam
set + the disjoint tilt indices it covers). It computes the excitation mask
(``|Sg| < sg_max`` and ``|g| < g_max``) from which the union sets are built,
reusing :func:`~diffBloch.core.dynamical.excitation_errors` and
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
from diffBloch.specs import PerTiltCoupling, UnionCoupling

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
    """Contiguous split boundaries into ``fixed_n_segments`` chunks over ``n_tilts``.

    The ``union_adaptive = False`` path: evenly sized chunks
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
    """Adaptive chunk boundaries by recursive bisection (the ``union_adaptive`` path).

    A range ``(a, b)`` is split at its midpoint only while the
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
    policy: UnionCoupling | PerTiltCoupling,
    candidate_beam_hkl: NDArray[np.int64],
    *,
    cell: NDArray[np.float64],
    orientation: NDArray[np.float64],
    tilts: NDArray[np.float64],
    energy: float,
    u0: float,
    split_cache: dict[int, NDArray[np.int64]] | None = None,
    sweeps: list[tuple[int, int]] | None = None,
) -> tuple[Segment, ...]:
    """Partition the rocking curve into boundary-union coupled segments (pure geometry).

    ``candidate_beam_hkl`` ``(G, 3)`` is the beam candidate pool to select from (the shared
    :class:`~diffBloch.engine.plan.StructureFactorGrid` ``structure_factor_hkl`` -- radius
    ``2 * g_max``, so the
    ``|g| < g_max`` mask filters it to each tilt's excited set). ``cell`` ``(3, 3)`` is the
    real-space basis, ``orientation`` ``(3, 3)`` the rotation's crystal orientation, ``tilts``
    ``(B, 3, 3)``
    the
    rocking-curve tilt matrices (each left-multiplying ``orientation``). ``energy`` (eV) and ``u0``
    (mean-inner-potential correction) set the Ewald geometry.

    Each boundary tilt's excited mask is ``|Sg| < sg_max`` and ``|g| < g_max`` with
    ``g = candidate_beam_hkl @ orientation_basis(cell, tilt @ orientation)``. Segment ``i`` couples the
    union of the masks at boundary tilts ``i`` and ``i + 1`` and covers the half-open tilt range
    between them; the final segment includes the end, so the covers tile ``0 .. B - 1`` exactly.

    That endpoint union only bounds the tilts between two boundaries when those tilts sweep
    monotonically. A tilt list composed of several sweeps (isotropic mosaicity puts one rocking
    sweep per mosaic sample, :func:`~diffBloch.preprocess.orientation.compose_mosaic_tilts`) is
    therefore split into its contiguous sweeps first (:func:`contiguous_sweeps`) and each sweep is
    segmented on its own, so no segment spans the jump between two sweeps.

    ``split_cache`` (``union_cost_select`` only) maps a sweep's first tilt to its chosen split
    boundaries. When given, a cached split is reused instead of searched again, so a search over
    many nearby orientations (the orientation fit) picks the split once, at its first trial.
    ``sweeps`` is the precomputed :func:`contiguous_sweeps` of ``tilts``, for a caller that couples
    the same tilts many times (the orientation fit); ``None`` derives it here.
    """
    candidate_beam_hkl = np.asarray(candidate_beam_hkl, dtype=np.int64)
    cell = np.asarray(cell, dtype=np.float64)
    orientation = np.asarray(orientation, dtype=np.float64)
    tilts = np.asarray(tilts, dtype=np.float64)
    if tilts.ndim != 3 or tilts.shape[1:] != (3, 3):
        raise ValueError(f"tilts must have shape (B, 3, 3), got {tilts.shape}")
    n_tilts = tilts.shape[0]

    # The per-tilt matrices are pure rotations (norm-preserving) and the orientation is constant
    # across the curve, so ``|g|`` is identical at every tilt -- the ``|g| < g_max`` cut selects the
    # SAME candidate subset at every tilt (verified: spread across tilts ~1e-15). Apply it once up
    # front to shrink the pool each per-tilt excitation mask scans: ``candidate_beam_hkl`` is the full
    # structure-factor grid, but only the ``|g| < g_max`` core can ever couple. Scanning the whole
    # grid at every boundary tilt was the dominant per-trial cost; one pass replaces it.
    #
    # NOTE the ``|g|`` here is in the *orientation* metric -- ``orientation`` is not guaranteed
    # exactly orthonormal (``u_matrix``: ``U = UB @ B^-1`` carries PETS's own small
    # UB-vs-cell-parameters fit residual; see ``preprocess.orientation``), so this |g| can differ
    # slightly from the authoritative-cell ``reciprocal_cell`` metric the grid is tabulated on. The
    # grid's ``2 * g_max + _SUPPORT_MARGIN`` shell (:meth:`StructureFactorGrid.from_cell_for_beam_cutoff`)
    # covers that residual; it is NOT a per-tilt variation and NOT an orthonormality bug -- coupling
    # in the orientation metric is the physically correct cut.
    g_nominal = candidate_beam_hkl @ orientation_basis(cell, orientation)  # constant across tilts
    radial_mask = np.linalg.norm(g_nominal, axis=1) < policy.g_max
    pool = candidate_beam_hkl[radial_mask]
    pool_g_nominal = g_nominal[radial_mask]

    _mask_cache: dict[int, NDArray[np.bool_]] = {}

    def excited_mask(tilt_index: int) -> NDArray[np.bool_]:
        cached = _mask_cache.get(tilt_index)
        if cached is not None:
            return cached
        # ``tilt`` is orthogonal, so
        # reciprocal(cell @ (tilt @ orientation).T) == reciprocal(cell @ orientation.T) @ tilt.T.
        # Rotate the already-materialised nominal g-vectors instead of repeating a pseudoinverse
        # and the full HKL @ basis product for every queried sub-tilt.
        sg = excitation_errors(pool_g_nominal @ tilts[tilt_index].T, energy, u0=u0)
        mask = np.abs(sg) < policy.sg_max  # |g| < g_max already guaranteed by the pool
        _mask_cache[tilt_index] = mask
        return mask

    if isinstance(policy, PerTiltCoupling):
        return tuple(
            Segment(
                union_hkl=pool[excited_mask(i)].copy(),
                covered_tilt_indices=(i,),
            )
            for i in range(n_tilts)
        )

    segments: list[Segment] = []
    for start, stop in contiguous_sweeps(tilts) if sweeps is None else sweeps:
        n_sweep = stop - start

        def sweep_mask(i: int, start: int = start) -> NDArray[np.bool_]:
            return excited_mask(start + i)

        if policy.union_adaptive:
            # Adaptive boundaries by recursive bisection (fixed_n_segments ignored). Each segment's
            # beam set is the union of its own inclusive endpoints; the covers tile every tilt once.
            if policy.union_cost_select and split_cache is not None and start in split_cache:
                segments.extend(
                    _fixed_segments(split_cache[start], start, n_sweep, sweep_mask, pool)
                )
                continue
            ranges = _adaptive_segment_ranges(n_sweep, sweep_mask, policy.union_max_new_beams_pct)
            if policy.union_cost_select:
                boundaries = _cost_segment_boundaries(
                    n_sweep, sweep_mask, len(ranges), policy.union_segment_overhead_beams
                )
                if split_cache is not None:
                    split_cache[start] = boundaries
                segments.extend(_fixed_segments(boundaries, start, n_sweep, sweep_mask, pool))
                continue
            segments.extend(
                Segment(
                    union_hkl=pool[sweep_mask(a) | sweep_mask(b)].copy(),
                    covered_tilt_indices=tuple(range(start + a, start + b + 1)),
                )
                for a, b in ranges
            )
            continue

        boundaries = _fixed_segment_ranges(n_sweep, policy.fixed_n_segments)
        segments.extend(_fixed_segments(boundaries, start, n_sweep, sweep_mask, pool))
    return tuple(segments)


def _fixed_segments(
    boundaries: NDArray[np.int64],
    start: int,
    n_sweep: int,
    sweep_mask: Callable[[int], NDArray[np.bool_]],
    pool: NDArray[np.int64],
) -> list[Segment]:
    """Endpoint-union segments for split ``boundaries`` of one sweep starting at tilt ``start``."""
    out: list[Segment] = []
    for i in range(len(boundaries) - 1):
        a, b = int(boundaries[i]), int(boundaries[i + 1])
        union = sweep_mask(a) | sweep_mask(b)
        cover = tuple(range(start + a, start + (n_sweep if i == len(boundaries) - 2 else b)))
        out.append(Segment(union_hkl=pool[union].copy(), covered_tilt_indices=cover))
    return out


def _union_counts(
    n_sweep: int, sweep_mask: Callable[[int], NDArray[np.bool_]]
) -> NDArray[np.float64]:
    """``(n_sweep, n_sweep)`` beam counts of ``mask(a) | mask(b)`` for every tilt pair.

    ``|A | B| = |A| + |B| - |A & B|``, and one product of the stacked masks gives every pairwise
    intersection count, so scoring a split is a table lookup instead of a fresh union and count per
    boundary pair (that recount dominated the orientation fit). The counts are integers far below
    ``2**53``, so they are exact in float64.
    """
    masks = np.stack([sweep_mask(i) for i in range(n_sweep)]).astype(np.float64)
    counts = masks.sum(axis=1)
    union: NDArray[np.float64] = counts[:, None] + counts[None, :] - masks @ masks.T
    return union


def _split_cost(
    boundaries: NDArray[np.int64],
    n_sweep: int,
    union_counts: NDArray[np.float64],
    overhead_beams: float,
) -> float:
    """Beam-count cost ``sum(K^3 + n_tilts * n_beams^3)`` of the even split at ``boundaries``.

    ``union_counts`` is the :func:`_union_counts` table of the sweep.
    """
    cost = 0.0
    last = len(boundaries) - 2
    for i in range(last + 1):
        a, b = int(boundaries[i]), int(boundaries[i + 1])
        n_beams = float(union_counts[a, b])
        n_covered = (n_sweep if i == last else b) - a
        cost += overhead_beams**3 + n_covered * n_beams**3
    return cost


def _cost_segment_boundaries(
    n_sweep: int,
    sweep_mask: Callable[[int], NDArray[np.bool_]],
    n_floor: int,
    overhead_beams: float,
) -> NDArray[np.int64]:
    """The even split of lowest beam cost with at least ``n_floor`` segments (``union_cost_select``).

    ``n_floor`` is the bisection rule's segment count: fewer even segments under-couple, since a
    fixed split's segments use endpoint unions only. Every count from ``n_floor`` up to one segment
    per tilt is scored with :func:`_split_cost`; ties keep the fewer segments.
    """
    union_counts = _union_counts(n_sweep, sweep_mask)
    best: NDArray[np.int64] | None = None
    best_cost = float("inf")
    seen: set[bytes] = set()
    for n in range(max(1, n_floor), max(n_floor, n_sweep) + 1):
        boundaries = _fixed_segment_ranges(n_sweep, n)
        key = boundaries.tobytes()
        if key in seen:
            continue
        seen.add(key)
        cost = _split_cost(boundaries, n_sweep, union_counts, overhead_beams)
        if cost < best_cost:
            best, best_cost = boundaries, cost
    assert best is not None
    return best


def contiguous_sweeps(tilts: NDArray[np.float64]) -> list[tuple[int, int]]:
    """Split ``tilts`` ``(B, 3, 3)`` into half-open ``[start, stop)`` runs of one even sweep.

    A rocking sweep steps by the same rotation angle between every pair of consecutive tilts, so
    the run continues while the step angle matches the run's first step; a different step (the
    jump from the end of one mosaic sample's sweep to the start of the next) begins a new run. A
    single sweep -- every tilt list without mosaicity -- is one run covering all ``B`` tilts.
    """
    n_tilts = tilts.shape[0]
    if n_tilts < 3:
        return [(0, n_tilts)]
    relative = np.einsum("bij,bkj->bik", tilts[1:], tilts[:-1])  # tilts[k+1] @ tilts[k].T
    cos_angle = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    steps = np.arccos(cos_angle)
    runs: list[tuple[int, int]] = []
    start = 0
    k = 0
    while k < n_tilts - 1:
        # steps[k] is the step from tilt k to k + 1; tilt `start` opens the run.
        if k > start and not np.isclose(steps[k], steps[start], rtol=1e-6, atol=1e-12):
            runs.append((start, k + 1))
            start = k + 1
        k += 1
    runs.append((start, n_tilts))
    return runs
