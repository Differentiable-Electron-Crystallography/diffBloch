"""Tilt-segment-union beam coupling: the per-tilt-chunk beam sets of the private rocking curve.

Rocking-curve integration solves the crystal at many slightly-tilted sub-orientations. A single
beam set for the whole curve either over-couples (every beam any tilt needs, slow) or drops beams a
tilt needs (a reflection drifts through the Ewald sphere as the crystal rocks). The
``diffBloch_private`` policy instead partitions the tilts into contiguous chunks and couples, within
each chunk, the **union** of the excited-beam sets at the chunk's two boundary tilts.

This module is the pure geometry of that partition: given a
:class:`~diffBloch.specs.TiltSegmentUnion`
policy and the rotation's tilt geometry, it returns the ordered :class:`Segment` list (each a beam
set + the disjoint tilt indices it covers). It computes the same excitation mask
(``|Sg| < sg_max`` and ``|g| < g_max``) the private's ``BlochNet.forward`` builds its
union sets from, reusing :func:`~diffBloch.core.dynamical.excitation_errors` and
:func:`~diffBloch.core.crystal.orientation_basis`. It does not build or solve anything -- an engine
step turns the segments into per-chunk plans and reassembles their curves before reduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from diffBloch.core.crystal import orientation_basis
from diffBloch.core.dynamical import excitation_errors
from diffBloch.specs import TiltSegmentUnion

__all__ = [
    "Segment",
    "tilt_segment_coupling",
]


@dataclass(frozen=True)
class Segment:
    """One tilt chunk's coupling: the beam set it solves and the tilt indices it covers.

    ``beam_hkl`` ``(n_beams, 3)`` is the union of the excited beams at the chunk's boundary tilts
    (includes ``(0, 0, 0)``, always excited). ``cover`` are the global rocking-curve tilt indices
    this segment is responsible for -- contiguous and, across a rotation's segments, disjoint and
    covering every tilt exactly once. The segment solves its beam set at each covered tilt; the
    per-tilt intensities are later scattered back onto each reflection's full rocking curve.
    """

    beam_hkl: NDArray[np.int64]
    cover: tuple[int, ...]


def _split_boundaries(n_tilts: int, n_splits: int) -> NDArray[np.int64]:
    """The private's contiguous split boundaries into ``n_splits`` chunks over ``n_tilts`` tilts.

    Ported verbatim from ``BlochNet.forward`` (``union_adaptive = False`` path): evenly sized chunks
    with the remainder front-loaded, boundaries clamped so the last index is ``n_tilts - 1``, then
    de-duplicated. Returns the strictly increasing boundary indices (length ``n_splits + 1`` before
    de-duplication), so ``n_splits`` segments span ``boundaries[i] .. boundaries[i + 1]``.
    """
    n_splits = max(1, n_splits)
    if n_splits >= n_tilts:
        boundaries = np.arange(n_tilts, dtype=np.int64)
    else:
        base, extra = divmod(n_tilts, n_splits)
        sizes = [base + 1] * extra + [base] * (n_splits - extra)
        acc = [0]
        for size in sizes:
            acc.append(acc[-1] + size)
        boundaries = np.asarray(acc, dtype=np.int64)
        boundaries[-1] = n_tilts - 1
    boundaries = np.unique(boundaries)
    if len(boundaries) < 2:
        boundaries = np.asarray([0, n_tilts - 1], dtype=np.int64)
    return boundaries


def tilt_segment_coupling(
    policy: TiltSegmentUnion,
    candidate_hkl: NDArray[np.int64],
    *,
    cell: NDArray[np.float64],
    orientation: NDArray[np.float64],
    tilts: NDArray[np.float64],
    energy: float,
    u0: float,
) -> tuple[Segment, ...]:
    """Partition the rocking curve into boundary-union coupled segments (pure geometry).

    ``candidate_hkl`` ``(G, 3)`` is the beam candidate pool to select from (the shared
    :class:`~diffBloch.engine.plan.ScatteringGrid` ``grid_hkl`` -- radius ``2 * g_max``, so the
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
    boundaries = _split_boundaries(n_tilts, policy.n_splits)

    # ``|g|`` is invariant under the orientation/tilt rotation (an orthogonal transform preserves
    # the vector norm), so the coupling cut ``|g| < g_max`` selects the SAME candidate subset at
    # every tilt. Apply it once up front to shrink the pool each per-tilt excitation mask scans:
    # ``candidate_hkl`` is the full structure-factor grid (radius ``2 * g_max``, sized to span the
    # g - h differences), but only the ``|g| < g_max`` core (~(1/2)^3 of the sphere) can ever
    # couple. Scanning the whole grid at every boundary tilt was the dominant per-trial cost of the
    # coupled fit; being orientation-invariant, one pass replaces the redundant full-grid scans.
    g_nominal = candidate_hkl @ orientation_basis(cell, orientation)  # any orientation: |g| invar.
    pool = candidate_hkl[np.linalg.norm(g_nominal, axis=1) < policy.g_max]

    def excited_mask(tilt_index: int) -> NDArray[np.bool_]:
        basis = orientation_basis(cell, tilts[tilt_index] @ orientation)
        sg = excitation_errors(pool @ basis, energy, u0=u0)
        return np.abs(sg) < policy.sg_max  # |g| < g_max already guaranteed by the pool

    masks = {int(b): excited_mask(int(b)) for b in boundaries}
    segments = []
    for i in range(len(boundaries) - 1):
        a, b = int(boundaries[i]), int(boundaries[i + 1])
        union = masks[a] | masks[b]
        cover = tuple(range(a, n_tilts if i == len(boundaries) - 2 else b))
        segments.append(Segment(beam_hkl=pool[union].copy(), cover=cover))
    return tuple(segments)
