"""Observables and typed product objects bridging propagation to losses.

``intensities`` is the pure observable ``|psi|^2``. On top of it sit three frozen, tensor-carrying
product objects:

- :class:`BlochSolution` -- the *calculated* side: amplitudes/intensities per thickness over a beam
  set (built from a :func:`core.solver.propagate` output).
- :class:`PatternBatch` -- the *observed* side: measured intensities/sigmas per reflection (built
  from an ``io.ExperimentalRecord``).
- :class:`AlignmentPlan` -- the precomputed hkl bridge between the two (mirrors ``BeamPlan``: built
  once from geometry, reused every step), consumed by :func:`align` to put calculated and observed
  on a common reflection axis ready for ``core.losses``.

hkl alignment here is exact (no symmetry merging); symmetry-equivalent merging is deferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from diffBloch.io.record import ExperimentalRecord

__all__ = [
    "AlignedIntensities",
    "AlignmentPlan",
    "BlochSolution",
    "MosaicAverage",
    "PLAIN_SUM",
    "PatternBatch",
    "PlainSum",
    "TiltReduction",
    "align",
    "build_alignment_plan",
    "intensities",
    "reduce_tilts",
]


def intensities(amplitudes: Tensor) -> Tensor:
    """Elastic diffracted intensity ``|psi|^2`` of complex exit-wave ``amplitudes``.

    Shape-preserving; returns a real tensor (the real dtype matching the complex input).
    Differentiable in ``amplitudes`` (hence back through ``A`` / ``Fgb``).
    """
    return amplitudes.abs().square()


@dataclass(frozen=True)
class PlainSum:
    """Incoherent sum over the rocking-curve tilts -- the default rotation-frame integration."""


@dataclass(frozen=True)
class MosaicAverage:
    """Normalized Gaussian mosaic-orientation weights for the expanded tilt geometry.

    ``weights`` aligns with the plan's tilt axis. Its sum equals the number of nominal rocking-curve
    samples, so mosaic averaging redistributes orientation weight without changing the integration
    scale. ``sigma_degrees`` records the PETS apparent mosaicity used to construct the geometry.
    """

    weights: tuple[float, ...]
    sigma_degrees: float

    def __post_init__(self) -> None:
        if not self.weights or not all(np.isfinite(value) and value >= 0.0 for value in self.weights):
            raise ValueError("mosaicity weights must be a non-empty sequence of finite values >= 0")
        if sum(self.weights) <= 0.0:
            raise ValueError("mosaicity weights must have a positive sum")
        if not np.isfinite(self.sigma_degrees) or self.sigma_degrees < 0.0:
            raise ValueError("mosaicity sigma_degrees must be finite and non-negative")


# The tilt-axis reduction of a rocking curve: a plain incoherent sum, or a mosaicity-broadened sum.
# Carried per-orientation on ``OrientationPlan`` (default ``PlainSum``) and applied by
# :meth:`BlochSolution.integrate`; a discriminated union rather than an optional ``window`` field.
TiltReduction = PlainSum | MosaicAverage

# Shared immutable default (``PlainSum`` is stateless), so signatures avoid a call-in-default.
PLAIN_SUM: TiltReduction = PlainSum()


def reduce_tilts(stacked: Tensor, reduction: TiltReduction) -> Tensor:
    """Reduce stacked per-tilt intensities ``(N_tilts, ...)`` over the leading tilt axis.

    The rocking-curve rotation-frame integration: :class:`PlainSum` sums the tilts;
    :class:`MosaicAverage` applies normalized PETS-derived orientation weights. Public because the
    tilt axis is reduced from two places -- a single shared beam set
    (:meth:`BlochSolution.integrate` / :meth:`BlochSolution.integrate_batched`) and the segmented
    coupling path, which reassembles each reflection's curve across per-chunk beam sets onto a
    shared
    union axis and reduces that ``(N_tilts, T, N_union)`` stack here.
    """
    match reduction:
        case PlainSum():
            return stacked.sum(dim=0)
        case MosaicAverage(weights=weights):
            if len(weights) != stacked.shape[0]:
                raise ValueError(
                    f"mosaicity has {len(weights)} weights for {stacked.shape[0]} tilts"
                )
            weight = stacked.new_tensor(weights)
            shape = (len(weights),) + (1,) * (stacked.ndim - 1)
            return (stacked * weight.reshape(shape)).sum(dim=0)


@dataclass(frozen=True)
class BlochSolution:
    """Calculated diffraction over a beam set: amplitudes and intensities per thickness.

    ``amplitudes`` / ``intensities`` are ``(T, N)`` (T thicknesses, N beams); ``beam_hkl`` is
    ``(N, 3)``; ``thicknesses`` is ``(T,)`` (Å). Build with :meth:`from_propagation` from a
    :func:`core.solver.propagate` output.
    """

    amplitudes: Tensor
    intensities: Tensor
    beam_hkl: Tensor
    thicknesses: Tensor

    @classmethod
    def from_propagation(cls, amplitudes: Tensor, beam_hkl: Tensor, thicknesses: Tensor) -> Self:
        """Wrap a ``(T, N)`` propagated wavefunction, deriving ``intensities = |amplitudes|^2``."""
        amplitudes = torch.as_tensor(amplitudes)
        beam_hkl = torch.as_tensor(beam_hkl, dtype=torch.int64)
        thicknesses = torch.as_tensor(thicknesses)
        if amplitudes.ndim != 2:
            raise ValueError(f"amplitudes must have shape (T, N), got {tuple(amplitudes.shape)}")
        n_thick, n_beams = amplitudes.shape
        if beam_hkl.shape != (n_beams, 3):
            raise ValueError(
                f"beam_hkl must have shape (N, 3) = ({n_beams}, 3) matching amplitudes"
            )
        if thicknesses.shape != (n_thick,):
            raise ValueError(f"thicknesses must have shape (T,) = ({n_thick},) matching amplitudes")
        return cls(amplitudes, intensities(amplitudes), beam_hkl, thicknesses)

    @classmethod
    def integrate(
        cls, solutions: Sequence[BlochSolution], *, reduction: TiltReduction = PLAIN_SUM
    ) -> Self:
        """Incoherently reduce tilt sub-solutions into one rocking-curve-integrated solution.

        Rocking-curve integration samples N slightly-tilted sub-orientations sharing one beam set
        and reduces their *intensities* ``|psi|^2`` over the tilt axis (an incoherent reduction, the
        physical rotation-frame integration -- not their amplitudes). ``reduction`` selects the
        tilt-axis reduction: :class:`PlainSum` sums the tilts; :class:`MosaicAverage` applies
        PETS-derived orientation weights. All sub-solutions must share the beam
        set (``beam_hkl``) and ``thicknesses``: the tilts reuse the one nominal beam set, varying
        only geometry. The integrated observable has no single exit-wave, so ``amplitudes`` is
        stored as the real effective amplitude ``sqrt(total intensity)`` (phase is physically lost
        in an incoherent reduction); only ``intensities`` feeds alignment/losses (``amplitudes`` has
        no downstream consumer). A single-element sequence returns an equivalent solution (the N=1
        identity is handled by the caller returning the sub-solution directly).
        """
        if not solutions:
            raise ValueError("integrate requires at least one solution")
        first = solutions[0]
        for other in solutions[1:]:
            if not torch.equal(other.beam_hkl, first.beam_hkl):
                raise ValueError("integrated solutions must share the same beam set")
        stacked = torch.stack([s.intensities for s in solutions])  # (N_tilts, T, N_beams)
        total = reduce_tilts(stacked, reduction)
        amplitudes = total.sqrt().to(first.amplitudes.dtype)
        return cls(amplitudes, total, first.beam_hkl, first.thicknesses)

    @classmethod
    def integrate_batched(
        cls,
        amplitudes: Tensor,
        beam_hkl: Tensor,
        thicknesses: Tensor,
        *,
        reduction: TiltReduction = PLAIN_SUM,
    ) -> Self:
        """Reduce a batched ``(N_tilts, T, N)`` propagation into one integrated solution.

        The batched-solver sibling of :meth:`integrate`: instead of stacking per-tilt sub-solutions,
        it takes the stacked exit-wave ``amplitudes`` a single batched
        :func:`core.solver.propagate` returns for all tilts at once (leading tilt axis), derives
        their ``intensities = |psi|^2``, and applies the same tilt-axis ``reduction``. Byte-for-byte
        equivalent to ``integrate`` on the corresponding per-tilt sub-solutions (identical stack,
        identical ``_reduce_tilts``); only the geometry of *how the tilts were solved* differs.
        ``amplitudes`` is ``(N_tilts, T, N)`` complex; ``beam_hkl`` ``(N, 3)``; ``thicknesses``
        ``(T,)`` (the shared beam set / thicknesses the tilts co-vary over).
        """
        amplitudes = torch.as_tensor(amplitudes)
        beam_hkl = torch.as_tensor(beam_hkl, dtype=torch.int64)
        thicknesses = torch.as_tensor(thicknesses)
        if amplitudes.ndim != 3:
            raise ValueError(
                f"amplitudes must have shape (N_tilts, T, N), got {tuple(amplitudes.shape)}"
            )
        _, n_thick, n_beams = amplitudes.shape
        if beam_hkl.shape != (n_beams, 3):
            raise ValueError(
                f"beam_hkl must have shape (N, 3) = ({n_beams}, 3) matching amplitudes"
            )
        if thicknesses.shape != (n_thick,):
            raise ValueError(f"thicknesses must have shape (T,) = ({n_thick},) matching amplitudes")
        total = reduce_tilts(intensities(amplitudes), reduction)  # (T, N)
        return cls(total.sqrt().to(amplitudes.dtype), total, beam_hkl, thicknesses)


@dataclass(frozen=True)
class PatternBatch:
    """Observed diffraction intensities: ``hkl`` ``(M, 3)``, ``intensities``/``sigmas`` ``(M,)``.

    Build with :meth:`from_experimental_record` from a validated ``io.ExperimentalRecord`` (optionally
    restricted to one PETS zone-axis row).
    """

    hkl: Tensor
    intensities: Tensor
    sigmas: Tensor
    rotation_index: int = 0

    @classmethod
    def from_experimental_record(
        cls,
        record: ExperimentalRecord,
        *,
        zone_axis_id: int | None = None,
        rotation_index: int = 0,
    ) -> Self:
        """Tensorise observed reflections, optionally filtering to one ``zone_axis_id``."""
        select = slice(None)
        if zone_axis_id is not None:
            select = np.asarray(record.reflection_zone_axis_ids) == zone_axis_id
            if not select.any():
                raise ValueError(f"no observed reflections for zone_axis_id {zone_axis_id}")
        select_hkl = np.asarray(record.hkl)[select].copy()
        select_i = np.asarray(record.intensities)[select].copy()
        select_s = np.asarray(record.sigmas)[select].copy()
        return cls(
            hkl=torch.as_tensor(select_hkl, dtype=torch.int64),
            intensities=torch.as_tensor(select_i, dtype=torch.float64),
            sigmas=torch.as_tensor(select_s, dtype=torch.float64),
            rotation_index=rotation_index,
        )


@dataclass(frozen=True)
class AlignmentPlan:
    """Precomputed hkl bridge between a :class:`BlochSolution` and a :class:`PatternBatch`.

    ``hkl`` ``(K, 3)`` lists the shared reflections (those observed *and* calculated, in observed
    order); ``solution_index`` / ``pattern_index`` ``(K,)`` gather the matching rows from
    ``BlochSolution.beam_hkl`` and ``PatternBatch.hkl`` respectively. Geometry-only and reusable.
    """

    hkl: Tensor
    solution_index: Tensor
    pattern_index: Tensor


def build_alignment_plan(
    solution_hkl: Tensor, pattern_hkl: Tensor, *, restrict_to: Tensor | None = None
) -> AlignmentPlan:
    """Match observed reflections to calculated beams by exact hkl (observed-order intersection).

    ``restrict_to`` ``(S, 3)`` optionally pins the **scored** reflection set: only pattern rows
    whose hkl is in ``restrict_to`` are eligible, so the result is
    ``pattern ∩ solution ∩ restrict_to``. This
    is how ``couple_beams`` keeps scoring on the ``select_beams`` selection while the *solve* set
    expands to the coupling union -- ``solution_hkl`` (the union) grows, but the scored axis stays
    the pre-couple set. It is an intersection, so a ``restrict_to`` reflection absent from
    ``solution_hkl`` is dropped: you can only score a reflection you solved, which is the
    ``scored ⊆ coupled`` invariant.
    ``None`` (the default) scores the whole ``pattern ∩ solution``, keeping the tilt-independent
    path unchanged. ``pattern_index`` indexes the full ``pattern_hkl`` regardless, so ``align`` is
    untouched.
    """
    solution = np.asarray(torch.as_tensor(solution_hkl, dtype=torch.int64))
    pattern = np.asarray(torch.as_tensor(pattern_hkl, dtype=torch.int64))
    if solution.ndim != 2 or solution.shape[1] != 3:
        raise ValueError(f"solution_hkl must have shape (N, 3), got {solution.shape}")
    if pattern.ndim != 2 or pattern.shape[1] != 3:
        raise ValueError(f"pattern_hkl must have shape (M, 3), got {pattern.shape}")

    scored: NDArray[np.int64] | None = None
    if restrict_to is not None:
        scored = np.asarray(torch.as_tensor(restrict_to, dtype=torch.int64))
        if scored.ndim != 2 or scored.shape[1] != 3:
            raise ValueError(f"restrict_to must have shape (S, 3), got {scored.shape}")

    arrays = [solution, pattern] if scored is None else [solution, pattern, scored]
    _, inverse = np.unique(np.concatenate(arrays, axis=0), axis=0, return_inverse=True)
    n_solution = len(solution)
    n_pattern = len(pattern)
    solution_codes = inverse[:n_solution]
    pattern_codes = inverse[n_solution : n_solution + n_pattern]
    lookup = np.full(int(inverse.max(initial=-1)) + 1, -1, dtype=np.int64)
    lookup[solution_codes] = np.arange(n_solution, dtype=np.int64)
    keep = lookup[pattern_codes] >= 0
    if scored is not None:
        scored_codes = inverse[n_solution + n_pattern :]
        allowed = np.zeros_like(lookup, dtype=np.bool_)
        allowed[scored_codes] = True
        keep &= allowed[pattern_codes]
    pat_idx = np.flatnonzero(keep)
    sol_idx = lookup[pattern_codes[pat_idx]]
    solution_index = torch.as_tensor(sol_idx, dtype=torch.int64)
    pattern_index = torch.as_tensor(pat_idx, dtype=torch.int64)
    return AlignmentPlan(
        hkl=torch.as_tensor(pattern[pat_idx], dtype=torch.int64),
        solution_index=solution_index,
        pattern_index=pattern_index,
    )


@dataclass(frozen=True)
class AlignedIntensities:
    """Calculated/observed intensities and sigmas on a common ``(T, K)`` reflection axis.

    ``observed``/``sigmas`` are broadcast across the ``T`` calculated thicknesses, so the trio drops
    straight into ``core.losses`` (e.g. ``rbragg(calculated, observed, sigmas)``).
    """

    calculated: Tensor
    observed: Tensor
    sigmas: Tensor


def align(
    solution: BlochSolution, pattern: PatternBatch, plan: AlignmentPlan
) -> AlignedIntensities:
    """Gather calculated and observed intensities onto the plan's shared reflection axis.

    Device-safe: the geometry-only plan indices may live on CPU, so (mirroring the gather/diagonal
    use sites in ``core.dynamical``) they are moved to each tensor's device, and observed/sigmas
    land on ``calculated.device`` so the trio is co-located for ``core.losses``.
    """
    calculated = solution.intensities[:, plan.solution_index.to(solution.intensities.device)]
    n_thick = calculated.shape[0]
    out_device = calculated.device
    pattern_index = plan.pattern_index.to(pattern.intensities.device)
    observed = pattern.intensities[pattern_index].to(out_device).expand(n_thick, -1)
    sigmas = pattern.sigmas[pattern_index].to(out_device).expand(n_thick, -1)
    return AlignedIntensities(calculated=calculated, observed=observed, sigmas=sigmas)
