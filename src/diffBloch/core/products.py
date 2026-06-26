"""Observables and typed product objects bridging propagation to losses.

``intensities`` is the pure observable ``|psi|^2`` (private ``dynamical.py`` ``torch.abs(psi)**2``,
lines 737/802). On top of it sit three frozen, tensor-carrying product objects that retire the
private ``DiffractionDataset``:

- :class:`BlochSolution` -- the *calculated* side: amplitudes/intensities per thickness over a beam
  set (built from a :func:`core.solver.propagate` output).
- :class:`PatternBatch` -- the *observed* side: measured intensities/sigmas per reflection (built
  from an ``io.ObservationRecord``).
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
from torch import Tensor

if TYPE_CHECKING:
    from diffBloch.io.record import ObservationRecord

__all__ = [
    "AlignedIntensities",
    "AlignmentPlan",
    "BlochSolution",
    "PatternBatch",
    "align",
    "build_alignment_plan",
    "intensities",
]


def intensities(amplitudes: Tensor) -> Tensor:
    """Elastic diffracted intensity ``|psi|^2`` of complex exit-wave ``amplitudes``.

    Shape-preserving; returns a real tensor (the real dtype matching the complex input).
    Differentiable in ``amplitudes`` (hence back through ``A`` / ``Fgb``).
    """
    return amplitudes.abs().square()


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


@dataclass(frozen=True)
class PatternBatch:
    """Observed diffraction intensities: ``hkl`` ``(M, 3)``, ``intensities``/``sigmas`` ``(M,)``.

    Build with :meth:`from_observation_record` from a validated ``io.ObservationRecord`` (optionally
    restricted to one PETS zone-axis row).
    """

    hkl: Tensor
    intensities: Tensor
    sigmas: Tensor

    @classmethod
    def from_observation_record(
        cls, record: ObservationRecord, *, zone_axis_id: int | None = None
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


def build_alignment_plan(solution_hkl: Tensor, pattern_hkl: Tensor) -> AlignmentPlan:
    """Match observed reflections to calculated beams by exact hkl (observed-order intersection)."""
    solution = np.asarray(torch.as_tensor(solution_hkl, dtype=torch.int64))
    pattern = np.asarray(torch.as_tensor(pattern_hkl, dtype=torch.int64))
    if solution.ndim != 2 or solution.shape[1] != 3:
        raise ValueError(f"solution_hkl must have shape (N, 3), got {solution.shape}")
    if pattern.ndim != 2 or pattern.shape[1] != 3:
        raise ValueError(f"pattern_hkl must have shape (M, 3), got {pattern.shape}")

    by_hkl = {tuple(int(c) for c in row): i for i, row in enumerate(solution)}
    sol_idx, pat_idx = [], []
    for pattern_pos, row in enumerate(pattern):
        solution_pos = by_hkl.get(tuple(int(c) for c in row))
        if solution_pos is not None:
            sol_idx.append(solution_pos)
            pat_idx.append(pattern_pos)
    solution_index = torch.tensor(sol_idx, dtype=torch.int64)
    pattern_index = torch.tensor(pat_idx, dtype=torch.int64)
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
    """Gather calculated and observed intensities onto the plan's shared reflection axis."""
    calculated = solution.intensities[:, plan.solution_index]  # (T, K)
    n_thick = calculated.shape[0]
    observed = pattern.intensities[plan.pattern_index].expand(n_thick, -1)
    sigmas = pattern.sigmas[plan.pattern_index].expand(n_thick, -1)
    return AlignedIntensities(calculated=calculated, observed=observed, sigmas=sigmas)
