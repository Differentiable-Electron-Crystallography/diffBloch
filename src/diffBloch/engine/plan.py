"""Refinement-invariant geometry plans: the shared scattering grid and per-orientation bundles.

These are the static (refinement-invariant) inputs the
:class:`~diffBloch.engine.engine.RefinementEngine` composes. The grid is owned once by
:class:`ScatteringGrid` and reused by both ``structure_factors`` and every ``BeamPlan``, so the two
sides cannot silently disagree on the ``Fgb`` support (the difference-support constraint is
validated when the beam plans are built).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from diffBloch.core.crystal import cell_volume as _cell_volume
from diffBloch.core.crystal import reciprocal_cell
from diffBloch.core.dynamical import BeamPlan, build_beam_plan
from diffBloch.core.products import AlignmentPlan, PatternBatch, build_alignment_plan
from diffBloch.core.reciprocal import make_hkl_grid, reciprocal_space_gpts

__all__ = [
    "OrientationPlan",
    "ScatteringGrid",
]


@dataclass(frozen=True)
class ScatteringGrid:
    """The shared ``Fgb`` support grid, owned once and reused by structure factors and beam plans.

    ``grid_hkl`` ``(G, 3)`` are the Miller indices ``Fgb`` is tabulated on (``|g| <= g_max``);
    ``reciprocal_basis`` ``(3, 3)`` and ``cell_volume`` fix the metric; ``gpts`` is the ravel box.
    ``g_max`` must span the beam difference support (~2x the beam ``g_max``) or beam-plan
    construction raises.
    """

    grid_hkl: Tensor
    reciprocal_basis: Tensor
    gpts: tuple[int, int, int]
    cell_volume: float
    g_max: float

    @classmethod
    def from_cell(cls, cell: NDArray[np.float64], g_max: float) -> ScatteringGrid:
        """Build the grid from a real-space cell ``(3, 3)`` and a structure-factor ``g_max``."""
        cell = np.asarray(cell, dtype=np.float64)
        return cls(
            grid_hkl=torch.tensor(make_hkl_grid(cell, g_max), dtype=torch.int64),
            reciprocal_basis=torch.tensor(reciprocal_cell(cell), dtype=torch.float64),
            gpts=reciprocal_space_gpts(cell, g_max),
            cell_volume=_cell_volume(cell),
            g_max=float(g_max),
        )


@dataclass(frozen=True)
class OrientationPlan:
    """The refinement-invariant plans for a single rotation/orientation.

    Bundles the geometry-only ``BeamPlan``, the observed ``PatternBatch``, the precomputed
    calculated<->observed ``AlignmentPlan``, and the beam Miller indices the solution is keyed on.
    """

    beam_hkl: Tensor
    beam_plan: BeamPlan
    pattern: PatternBatch
    alignment: AlignmentPlan

    @classmethod
    def build(
        cls,
        grid: ScatteringGrid,
        beam_hkl: NDArray[np.int64],
        pattern: PatternBatch,
        *,
        energy: float,
        u0: float = 0.0,
    ) -> OrientationPlan:
        """Assemble an orientation's plans against the shared grid (enforces grid coupling)."""
        beam_hkl = np.asarray(beam_hkl, dtype=np.int64)
        beam_plan = build_beam_plan(
            beam_hkl,
            np.asarray(grid.grid_hkl),
            np.asarray(grid.reciprocal_basis),
            energy=energy,
            gpts=grid.gpts,
            u0=u0,
        )
        beam_hkl_t = torch.tensor(beam_hkl, dtype=torch.int64)
        return cls(
            beam_hkl=beam_hkl_t,
            beam_plan=beam_plan,
            pattern=pattern,
            alignment=build_alignment_plan(beam_hkl_t, pattern.hkl),
        )
