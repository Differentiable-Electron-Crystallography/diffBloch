"""Refinement-invariant geometry plans: the shared scattering grid and per-orientation bundles.

These are the static (refinement-invariant) inputs the
:class:`~diffBloch.engine.forward.RefinementEngine` composes. The grid is owned once by
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
from diffBloch.core.crystal import orientation_basis, reciprocal_cell
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
    ``cell`` ``(3, 3)`` is the real-space basis and ``reciprocal_basis`` ``(3, 3)`` /
    ``cell_volume`` the metric it derives (kept together; only :meth:`from_cell` constructs them, so
    they cannot desync). ``gpts`` is the ravel box. ``g_max`` must span the beam difference support
    (~2x the beam ``g_max``) or beam-plan construction raises.
    """

    grid_hkl: Tensor
    cell: Tensor
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
            cell=torch.tensor(cell, dtype=torch.float64),
            reciprocal_basis=torch.tensor(reciprocal_cell(cell), dtype=torch.float64),
            gpts=reciprocal_space_gpts(cell, g_max),
            cell_volume=_cell_volume(cell),
            g_max=float(g_max),
        )


@dataclass(frozen=True)
class OrientationPlan:
    """The refinement-invariant plans for a single rotation/orientation.

    Self-describing: it carries both its **source / rebuild inputs** (``orientation``, ``energy``,
    ``u0`` -- what ``preprocess`` steps like ``select_beams`` / ``fit_orientation`` consume to
    recompile) and the **compiled geometry** (``beam_plan``, ``alignment`` -- what
    ``engine.simulate`` consumes). Source and compiled are only ever set together by :meth:`build`,
    so they cannot desync (see ``design/decisions/plan-shape-and-step-ordering.md``).
    ``orientation`` is the source of truth; the lab-frame basis is derived from it, never stored.
    """

    orientation: Tensor
    energy: float
    u0: float
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
        orientation: Tensor | NDArray[np.float64] | None = None,
    ) -> OrientationPlan:
        """Assemble an orientation's plans against the shared grid (enforces grid coupling).

        ``orientation`` ``(3, 3)`` is the crystal orientation matrix for this rotation; the
        lab-frame reciprocal cell is derived from it and the grid's real-space ``cell`` via
        ``orientation_basis(grid.cell, orientation) = reciprocal_cell(cell @ orientation.T)`` and
        drives ``g`` -> ``Sg`` / ``Mii`` only. When ``None`` the orientation is the identity and the
        shared ``grid.reciprocal_basis`` is used directly (the untilted / single-orientation case),
        making that path byte-identical to the unoriented build. The rotation convention (and the
        measured-cell correction folded into ``orientation``) is derived upstream in ``preprocess``;
        the ``Fgb`` gather is keyed on ``grid.grid_hkl`` and is unaffected.

        ``orientation`` accepts either a NumPy array or a ``Tensor`` (e.g. a prior plan's stored
        ``orientation``), so a later ``Plan -> Plan`` rebuild can pass ``old_plan.orientation``
        directly without ad-hoc conversion.
        """
        beam_hkl = np.asarray(beam_hkl, dtype=np.int64)
        if orientation is None:
            rotation = np.eye(3, dtype=np.float64)
            basis = np.asarray(grid.reciprocal_basis)
        else:
            rotation = np.asarray(orientation, dtype=np.float64)
            basis = orientation_basis(np.asarray(grid.cell), rotation)
        beam_plan = build_beam_plan(
            beam_hkl,
            np.asarray(grid.grid_hkl),
            basis,
            energy=energy,
            gpts=grid.gpts,
            u0=u0,
        )
        beam_hkl_t = torch.tensor(beam_hkl, dtype=torch.int64)
        return cls(
            orientation=torch.tensor(rotation, dtype=torch.float64),
            energy=float(energy),
            u0=float(u0),
            beam_hkl=beam_hkl_t,
            beam_plan=beam_plan,
            pattern=pattern,
            alignment=build_alignment_plan(beam_hkl_t, pattern.hkl),
        )
