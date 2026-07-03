"""Refinement-invariant geometry plans: the shared scattering grid and per-orientation bundles.

These are the static (refinement-invariant) inputs the
:class:`~diffBloch.engine.forward.RefinementEngine` composes. The grid is owned once by
:class:`ScatteringGrid` and reused by both ``structure_factors`` and every ``BeamPlan``, so the two
sides cannot silently disagree on the ``Fgb`` support (the difference-support constraint is
validated when the beam plans are built).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from diffBloch.core.crystal import cell_volume as _cell_volume
from diffBloch.core.crystal import orientation_basis, reciprocal_cell
from diffBloch.core.dynamical import (
    BeamPlan,
    StructureFactorGather,
    build_beam_plan,
    build_structure_factor_gather,
)
from diffBloch.core.products import (
    PLAIN_SUM,
    AlignmentPlan,
    PatternBatch,
    TiltReduction,
    build_alignment_plan,
)
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
    ``u0``, ``thickness`` -- what ``preprocess`` steps like ``select_beams`` / ``fit_orientation`` /
    ``fit_thickness`` consume to recompile) and the **compiled geometry** (``beam_plan``,
    ``alignment`` -- what ``engine.simulate`` consumes). Source and compiled are only ever set
    together by :meth:`build`, so they cannot desync (see
    ``design/decisions/plan-shape-and-step-ordering.md``). ``orientation`` is the source of truth;
    the lab-frame basis is derived from it, never stored. ``tilts`` ``(N, 3, 3)`` is the
    rocking-curve integration tilt set (source): N goniometer sub-orientations, each compiled into
    the matching entry of ``beam_plans`` (``N = len(beam_plans)``). The default is a single identity
    tilt ``(1, 3, 3)`` -- one static solve, byte-identical to the pre-integration plan; a longer set
    is baked by ``integrate_rocking_curve`` and summed as ``|psi|^2`` over the tilts by the engine.
    ``thickness`` ``(T,)`` is the specimen's
    thickness for this rotation (its beam path length at this tilt), held fixed during refinement.
    It is seeded from the sample thickness and later replaced by the best-fitting value
    ``fit_thickness`` finds. The forward model uses it for this orientation unless the caller is
    refining thickness directly (see
    :meth:`~diffBloch.engine.forward.RefinementEngine._thickness_for`).
    """

    orientation: Tensor
    tilts: Tensor
    energy: float
    u0: float
    thickness: Tensor
    beam_hkl: Tensor
    beam_plans: tuple[BeamPlan, ...]
    pattern: PatternBatch
    alignment: AlignmentPlan
    tilt_reduction: TiltReduction = PLAIN_SUM

    @classmethod
    def build(
        cls,
        grid: ScatteringGrid,
        beam_hkl: NDArray[np.int64],
        pattern: PatternBatch,
        *,
        energy: float,
        thickness: Tensor | NDArray[np.float64] | Sequence[float],
        u0: float = 0.0,
        orientation: Tensor | NDArray[np.float64] | None = None,
        tilts: NDArray[np.float64] | None = None,
        tilt_reduction: TiltReduction = PLAIN_SUM,
        gather: StructureFactorGather | None = None,
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

        ``thickness`` ``(T,)`` is required: this rotation's frozen per-rotation conditioning,
        coerced to a 1-D float64 tensor. A rebuild threads ``old_plan.thickness`` through unchanged;
        ``fit_thickness`` bakes the single gridsearch winner ``(1,)``.

        ``tilts`` ``(N, 3, 3)`` is the optional rocking-curve integration set: N goniometer
        rotations, each left-multiplying ``orientation`` (``R_tilt @ orientation``) into its own
        compiled ``beam_plan``, sharing this orientation's one beam set. ``None`` (the default) is a
        single identity tilt, so ``beam_plans`` has length 1 and the untilted path is byte-identical
        to before; ``integrate_rocking_curve`` passes the tilt matrices from
        :func:`~diffBloch.preprocess.orientation.rocking_curve_tilts`.

        ``tilt_reduction`` selects how the engine reduces the tilt sub-solutions over the rocking
        curve: :class:`~diffBloch.core.products.PlainSum` (the default) sums them;
        :class:`~diffBloch.core.products.MosaicSmoothed` applies mosaicity broadening first. It is a
        rebuild-preserved attribute (geometry-independent), set by the ``mosaicity`` step.

        ``gather`` may be a precomputed :class:`~diffBloch.core.dynamical.StructureFactorGather` for
        this beam set against the shared grid. The F-gather is basis- and orientation-free, so all N
        tilts here share one, and a caller rebuilding this plan over a fixed beam set (rocking
        integration, orientation-search trials) passes the seed plan's gather
        (``op.beam_plans[0].gather``) to skip re-deriving it on every rebuild -- the dominant
        preprocess cost (see ``KNOWN_ISSUES.md`` / private 6bb3031). When ``None`` it is built once
        here and shared across the tilts.
        """
        beam_hkl = np.asarray(beam_hkl, dtype=np.int64)
        if gather is None:
            gather = build_structure_factor_gather(np.asarray(grid.grid_hkl), beam_hkl, grid.gpts)
        thickness_t = torch.as_tensor(
            np.atleast_1d(np.asarray(thickness, dtype=np.float64)), dtype=torch.float64
        )
        if orientation is None:
            rotation = np.eye(3, dtype=np.float64)
            nominal_basis = np.asarray(grid.reciprocal_basis)
        else:
            rotation = np.asarray(orientation, dtype=np.float64)
            nominal_basis = orientation_basis(np.asarray(grid.cell), rotation)
        if tilts is None:
            tilt_mats = np.eye(3, dtype=np.float64)[None]
            # No tilts: reuse the nominal basis exactly, keeping the untilted build byte-identical.
            bases = [nominal_basis]
        else:
            tilt_mats = np.asarray(tilts, dtype=np.float64)
            if tilt_mats.ndim != 3 or tilt_mats.shape[1:] != (3, 3):
                raise ValueError(f"tilts must have shape (N, 3, 3), got {tilt_mats.shape}")
            cell = np.asarray(grid.cell)
            bases = [orientation_basis(cell, tilt @ rotation) for tilt in tilt_mats]
        beam_plans = tuple(
            build_beam_plan(
                beam_hkl,
                np.asarray(grid.grid_hkl),
                basis,
                energy=energy,
                gpts=grid.gpts,
                u0=u0,
                gather=gather,
            )
            for basis in bases
        )
        beam_hkl_t = torch.tensor(beam_hkl, dtype=torch.int64)
        return cls(
            orientation=torch.tensor(rotation, dtype=torch.float64),
            tilts=torch.tensor(tilt_mats, dtype=torch.float64),
            energy=float(energy),
            u0=float(u0),
            thickness=thickness_t,
            beam_hkl=beam_hkl_t,
            beam_plans=beam_plans,
            pattern=pattern,
            alignment=build_alignment_plan(beam_hkl_t, pattern.hkl),
            tilt_reduction=tilt_reduction,
        )
