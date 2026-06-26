"""Stateless refinement forward: raw parameters -> simulated diffraction -> scalar objective.

The composition spine that ties stages 1-9 together. A :class:`RefinementEngine` holds the
refinement-invariant plans (constraint spec, ASU-expansion plan, the shared scattering grid, and one
:class:`OrientationPlan` per rotation) and maps :class:`~diffBloch.params.RefinableParams` to a
differentiable objective:

    constrain -> expand ASU -> structure_factors (Fgb on the shared grid)
              -> per orientation: build_bloch_system -> propagate -> intensities -> align -> loss

The grid is owned once by :class:`ScatteringGrid` and reused by both ``structure_factors`` and every
``BeamPlan``, so the two sides cannot silently disagree on the Fgb support (the difference-support
constraint is validated when the beam plans are built).

This slice is deliberately stateless: no optimizer, no history. ``from_config`` /
``from_experiment`` construction is deferred until beam selection (stage 11) exists; engines are
assembled here from explicit per-orientation beam sets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from diffBloch.core.crystal import cell_volume as _cell_volume
from diffBloch.core.crystal import reciprocal_cell
from diffBloch.core.dynamical import BeamPlan, build_beam_plan, build_bloch_system
from diffBloch.core.products import (
    AlignedIntensities,
    AlignmentPlan,
    BlochSolution,
    PatternBatch,
    align,
    build_alignment_plan,
)
from diffBloch.core.reciprocal import make_hkl_grid, reciprocal_space_gpts
from diffBloch.core.scattering import structure_factors
from diffBloch.core.solver import Method, propagate
from diffBloch.core.symmetry import AsuExpansionPlan, expand_asu
from diffBloch.params import ConstraintSpec, RefinableParams, constrain

__all__ = [
    "Objective",
    "OrientationPlan",
    "RefinementEngine",
    "ScatteringGrid",
]

# An objective reduces one orientation's aligned intensities to a scalar (its own loss + reduction).
type Objective = Callable[[AlignedIntensities], Tensor]


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


@dataclass(frozen=True)
class RefinementEngine:
    """Stateless forward from raw parameters to a differentiable scalar objective.

    Holds the refinement-invariant context: the constraint ``spec``, the ASU-expansion ``asu_plan``,
    the ASU atomic ``numbers``, the shared ``grid``, the per-rotation ``orientations``, the sample
    ``thicknesses``, the ``objective``, and the propagation ``method``.
    """

    spec: ConstraintSpec
    asu_plan: AsuExpansionPlan
    numbers: Tensor
    grid: ScatteringGrid
    orientations: tuple[OrientationPlan, ...]
    thicknesses: Tensor
    objective: Objective
    method: Method = "matrix_exp"

    def simulate(self, params: RefinableParams) -> tuple[BlochSolution, ...]:
        """Return the calculated :class:`BlochSolution` for every orientation (no loss)."""
        fgb = self._structure_factors(params)
        return tuple(self._solve(orientation, fgb) for orientation in self.orientations)

    def forward(self, params: RefinableParams) -> Tensor:
        """Return the scalar objective summed over orientations (differentiable in ``params``)."""
        if not self.orientations:
            raise ValueError("engine has no orientations to evaluate")
        fgb = self._structure_factors(params)
        total = params.asu_positions.new_zeros(())
        for orientation in self.orientations:
            solution = self._solve(orientation, fgb)
            aligned = align(solution, orientation.pattern, orientation.alignment)
            total = total + self.objective(aligned)
        return total

    def _structure_factors(self, params: RefinableParams) -> Tensor:
        state = constrain(params, self.spec)
        device = state.positions.device  # the active (params) device; co-locate invariants here
        expanded = expand_asu(
            self.asu_plan,
            state.positions,
            numbers=self.numbers.to(device),
            uij=state.uij_star,
            occupancies=state.occupancies,
        )
        assert expanded.numbers is not None and expanded.uij is not None
        assert expanded.occupancies is not None
        return structure_factors(
            expanded.positions,
            expanded.numbers,
            expanded.occupancies,
            expanded.uij,
            hkl=self.grid.grid_hkl.to(device),
            reciprocal_basis=self.grid.reciprocal_basis.to(device),
            cell_volume=self.grid.cell_volume,
            g_max=self.grid.g_max,
        )

    def _solve(self, orientation: OrientationPlan, fgb: Tensor) -> BlochSolution:
        device = fgb.device  # fgb is param-derived; thicknesses/beam_hkl must co-locate with it
        thicknesses = self.thicknesses.to(device)
        system = build_bloch_system(orientation.beam_plan, fgb)
        psi = propagate(system, thicknesses, method=self.method)
        return BlochSolution.from_propagation(psi, orientation.beam_hkl.to(device), thicknesses)
