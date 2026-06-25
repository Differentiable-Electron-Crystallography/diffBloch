"""Raw refinable parameters and the constraint seam to physical tensors."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from diffBloch.core.adp import cholesky_adp
from diffBloch.core.constraints import apply_symmetry_mask, positive, unit_interval


@dataclass(frozen=True)
class RefinableParams:
    """The refinable surface in unconstrained tensor space."""

    asu_positions: Tensor
    uij_raw: Tensor
    occupancy_raw: Tensor | None = None
    Fgb: Tensor | None = None
    thickness_raw: Tensor | None = None
    b_dose_raw: Tensor | None = None


@dataclass(frozen=True)
class StructureSpec:
    """Static structure metadata needed to constrain raw parameters."""

    fixed_positions: Tensor
    position_mask: Tensor
    occupancies: Tensor


@dataclass(frozen=True)
class PhysicalState:
    """Constrained physical tensors consumed by later physics stages."""

    positions: Tensor
    uij_cif: Tensor
    occupancies: Tensor
    Fgb: Tensor | None = None
    thicknesses: Tensor | None = None
    b_dose: Tensor | None = None


def constrain(params: RefinableParams, spec: StructureSpec) -> PhysicalState:
    """Map raw refinable tensors to constrained physical tensors."""
    _validate_shapes(params, spec)
    occupancies = (
        unit_interval(params.occupancy_raw)
        if params.occupancy_raw is not None
        else spec.occupancies.to(
            dtype=params.asu_positions.dtype, device=params.asu_positions.device
        )
    )
    return PhysicalState(
        positions=apply_symmetry_mask(
            params.asu_positions,
            mask=spec.position_mask,
            fixed=spec.fixed_positions,
        ),
        uij_cif=cholesky_adp(params.uij_raw),
        occupancies=occupancies,
        Fgb=params.Fgb,
        thicknesses=positive(params.thickness_raw) if params.thickness_raw is not None else None,
        b_dose=positive(params.b_dose_raw) if params.b_dose_raw is not None else None,
    )


def _validate_shapes(params: RefinableParams, spec: StructureSpec) -> None:
    n_atoms = int(params.asu_positions.shape[0])
    if params.asu_positions.shape != (n_atoms, 3):
        raise ValueError("asu_positions must have shape (N, 3)")
    if spec.fixed_positions.shape != params.asu_positions.shape:
        raise ValueError("fixed_positions must match asu_positions")
    if spec.position_mask.shape != params.asu_positions.shape:
        raise ValueError("position_mask must match asu_positions")
    if params.uij_raw.shape != (n_atoms, 3, 3):
        raise ValueError("uij_raw must have shape (N, 3, 3)")
    if spec.occupancies.shape != (n_atoms,):
        raise ValueError("occupancies must have shape (N,)")
    if params.occupancy_raw is not None and params.occupancy_raw.shape != (n_atoms,):
        raise ValueError("occupancy_raw must have shape (N,)")
