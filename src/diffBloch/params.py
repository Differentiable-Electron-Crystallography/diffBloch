"""Raw refinable parameters and the constraint seam to physical tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from diffBloch.core.adp import (
    cartesian_adp_to_star,
    cholesky_adp,
    cif_adp_to_star,
    isotropic_adp,
)
from diffBloch.core.constraints import apply_symmetry_mask, positive, unit_interval

type AdpKind = Literal["Uiso", "Uani", "missing"]


@dataclass(frozen=True)
class RefinableParams:
    """The refinable surface in unconstrained tensor space."""

    asu_positions: Tensor
    uij_raw: Tensor | None = None
    u_iso_raw: Tensor | None = None
    occupancy_raw: Tensor | None = None
    Fgb: Tensor | None = None
    thickness_raw: Tensor | None = None
    b_dose_raw: Tensor | None = None


@dataclass(frozen=True)
class ConstraintSpec:
    """Static constraint metadata needed to constrain raw parameters.

    ``reciprocal_basis`` (``B`` = ``reciprocal_cell``, rows ``a*, b*, c*``) is required whenever
    ADPs are constrained: it carries the cell frame used to map raw ADPs into the reciprocal ``U*``
    frame that :func:`diffBloch.core.scattering.structure_factors` consumes.

    ``refinable_position_mask`` is a shape-preserving ``(N, 3)`` gate with polarity
    ``1 = refinable``, ``0 = frozen`` (applied as ``raw * mask + fixed * (1 - mask)`` in
    :func:`diffBloch.core.constraints.apply_symmetry_mask`). Naming convention in this codebase:
    *mask* = a shape-preserving 0/1 selector named by its subject and polarity; *filter* /
    *select* = an operation that shrinks a set.
    """

    fixed_positions: Tensor
    refinable_position_mask: Tensor
    occupancies: Tensor
    adp_kind: tuple[AdpKind, ...] | None = None
    reciprocal_basis: Tensor | None = None


@dataclass(frozen=True)
class PhysicalState:
    """Constrained physical tensors consumed by later physics stages.

    ``uij_star`` is the ASU ADP tensor already mapped into the reciprocal ``U*`` frame (Uani via the
    ``d*`` relation, Uiso via ``Uiso G*``), so :func:`diffBloch.core.scattering.structure_factors`
    can consume it directly with no further frame hand-off.
    """

    positions: Tensor
    uij_star: Tensor
    occupancies: Tensor
    Fgb: Tensor | None = None
    thicknesses: Tensor | None = None
    b_dose: Tensor | None = None


def constrain(params: RefinableParams, spec: ConstraintSpec) -> PhysicalState:
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
            mask=spec.refinable_position_mask,
            fixed=spec.fixed_positions,
        ),
        uij_star=_constrain_adps(params, spec),
        occupancies=occupancies,
        Fgb=params.Fgb,
        thicknesses=positive(params.thickness_raw) if params.thickness_raw is not None else None,
        b_dose=positive(params.b_dose_raw) if params.b_dose_raw is not None else None,
    )


def _validate_shapes(params: RefinableParams, spec: ConstraintSpec) -> None:
    n_atoms = int(params.asu_positions.shape[0])
    if params.asu_positions.shape != (n_atoms, 3):
        raise ValueError("asu_positions must have shape (N, 3)")
    if spec.fixed_positions.shape != params.asu_positions.shape:
        raise ValueError("fixed_positions must match asu_positions")
    if spec.refinable_position_mask.shape != params.asu_positions.shape:
        raise ValueError("refinable_position_mask must match asu_positions")
    if spec.occupancies.shape != (n_atoms,):
        raise ValueError("occupancies must have shape (N,)")
    if params.occupancy_raw is not None and params.occupancy_raw.shape != (n_atoms,):
        raise ValueError("occupancy_raw must have shape (N,)")
    if spec.adp_kind is not None and len(spec.adp_kind) != n_atoms:
        raise ValueError("adp_kind must have one entry per atom")
    if spec.adp_kind is not None and "missing" in spec.adp_kind:
        raise ValueError("missing ADPs require an explicit initialization policy")
    if (_requires_uani(spec) or spec.adp_kind is None) and (
        params.uij_raw is None or params.uij_raw.shape != (n_atoms, 3, 3)
    ):
        raise ValueError("uij_raw must have shape (N, 3, 3)")
    if _requires_uiso(spec) and (params.u_iso_raw is None or params.u_iso_raw.shape != (n_atoms,)):
        raise ValueError("u_iso_raw must have shape (N,)")
    if _constrains_adps(spec) and spec.reciprocal_basis is None:
        raise ValueError("reciprocal_basis is required to map ADPs into the U* frame")
    if spec.reciprocal_basis is not None and spec.reciprocal_basis.shape != (3, 3):
        raise ValueError("reciprocal_basis must have shape (3, 3)")


def _constrain_adps(params: RefinableParams, spec: ConstraintSpec) -> Tensor:
    n_atoms = int(params.asu_positions.shape[0])
    dtype = params.asu_positions.dtype
    device = params.asu_positions.device
    assert spec.reciprocal_basis is not None  # guaranteed by _validate_shapes when ADPs are present
    reciprocal_basis = spec.reciprocal_basis.to(dtype=dtype, device=device)
    reciprocal_lengths = torch.linalg.norm(reciprocal_basis, dim=1)

    if spec.adp_kind is None:
        if params.uij_raw is None:
            raise ValueError("uij_raw is required when adp_kind is not provided")
        return cif_adp_to_star(cholesky_adp(params.uij_raw), reciprocal_lengths)

    uij = torch.zeros((n_atoms, 3, 3), dtype=dtype, device=device)

    if _requires_uani(spec):
        if params.uij_raw is None:
            raise ValueError("uij_raw is required for Uani ADPs")
        mask = _kind_mask(spec.adp_kind, "Uani", device=device)
        star = cif_adp_to_star(cholesky_adp(params.uij_raw), reciprocal_lengths)
        uij = torch.where(mask[:, None, None], star, uij)

    if _requires_uiso(spec):
        if params.u_iso_raw is None:
            raise ValueError("u_iso_raw is required for Uiso ADPs")
        mask = _kind_mask(spec.adp_kind, "Uiso", device=device)
        star = cartesian_adp_to_star(isotropic_adp(positive(params.u_iso_raw)), reciprocal_basis)
        uij = torch.where(mask[:, None, None], star, uij)

    return uij


def _constrains_adps(spec: ConstraintSpec) -> bool:
    """Whether ``constrain`` will produce ADPs (and therefore needs a cell frame)."""
    return spec.adp_kind is None or _requires_uani(spec) or _requires_uiso(spec)


def _requires_uani(spec: ConstraintSpec) -> bool:
    return spec.adp_kind is not None and "Uani" in spec.adp_kind


def _requires_uiso(spec: ConstraintSpec) -> bool:
    return spec.adp_kind is not None and "Uiso" in spec.adp_kind


def _kind_mask(kinds: tuple[AdpKind, ...], target: AdpKind, *, device: torch.device) -> Tensor:
    return torch.tensor([kind == target for kind in kinds], dtype=torch.bool, device=device)
