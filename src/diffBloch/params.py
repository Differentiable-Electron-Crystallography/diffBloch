"""The adjustable model parameters, and the transform that turns them into physical quantities.

Refinement varies a set of unbounded numbers, but the physical quantities they stand for are
bounded: atomic displacement parameters (ADPs) must stay positive-definite, site occupancies lie in
[0, 1], and atoms sitting on symmetry elements have some coordinates fixed. Rather than restrict
the optimizer, we store each quantity as an unbounded "raw" number and apply a fixed transform
(:func:`constrain`) that maps it onto its physical range.
:class:`RefinableParams` holds the raw numbers the optimizer varies; :class:`PhysicalState` holds
the physical quantities the diffraction calculation consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import Tensor

from diffBloch.core.adp import (
    cartesian_adp_to_star,
    cholesky_adp,
    cif_adp_to_star,
    isotropic_adp,
)
from diffBloch.core.constraints import (
    AdpConstraints,
    apply_adp_constraints,
    apply_symmetry_projection,
    positive,
    unit_interval,
)

type AdpKind = Literal["Uiso", "Uani", "missing"]
type Device = torch.device | str


@dataclass(frozen=True)
class RefinableParams:
    """The adjustable numbers the optimizer varies, before physical bounds are applied.

    Each field is an unbounded tensor; the physical bounds (positivity, [0, 1], positive-definite
    ADPs) are applied later by :func:`constrain`. An optional field is ``None`` when that quantity
    is not being refined.
    """

    asu_positions: Tensor
    uij_raw: Tensor | None = None
    u_iso_raw: Tensor | None = None
    occupancy_raw: Tensor | None = None
    Fgb: Tensor | None = None

    def to(self, device: Device) -> RefinableParams:
        """Move every present parameter tensor to ``device`` (the device knob's single primitive).

        ``params.asu_positions.device`` is the authoritative device for the whole forward model --
        :func:`constrain` and the engine co-locate every invariant (spec projector, ASU plan,
        scattering grid, beam sets) onto it at the use site -- so placing the params on an
        accelerator is all it takes to run there. A no-op (returns tensors already on ``device``)
        when nothing moves, so ``to("cpu")`` on a CPU params is an identity.
        """

        def move(t: Tensor | None) -> Tensor | None:
            return None if t is None else t.to(device)

        return replace(
            self,
            asu_positions=self.asu_positions.to(device),
            uij_raw=move(self.uij_raw),
            u_iso_raw=move(self.u_iso_raw),
            occupancy_raw=move(self.occupancy_raw),
            Fgb=move(self.Fgb),
        )


@dataclass(frozen=True)
class ConstraintSpec:
    """The fixed information :func:`constrain` needs to turn raw numbers into physical quantities.

    ``reciprocal_basis`` (``B`` = ``reciprocal_cell``, rows ``a*, b*, c*``) is required whenever
    ADPs are converted: it carries the cell frame used to express the ADPs in the reciprocal ``U*``
    frame that :func:`diffBloch.core.scattering.structure_factors` expects.

    ``position_projection`` is a ``(N, 3, 3)`` per-atom site-symmetry projector ``P`` and
    ``position_offset`` the ``(N, 3)`` on-site offset ``(I - P) @ x0``; together they constrain the
    atomic coordinates via ``P @ raw + offset`` in
    :func:`diffBloch.core.constraints.apply_symmetry_projection`, holding each atom on its
    special-position manifold. A general-position atom has ``P = I`` (unconstrained); an
    axis-aligned special position reduces to a diagonal ``P`` (the freeze mask); a coupled site
    (``x = y``) needs
    the off-diagonal projector. Built by :func:`diffBloch.io.symmetry_setup.symmetry_constraints`.

    ``adp_constraints`` holds, per atom, the site-symmetry ``Uij`` equalities
    ``Uij[i,j] = coeff * Uij[src_i, src_j]`` (empty per atom when the ADP is unconstrained;
    ``None`` when no ADP constraints apply at all), enforced by
    :func:`diffBloch.core.constraints.apply_adp_constraints` in the CIF ``Uij`` frame.
    """

    position_projection: Tensor
    position_offset: Tensor
    occupancies: Tensor
    adp_kind: tuple[AdpKind, ...] | None = None
    adp_constraints: AdpConstraints | None = None
    reciprocal_basis: Tensor | None = None


@dataclass(frozen=True)
class PhysicalState:
    """The physical quantities the diffraction calculation consumes, after bounds are applied.

    ``uij_star`` is the ADP tensor for the asymmetric-unit atoms, already expressed in the
    reciprocal ``U*`` frame (Uani via the ``d*`` relation, Uiso via ``Uiso G*``), so
    :func:`diffBloch.core.scattering.structure_factors` can use it directly with no further frame
    conversion.
    """

    positions: Tensor
    uij_star: Tensor
    occupancies: Tensor
    Fgb: Tensor | None = None


def constrain(params: RefinableParams, spec: ConstraintSpec) -> PhysicalState:
    """Apply the *crystallographic* constraints, mapping raw parameters to the physical state.

    This is the hard-constraint layer on the raw parameters: per-atom site-symmetry position
    projection, ADP site-symmetry equalities, and positivity/bounded transforms (occupancy, ADP).
    It produces the crystallographically valid :class:`PhysicalState`, which the
    refinement objective may further transform -- *molecular* hard constraints (e.g. hydrogen
    riding, a future ConstraintTransform layer) -- before the diffraction term and soft penalties
    (:meth:`diffBloch.engine.forward.RefinementEngine.objective_value`).
    """
    _validate_shapes(params, spec)
    dtype, device = params.asu_positions.dtype, params.asu_positions.device
    occupancies = (
        unit_interval(params.occupancy_raw)
        if params.occupancy_raw is not None
        else spec.occupancies.to(dtype=dtype, device=device)
    )
    return PhysicalState(
        # Co-locate the (device-invariant) spec projector onto the param device/dtype, as with
        # occupancies / reciprocal_basis below: params.asu_positions.device is the authoritative
        # device, so a GPU params + CPU-built spec runs on the GPU without the caller moving spec.
        positions=apply_symmetry_projection(
            params.asu_positions,
            projection=spec.position_projection.to(dtype=dtype, device=device),
            offset=spec.position_offset.to(dtype=dtype, device=device),
        ),
        uij_star=_constrain_adps(params, spec),
        occupancies=occupancies,
        Fgb=params.Fgb,
    )


def _validate_shapes(params: RefinableParams, spec: ConstraintSpec) -> None:
    n_atoms = int(params.asu_positions.shape[0])
    if params.asu_positions.shape != (n_atoms, 3):
        raise ValueError("asu_positions must have shape (N, 3)")
    if spec.position_projection.shape != (n_atoms, 3, 3):
        raise ValueError("position_projection must have shape (N, 3, 3)")
    if spec.position_offset.shape != params.asu_positions.shape:
        raise ValueError("position_offset must match asu_positions")
    if spec.occupancies.shape != (n_atoms,):
        raise ValueError("occupancies must have shape (N,)")
    if params.occupancy_raw is not None and params.occupancy_raw.shape != (n_atoms,):
        raise ValueError("occupancy_raw must have shape (N,)")
    if spec.adp_kind is not None and len(spec.adp_kind) != n_atoms:
        raise ValueError("adp_kind must have one entry per atom")
    if spec.adp_constraints is not None and len(spec.adp_constraints) != n_atoms:
        raise ValueError("adp_constraints must have one entry per atom")
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
        return cif_adp_to_star(_constrained_cif_uij(params.uij_raw, spec), reciprocal_lengths)

    uij = torch.zeros((n_atoms, 3, 3), dtype=dtype, device=device)

    if _requires_uani(spec):
        if params.uij_raw is None:
            raise ValueError("uij_raw is required for Uani ADPs")
        mask = _kind_mask(spec.adp_kind, "Uani", device=device)
        star = cif_adp_to_star(_constrained_cif_uij(params.uij_raw, spec), reciprocal_lengths)
        uij = torch.where(mask[:, None, None], star, uij)

    if _requires_uiso(spec):
        if params.u_iso_raw is None:
            raise ValueError("u_iso_raw is required for Uiso ADPs")
        mask = _kind_mask(spec.adp_kind, "Uiso", device=device)
        star = cartesian_adp_to_star(isotropic_adp(positive(params.u_iso_raw)), reciprocal_basis)
        uij = torch.where(mask[:, None, None], star, uij)

    return uij


def _constrained_cif_uij(uij_raw: Tensor, spec: ConstraintSpec) -> Tensor:
    """Expand the raw Cholesky ADP parameters to the CIF ``Uij`` tensor, applying site-symmetry.

    The site-symmetry ``Uij`` equalities act in the CIF frame (before the ``U*`` transform), so
    special-position atoms are not over-parameterized: any constrained component follows its base
    component and stops carrying an independent gradient.
    """
    uij = cholesky_adp(uij_raw)
    if spec.adp_constraints is not None:
        uij = apply_adp_constraints(uij, spec.adp_constraints)
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
