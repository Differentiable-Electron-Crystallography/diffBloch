"""``from_experiment``: the boundary constructor from parsed records + config to refinement inputs.

This is the *initial total construction* of the preprocess pipeline (it is not ``Plan -> Plan`` --
there is no ``Plan`` yet). It assembles two separable products from the same records/config:

- a :class:`~diffBloch.preprocess.plan.Plan` pair (``train`` / ``validation``) -- the invariant
  geometry the preprocess steps then sharpen and ``refine`` consumes (added in the next slice);
- a :class:`RefinementSetup` -- the structure-side static + refinable inputs the
  :class:`~diffBloch.engine.forward.RefinementEngine` needs (ASU expansion, constraint spec, initial
  parameters, atomic numbers, thicknesses).

The structure side lives here so the structure/observation split mirrors the two parsed records.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from diffBloch.core.adp import cholesky_raw_from_adp
from diffBloch.core.crystal import reciprocal_cell
from diffBloch.core.symmetry import AsuExpansionPlan, build_asu_expansion_plan
from diffBloch.io.record import AdpRecord, StructureRecord
from diffBloch.params import ConstraintSpec, RefinableParams

__all__ = [
    "RefinementSetup",
    "refinement_setup",
]


@dataclass(frozen=True)
class RefinementSetup:
    """The structure-side inputs a :class:`RefinementEngine` is built from.

    Kept separate from the geometry :class:`~diffBloch.preprocess.plan.Plan` (which carries the grid
    + orientations): the ``Plan`` flows through the ``Plan -> Plan`` preprocess steps, while this
    static structure context is handed to the engine at refinement time. ``params`` are the initial
    refinable parameters seeded from the CIF (positions at their CIF values, ADPs inverted from the
    CIF ADPs); ``spec`` freezes the constraint metadata (fixed positions, occupancies, ADP kinds,
    the reciprocal frame ADPs map through).
    """

    asu_plan: AsuExpansionPlan
    spec: ConstraintSpec
    params: RefinableParams
    numbers: Tensor
    thicknesses: Tensor


def refinement_setup(
    structure: StructureRecord, *, thicknesses: Sequence[float]
) -> RefinementSetup:
    """Assemble the structure-side refinement inputs from a parsed :class:`StructureRecord`.

    Positions and symmetry use the *ideal* CIF cell (the measured-lattice correction enters only
    through the per-orientation matrices in the geometry plan). ADPs are mapped to the reciprocal
    ``U*`` frame of this ideal cell by :func:`diffBloch.params.constrain`.

    .. note::
       The ``position_mask`` is all-free for now: special-position degree-of-freedom constraints
       (the diffpy-backed expansion behind :mod:`diffBloch.io.symmetry_setup`) are a later
       constraints stage. Until then a special-position atom is over-parameterized and may drift off
       its site under refinement. Recorded in ``KNOWN_ISSUES.md``.
    """
    if not thicknesses:
        raise ValueError("thicknesses must contain at least one value")

    positions = torch.tensor(structure.frac_positions, dtype=torch.float64)
    uij_raw, u_iso_raw = _initial_adp_params(structure.adp)
    spec = ConstraintSpec(
        fixed_positions=positions,
        position_mask=torch.ones_like(positions),
        occupancies=torch.tensor(structure.occupancies, dtype=torch.float64),
        adp_kind=structure.adp.kind,
        reciprocal_basis=torch.tensor(reciprocal_cell(structure.unit_cell), dtype=torch.float64),
    )
    return RefinementSetup(
        asu_plan=build_asu_expansion_plan(
            structure.frac_positions, structure.symops_R, structure.symops_t
        ),
        spec=spec,
        params=RefinableParams(
            asu_positions=positions.clone(), uij_raw=uij_raw, u_iso_raw=u_iso_raw
        ),
        numbers=torch.tensor(structure.numbers, dtype=torch.int64),
        thicknesses=torch.tensor(list(thicknesses), dtype=torch.float64),
    )


def _initial_adp_params(adp: AdpRecord) -> tuple[Tensor | None, Tensor | None]:
    """Invert CIF ADPs into the raw parameters ``constrain`` re-expands (Cholesky / softplus).

    Returns ``(uij_raw, u_iso_raw)``, each present only when its ADP kind occurs. ``constrain``
    builds the full ADP tensor and masks by kind, so the raw tensor must span every atom; rows of
    the other kind are filled with a harmless valid placeholder (identity / unit) that is masked out
    downstream.
    """
    kinds = adp.kind
    if "missing" in kinds:
        raise ValueError("missing ADPs require an explicit initialization policy")

    uij_raw: Tensor | None = None
    u_iso_raw: Tensor | None = None
    n_atoms = len(kinds)

    if "Uani" in kinds:
        uani = torch.tensor([kind == "Uani" for kind in kinds])
        uij_cif = torch.tensor(adp.uij_cif, dtype=torch.float64)
        identity = torch.eye(3, dtype=torch.float64).expand(n_atoms, 3, 3)
        uij = torch.where(uani[:, None, None], uij_cif, identity)
        uij_raw = cholesky_raw_from_adp(uij)

    if "Uiso" in kinds:
        uiso = torch.tensor([kind == "Uiso" for kind in kinds])
        u_iso = torch.tensor(adp.u_iso, dtype=torch.float64)
        filled = torch.where(uiso, u_iso, torch.ones_like(u_iso))
        u_iso_raw = torch.log(torch.expm1(filled))  # inverse softplus (constrain re-applies it)

    return uij_raw, u_iso_raw
