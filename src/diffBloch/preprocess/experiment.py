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

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from diffBloch.config.schema import DataSplitConfig, ExperimentConfig
from diffBloch.core.adp import cholesky_raw_from_adp
from diffBloch.core.crystal import reciprocal_cell
from diffBloch.core.dynamical import wavelength2energy
from diffBloch.core.products import PatternBatch
from diffBloch.core.reciprocal import gmax_mask
from diffBloch.core.symmetry import AsuExpansionPlan, build_asu_expansion_plan
from diffBloch.engine.plan import OrientationPlan, ScatteringGrid
from diffBloch.io.record import AdpRecord, ObservationRecord, StructureRecord
from diffBloch.params import ConstraintSpec, RefinableParams
from diffBloch.preprocess.orientation import orientation_matrices
from diffBloch.preprocess.plan import Plan

__all__ = [
    "ExperimentSetup",
    "PlanSplit",
    "RefinementSetup",
    "from_experiment",
    "refinement_setup",
]


@dataclass(frozen=True)
class PlanSplit:
    """A ``train`` / ``validation`` :class:`Plan` pair sharing one :class:`ScatteringGrid`.

    ``from_experiment`` splits the rotations into the two plans; the preprocess ``Plan -> Plan``
    steps sharpen each, and ``refine`` fits on ``train`` while scoring the held-out ``validation``.
    Both plans reference the *same* grid object, so the shared ``Fgb`` support cannot diverge.
    """

    train: Plan
    validation: Plan


@dataclass(frozen=True)
class ExperimentSetup:
    """The full product of ``from_experiment``: the geometry ``plans`` + structure ``refinement``.

    Two separable concerns from the same records/config: ``plans`` (the ``Plan -> Plan`` geometry
    spine) and ``refinement`` (the static structure context the engine is built from). Kept distinct
    so only the ``Plan`` pair flows through the preprocess pipeline.
    """

    plans: PlanSplit
    refinement: RefinementSetup


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


def from_experiment(
    structure: StructureRecord,
    observations: ObservationRecord,
    config: ExperimentConfig,
) -> ExperimentSetup:
    """Construct the geometry ``Plan`` pair + structure ``RefinementSetup`` from parsed inputs.

    The *initial total construction* of the preprocess pipeline (not ``Plan -> Plan`` -- there is no
    ``Plan`` yet). The shared grid is sized from the ideal CIF cell and ``numerics.g_max``; the beam
    energy is derived from the PETS wavelength; one :class:`OrientationPlan` per rotation carries
    its crystal orientation matrix (native PETS derivation, no side-car file) and the observed
    pattern for that zone axis. Rotations split into ``train`` / ``validation`` plans sharing it.

    Each orientation is seeded with the orientation-independent, difference-safe beam set
    ``{hkl in grid : |g| <= numerics.g_max_refine}`` (so beam differences stay within the
    ``g_max`` grid and the 000 transmitted beam is always present). The faithful per-orientation
    ``sg_max`` / rsg-dsg pruning is the later ``select_beams`` step.
    """
    grid = ScatteringGrid.from_cell(structure.unit_cell, g_max=config.numerics.g_max)
    energy = wavelength2energy(observations.wavelength)
    beam_hkl = _seed_beam_hkl(grid, g_max_refine=config.numerics.g_max_refine)
    orientations = orientation_matrices(
        observations.ub_matrix,
        observations.cell_parameters,
        observations.alphas,
        observations.betas,
        observations.omegas,
    )
    plans = tuple(
        OrientationPlan.build(
            grid,
            beam_hkl,
            PatternBatch.from_observation_record(observations, zone_axis_id=int(zone_id)),
            energy=energy,
            orientation=orientations[index],
        )
        for index, zone_id in enumerate(observations.zone_axis_ids)
    )

    validation = _validation_mask(len(plans), config.refinement.split)
    train_orientations = tuple(p for p, v in zip(plans, validation, strict=True) if not v)
    val_orientations = tuple(p for p, v in zip(plans, validation, strict=True) if v)
    return ExperimentSetup(
        plans=PlanSplit(
            train=Plan(grid=grid, orientations=train_orientations),
            validation=Plan(grid=grid, orientations=val_orientations),
        ),
        refinement=refinement_setup(structure, thicknesses=config.sample.thicknesses),
    )


def _seed_beam_hkl(grid: ScatteringGrid, *, g_max_refine: float) -> NDArray[np.int64]:
    """Difference-safe seed beams: the grid reflections within ``g_max_refine`` (includes 000)."""
    grid_hkl = np.asarray(grid.grid_hkl)
    beams: NDArray[np.int64] = grid_hkl[
        gmax_mask(grid_hkl, np.asarray(grid.reciprocal_basis), g_max_refine)
    ]
    return beams


def _validation_mask(n_rotations: int, split: DataSplitConfig) -> NDArray[np.bool_]:
    """Boolean per-rotation validation mask from the split policy.

    Only the Stage-1 policies are implemented: ``train='all_except_validation'`` with
    ``validation='every_10th_rotation'`` (every 10th rotation by 1-based count -> 0-based indices
    where ``(i + 1) % 10 == 0``). Other selector strings are rejected rather than silently ignored.
    """
    if split.train != "all_except_validation":
        raise ValueError(f"unsupported train split policy: {split.train!r}")
    if split.validation != "every_10th_rotation":
        raise ValueError(f"unsupported validation split policy: {split.validation!r}")
    mask: NDArray[np.bool_] = (np.arange(n_rotations) + 1) % 10 == 0
    return mask


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
