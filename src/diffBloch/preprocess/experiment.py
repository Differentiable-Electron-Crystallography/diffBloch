"""``from_experiment``: the boundary constructor from parsed records + config to refinement inputs.

This is the *initial total construction* of the preprocess pipeline (it is not ``Plan -> Plan`` --
there is no ``Plan`` yet). It assembles two separable products from the same records/config:

- a :class:`~diffBloch.preprocess.plan.Plan` pair (``train`` / ``validation``) -- the invariant
  geometry the preprocess steps then sharpen and ``refine`` consumes;
- a :class:`RefinementSetup` -- the structure-side static + refinable inputs the
  :class:`~diffBloch.engine.forward.RefinementEngine` needs (ASU expansion, constraint spec, initial
  parameters, atomic numbers).

The structure side lives here so the structure/observation split mirrors the two parsed records.
"""

from __future__ import annotations

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
from diffBloch.engine.plan import StructureFactorGrid
from diffBloch.io.record import AdpRecord, ObservationRecord, StructureRecord
from diffBloch.io.symmetry_setup import symmetry_constraints
from diffBloch.params import ConstraintSpec, RefinableParams
from diffBloch.preprocess.orientation import orientation_matrices
from diffBloch.preprocess.plan import CandidatePlan, Plan
from diffBloch.specs import IntegrationGeometry

__all__ = [
    "ExperimentSetup",
    "PlanSplit",
    "RefinementSetup",
    "from_experiment",
    "seed_beam_hkl",
]


@dataclass(frozen=True)
class PlanSplit:
    """A ``train`` / ``validation`` :class:`Plan` pair sharing one :class:`StructureFactorGrid`.

    ``from_experiment`` splits the rotations into the two plans (``validation`` = every 10th
    rotation by default); both reference the *same* grid object, so the shared ``Fgb`` support
    cannot diverge.

    The split is currently **dormant**: nothing downstream distinguishes ``train`` from
    ``validation`` (inference and the engine take a single :class:`Plan`), and whole-experiment
    work uses :attr:`combined` (all rotations). A whole-*rotation* holdout is a weak
    cross-validation guard for over-determined physics refinement anyway -- the principled analog
    holds out *reflections* (R_free), not orientations. It is retained because it becomes
    informative for learned modes, where a learned ``theta -> thickness`` can overfit per rotation.
    """

    train: Plan
    validation: Plan

    @property
    def combined(self) -> Plan:
        """One :class:`Plan` over *all* rotations (train + validation) on the shared grid.

        For whole-experiment evaluation (e.g. inference over every rotation), where the train/val
        split is irrelevant. ``train`` orientations come first, then ``validation``.
        """
        return Plan(
            structure_factor_grid=self.train.structure_factor_grid,
            orientations=self.train.orientations + self.validation.orientations,
        )


@dataclass(frozen=True)
class ExperimentSetup:
    """The full product of ``from_experiment``: the geometry ``plans`` + structure ``refinement``.

    Two separable concerns from the same records/config: ``plans`` (the ``Plan -> Plan`` geometry
    spine) and ``refinement`` (the static structure context the engine is built from). Kept distinct
    so only the ``Plan`` pair flows through the preprocess pipeline.
    """

    plans: PlanSplit
    refinement: RefinementSetup
    integration: IntegrationGeometry


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

    @classmethod
    def from_structure(cls, structure: StructureRecord) -> RefinementSetup:
        """Assemble the structure-side refinement inputs from a parsed :class:`StructureRecord`.

        Positions and symmetry use the *ideal* CIF cell (the measured-lattice correction enters only
        through the per-orientation matrices in the geometry plan). ADPs are mapped to the
        reciprocal ``U*`` frame of this ideal cell by :func:`diffBloch.params.constrain`.
        Per-rotation thickness lives elsewhere -- on each per-rotation plan (seeded by
        ``from_experiment``, fitted by ``fit_thickness``) -- because it varies per rotation, not per
        structure.

        Special-position degrees of freedom are constrained by the site-symmetry projector built
        natively from the structure's symmetry operators (:func:`symmetry_constraints`): an atom
        special position is held on its site under refinement, so it is neither over-parameterized
        nor free to drift off. A general-position atom gets the identity projector (unconstrained).
        """
        positions = torch.tensor(structure.frac_positions, dtype=torch.float64)
        uij_raw, u_iso_raw = _initial_adp_params(structure.adp)
        constraints = symmetry_constraints(structure)
        spec = ConstraintSpec(
            position_projection=torch.tensor(constraints.position_projection, dtype=torch.float64),
            position_offset=torch.tensor(constraints.position_offset, dtype=torch.float64),
            occupancies=torch.tensor(structure.occupancies, dtype=torch.float64),
            adp_kind=structure.adp.kind,
            adp_constraints=constraints.adp_constraints,
            reciprocal_basis=torch.tensor(
                reciprocal_cell(structure.unit_cell), dtype=torch.float64
            ),
        )
        return cls(
            asu_plan=build_asu_expansion_plan(
                structure.frac_positions, structure.symops_R, structure.symops_t
            ),
            spec=spec,
            params=RefinableParams(
                asu_positions=positions.clone(), uij_raw=uij_raw, u_iso_raw=u_iso_raw
            ),
            numbers=torch.tensor(structure.numbers, dtype=torch.int64),
        )


def from_experiment(
    structure: StructureRecord,
    observations: ObservationRecord,
    config: ExperimentConfig,
) -> ExperimentSetup:
    """Construct the geometry ``Plan`` pair + structure ``RefinementSetup`` from parsed inputs.

    The *initial total construction* of the preprocess pipeline (not ``Plan -> Plan`` -- there is no
    ``Plan`` yet). The shared structure-factor grid is *derived* from the solve cutoff rather than
    hand-declared: :func:`~diffBloch.engine.plan.StructureFactorGrid.from_cell_for_beam_cutoff`
    sizes it
    to ``2x`` the cutoff so it spans every coupled ``g - h`` difference. The solve cutoff is the
    ``blochwave`` radius when the run couples, else the seed pool ``g_max_refine`` (a
    tilt-independent run solves only the seed set). The beam energy is derived from the PETS
    wavelength; one :class:`CandidatePlan` per rotation carries its crystal orientation matrix
    (native PETS derivation, no side-car file) and the observed pattern for that zone axis.
    Rotations split into ``train`` / ``validation`` plans sharing the grid.

    Each orientation is seeded with the orientation-independent, difference-safe beam set
    ``{hkl in grid : |g| <= blochwave.g_max_refine}`` (so beam differences stay within the derived
    grid and the 000 transmitted beam is present). The per-orientation ``sg_max`` / rsg-dsg
    pruning is the later ``select_beams`` step.

    This is intentionally a module-level function rather than ``ExperimentSetup.from_*``: it is the
    single documented public boundary of the preprocess pipeline (records + config -> setup), and it
    returns a *composite* of two products rather than constructing one domain object. The per-object
    constructors it delegates to follow the classmethod idiom
    (``StructureFactorGrid.from_cell_for_beam_cutoff``, ``CandidatePlan.seed``,
    ``RefinementSetup.from_structure``).
    """
    solve_cutoff = config.blochwave.g_max
    grid = StructureFactorGrid.from_cell_for_beam_cutoff(structure.unit_cell, solve_cutoff)
    energy = wavelength2energy(observations.wavelength)
    beam_hkl = seed_beam_hkl(grid, g_max_refine=config.blochwave.g_max_refine)
    orientations = orientation_matrices(
        observations.ub_matrix,
        observations.cell_parameters,
        observations.alphas,
        observations.betas,
        observations.omegas,
    )
    all_plans = tuple(
        CandidatePlan.seed(
            beam_hkl,
            PatternBatch.from_observation_record(observations, zone_axis_id=int(zone_id)),
            energy=energy,
            thickness=config.sample.thicknesses,
            orientation=orientations[index],
        )
        for index, zone_id in enumerate(observations.zone_axis_ids)
    )

    selection = config.blochwave.to_orientation_selection()
    ignored = set(selection.ignore_orientations)
    out_of_range = sorted(index for index in ignored if index >= len(all_plans))
    if out_of_range:
        raise ValueError(
            "ignore_orientations contains indices outside the PETS rotation range "
            f"0..{len(all_plans) - 1}: {out_of_range}"
        )
    selected = tuple((index, plan) for index, plan in enumerate(all_plans) if index not in ignored)
    if not selected:
        raise ValueError("ignore_orientations excludes every PETS rotation")

    # Split membership is defined on the original PETS order. Ignoring a rotation must not renumber
    # later frames and silently move them between train and validation.
    validation = _validation_mask(len(all_plans), config.refinement.split)
    train_orientations = tuple(plan for index, plan in selected if not validation[index])
    val_orientations = tuple(plan for index, plan in selected if validation[index])
    return ExperimentSetup(
        plans=PlanSplit(
            train=Plan(structure_factor_grid=grid, orientations=train_orientations),
            validation=Plan(structure_factor_grid=grid, orientations=val_orientations),
        ),
        refinement=RefinementSetup.from_structure(structure),
        integration=IntegrationGeometry(semiangle=observations.integration_semiangle),
    )


def seed_beam_hkl(grid: StructureFactorGrid, *, g_max_refine: float) -> NDArray[np.int64]:
    """Difference-safe seed beams: the grid reflections within ``g_max_refine`` (includes 000).

    The orientation-independent candidate pool ``{hkl in grid : |g| <= g_max_refine}`` that
    ``from_experiment`` lays down and the pool-lever convergence step (``converge_pool``) re-seeds
    at a wider radius. Selecting from the shared ``grid`` keeps every beam difference inside the
    ``Fgb`` support as long as ``2 * g_max_refine <= grid.g_max`` (the caller's responsibility).
    """
    structure_factor_hkl = np.asarray(grid.structure_factor_hkl)
    beams: NDArray[np.int64] = structure_factor_hkl[
        gmax_mask(structure_factor_hkl, np.asarray(grid.reciprocal_basis), g_max_refine)
    ]
    return beams


def _validation_mask(n_rotations: int, split: DataSplitConfig) -> NDArray[np.bool_]:
    """Boolean per-rotation validation mask from the split policy.

    The implemented policies are ``train='all_except_validation'`` with
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
