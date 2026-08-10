"""``from_experiment``: the boundary constructor from parsed records + config to refinement inputs.

This is the *initial total construction* of the preprocess pipeline (it is not ``Plan -> Plan`` --
there is no ``Plan`` yet). It assembles two separable products from the same records/config:

- a :class:`~diffBloch.preprocess.plan.Plan` pair (``train`` / ``validation``) -- the invariant
  geometry the preprocess steps then sharpen and ``refine`` consumes;
- a :class:`RefinementSetup` -- the structure-side static + refinable inputs the
  :class:`~diffBloch.engine.forward.RefinementEngine` needs (ASU expansion, constraint spec, initial
  parameters, atomic numbers).

The structure side lives here so the structure/experimental split mirrors the two parsed records.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from diffBloch.config.schema import DataSplitConfig, ExperimentConfig
from diffBloch.core.adp import cholesky_raw_from_adp
from diffBloch.core.crystal import reciprocal_cell
from diffBloch.core.dynamical import (
    energy2sigma,
    energy2wavelength,
    kappa,
    snap_to_standard_energy,
    wavelength2energy,
)
from diffBloch.core.products import PatternBatch
from diffBloch.core.reciprocal import gmax_mask
from diffBloch.core.scattering import structure_factors
from diffBloch.core.symmetry import AsuExpansionPlan, build_asu_expansion_plan, expand_asu
from diffBloch.engine.plan import StructureFactorGrid
from diffBloch.io.record import AdpRecord, ExperimentalRecord, StructureRecord
from diffBloch.io.symmetry_setup import symmetry_constraints
from diffBloch.params import ConstraintSpec, RefinableParams, constrain
from diffBloch.preprocess.orientation import orientation_matrices
from diffBloch.preprocess.plan import CandidatePlan, Plan
from diffBloch.specs import NO_ABSORPTION, Absorption, IntegrationGeometry

__all__ = [
    "ExperimentSetup",
    "PlanSplit",
    "RefinementSetup",
    "from_experiment",
    "seed_beam_hkl",
]

_log = logging.getLogger(__name__)

_CELL_PARAMETER_NAMES = ("a", "b", "c", "alpha", "beta", "gamma")
_CELL_PARAMETER_RTOL = 0.01


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
        ``from_experiment``, fitted by ``optimize_thickness``) -- because it varies per rotation, not per
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
    experimental_data: ExperimentalRecord | Sequence[ExperimentalRecord],
    config: ExperimentConfig,
) -> ExperimentSetup:
    """Construct the geometry ``Plan`` pair + structure ``RefinementSetup`` from parsed inputs.

    The *initial total construction* of the preprocess pipeline (not ``Plan -> Plan`` -- there is no
    ``Plan`` yet). The shared structure-factor grid is *derived* from the solve cutoff rather than
    hand-declared: :func:`~diffBloch.engine.plan.StructureFactorGrid.from_cell_for_beam_cutoff`
    sizes it
    to ``2x`` the cutoff so it spans every coupled ``g - h`` difference. The beam energy is derived
    from the PETS wavelength and snapped onto the nearest standard TEM voltage when close
    (:func:`~diffBloch.core.dynamical.snap_to_standard_energy`) -- PETS records wavelength to only
    4-5 significant figures, so the exact inverse lands a few hundred eV off 100/200/300 kV rather
    than on it. One :class:`CandidatePlan` per rotation carries its crystal orientation matrix
    (native PETS derivation, no side-car file) and the observed pattern for that zone axis.
    Rotations split into ``train`` / ``validation`` plans sharing the grid.

    Each orientation is seeded with the orientation-independent, difference-safe beam set
    ``{hkl in grid : |g| <= blochwave.g_max}`` (so beam differences stay within the derived
    grid and the 000 transmitted beam is present). The per-orientation ``sg_max`` / rsg-dsg
    pruning is the later ``select_beams`` step.

    ``experimental_data`` is a single :class:`~diffBloch.io.record.ExperimentalRecord` for the
    ordinary single-dataset experiment, or a sequence of them (``inputs.multi_dataset``) to combine
    rotations from several PETS files into one experiment -- e.g. a damage series or repeat
    measurements of the same crystal. Each file keeps its own wavelength-derived energy, UB
    matrix/cell, and mean-inner-potential correction; rotations are concatenated file-by-file with a
    running offset, so ``rotation_index`` stays globally unique across the combined set (rather than
    restarting at 0 per file) -- every place that keys off it (``ignore_orientations``, the
    train/validation split, orientation-CSV import, the thickness-NN's per-rotation lookup) then
    keeps working unmodified. Combined files must share one rocking-curve integration semiangle: the
    recipe builds a single shared rocking-curve/beam-selection geometry for the whole experiment, so
    files recorded with different precession angles cannot currently be mixed.

    This is intentionally a module-level function rather than ``ExperimentSetup.from_*``: it is the
    single documented public boundary of the preprocess pipeline (records + config -> setup), and it
    returns a *composite* of two products rather than constructing one domain object. The per-object
    constructors it delegates to follow the classmethod idiom
    (``StructureFactorGrid.from_cell_for_beam_cutoff``, ``CandidatePlan.seed``,
    ``RefinementSetup.from_structure``).
    """
    records = (
        (experimental_data,)
        if isinstance(experimental_data, ExperimentalRecord)
        else tuple(experimental_data)
    )
    if len(records) > 1:
        semiangles = [record.integration_semiangle for record in records]
        if not math.isclose(max(semiangles), min(semiangles)):
            raise ValueError(
                "combined datasets (inputs.multi_dataset) must share one rocking-curve integration "
                f"semiangle; got {sorted(set(semiangles))}"
            )
    _check_cell_parameters_agree(structure, records)

    solve_cutoff = config.blochwave.g_max
    grid = StructureFactorGrid.from_cell_for_beam_cutoff(structure.unit_cell, solve_cutoff)
    beam_hkl = seed_beam_hkl(grid, g_max=solve_cutoff)
    refinement_setup = RefinementSetup.from_structure(structure)
    absorption = config.blochwave.to_absorption()

    combined_plans: list[CandidatePlan] = []
    rotation_offset = 0
    for record_index, record in enumerate(records):
        if config.blochwave.mosaicity and record.mosaicity_degrees is None:
            source = record.source_path if record.source_path is not None else "<experimental data>"
            raise ValueError(
                f"blochwave.mosaicity=true requires a 'mosaicity:' value in {source}"
            )
        energy = snap_to_standard_energy(wavelength2energy(record.wavelength))
        orientations = orientation_matrices(
            record.ub_matrix,
            record.cell_parameters,
            record.alphas,
            record.betas,
            record.omegas,
        )
        u0 = _mean_inner_potential(grid, refinement_setup, energy=energy, absorption=absorption)
        # A per-dataset seed_thicknesses list (inputs.multi_dataset) gives each combined file its
        # own starting thickness instead of every rotation sharing one value regardless of which
        # specimen/region it actually came from.
        seed_thickness = (
            config.sample.thicknesses[record_index]
            if isinstance(config.sample.thicknesses, list)
            else config.sample.thicknesses
        )
        combined_plans.extend(
            CandidatePlan.seed(
                beam_hkl,
                PatternBatch.from_experimental_record(
                    record,
                    zone_axis_id=int(zone_id),
                    rotation_index=rotation_offset + local_index,
                ),
                energy=energy,
                thickness=seed_thickness,
                orientation=orientations[local_index],
                u0=u0,
                mosaicity_degrees=record.mosaicity_degrees,
            )
            for local_index, zone_id in enumerate(record.zone_axis_ids)
        )
        rotation_offset += len(record.zone_axis_ids)

    all_plans = tuple(combined_plans)
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

    # Split membership is defined on the original (combined, file-by-file) PETS order. Ignoring a
    # rotation must not renumber later frames and silently move them between train and validation.
    validation = _validation_mask(len(all_plans), config.refinement.split)
    train_orientations = tuple(plan for index, plan in selected if not validation[index])
    val_orientations = tuple(plan for index, plan in selected if validation[index])
    return ExperimentSetup(
        plans=PlanSplit(
            train=Plan(structure_factor_grid=grid, orientations=train_orientations),
            validation=Plan(structure_factor_grid=grid, orientations=val_orientations),
        ),
        refinement=refinement_setup,
        integration=IntegrationGeometry(semiangle=records[0].integration_semiangle),
    )


def _check_cell_parameters_agree(
    structure: StructureRecord, records: Sequence[ExperimentalRecord]
) -> None:
    """Warn when the structure CIF and a combined PETS ``.cif_pets`` disagree on unit-cell parameters.

    Refinement pins the structure's own cell (``structure.unit_cell``), so a mismatch here never
    corrupts the simulation numerically -- but a mismatch beyond ordinary refinement drift usually
    means the two files describe different crystals/settings entirely (wrong file paired up, or a
    unit mix-up), the same class of setting mismatch behind the NaN-simulation failures the
    fail-fast checks elsewhere in the pipeline guard against. Flagged only beyond 1% relative
    difference on any of ``a, b, c, alpha, beta, gamma`` -- day-to-day refinement/measurement drift
    stays well under that. Checked against every combined file independently (``inputs.multi_dataset``
    can mix a good file with a mismatched one), named by its ``source_path`` in the warning.
    """
    structure_cell = structure.cell_parameters
    for record in records:
        experimental_cell = record.cell_parameters
        relative_diff = np.abs(structure_cell - experimental_cell) / np.abs(experimental_cell)
        mismatched = relative_diff > _CELL_PARAMETER_RTOL
        if not np.any(mismatched):
            continue
        details = ", ".join(
            f"{name}: structure={structure_value:.6g} vs experimental={experimental_value:.6g} "
            f"({100.0 * diff:.2f}% off)"
            for name, structure_value, experimental_value, diff in zip(
                _CELL_PARAMETER_NAMES, structure_cell, experimental_cell, relative_diff, strict=True
            )
            if diff > _CELL_PARAMETER_RTOL
        )
        source = record.source_path if record.source_path is not None else "<experimental data>"
        _log.warning(
            "structure CIF and %s unit cells disagree by more than 1%% on %d parameter(s): %s -- "
            "check that the structure and experimental-data files describe the same crystal "
            "setting.",
            source,
            int(np.count_nonzero(mismatched)),
            details,
        )


def _mean_inner_potential(
    grid: StructureFactorGrid,
    refinement: RefinementSetup,
    *,
    energy: float,
    absorption: Absorption = NO_ABSORPTION,
) -> float:
    """The mean-inner-potential correction ``U0 = |Fgb(000)| * prefactor``, from the seed structure.

    Corrects the in-crystal wavevector magnitude (``core.dynamical.wavevector_magnitude``'s
    ``u0``), computed once from the CIF-seeded structure (not the trainable state a refinement
    later reaches) and carried fixed on every :class:`~diffBloch.preprocess.plan.CandidatePlan`
    thereafter -- mirroring the reference implementation, which computes it once in its model
    constructor and never updates it during a refinement run. ``Fgb(000)`` includes the
    absorptive component when ``absorption`` is enabled, so ``u0`` folds in both the elastic and
    absorptive forward-scattering contribution, matching ``abs(complex Fgb(000))``.
    """
    state = constrain(refinement.params, refinement.spec)
    expanded = expand_asu(
        refinement.asu_plan,
        state.positions,
        numbers=refinement.numbers,
        uij=state.uij_star,
        occupancies=state.occupancies,
    )
    assert expanded.numbers is not None and expanded.uij is not None
    assert expanded.occupancies is not None
    fgb_000 = structure_factors(
        expanded.positions,
        expanded.numbers,
        expanded.occupancies,
        expanded.uij,
        hkl=torch.zeros((1, 3), dtype=torch.int64),
        reciprocal_basis=grid.reciprocal_basis,
        cell_volume=grid.cell_volume,
        g_max=grid.g_max,
        absorption=absorption,
        energy=energy,
    )
    prefactor = energy2sigma(energy) / (kappa * energy2wavelength(energy) * float(np.pi))
    return float(torch.abs(fgb_000[0]) * prefactor)


def seed_beam_hkl(grid: StructureFactorGrid, *, g_max: float) -> NDArray[np.int64]:
    """Difference-safe seed beams: the grid reflections within ``g_max`` (includes 000).

    The orientation-independent candidate pool ``{hkl in grid : |g| <= g_max}`` that
    ``from_experiment`` lays down. Selecting from the shared ``grid`` keeps every beam difference
    inside the ``Fgb`` support as long as ``2 * g_max <= grid.g_max`` (the caller's responsibility).
    """
    structure_factor_hkl = np.asarray(grid.structure_factor_hkl)
    beams: NDArray[np.int64] = structure_factor_hkl[
        gmax_mask(structure_factor_hkl, np.asarray(grid.reciprocal_basis), g_max)
    ]
    return beams


def _validation_mask(n_rotations: int, split: DataSplitConfig) -> NDArray[np.bool_]:
    """Boolean per-rotation validation mask from the split policy.

    ``train_test=False`` holds out nothing (every rotation trains). Otherwise every
    ``round(1 / val_frac)``-th rotation (1-based count -> 0-based indices) is held out for
    validation, e.g. ``val_frac=0.2`` holds out every 5th rotation.
    """
    if not split.train_test:
        return np.zeros(n_rotations, dtype=np.bool_)
    step = round(1.0 / split.val_frac)
    mask: NDArray[np.bool_] = (np.arange(n_rotations) + 1) % step == 0
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
