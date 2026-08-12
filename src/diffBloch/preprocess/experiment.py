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
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from diffBloch.config.schema import DataSplitConfig, ExperimentConfig
from diffBloch.core.adp import cholesky_raw_from_adp
from diffBloch.core.crystal import cell_matrix_from_parameters, reciprocal_cell
from diffBloch.core.dynamical import (
    energy2sigma,
    energy2wavelength,
    kappa,
    snap_to_standard_energy,
    wavelength2energy,
)
from diffBloch.core.products import MosaicSmoothed, PatternBatch
from diffBloch.core.reciprocal import gmax_mask
from diffBloch.core.scattering import structure_factors
from diffBloch.core.symmetry import AsuExpansionPlan, build_asu_expansion_plan, expand_asu
from diffBloch.engine.plan import StructureFactorGrid
from diffBloch.io.record import AdpRecord, ExperimentalRecord, StructureRecord
from diffBloch.io.symmetry_setup import symmetry_constraints
from diffBloch.params import ConstraintSpec, RefinableParams, constrain
from diffBloch.preprocess.orientation import orientation_matrices
from diffBloch.preprocess.plan import CandidatePlan, Plan
from diffBloch.specs import NO_ABSORPTION, Absorption, IntegrationGeometry, RockingCurve

__all__ = [
    "DatasetSetup",
    "ExperimentSetup",
    "PlanSplit",
    "RefinementSetup",
    "from_experiment",
    "resolve_dataset_mosaicity",
    "seed_beam_hkl",
    "setup_datasets",
    "validation_mask",
]

_log = logging.getLogger(__name__)

_CELL_PARAMETER_NAMES = ("a", "b", "c", "alpha", "beta", "gamma")
_CELL_WARN_RTOL = 0.01  # relative difference past which a cell mismatch warns
_CELL_FAIL_RTOL = 0.05  # relative difference past which a cell mismatch is a hard error


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
    mosaicity: MosaicSmoothed | None


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
    # The (a, b, c, alpha, beta, gamma) cell spec.reciprocal_basis was built from. PETS supplies
    # this on experiment-derived setups; None means direct construction with no PETS record in scope.
    cell_parameters: NDArray[np.float64] | None = None

    @classmethod
    def from_structure(
        cls,
        structure: StructureRecord,
        *,
        cell_parameters: NDArray[np.float64] | None = None,
    ) -> RefinementSetup:
        """Assemble the structure-side refinement inputs from a parsed :class:`StructureRecord`.

        Positions and symmetry are read directly off the CIF's fractional coordinates and symmetry
        operators -- both are metric-independent. ADPs are mapped to the reciprocal ``U*`` frame by
        :func:`diffBloch.params.constrain`, using ``cell_parameters`` as the metric when given
        (PETS's authoritative cell from :func:`setup_datasets`); otherwise the structure's own CIF
        cell, for direct construction with no PETS record in scope. Per-rotation thickness lives
        elsewhere -- on each per-rotation plan (seeded by ``from_experiment``, fitted by
        ``optimize_thickness``) -- because it varies per rotation, not per structure.

        Special-position degrees of freedom are constrained by the site-symmetry projector built
        natively from the structure's symmetry operators (:func:`symmetry_constraints`): an atom
        special position is held on its site under refinement, so it is neither over-parameterized
        nor free to drift off. A general-position atom gets the identity projector (unconstrained).
        """
        positions = torch.tensor(structure.frac_positions, dtype=torch.float64)
        uij_raw, u_iso_raw = _initial_adp_params(structure.adp)
        constraints = symmetry_constraints(structure)
        resolved_cell_parameters = (
            structure.cell_parameters if cell_parameters is None else np.asarray(cell_parameters)
        )
        unit_cell = cell_matrix_from_parameters(resolved_cell_parameters)
        spec = ConstraintSpec(
            position_projection=torch.tensor(constraints.position_projection, dtype=torch.float64),
            position_offset=torch.tensor(constraints.position_offset, dtype=torch.float64),
            occupancies=torch.tensor(structure.occupancies, dtype=torch.float64),
            adp_kind=structure.adp.kind,
            adp_constraints=constraints.adp_constraints,
            reciprocal_basis=torch.tensor(reciprocal_cell(unit_cell), dtype=torch.float64),
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
            cell_parameters=cell_parameters,
        )


@dataclass(frozen=True)
class DatasetSetup:
    """One dataset's seeded geometry: the candidate ``plan`` plus its own measurement context.

    The per-dataset product of :func:`setup_datasets`. ``plan`` holds one
    :class:`~diffBloch.preprocess.plan.CandidatePlan` per *non-ignored* rotation, with
    ``rotation_index`` **file-local** (the rotation's original position within this dataset's own
    PETS file, so ignoring a rotation leaves a gap rather than renumbering later frames);
    :func:`~diffBloch.preprocess.pool.pool` maps these onto the pooled global index space.
    ``integration`` is this file's own rocking-curve semiangle -- pooled datasets may differ (each
    file's recipe runs with its own geometry). ``energy`` is the snapped beam energy
    (:func:`~diffBloch.core.dynamical.snap_to_standard_energy` over the PETS wavelength);
    ``pool`` guards that pooled datasets agree, since the engine solves one experiment at one
    energy. ``n_rotations`` is the *full* pre-ignore rotation count (the pooled offset arithmetic
    and the train/val mask both run over original counts), and ``ignored_rotations`` the sorted
    file-local ignore slice (part of this dataset's checkpoint-lock identity).
    """

    plan: Plan
    integration: IntegrationGeometry
    mosaicity: MosaicSmoothed | None
    energy: float
    n_rotations: int
    ignored_rotations: tuple[int, ...]


def setup_datasets(
    structure: StructureRecord,
    records: Sequence[ExperimentalRecord],
    config: ExperimentConfig,
) -> tuple[RefinementSetup, tuple[DatasetSetup, ...]]:
    """Seed one candidate :class:`~diffBloch.preprocess.plan.Plan` per dataset + the structure side.

    The *initial total construction* of the preprocess pipeline (not ``Plan -> Plan`` -- there is no
    ``Plan`` yet), generalized over one or more PETS files (``inputs.multi_dataset``). The
    structure-side products are dataset-independent and built once, shared by every dataset's plan:
    the structure-factor grid is *derived* from the solve cutoff
    (:func:`~diffBloch.engine.plan.StructureFactorGrid.from_cell_for_beam_cutoff` sizes it to ``2x``
    the cutoff so it spans every coupled ``g - h`` difference -- the same grid *object* rides on
    every per-dataset plan, so their ``Fgb`` support cannot diverge), as are the difference-safe
    seed beams and the :class:`RefinementSetup`.

    Per dataset: the beam energy is derived from that file's PETS wavelength and snapped onto the
    nearest standard TEM voltage when close (PETS records wavelength to only 4-5 significant
    figures, so the exact inverse lands a few hundred eV off 100/200/300 kV rather than on it), and
    the mean-inner-potential ``u0`` follows the energy -- computed once per *distinct* snapped
    energy, since it depends on nothing else that varies between datasets. One
    :class:`CandidatePlan` per rotation carries its crystal orientation matrix (native PETS
    derivation, no side-car file) and the observed pattern for that zone axis.

    ``blochwave.ignore_orientations`` indexes the *pooled* rotation space (files concatenated in
    ``records`` order, original pre-ignore counts); it is validated against the pooled total and
    translated to each dataset's file-local slice here. A dataset whose every rotation is ignored
    raises -- its recipe would have nothing to fit.

    This is intentionally a module-level function rather than a classmethod: it is the single
    documented public boundary of the preprocess pipeline (records + config -> setups), and it
    returns a *composite* of products rather than constructing one domain object. The per-object
    constructors it delegates to follow the classmethod idiom
    (``StructureFactorGrid.from_cell_for_beam_cutoff``, ``CandidatePlan.seed``,
    ``RefinementSetup.from_structure``).
    """
    records = tuple(records)
    if not records:
        raise ValueError("no experimental data: setup_datasets needs at least one PETS record")
    authoritative_cell_parameters = _resolve_authoritative_cell(structure, records)
    authoritative_unit_cell = cell_matrix_from_parameters(authoritative_cell_parameters)
    solve_cutoff = config.blochwave.g_max
    grid = StructureFactorGrid.from_cell_for_beam_cutoff(authoritative_unit_cell, solve_cutoff)
    beam_hkl = seed_beam_hkl(grid, g_max=solve_cutoff)
    refinement_setup = RefinementSetup.from_structure(
        structure, cell_parameters=authoritative_cell_parameters
    )
    absorption = config.blochwave.to_absorption()

    counts = [len(record.zone_axis_ids) for record in records]
    total = sum(counts)
    ignored = set(config.blochwave.to_orientation_selection().ignore_orientations)
    out_of_range = sorted(index for index in ignored if index >= total)
    if out_of_range:
        raise ValueError(
            "ignore_orientations contains indices outside the PETS rotation range "
            f"0..{total - 1}: {out_of_range}"
        )

    u0_by_energy: dict[float, float] = {}
    dataset_refs = (
        tuple(config.inputs.exp_data)
        if isinstance(config.inputs.exp_data, list)
        else (config.inputs.exp_data,)
    )
    datasets: list[DatasetSetup] = []
    offset = 0
    for dataset_index, record in enumerate(records):
        seed_thicknesses = config.sample.thicknesses
        if config.sample.mean_thickness_by_dataset:
            seed_thicknesses = config.sample.seed_thicknesses_for(dataset_refs[dataset_index])
        count = counts[dataset_index]
        local_ignored = tuple(
            sorted(index - offset for index in ignored if offset <= index < offset + count)
        )
        if len(local_ignored) == count:
            raise ValueError(
                f"ignore_orientations excludes every PETS rotation of dataset {dataset_index}"
            )
        integration = IntegrationGeometry(
            semiangle=record.integration_semiangle,
            geometry=record.data_collection_geometry,
        )
        mosaicity = resolve_dataset_mosaicity(
            config.blochwave.mosaicity,
            record,
            config.blochwave.to_rocking_curve(integration),
        )
        energy = snap_to_standard_energy(wavelength2energy(record.wavelength))
        if energy not in u0_by_energy:
            u0_by_energy[energy] = _mean_inner_potential(
                grid, refinement_setup, energy=energy, absorption=absorption
            )
        u0 = u0_by_energy[energy]
        orientations = orientation_matrices(
            record.ub_matrix,
            record.cell_parameters,
            record.alphas,
            record.betas,
            record.omegas,
        )
        local_ignore_set = set(local_ignored)
        plans = tuple(
            CandidatePlan.seed(
                beam_hkl,
                PatternBatch.from_experimental_record(
                    record,
                    zone_axis_id=int(zone_id),
                    rotation_index=local_index,
                ),
                energy=energy,
                thickness=seed_thicknesses,
                orientation=orientations[local_index],
                u0=u0,
            )
            for local_index, zone_id in enumerate(record.zone_axis_ids)
            if local_index not in local_ignore_set
        )
        datasets.append(
            DatasetSetup(
                plan=Plan(structure_factor_grid=grid, orientations=plans),
                integration=integration,
                mosaicity=mosaicity,
                energy=energy,
                n_rotations=count,
                ignored_rotations=local_ignored,
            )
        )
        offset += count
    return refinement_setup, tuple(datasets)


def resolve_dataset_mosaicity(
    enabled: bool,
    record: ExperimentalRecord,
    rocking: RockingCurve,
) -> MosaicSmoothed | None:
    """Resolve PETS apparent mosaicity to the applied tilt reduction, or ``None`` when disabled."""
    if not enabled:
        return None
    source = record.source_path if record.source_path is not None else "<experimental data>"
    mosaicity_degrees = record.mosaicity_degrees
    if mosaicity_degrees is None:
        raise ValueError(
            f"blochwave.mosaicity=true requires a 'mosaicity:' value in {source}; "
            "add the PETS apparent mosaicity or set blochwave.mosaicity: false"
        )
    if not np.isfinite(mosaicity_degrees) or mosaicity_degrees < 0.0:
        raise ValueError(
            f"blochwave.mosaicity=true requires a finite, non-negative PETS mosaicity in "
            f"{source}; got {mosaicity_degrees!r}"
        )
    if mosaicity_degrees == 0.0:
        return None
    degrees_per_sample = 2.0 * rocking.integration.semiangle / rocking.sampling
    samples = round(mosaicity_degrees / degrees_per_sample)
    if samples <= 1:
        return None
    if samples > rocking.sampling:
        raise ValueError(
            f"blochwave.mosaicity=true resolves PETS mosaicity {mosaicity_degrees:g} degrees "
            f"in {source} to a smoothing span of {samples} samples, which exceeds the "
            f"{rocking.sampling} sampled rocking-curve tilts across +/-"
            f"{rocking.integration.semiangle:g} degrees. Increase the PETS integration "
            "coverage or set blochwave.mosaicity: false."
        )
    return MosaicSmoothed(samples=samples)


def from_experiment(
    structure: StructureRecord,
    experimental_data: ExperimentalRecord,
    config: ExperimentConfig,
) -> ExperimentSetup:
    """Construct the geometry ``Plan`` pair + structure ``RefinementSetup`` from parsed inputs.

    The single-dataset public boundary, kept for API users and the inference/e2e paths: it is
    :func:`setup_datasets` over one record, with the train/validation split applied on top.
    Rotations split into ``train`` / ``validation`` plans sharing the grid; split membership is
    defined on the original PETS order (ignoring a rotation must not renumber later frames and
    silently move them between train and validation). The app's preprocess spine does not come
    through here -- it runs :func:`setup_datasets` per dataset and applies the split after
    :func:`~diffBloch.preprocess.pool.pool`.
    """
    refinement_setup, (dataset,) = setup_datasets(structure, (experimental_data,), config)
    grid = dataset.plan.structure_factor_grid
    validation = validation_mask(dataset.n_rotations, config.refinement.split)
    train_orientations = tuple(
        plan for plan in dataset.plan.orientations if not validation[plan.pattern.rotation_index]
    )
    val_orientations = tuple(
        plan for plan in dataset.plan.orientations if validation[plan.pattern.rotation_index]
    )
    return ExperimentSetup(
        plans=PlanSplit(
            train=Plan(structure_factor_grid=grid, orientations=train_orientations),
            validation=Plan(structure_factor_grid=grid, orientations=val_orientations),
        ),
        refinement=refinement_setup,
        integration=dataset.integration,
        mosaicity=dataset.mosaicity,
    )


def _resolve_authoritative_cell(
    structure: StructureRecord, records: Sequence[ExperimentalRecord]
) -> NDArray[np.float64]:
    """Resolve the PETS cell used for all shared simulation/refinement geometry.

    PETS's own cell -- not the structure CIF's -- drives the structure-factor grid, reciprocal
    basis, cell volume, ADP ``U*`` frame conversion, and beam geometry. The CIF supplies atomic
    content: fractional coordinates, atom types, occupancies, ADPs, and symmetry operators. For a
    pooled experiment, the first ``inputs.exp_data`` file is the anchor cell every other dataset is
    checked against; each dataset still derives its orientation from its own UB matrix and PETS cell.

    The CIF cell and every later PETS cell are checked against the authoritative cell. A mismatch
    over 1% on any cell parameter logs a warning; over 5% raises. The authoritative cell is never
    adjusted to fit the other files.
    """
    authoritative_cell = np.asarray(records[0].cell_parameters, dtype=np.float64)
    authoritative_label = (
        str(records[0].source_path) if records[0].source_path is not None else "<experimental data>"
    )
    _check_cell_agreement(
        authoritative_cell, authoritative_label, structure.cell_parameters, "structure CIF"
    )
    for record in records[1:]:
        other_label = (
            str(record.source_path) if record.source_path is not None else "<experimental data>"
        )
        _check_cell_agreement(
            authoritative_cell,
            authoritative_label,
            np.asarray(record.cell_parameters, dtype=np.float64),
            other_label,
        )
    return authoritative_cell


def _check_cell_agreement(
    authoritative_cell: NDArray[np.float64],
    authoritative_label: str,
    other_cell: NDArray[np.float64],
    other_label: str,
) -> None:
    """Compare ``other_cell`` against the authoritative cell: warn past 1%, raise past 5%."""
    relative_diff = np.abs(other_cell - authoritative_cell) / np.abs(authoritative_cell)

    def offending(threshold: float) -> str:
        return ", ".join(
            f"{name}: {other_label}={other_value:.6g} vs {authoritative_label}="
            f"{authoritative_value:.6g} ({100.0 * diff:.2f}% off)"
            for name, other_value, authoritative_value, diff in zip(
                _CELL_PARAMETER_NAMES, other_cell, authoritative_cell, relative_diff, strict=True
            )
            if diff > threshold
        )

    fail_mask = relative_diff > _CELL_FAIL_RTOL
    if np.any(fail_mask):
        raise ValueError(
            f"{other_label} and {authoritative_label} unit cells disagree by more than 5% on "
            f"{int(np.count_nonzero(fail_mask))} parameter(s): {offending(_CELL_FAIL_RTOL)} -- "
            "refusing to continue; check that these files describe the same crystal setting."
        )
    warn_mask = relative_diff > _CELL_WARN_RTOL
    if np.any(warn_mask):
        _log.warning(
            "%s and %s unit cells disagree by more than 1%% on %d parameter(s): %s -- %s is "
            "authoritative and overrides %s for the simulation geometry.",
            other_label,
            authoritative_label,
            int(np.count_nonzero(warn_mask)),
            offending(_CELL_WARN_RTOL),
            authoritative_label,
            other_label,
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


def validation_mask(n_rotations: int, split: DataSplitConfig) -> NDArray[np.bool_]:
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
