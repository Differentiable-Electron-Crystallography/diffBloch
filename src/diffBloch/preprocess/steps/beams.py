"""``select_beams``: prune each orientation's beams to its active set (Klar et al. 2023 filter).

A ``Plan -> Plan`` step that re-picks every :class:`~diffBloch.engine.plan.OrientationPlan`'s active
``beam_hkl`` using the relative-/minimum-excitation-error criterion of the SI of Klar et al. (2023),
then rebuilds its ``BeamPlan`` + ``AlignmentPlan`` against the shared grid (``pattern`` unchanged).
This is the per-orientation selection that replaces the orientation-independent
``g_max_refine`` seed laid down by ``from_experiment``.

``sg_max`` is the excitation-error span a reflection sweeps *during the actual integration*, so its
transverse lever arm is set by the tilt geometry (``BeamSelection.geometry``), which must match the
integrator's (:class:`~diffBloch.specs.RockingCurve`). For ``continuous_rotation`` the crystal rocks
about the goniometer axis (``x`` in the PETS frame; ``rocking_curve_tilts`` builds ``R_x``), so the
swept excitation error has amplitude ``|(g_y, g_z)|`` -- the distance from the rock axis -- and a
reflection *on* that axis (``g_y = g_z = 0``) never sweeps and is correctly dropped. For
``precession`` (an isotropic cone about the ``-z`` beam) the lever arm is instead ``|(g_x, g_y)|``,
the distance from the beam. The frame convention has the beam along ``-z`` and the rock axis along
``x``.
"""

from __future__ import annotations

from _thread import LockType
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Lock
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from diffBloch.core.crystal import orientation_basis
from diffBloch.core.dynamical import (
    StructureFactorGather,
    build_structure_factor_gather,
    excitation_errors,
    grid_source_indices,
)
from diffBloch.core.products import PLAIN_SUM, MosaicSmoothed, TiltReduction
from diffBloch.core.reciprocal import g_vectors
from diffBloch.engine.plan import CoupledOrientationPlan, OrientationPlan, StructureFactorGrid
from diffBloch.preprocess.coupling import build_coupling_segments
from diffBloch.preprocess.experiment import seed_beam_hkl
from diffBloch.preprocess.orientation import rocking_curve_tilts
from diffBloch.preprocess.pipeline import PlanStep, as_step
from diffBloch.preprocess.plan import CandidatePlan, Plan, require_candidate_plans
from diffBloch.specs import (
    BeamSelection,
    Mosaicity,
    PerTiltCoupling,
    RockingCurve,
    UnionCoupling,
    assert_grid_covers_coupling,
)

__all__ = ["build_orientation_plans", "klar_beam_mask", "reseed_pool", "select_beams"]


def select_beams(selection: BeamSelection) -> PlanStep:
    """Return a ``Plan -> Plan`` step that prunes each candidate to its Klar active beam set.

    A *source-level* prune on the :class:`~diffBloch.preprocess.plan.CandidatePlan` phase: for every
    orientation the candidate ``beam_hkl`` is re-selected by :func:`klar_beam_mask` against that
    orientation's lab-frame ``g`` (derived from its stored ``orientation`` and the grid cell),
    keeping only the active set. No geometry is built here -- the structure-factor gather is built
    later, over the pruned set, by :func:`build_orientation_plans`. ``selection`` is a pre-validated
    :class:`~diffBloch.specs.BeamSelection` (``rsg`` relative excitation-error cutoff, ``dsg``
    minimum margin, ``integration_semiangle`` in degrees); invalid cutoffs are unrepresentable, so
    this step never re-validates. The observed ``pattern`` is untouched.

    The 000 transmitted beam is retained whenever present (the ``from_experiment`` seed always
    includes it): ``BeamPlan`` anchors ``psi0`` on ``hkl == 000``, and 000 has ``g = 0`` so its
    ``sg_max = 0`` would otherwise reject it. Beams stay within ``g_max_refine < g_max``, so the
    ``Fgb`` difference support remains valid once ``build_orientation_plans`` runs.
    """

    def run(plan: Plan) -> Plan:
        cell = np.asarray(plan.structure_factor_grid.cell)
        candidates = tuple(_reselect(cell, cp, selection) for cp in require_candidate_plans(plan))
        return replace(plan, orientations=candidates)

    return as_step("select_beams", selection, run)


def build_orientation_plans(
    rocking: RockingCurve | None = None,
    mosaicity: Mosaicity | None = None,
    *,
    coupling: UnionCoupling | PerTiltCoupling | None = None,
    scoring_selection: BeamSelection | None = None,
    workers: int = 1,
) -> PlanStep:
    """Build each candidate's final tilted Bloch geometry and intensity reduction.

    The single *build* boundary of the preprocess pipeline: it materialises each orientation's
    structure-factor gather (the dominant cost) over its beam set via ``OrientationPlan.build``, and
    the rebuilt ``AlignmentPlan`` re-bridges the simulator output to the observed ``pattern``. A
    custom pipeline may compose it after :func:`select_beams`; the default coupled path instead
    derives the SOLVE beams directly from ``g_max``/``sg_max`` and the explicit sub-tilts. The
    engine consumes only these built plans; a
    :class:`~diffBloch.preprocess.plan.CandidatePlan` has no ``beam_plans`` and is unsolvable by
    construction.

    When ``rocking`` is supplied, the builder directly creates its complete sub-tilt geometry
    instead of first building a temporary central-orientation plan and rebuilding it later.
    ``mosaicity`` selects the reduction applied to those sub-tilt intensities and therefore requires
    ``rocking``. When ``coupling`` is supplied, each segment's beam set is selected from the full
    support grid by ``|g| < g_max`` and ``|Sg| < sg_max`` at its boundary tilts, then the ordinary
    alignment intersects the resulting simulator HKLs with the PETS observations.
    ``scoring_selection`` optionally applies the former Klar ``rsg``/``dsg``/semiangle filter to
    the candidate scoring pool before that intersection; it does not alter the coupled SOLVE beams.
    ``workers`` fans independent rotation builds over threads while preserving input order. It is
    execution-only and therefore intentionally absent from the step's provenance record.
    Omitting ``coupling`` preserves the simple builder used by focused APIs/tests.
    """
    if mosaicity is not None and rocking is None:
        raise ValueError("mosaicity requires rocking-curve geometry")
    if coupling is not None and rocking is None:
        raise ValueError("coupling requires rocking-curve geometry")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    tilts = (
        None
        if rocking is None
        else rocking_curve_tilts(
            rocking.integration.semiangle,
            rocking.sampling,
            geometry=rocking.integration.geometry,
        )
    )
    if mosaicity is not None:
        assert rocking is not None  # narrowed by the construction guard above
        if mosaicity.window > rocking.sampling:
            raise ValueError(
                f"mosaicity window {mosaicity.window} exceeds the {rocking.sampling} "
                "rocking-curve tilts"
            )
    reduction = PLAIN_SUM if mosaicity is None else MosaicSmoothed(mosaicity.window)

    def run(plan: Plan) -> Plan:
        candidates = require_candidate_plans(plan)
        built: tuple[OrientationPlan | CoupledOrientationPlan, ...]
        if coupling is None:
            built = tuple(
                OrientationPlan.build(
                    plan.structure_factor_grid,
                    np.asarray(cp.beam_hkl),
                    cp.pattern,
                    energy=cp.energy,
                    thickness=cp.thickness,
                    u0=cp.u0,
                    orientation=cp.orientation,
                    tilts=tilts,
                    tilt_reduction=reduction,
                )
                for cp in candidates
            )
        else:
            assert tilts is not None
            grid = plan.structure_factor_grid
            assert_grid_covers_coupling(coupling, grid.g_max)
            structure_factor_hkl = np.asarray(grid.structure_factor_hkl, dtype=np.int64)
            source = grid_source_indices(structure_factor_hkl, grid.gpts)
            gather_cache: dict[bytes, StructureFactorGather] = {}
            gather_cache_lock = Lock()

            def build_one(candidate: CandidatePlan) -> CoupledOrientationPlan:
                return _build_coupled_candidate(
                    grid,
                    candidate,
                    tilts,
                    reduction,
                    coupling,
                    scoring_selection=scoring_selection,
                    structure_factor_hkl=structure_factor_hkl,
                    structure_factor_indices=source,
                    gather_cache=gather_cache,
                    gather_cache_lock=gather_cache_lock,
                )

            if workers == 1:
                built = tuple(build_one(candidate) for candidate in candidates)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    built = tuple(pool.map(build_one, candidates))
        return replace(plan, orientations=built)

    return as_step(
        "build_orientation_plans",
        (
            None
            if rocking is None and coupling is None
            else {
                "rocking": rocking,
                "mosaicity": mosaicity,
                "coupling": coupling,
                "scoring_selection": scoring_selection,
            }
        ),
        run,
    )


def _build_coupled_candidate(
    grid: StructureFactorGrid,
    candidate: CandidatePlan,
    tilts: NDArray[np.float64],
    reduction: TiltReduction,
    coupling: UnionCoupling | PerTiltCoupling,
    *,
    scoring_selection: BeamSelection | None,
    structure_factor_hkl: NDArray[np.int64],
    structure_factor_indices: NDArray[np.int64],
    gather_cache: dict[bytes, StructureFactorGather],
    gather_cache_lock: LockType,
) -> CoupledOrientationPlan:
    """Build one simulator plan from geometric coupling only; alignment handles observations."""
    segments = build_coupling_segments(
        coupling,
        np.asarray(grid.structure_factor_hkl, dtype=np.int64),
        cell=np.asarray(grid.cell, dtype=np.float64),
        orientation=np.asarray(candidate.orientation, dtype=np.float64),
        tilts=tilts,
        energy=candidate.energy,
        u0=candidate.u0,
    )
    scored_hkl = (
        None
        if scoring_selection is None
        else np.asarray(
            _reselect(np.asarray(grid.cell), candidate, scoring_selection).beam_hkl,
            dtype=np.int64,
        )
    )
    gathers: list[StructureFactorGather] = []
    for segment in segments:
        key = np.ascontiguousarray(segment.union_hkl).tobytes()
        with gather_cache_lock:
            gather = gather_cache.get(key)
        if gather is None:
            candidate_gather = build_structure_factor_gather(
                structure_factor_hkl,
                segment.union_hkl,
                grid.gpts,
                validate=False,
                structure_factor_indices=structure_factor_indices,
            )
            with gather_cache_lock:
                gather = gather_cache.setdefault(key, candidate_gather)
        gathers.append(gather)
    return CoupledOrientationPlan.build(
        grid,
        [(segment.union_hkl, segment.covered_tilt_indices) for segment in segments],
        candidate.pattern,
        energy=candidate.energy,
        thickness=candidate.thickness,
        u0=candidate.u0,
        orientation=candidate.orientation,
        tilts=tilts,
        tilt_reduction=reduction,
        scored_hkl=scored_hkl,
        gathers=gathers,
    )


def reseed_pool(seed: Plan, selection: BeamSelection, *, g_max_refine: float) -> Plan:
    """Re-seed each orientation from the grid at ``g_max_refine``, then apply the Klar window.

    The shared build step for the pool levers
    (:func:`~diffBloch.preprocess.steps.convergence.converge_pool`,
    :func:`~diffBloch.preprocess.steps.coverage.cover_pool`) and the convergence driver: each
    orientation's *candidate* reflections are re-seeded from the shared grid at ``g_max_refine``
    (:func:`~diffBloch.preprocess.experiment.seed_beam_hkl`), then :func:`select_beams` applies the
    fixed Klar window and :func:`build_orientation_plans` builds the (small) active set -- so the
    returned plan is solvable, with active set
    ``seed(g_max_refine) intersect Klar-window(selection)``. Re-seeding from the shared grid (not a
    previous pruned ``Plan``) lets a widening pool recover beams a narrower one clipped.

    The pool stays inside the existing ``Fgb`` difference support while
    ``2 * g_max_refine <= grid.g_max``; a candidate past that raises rather than silently truncating
    (dependent grid resizing is unimplemented).
    """
    if 2.0 * g_max_refine > seed.structure_factor_grid.g_max:
        raise ValueError(
            f"g_max_refine={g_max_refine:.4g} exceeds the grid's beam-difference support "
            f"(g_max={seed.structure_factor_grid.g_max:.4g}); "
            "dependent grid resizing is not implemented"
        )
    beam_hkl = seed_beam_hkl(seed.structure_factor_grid, g_max_refine=g_max_refine)
    candidates = tuple(
        CandidatePlan.seed(
            beam_hkl,
            op.pattern,
            energy=op.energy,
            thickness=op.thickness,
            u0=op.u0,
            orientation=op.orientation,
        )
        for op in seed.orientations
    )
    reseeded = replace(seed, orientations=candidates)
    return build_orientation_plans()(select_beams(selection)(reseeded))


def _reselect(
    cell: NDArray[np.float64],
    cp: CandidatePlan,
    selection: BeamSelection,
) -> CandidatePlan:
    beam_hkl = np.asarray(cp.beam_hkl, dtype=np.int64)
    basis = orientation_basis(cell, np.asarray(cp.orientation))
    g = g_vectors(beam_hkl, basis)
    keep = klar_beam_mask(
        g,
        energy=cp.energy,
        u0=cp.u0,
        rsg=selection.rsg,
        dsg=selection.dsg,
        semiangle=selection.integration.semiangle,
        geometry=selection.integration.geometry,
    )
    keep |= (beam_hkl == 0).all(axis=1)  # 000 anchors psi0; retained when present
    return replace(cp, beam_hkl=beam_hkl[keep])


def klar_beam_mask(
    g: NDArray[np.float64],
    *,
    energy: float,
    u0: float = 0.0,
    rsg: float,
    dsg: float,
    semiangle: float,
    geometry: Literal["continuous_rotation", "precession"] = "continuous_rotation",
) -> NDArray[np.bool_]:
    """Boolean keep-mask for reflections ``g`` ``(N, 3)`` under the Klar (2023) rsg/dsg filter.

    Each reflection's excitation error ``|Sg|`` (Spence & Zuo, via :func:`excitation_errors`; beam
    along ``-z``) is compared against ``sg_max``, the excitation-error span it sweeps during
    integration: ``sg_max = |g_lever| * deg2rad(semiangle)``. The lever arm depends on ``geometry``
    -- for ``continuous_rotation`` the rock is about the goniometer ``x`` axis, so
    ``g_lever = (g_y, g_z)`` (distance from the rock axis); for ``precession`` (cone about the beam)
    it is ``g_lever = (g_x, g_y)`` (distance from the ``-z`` beam). A reflection is kept when both
    ``|Sg| / sg_max < rsg`` (relative excitation error small) and ``sg_max - |Sg| > dsg`` (a minimum
    absolute margin). Reflections with ``sg_max = 0`` (on the rock axis, resp. optic axis) fail the
    relative test and are dropped -- they never sweep through the Ewald sphere; the 000-beam
    retention required by the Bloch system is handled by the caller.
    """
    g_array = np.asarray(g, dtype=np.float64)
    if g_array.ndim != 2 or g_array.shape[1] != 3:
        raise ValueError("g must have shape (N, 3)")
    if geometry == "continuous_rotation":
        g_lever = g_array[:, 1:]  # (g_y, g_z): distance from the x goniometer rock axis
    elif geometry == "precession":
        g_lever = g_array[:, :2]  # (g_x, g_y): distance from the -z beam
    else:
        raise ValueError("geometry must be 'continuous_rotation' or 'precession'")
    sg = np.abs(excitation_errors(g_array, energy, u0=u0))
    sg_max = np.linalg.norm(g_lever, axis=1) * np.deg2rad(semiangle)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error = np.where(sg_max > 0.0, sg / sg_max, np.inf)
    mask: NDArray[np.bool_] = (rel_error < rsg) & (sg_max - sg > dsg)
    return mask
