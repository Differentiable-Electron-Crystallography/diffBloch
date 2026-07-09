"""``select_beams``: prune each orientation's beams to its active set (Klar et al. 2023 filter).

A ``Plan -> Plan`` step that re-picks every :class:`~diffBloch.engine.plan.OrientationPlan`'s active
``beam_hkl`` using the relative-/minimum-excitation-error criterion of the SI of Klar et al. (2023),
then rebuilds its ``BeamPlan`` + ``AlignmentPlan`` against the shared grid (``pattern`` unchanged).
This is the faithful per-orientation selection that replaces the orientation-independent
``g_max_refine`` seed laid down by ``from_experiment``.

``sg_max`` is the excitation-error span a reflection sweeps *during the actual integration*, so its
transverse lever arm is set by the tilt geometry (``BeamSelection.geometry``), which must match the
integrator's (:class:`~diffBloch.specs.RockingCurve`). For ``continuous_rotation`` the crystal rocks
about the goniometer axis (``x`` in the PETS frame; ``rocking_curve_tilts`` builds ``R_x``), so the
swept excitation error has amplitude ``|(g_y, g_z)|`` -- the distance from the rock axis -- and a
reflection *on* that axis (``g_y = g_z = 0``) never sweeps and is correctly dropped. For
``precession`` (an isotropic cone about the ``-z`` beam) the lever arm is instead ``|(g_x, g_y)|``,
the distance from the beam. This matches ``diffBloch_private`` ``filter_hkls`` (``norm(k[:, 1:])``
for its continuous-rotation data), whose beam is ``-z`` and rock axis ``x`` (same frame as ours).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from diffBloch.core.crystal import orientation_basis
from diffBloch.core.dynamical import excitation_errors
from diffBloch.core.reciprocal import g_vectors
from diffBloch.engine.plan import OrientationPlan
from diffBloch.preprocess.experiment import seed_beam_hkl
from diffBloch.preprocess.pipeline import PlanStep, as_step
from diffBloch.preprocess.plan import CandidatePlan, Plan, require_candidate_plans
from diffBloch.specs import BeamSelection

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
        cell = np.asarray(plan.grid.cell)
        candidates = tuple(_reselect(cell, cp, selection) for cp in require_candidate_plans(plan))
        return replace(plan, orientations=candidates)

    return as_step("select_beams", selection, run)


def build_orientation_plans() -> PlanStep:
    """Return a ``Plan -> Plan`` step that builds each candidate into a solvable orientation plan.

    The single *build* boundary of the preprocess pipeline: it materialises each orientation's
    structure-factor gather (the dominant cost) over its beam set via ``OrientationPlan.build``, and
    the rebuilt ``AlignmentPlan`` re-bridges the observed ``pattern`` to that set. Composed *after*
    :func:`select_beams`, so the gather is built once over the small Klar-active set -- never the
    full candidate pool ``from_experiment`` lays down (which is intractable for a large cell). The
    engine consumes only these built plans; a :class:`~diffBloch.preprocess.plan.CandidatePlan` has
    no ``beam_plans`` and is unsolvable by construction.
    """

    def run(plan: Plan) -> Plan:
        built = tuple(
            OrientationPlan.build(
                plan.grid,
                np.asarray(cp.beam_hkl),
                cp.pattern,
                energy=cp.energy,
                thickness=cp.thickness,
                u0=cp.u0,
                orientation=cp.orientation,
            )
            for cp in require_candidate_plans(plan)
        )
        return replace(plan, orientations=built)

    return as_step("build_orientation_plans", None, run)


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
    if 2.0 * g_max_refine > seed.grid.g_max:
        raise ValueError(
            f"g_max_refine={g_max_refine:.4g} exceeds the grid's beam-difference support "
            f"(g_max={seed.grid.g_max:.4g}); dependent grid resizing is not implemented"
        )
    beam_hkl = seed_beam_hkl(seed.grid, g_max_refine=g_max_refine)
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
