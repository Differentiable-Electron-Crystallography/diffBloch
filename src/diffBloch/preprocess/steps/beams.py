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
from diffBloch.engine.plan import OrientationPlan, ScatteringGrid
from diffBloch.preprocess.experiment import seed_beam_hkl
from diffBloch.preprocess.pipeline import PlanStep
from diffBloch.preprocess.plan import Plan
from diffBloch.specs import BeamSelection

__all__ = ["klar_beam_mask", "reseed_pool", "select_beams"]


def select_beams(selection: BeamSelection) -> PlanStep:
    """Return a ``Plan -> Plan`` step that prunes each orientation to its Klar active beam set.

    For every orientation the beams are re-selected by :func:`klar_beam_mask` against that
    orientation's lab-frame ``g`` (derived from its stored ``orientation`` and the grid cell), and
    the plan is rebuilt via the self-describing ``OrientationPlan.build`` path. ``selection`` is a
    pre-validated :class:`~diffBloch.specs.BeamSelection` (``rsg`` relative excitation-error cutoff,
    ``dsg`` minimum margin, ``integration_semiangle`` in degrees); invalid cutoffs are
    unrepresentable, so this step never re-validates. The observed ``pattern`` is untouched; the
    rebuilt ``AlignmentPlan`` re-bridges it to the pruned beam set.

    The 000 transmitted beam is retained whenever it is present (the ``from_experiment`` seed always
    includes it): ``BeamPlan`` anchors ``psi0`` on ``hkl == 000``, and 000 has ``g = 0`` so its
    ``sg_max = 0`` would otherwise reject it. ``select_beams`` retains 000 rather than synthesising
    it -- a 000-less input set would still fail ``build_beam_plan``'s ``psi0`` anchor. The grid is
    sized from ``g_max`` and beams stay within ``g_max_refine < g_max``, so the ``Fgb`` difference
    support remains valid after reselection.
    """

    def run(plan: Plan) -> Plan:
        cell = np.asarray(plan.grid.cell)
        orientations = tuple(_reselect(plan.grid, cell, op, selection) for op in plan.orientations)
        return replace(plan, orientations=orientations)

    return run


def reseed_pool(seed: Plan, selection: BeamSelection, *, g_max_refine: float) -> Plan:
    """Re-seed each orientation from the grid at ``g_max_refine``, then apply the Klar window.

    The shared build step for the pool levers
    (:func:`~diffBloch.preprocess.steps.convergence.converge_pool`,
    :func:`~diffBloch.preprocess.steps.coverage.cover_pool`) and the convergence driver: each
    orientation's *candidate* reflections are re-seeded from the shared grid at ``g_max_refine``
    (:func:`~diffBloch.preprocess.experiment.seed_beam_hkl`), every
    :class:`~diffBloch.engine.plan.OrientationPlan` is rebuilt on that seed, then
    :func:`select_beams`
    applies the fixed Klar window -- so the active set is
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
    reseeded = tuple(
        OrientationPlan.build(
            seed.grid,
            beam_hkl,
            op.pattern,
            energy=op.energy,
            thickness=op.thickness,
            u0=op.u0,
            orientation=op.orientation,
        )
        for op in seed.orientations
    )
    return select_beams(selection)(replace(seed, orientations=reseeded))


def _reselect(
    grid: ScatteringGrid,
    cell: NDArray[np.float64],
    op: OrientationPlan,
    selection: BeamSelection,
) -> OrientationPlan:
    beam_hkl = np.asarray(op.beam_hkl, dtype=np.int64)
    basis = orientation_basis(cell, np.asarray(op.orientation))
    g = g_vectors(beam_hkl, basis)
    keep = klar_beam_mask(
        g,
        energy=op.energy,
        u0=op.u0,
        rsg=selection.rsg,
        dsg=selection.dsg,
        semiangle=selection.integration_semiangle,
        geometry=selection.geometry,
    )
    keep |= (beam_hkl == 0).all(axis=1)  # 000 anchors psi0; retained when present
    return OrientationPlan.build(
        grid,
        beam_hkl[keep],
        op.pattern,
        energy=op.energy,
        thickness=op.thickness,
        u0=op.u0,
        orientation=op.orientation,
    )


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
