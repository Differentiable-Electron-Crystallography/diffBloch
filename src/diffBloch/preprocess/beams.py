"""``select_beams``: prune each orientation's beams to its active set (Klar et al. 2023 filter).

A ``Plan -> Plan`` step that re-picks every :class:`~diffBloch.engine.plan.OrientationPlan`'s active
``beam_hkl`` using the relative-/minimum-excitation-error criterion of the SI of Klar et al. (2023),
then rebuilds its ``BeamPlan`` + ``AlignmentPlan`` against the shared grid (``pattern`` unchanged).
This is the faithful per-orientation selection that replaces the orientation-independent
``g_max_refine`` seed laid down by ``from_experiment``.

Divergence from ``diffBloch_private`` (recorded in ``DIVERGENCE.md``): the
private ``diffraction_dataset.filter_hkls`` builds ``sg_max`` from ``norm(k[:, 1:])`` -- columns
``(y, z)`` -- while its ``excitation_errors`` fixes the beam along ``-z`` (so the transverse plane
is ``(x, y)``). Mixing the along-beam component ``g_z`` into the transverse distance over-weights
HOLZ reflections. We use the geometrically consistent transverse component ``(g_x, g_y)`` instead.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from diffBloch.core.crystal import orientation_basis
from diffBloch.core.dynamical import excitation_errors
from diffBloch.core.reciprocal import g_vectors
from diffBloch.engine.plan import OrientationPlan, ScatteringGrid
from diffBloch.preprocess.pipeline import PlanStep
from diffBloch.preprocess.plan import Plan

__all__ = ["klar_beam_mask", "select_beams"]


def select_beams(*, rsg: float, dsg: float, semiangle: float) -> PlanStep:
    """Return a ``Plan -> Plan`` step that prunes each orientation to its Klar active beam set.

    For every orientation the beams are re-selected by :func:`klar_beam_mask` against that
    orientation's lab-frame ``g`` (derived from its stored ``orientation`` and the grid cell), and
    the plan is rebuilt via the self-describing ``OrientationPlan.build`` path. ``rsg`` (relative
    excitation-error cutoff), ``dsg`` (minimum excitation-error margin), and ``semiangle`` (the
    integration semi-angle in degrees) come from ``NumericsConfig``. The observed ``pattern`` is
    untouched; the rebuilt ``AlignmentPlan`` re-bridges it to the pruned beam set.

    The 000 transmitted beam is always retained regardless of the filter: ``BeamPlan`` anchors
    ``psi0`` on ``hkl == 000``, and 000 has ``g = 0`` so its ``sg_max = 0`` would otherwise reject
    it. The grid is sized from ``g_max`` and beams stay within ``g_max_refine < g_max``, so the
    ``Fgb`` difference support remains valid after reselection.
    """

    def step(plan: Plan) -> Plan:
        cell = np.asarray(plan.grid.cell)
        orientations = tuple(
            _reselect(plan.grid, cell, op, rsg=rsg, dsg=dsg, semiangle=semiangle)
            for op in plan.orientations
        )
        return replace(plan, orientations=orientations)

    return step


def _reselect(
    grid: ScatteringGrid,
    cell: NDArray[np.float64],
    op: OrientationPlan,
    *,
    rsg: float,
    dsg: float,
    semiangle: float,
) -> OrientationPlan:
    beam_hkl = np.asarray(op.beam_hkl, dtype=np.int64)
    basis = orientation_basis(cell, np.asarray(op.orientation))
    g = g_vectors(beam_hkl, basis)
    keep = klar_beam_mask(g, energy=op.energy, u0=op.u0, rsg=rsg, dsg=dsg, semiangle=semiangle)
    keep |= (beam_hkl == 0).all(axis=1)  # 000 anchors psi0; always retained
    return OrientationPlan.build(
        grid,
        beam_hkl[keep],
        op.pattern,
        energy=op.energy,
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
) -> NDArray[np.bool_]:
    """Boolean keep-mask for reflections ``g`` ``(N, 3)`` under the Klar (2023) rsg/dsg filter.

    Each reflection's excitation error ``|Sg|`` (Spence & Zuo, via :func:`excitation_errors`) is
    compared against ``sg_max = |g_transverse| * deg2rad(semiangle)`` -- the excitation-error spread
    swept as the beam tilts over the integration cone -- where ``g_transverse = (g_x, g_y)`` is the
    component perpendicular to the ``-z`` beam. A reflection is kept when both
    ``|Sg| / sg_max < rsg`` (relative excitation error small) and ``sg_max - |Sg| > dsg`` (a minimum
    absolute margin). Reflections with ``sg_max = 0`` (on the optic axis) fail the relative test and
    are dropped; the 000-beam retention required by the Bloch system is handled by the caller.
    """
    g_array = np.asarray(g, dtype=np.float64)
    if g_array.ndim != 2 or g_array.shape[1] != 3:
        raise ValueError("g must have shape (N, 3)")
    sg = np.abs(excitation_errors(g_array, energy, u0=u0))
    sg_max = np.linalg.norm(g_array[:, :2], axis=1) * np.deg2rad(semiangle)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error = np.where(sg_max > 0.0, sg / sg_max, np.inf)
    mask: NDArray[np.bool_] = (rel_error < rsg) & (sg_max - sg > dsg)
    return mask
