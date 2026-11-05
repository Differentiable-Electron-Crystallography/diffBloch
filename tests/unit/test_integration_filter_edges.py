"""The first and last frame of a dataset keep crossings only up to half a frame step on their open side."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from diffBloch.core.crystal import orientation_basis
from diffBloch.core.reciprocal import g_vectors
from diffBloch.preprocess.integrated import bragg_crossing_angles
from diffBloch.preprocess.plan import require_built_plans
from diffBloch.preprocess.serialize import read_plan
from diffBloch.preprocess.steps.integrated_scoring import _inside_outer_limit
from diffBloch.specs import IntegrationGeometry, RockingCurve

_PLAN = (
    Path(__file__).parent.parent
    / "fixtures"
    / "quartz_anchor"
    / "reproducibility"
    / "plan.exp_data.npz"
)


def _setup() -> tuple:
    plan = read_plan(_PLAN)
    ops = require_built_plans(plan)
    cell = np.asarray(plan.structure_factor_grid.cell)
    step = float(ops[1].pattern.alpha_degrees) - float(ops[0].pattern.alpha_degrees)
    rocking = RockingCurve(integration=IntegrationGeometry(semiangle=abs(step)), sampling=5)
    return ops, cell, step, rocking


def _crossings(op, cell, rocking):
    hkl = op.pattern.hkl.cpu().numpy().astype(np.int64)
    g = g_vectors(hkl, orientation_basis(cell, np.asarray(op.orientation)))
    semi = rocking.integration.semiangle
    return g, bragg_crossing_angles(g, energy=op.energy, u0=op.u0, semiangle=semi)


def test_edge_frames_drop_only_the_open_side_beyond_half_a_step() -> None:
    ops, cell, _, rocking = _setup()
    for index, neighbour in ((0, 1), (len(ops) - 1, len(ops) - 2)):
        local = float(ops[neighbour].pattern.alpha_degrees) - float(
            ops[index].pattern.alpha_degrees
        )
        g, crossing = _crossings(ops[index], cell, rocking)
        keep = _inside_outer_limit(ops, index, g, ops[index], rocking)
        inward = np.sign(local) * crossing
        finite = np.isfinite(crossing)
        assert (~keep[finite] == (inward[finite] < -abs(local) / 2)).all()
        assert (~keep).any() and keep[finite].any()
        assert keep[~finite].all()


def test_interior_frames_are_unrestricted() -> None:
    ops, cell, _, rocking = _setup()
    g, _ = _crossings(ops[1], cell, rocking)
    assert _inside_outer_limit(ops, 1, g, ops[1], rocking).all()
