"""Shared fast synthetic silicon system for the convergence / coverage / driver tests.

One silicon atom on a cubic 5 A cell -- no heavy fixture simulation. ``seed_system`` builds a
deliberately rich seed ``Plan`` (the full ``|h|,|k|,|l| <= 3`` hkl cube on a ``g_max = 2.2`` grid)
so that widening a beam knob admits beams progressively and then saturates, which is exactly the
shape the convergence/coverage sweeps need to exercise their stopping rules.

A plain helper module (not a conftest, not collected): test modules import from here instead of
from each other, so no test file depends on another test file's internals.
"""

from __future__ import annotations

import numpy as np
import torch

from diffBloch.core.products import PatternBatch
from diffBloch.core.symmetry import build_asu_expansion_plan
from diffBloch.engine import OrientationPlan, ScatteringGrid
from diffBloch.params import ConstraintSpec, RefinableParams
from diffBloch.preprocess import RefinementSetup
from diffBloch.preprocess.plan import Plan

ENERGY = 200e3
CELL = np.eye(3, dtype=np.float64) * 5.0
THICKNESS = 300.0

# A seed rich enough that widening the Klar window admits beams progressively then saturates: an
# hkl cube whose outer beams are only kept at wider integration angles.
SEED_BEAMS = np.array(
    [
        [hkl_h, hkl_k, hkl_l]
        for hkl_h in range(-3, 4)
        for hkl_k in range(-3, 4)
        for hkl_l in range(-3, 4)
    ],
    dtype=np.int64,
)


def silicon_params() -> RefinableParams:
    return RefinableParams(
        asu_positions=torch.zeros((1, 3), dtype=torch.float64),
        uij_raw=torch.eye(3, dtype=torch.float64)[None] * 0.1,
    )


def seed_system() -> tuple[RefinementSetup, Plan]:
    grid = ScatteringGrid.from_cell(CELL, g_max=2.2)
    asu_plan = build_asu_expansion_plan(np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, 3)))
    spec = ConstraintSpec(
        fixed_positions=torch.zeros((1, 3), dtype=torch.float64),
        refinable_position_mask=torch.ones((1, 3), dtype=torch.float64),
        occupancies=torch.ones(1, dtype=torch.float64),
        reciprocal_basis=grid.reciprocal_basis,
    )
    refinement = RefinementSetup(
        asu_plan=asu_plan,  # type: ignore[arg-type]
        spec=spec,
        params=silicon_params(),
        numbers=torch.tensor([14], dtype=torch.int64),
    )
    pattern = PatternBatch(
        hkl=torch.tensor(SEED_BEAMS, dtype=torch.int64),
        intensities=torch.zeros(len(SEED_BEAMS), dtype=torch.float64),
        sigmas=torch.ones(len(SEED_BEAMS), dtype=torch.float64),
    )
    op = OrientationPlan.build(grid, SEED_BEAMS, pattern, energy=ENERGY, thickness=(THICKNESS,))
    return refinement, Plan(grid=grid, orientations=(op,))


def beam_count(plan: Plan) -> int:
    return len(plan.orientations[0].beam_hkl)
