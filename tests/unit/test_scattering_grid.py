"""``ScatteringGrid`` construction: explicit support radius vs derived-from-solve-cutoff."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diffBloch.engine.plan import ScatteringGrid

_CELL = np.eye(3, dtype=np.float64) * 5.0  # 5 A cubic -> reciprocal basis (1/5) I


def test_from_cell_for_solve_cutoff_derives_double_support() -> None:
    # The solve cutoff is the beam radius; the SF support it needs is 2x that (g - h differences),
    # so the derived grid is identical to hand-passing the doubled radius to from_cell -- without
    # the caller having to know the doubling convention.
    solve = ScatteringGrid.from_cell_for_solve_cutoff(_CELL, solve_g_max=1.0)
    explicit = ScatteringGrid.from_cell(_CELL, g_max=2.0)

    assert solve.g_max == 2.0
    assert solve.gpts == explicit.gpts
    assert torch.equal(solve.grid_hkl, explicit.grid_hkl)


def test_from_cell_for_solve_cutoff_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="solve_g_max must be positive"):
        ScatteringGrid.from_cell_for_solve_cutoff(_CELL, solve_g_max=0.0)
