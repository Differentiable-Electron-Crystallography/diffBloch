"""``StructureFactorGrid`` construction: explicit support radius vs derived-from-solve-cutoff."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diffBloch.core.dynamical import build_structure_factor_gather, gather_structure_factors
from diffBloch.core.reciprocal import make_hkl_grid
from diffBloch.engine.plan import StructureFactorGrid

_CELL = np.eye(3, dtype=np.float64) * 5.0  # 5 A cubic -> reciprocal basis (1/5) I
_SOLVE_G_MAX = 0.45  # beams to |g| <= 0.45; their g - h differences reach 0.9


def test_from_cell_for_beam_cutoff_derives_double_support_plus_margin() -> None:
    # The solve cutoff is the beam radius; the SF support it needs is 2x that (g - h differences)
    # plus a half-Angstrom shell (the private's 2*g_max + 0.5) for orientation-metric wobble. So the
    # derived grid matches hand-passing 2*solve + 0.5 to from_cell -- without the caller knowing the
    # convention.
    solve = StructureFactorGrid.from_cell_for_beam_cutoff(_CELL, solve_g_max=1.0)
    explicit = StructureFactorGrid.from_cell(_CELL, g_max=2.0 + 0.5)

    assert solve.g_max == 2.5
    assert solve.gpts == explicit.gpts
    assert torch.equal(solve.structure_factor_hkl, explicit.structure_factor_hkl)


def test_from_cell_for_beam_cutoff_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="solve_g_max must be positive"):
        StructureFactorGrid.from_cell_for_beam_cutoff(_CELL, solve_g_max=0.0)


def test_derived_support_covers_every_solve_cutoff_beam_difference() -> None:
    # The whole point of the derivation: a grid built for the solve cutoff spans every g - h of a
    # beam set bounded by that cutoff, so the gather's coverage check passes -- while a grid sized
    # to the solve cutoff itself (the pre-#154 mistake of one radius for both jobs) does not.
    beam_hkl = make_hkl_grid(_CELL, _SOLVE_G_MAX)

    derived = StructureFactorGrid.from_cell_for_beam_cutoff(_CELL, solve_g_max=_SOLVE_G_MAX)
    build_structure_factor_gather(  # does not raise: support covers the differences
        derived.structure_factor_hkl.numpy(), beam_hkl, derived.gpts, validate=True
    )

    undersized = StructureFactorGrid.from_cell(_CELL, g_max=_SOLVE_G_MAX)
    with pytest.raises(ValueError, match="difference"):
        build_structure_factor_gather(
            undersized.structure_factor_hkl.numpy(), beam_hkl, undersized.gpts, validate=True
        )


def test_gather_through_derived_support_preserves_gradients() -> None:
    # End-to-end at the semantic-split level: Fgb flows a gradient through the gather off the
    # derived grid, so refinement of the structure factors is differentiable on this path.
    beam_hkl = make_hkl_grid(_CELL, _SOLVE_G_MAX)
    grid = StructureFactorGrid.from_cell_for_beam_cutoff(_CELL, solve_g_max=_SOLVE_G_MAX)
    gather = build_structure_factor_gather(grid.structure_factor_hkl.numpy(), beam_hkl, grid.gpts)

    fgb = torch.randn(
        grid.structure_factor_hkl.shape[0], dtype=torch.complex128, requires_grad=True
    )
    gather_structure_factors(gather, fgb).abs().pow(2).sum().backward()

    assert fgb.grad is not None
    assert torch.count_nonzero(fgb.grad) > 0


def test_reused_gather_object_is_geometry_only() -> None:
    # #154 disabled reuse of a detached A(Fgb) cache under autograd. Public has no such cache to
    # guard: the reused precompute is StructureFactorGather, whose tensor fields are integer index
    # maps (no Fgb, no float/complex payload, no grad), so Fgb always re-flows through the gather.
    beam_hkl = make_hkl_grid(_CELL, _SOLVE_G_MAX)
    grid = StructureFactorGrid.from_cell_for_beam_cutoff(_CELL, solve_g_max=_SOLVE_G_MAX)
    gather = build_structure_factor_gather(grid.structure_factor_hkl.numpy(), beam_hkl, grid.gpts)

    for field in (gather.structure_factor_indices, gather.beam_difference_indices):
        assert field.dtype == torch.long
        assert not field.is_floating_point() and not field.is_complex()
        assert not field.requires_grad
