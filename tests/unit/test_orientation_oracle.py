"""Real-orientation excitation-error anchor: per-orientation reciprocal-basis geometry.

Pins the native geometry path (``reciprocal_cell(cell @ orientation.T)`` -> ``g`` -> ``Sg``) against
a golden extracted from the ``diffBloch_private`` implementation on real quartz orientation matrices
(``tests/fixtures/quartz_anchor/orientation_oracle.npz``; regenerate via the co-located
``generate_orientation_oracle.py``, provenance in ``orientation_oracle_provenance.json``).

The orientation matrices are non-orthonormal (they fold a ~1% anisotropic measured-vs-ideal cell
correction), so ``orientation^-1 != orientation^T`` and the convention is observable on real data:
``reciprocal_cell(cell @ M.T)`` is faithful while ``reciprocal_basis @ M.T`` is off by ~0.008 A^-1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from diffBloch.core.crystal import cell_matrix_from_parameters, reciprocal_cell
from diffBloch.core.dynamical import excitation_errors
from diffBloch.core.products import PatternBatch
from diffBloch.engine import OrientationPlan, StructureFactorGrid

_ORACLE = Path(__file__).parent.parent / "fixtures" / "quartz_anchor" / "orientation_oracle.npz"


def _oracle() -> dict[str, np.ndarray]:
    data = np.load(_ORACLE)
    return {k: data[k] for k in data.files}


def test_native_per_orientation_excitation_matches_private_golden() -> None:
    o = _oracle()
    cell = cell_matrix_from_parameters(o["cellpar"])
    energy = float(o["energy"])
    for i, orientation in enumerate(o["orientation"]):
        rotated_basis = reciprocal_cell(cell @ orientation.T)
        sg = excitation_errors(o["hkl"] @ rotated_basis, energy)
        # Sg golden differs only by the abTEM-pinned wavelength constant (~1e-8); 1e-7 is ample.
        assert np.allclose(sg, o["sg"][i], atol=1e-7)


def test_wrong_transpose_convention_is_observably_off() -> None:
    # Guards the trap: M^T (vs reciprocal_cell(cell @ M.T)) is wrong on real, non-orthonormal
    # orientation matrices -- this assertion fails loudly if anyone "simplifies" to a rotation.
    o = _oracle()
    cell = cell_matrix_from_parameters(o["cellpar"])
    energy = float(o["energy"])
    untilted = reciprocal_cell(cell)
    orientation = o["orientation"][0]
    sg_wrong = excitation_errors(o["hkl"] @ (untilted @ orientation.T), energy)
    assert not np.allclose(sg_wrong, o["sg"][0], atol=1e-3)


def _grid(o: dict[str, np.ndarray]) -> StructureFactorGrid:
    return StructureFactorGrid.from_cell(cell_matrix_from_parameters(o["cellpar"]), g_max=4.5)


def _pattern(beam_hkl: np.ndarray) -> PatternBatch:
    hkl = torch.tensor(beam_hkl, dtype=torch.int64)
    n = hkl.shape[0]
    return PatternBatch(
        hkl=hkl,
        intensities=torch.zeros(n, dtype=torch.float64),
        sigmas=torch.ones(n, dtype=torch.float64),
    )


def test_orientation_plan_default_basis_is_byte_identical_to_grid() -> None:
    o = _oracle()
    grid = _grid(o)
    beam_hkl = o["hkl"][:24]
    pattern = _pattern(beam_hkl)
    default = OrientationPlan.build(
        grid, beam_hkl, pattern, energy=float(o["energy"]), thickness=(300.0,)
    )
    explicit = OrientationPlan.build(
        grid,
        beam_hkl,
        pattern,
        energy=float(o["energy"]),
        thickness=(300.0,),
        orientation=np.eye(3),
    )
    assert torch.equal(default.beam_plans[0].diagonal, explicit.beam_plans[0].diagonal)


def test_orientation_plan_per_orientation_basis_shifts_excitation() -> None:
    o = _oracle()
    grid = _grid(o)
    beam_hkl = o["hkl"][:24]
    pattern = _pattern(beam_hkl)
    untilted = OrientationPlan.build(
        grid, beam_hkl, pattern, energy=float(o["energy"]), thickness=(300.0,)
    )
    tilted = OrientationPlan.build(
        grid,
        beam_hkl,
        pattern,
        energy=float(o["energy"]),
        thickness=(300.0,),
        orientation=o["orientation"][0],
    )
    # diagonal = 2 k_n Sg Mii: a real orientation moves it well clear of the untilted case.
    assert not torch.allclose(untilted.beam_plans[0].diagonal, tilted.beam_plans[0].diagonal)


def test_orientation_plan_is_self_describing() -> None:
    # The plan carries its source/rebuild inputs (orientation/energy/u0), not just compiled
    # geometry, so a later Plan->Plan step can recompile without the original record. orientation
    # is the source of truth; None records the identity (untilted).
    o = _oracle()
    grid = _grid(o)
    beam_hkl = o["hkl"][:24]
    pattern = _pattern(beam_hkl)
    m = o["orientation"][0]
    plan = OrientationPlan.build(
        grid,
        beam_hkl,
        pattern,
        energy=float(o["energy"]),
        thickness=(300.0,),
        u0=1.5,
        orientation=m,
    )
    assert plan.energy == float(o["energy"])
    assert plan.u0 == 1.5
    assert torch.allclose(plan.orientation, torch.tensor(m, dtype=torch.float64))
    untilted = OrientationPlan.build(
        grid, beam_hkl, pattern, energy=float(o["energy"]), thickness=(300.0,)
    )
    assert torch.equal(untilted.orientation, torch.eye(3, dtype=torch.float64))


def test_orientation_plan_rebuilds_from_its_own_tensor_orientation() -> None:
    # A later Plan->Plan step recompiles from the stored Tensor orientation -- no record, no copy.
    o = _oracle()
    grid = _grid(o)
    beam_hkl = o["hkl"][:24]
    pattern = _pattern(beam_hkl)
    plan = OrientationPlan.build(
        grid,
        beam_hkl,
        pattern,
        energy=float(o["energy"]),
        thickness=(300.0,),
        orientation=o["orientation"][0],
    )
    rebuilt = OrientationPlan.build(
        grid,
        beam_hkl,
        pattern,
        energy=plan.energy,
        thickness=plan.thickness,
        u0=plan.u0,
        orientation=plan.orientation,
    )
    assert torch.equal(rebuilt.beam_plans[0].diagonal, plan.beam_plans[0].diagonal)
    assert torch.equal(rebuilt.orientation, plan.orientation)
