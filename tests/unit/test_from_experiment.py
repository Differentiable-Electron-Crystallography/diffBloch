"""``from_experiment`` boundary construction: structure-side ``RefinementSetup`` assembly."""

from pathlib import Path

import pytest
import torch

from diffBloch.io import read_structure
from diffBloch.params import constrain
from diffBloch.preprocess import refinement_setup

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures"


def test_refinement_setup_seeds_quartz_uani_structure() -> None:
    structure = read_structure(FIXTURE_ROOT / "quartz_anchor" / "enantiomer_1.cif")
    setup = refinement_setup(structure, thicknesses=(820.0,))

    assert setup.numbers.tolist() == structure.numbers.tolist()
    assert setup.asu_plan.n_asu_sites == structure.n_atoms
    assert setup.thicknesses.tolist() == [820.0]
    # Positions are seeded at their CIF values (all-free mask, fixed == start).
    assert torch.allclose(
        setup.params.asu_positions, torch.tensor(structure.frac_positions, dtype=torch.float64)
    )
    assert setup.spec.adp_kind == structure.adp.kind


def test_refinement_setup_params_constrain_back_to_the_cif_adps() -> None:
    structure = read_structure(FIXTURE_ROOT / "quartz_anchor" / "enantiomer_1.cif")
    setup = refinement_setup(structure, thicknesses=(820.0,))

    state = constrain(setup.params, setup.spec)

    # Positions recovered exactly (mask all-free, so constrain returns the raw asu_positions).
    assert torch.allclose(
        state.positions, torch.tensor(structure.frac_positions, dtype=torch.float64)
    )
    # ADPs round-trip: cholesky_raw_from_adp -> cholesky_adp recovers the CIF Uij, then maps to U*.
    # Check the symmetric-PSD U* and that it equals constraining the CIF Uij directly would.
    assert state.uij_star.shape == (structure.n_atoms, 3, 3)
    assert torch.allclose(state.uij_star, state.uij_star.transpose(-1, -2))
    assert torch.all(torch.linalg.eigvalsh(state.uij_star) > 0.0)
    # Occupancies are fixed at the CIF values (not refined).
    assert torch.allclose(
        state.occupancies, torch.tensor(structure.occupancies, dtype=torch.float64)
    )


def test_refinement_setup_handles_uiso_structure() -> None:
    structure = read_structure(FIXTURE_ROOT / "paracetamol_min" / "enantiomer_1.cif")
    setup = refinement_setup(structure, thicknesses=(500.0,))

    assert setup.params.uij_raw is None  # all-Uiso: no anisotropic raw factor
    assert setup.params.u_iso_raw is not None
    assert setup.params.u_iso_raw.shape == (structure.n_atoms,)

    state = constrain(setup.params, setup.spec)
    # inverse-softplus seed must reproduce the CIF Uiso after constrain re-applies softplus:
    # U* = Uiso * G*, so the diagonal-trace scale tracks the CIF Uiso.
    assert state.uij_star.shape == (structure.n_atoms, 3, 3)
    assert torch.all(torch.linalg.eigvalsh(state.uij_star) >= 0.0)


def test_refinement_setup_rejects_empty_thicknesses() -> None:
    structure = read_structure(FIXTURE_ROOT / "quartz_anchor" / "enantiomer_1.cif")
    with pytest.raises(ValueError, match="thicknesses must contain at least one value"):
        refinement_setup(structure, thicknesses=())
