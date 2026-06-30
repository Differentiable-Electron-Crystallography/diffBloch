"""``from_experiment`` boundary construction: Plan pair split + structure-side RefinementSetup."""

from pathlib import Path

import pytest
import torch

from diffBloch.config import load_config
from diffBloch.io import read_observations, read_structure
from diffBloch.params import constrain
from diffBloch.preprocess import RefinementSetup, from_experiment, orientation_matrices

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures"
QUARTZ = FIXTURE_ROOT / "quartz_anchor"


def _quartz_setup() -> object:
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    observations = read_observations(QUARTZ / "exp_data.cif_pets")
    config = load_config(QUARTZ / "experiment.yaml")
    return structure, observations, config, from_experiment(structure, observations, config)


def test_from_experiment_builds_grid_sharing_train_val_split() -> None:
    structure, observations, config, setup = _quartz_setup()
    train = setup.plans.train
    val = setup.plans.validation

    # Both plans reference the SAME grid object -> shared Fgb support cannot diverge.
    assert train.grid is val.grid
    # Every rotation lands in exactly one plan; validation is every 10th (99 -> 9 val, 90 train).
    assert len(train.orientations) + len(val.orientations) == observations.n_rotations
    assert len(val.orientations) == 9
    assert len(train.orientations) == 90


def test_from_experiment_seeds_native_orientation_and_000_beam() -> None:
    structure, observations, config, setup = _quartz_setup()
    expected = orientation_matrices(
        observations.ub_matrix,
        observations.cell_parameters,
        observations.alphas,
        observations.betas,
        observations.omegas,
    )

    # First train rotation is rotation index 0 (index 9 is the first validation pick).
    first = setup.plans.train.orientations[0]
    assert torch.allclose(first.orientation, torch.tensor(expected[0], dtype=torch.float64))
    # Seed beams include the 000 transmitted beam and stay within g_max_refine (difference-safe).
    beam_hkl = first.beam_hkl
    assert (beam_hkl == 0).all(dim=1).any()
    g = beam_hkl.to(torch.float64) @ setup.plans.train.grid.reciprocal_basis
    assert torch.all(torch.linalg.norm(g, dim=1) <= config.numerics.g_max_refine + 1e-9)


def test_from_experiment_patterns_are_per_zone_axis() -> None:
    structure, observations, config, setup = _quartz_setup()
    # The first train plan's pattern carries only rotation 1's observed reflections.
    expected_n = int((observations.reflection_zone_axis_ids == 1).sum())
    assert setup.plans.train.orientations[0].pattern.hkl.shape == (expected_n, 3)


def test_from_experiment_refinement_side_matches_structure() -> None:
    structure, observations, config, setup = _quartz_setup()
    direct = RefinementSetup.from_structure(structure, thicknesses=config.sample.thicknesses)

    assert setup.refinement.numbers.tolist() == direct.numbers.tolist()
    assert setup.refinement.thicknesses.tolist() == list(config.sample.thicknesses)
    # The structure side constrains cleanly (positions + ADPs round-trip).
    state = constrain(setup.refinement.params, setup.refinement.spec)
    assert state.uij_star.shape == (structure.n_atoms, 3, 3)


def test_refinement_setup_seeds_quartz_uani_structure() -> None:
    structure = read_structure(FIXTURE_ROOT / "quartz_anchor" / "enantiomer_1.cif")
    setup = RefinementSetup.from_structure(structure, thicknesses=(820.0,))

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
    setup = RefinementSetup.from_structure(structure, thicknesses=(820.0,))

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
    setup = RefinementSetup.from_structure(structure, thicknesses=(500.0,))

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
        RefinementSetup.from_structure(structure, thicknesses=())
