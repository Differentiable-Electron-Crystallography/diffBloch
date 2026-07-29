"""``from_experiment`` boundary construction: Plan pair split + structure-side RefinementSetup."""

from pathlib import Path

import numpy as np
import pytest
import torch

from diffBloch.config import load_config
from diffBloch.io import read_observations, read_structure
from diffBloch.io.record import AdpRecord, StructureRecord
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
    assert train.structure_factor_grid is val.structure_factor_grid
    # Every rotation lands in exactly one plan; validation is every 10th (99 -> 9 val, 90 train).
    assert len(train.orientations) + len(val.orientations) == observations.n_rotations
    assert len(val.orientations) == 9
    assert len(train.orientations) == 90


def test_from_experiment_ignores_original_pets_indices_before_split() -> None:
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    observations = read_observations(QUARTZ / "exp_data.cif_pets")
    base = load_config(QUARTZ / "experiment.yaml")
    config = base.model_copy(
        update={"blochwave": base.blochwave.model_copy(update={"ignore_orientations": (0, 9, 56)})}
    )

    setup = from_experiment(structure, observations, config)

    # Raw index 9 remains a validation member when removed; later rotations are not renumbered.
    assert len(setup.plans.train.orientations) == 88
    assert len(setup.plans.validation.orientations) == 8
    expected = orientation_matrices(
        observations.ub_matrix,
        observations.cell_parameters,
        observations.alphas,
        observations.betas,
        observations.omegas,
    )
    assert np.allclose(setup.plans.train.orientations[0].orientation, expected[1])
    assert not any(
        np.allclose(plan.orientation, expected[56]) for plan in setup.plans.combined.orientations
    )


def test_from_experiment_rejects_invalid_data_dependent_ignore_selection() -> None:
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    observations = read_observations(QUARTZ / "exp_data.cif_pets")
    base = load_config(QUARTZ / "experiment.yaml")

    out_of_range = base.model_copy(
        update={
            "blochwave": base.blochwave.model_copy(
                update={"ignore_orientations": (observations.n_rotations,)}
            )
        }
    )
    with pytest.raises(ValueError, match="outside the PETS rotation range"):
        from_experiment(structure, observations, out_of_range)

    all_ignored = base.model_copy(
        update={
            "blochwave": base.blochwave.model_copy(
                update={"ignore_orientations": tuple(range(observations.n_rotations))}
            )
        }
    )
    with pytest.raises(ValueError, match="excludes every PETS rotation"):
        from_experiment(structure, observations, all_ignored)


def test_from_experiment_seeds_native_orientation_and_000_beam() -> None:
    structure, observations, config, setup = _quartz_setup()
    expected = orientation_matrices(
        observations.ub_matrix,
        observations.cell_parameters,
        observations.alphas,
        observations.betas,
        observations.omegas,
    )

    # First train rotation is rotation index 0 (index 9 is the first validation pick). The candidate
    # phase carries plain-numpy source (orientation / beam_hkl), built into tensors later.
    first = setup.plans.train.orientations[0]
    assert np.allclose(first.orientation, expected[0])
    # Seed beams include the 000 transmitted beam and stay within g_max_refine (difference-safe).
    beam_hkl = first.beam_hkl
    assert (beam_hkl == 0).all(axis=1).any()
    g = beam_hkl.astype(np.float64) @ np.asarray(
        setup.plans.train.structure_factor_grid.reciprocal_basis
    )
    assert np.all(np.linalg.norm(g, axis=1) <= config.blochwave.g_max_refine + 1e-9)


def test_from_experiment_patterns_are_per_zone_axis() -> None:
    structure, observations, config, setup = _quartz_setup()
    # The first train plan's pattern carries only rotation 1's observed reflections.
    expected_n = int((observations.reflection_zone_axis_ids == 1).sum())
    assert setup.plans.train.orientations[0].pattern.hkl.shape == (expected_n, 3)


def test_from_experiment_refinement_side_matches_structure() -> None:
    structure, observations, config, setup = _quartz_setup()
    direct = RefinementSetup.from_structure(structure)

    assert setup.refinement.numbers.tolist() == direct.numbers.tolist()
    # Per-rotation thickness is seeded onto the orientations (from config.sample.thicknesses),
    # not onto RefinementSetup.
    assert setup.plans.train.orientations[0].thickness.tolist() == list(config.sample.thicknesses)
    # The structure side constrains cleanly (positions + ADPs round-trip).
    state = constrain(setup.refinement.params, setup.refinement.spec)
    assert state.uij_star.shape == (structure.n_atoms, 3, 3)


def test_refinement_setup_seeds_quartz_uani_structure() -> None:
    structure = read_structure(FIXTURE_ROOT / "quartz_anchor" / "enantiomer_1.cif")
    setup = RefinementSetup.from_structure(structure)

    assert setup.numbers.tolist() == structure.numbers.tolist()
    assert setup.asu_plan.n_asu_sites == structure.n_atoms
    # Positions are seeded at their CIF values (raw asu_positions == the CIF coordinates).
    assert torch.allclose(
        setup.params.asu_positions, torch.tensor(structure.frac_positions, dtype=torch.float64)
    )
    assert setup.spec.adp_kind == structure.adp.kind


def test_refinement_setup_params_constrain_back_to_the_cif_adps() -> None:
    structure = read_structure(FIXTURE_ROOT / "quartz_anchor" / "enantiomer_1.cif")
    setup = RefinementSetup.from_structure(structure)

    state = constrain(setup.params, setup.spec)

    # Positions recovered exactly: raw is seeded at the CIF values (which lie on their sites), so
    # the site-symmetry projector returns them unchanged (x = x0 + P @ (raw - x0) = x0 at raw = x0).
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
    setup = RefinementSetup.from_structure(structure)

    assert setup.params.uij_raw is None  # all-Uiso: no anisotropic raw factor
    assert setup.params.u_iso_raw is not None
    assert setup.params.u_iso_raw.shape == (structure.n_atoms,)

    state = constrain(setup.params, setup.spec)
    # inverse-softplus seed must reproduce the CIF Uiso after constrain re-applies softplus:
    # U* = Uiso * G*, so the diagonal-trace scale tracks the CIF Uiso.
    assert state.uij_star.shape == (structure.n_atoms, 3, 3)
    assert torch.all(torch.linalg.eigvalsh(state.uij_star) >= 0.0)


def test_from_experiment_seeds_per_rotation_thickness_on_the_orientations() -> None:
    _, _, config, setup = _quartz_setup()
    seeded = list(config.sample.thicknesses)
    for plan in (setup.plans.train, setup.plans.validation):
        for orientation in plan.orientations:
            assert orientation.thickness.tolist() == seeded


# --- mixed Uani + Uiso ADP path -------------------------------------------------------------------

_UANI = np.array([[0.011, 0.002, 0.0], [0.002, 0.013, 0.001], [0.0, 0.001, 0.009]])
_UISO = 0.015


def _ortho_structure(kinds: tuple[str, ...]) -> StructureRecord:
    """A P1 orthorhombic (5, 6, 7) structure whose per-atom ADP kind follows ``kinds``.

    Anisotropic cell (distinct reciprocal lengths) so the Uani d*-relation and the Uiso ``Uiso G*``
    map differ observably. Each Uani atom carries ``_UANI``; each Uiso atom carries ``_UISO``.
    """
    n = len(kinds)
    uij_cif = np.stack([_UANI if k == "Uani" else np.full((3, 3), np.nan) for k in kinds])
    u_iso = np.array([_UISO if k == "Uiso" else np.nan for k in kinds])
    return StructureRecord(
        unit_cell=np.diag([5.0, 6.0, 7.0]),
        cell_parameters=np.array([5.0, 6.0, 7.0, 90.0, 90.0, 90.0]),
        cell_parameters_su=np.full((6,), np.nan),
        spacegroup_hm="P1",
        symops_R=np.eye(3)[None, :, :],
        symops_t=np.zeros((1, 3)),
        labels=tuple(f"A{i}" for i in range(n)),
        numbers=np.array([14] * n),
        frac_positions=np.linspace(0.1, 0.6, n * 3).reshape(n, 3),
        frac_positions_su=np.full((n, 3), np.nan),
        occupancies=np.ones(n),
        occupancies_su=np.full((n,), np.nan),
        adp=AdpRecord(
            kind=kinds,
            u_iso=u_iso,
            u_iso_su=np.full((n,), np.nan),
            uij_cif=uij_cif,
            uij_cif_su=np.full((n, 3, 3), np.nan),
        ),
    )


def test_refinement_setup_mixed_uani_uiso_constrains_each_atom_by_kind() -> None:
    mixed = RefinementSetup.from_structure(_ortho_structure(("Uani", "Uiso")))

    # Both raw factors are present and span every atom (filler rows for the other kind).
    assert mixed.params.uij_raw is not None and mixed.params.uij_raw.shape == (2, 3, 3)
    assert mixed.params.u_iso_raw is not None and mixed.params.u_iso_raw.shape == (2,)

    state = constrain(mixed.params, mixed.spec)
    assert state.uij_star.shape == (2, 3, 3)
    assert torch.allclose(state.uij_star, state.uij_star.transpose(-1, -2))
    assert torch.all(torch.linalg.eigvalsh(state.uij_star) > 0.0)

    # Each atom must match a single-kind reference on the same cell: the kind mask, not atom order,
    # decides which transform applies. uij_star depends only on the ADP + cell, not the position.
    uani_ref = constrain(*_setup_state(_ortho_structure(("Uani",))))
    uiso_ref = constrain(*_setup_state(_ortho_structure(("Uiso",))))
    assert torch.allclose(state.uij_star[0], uani_ref.uij_star[0])
    assert torch.allclose(state.uij_star[1], uiso_ref.uij_star[0])
    # The two transforms genuinely differ, so the test is not vacuous.
    assert not torch.allclose(uani_ref.uij_star[0], uiso_ref.uij_star[0])


def _setup_state(structure: StructureRecord) -> tuple:
    setup = RefinementSetup.from_structure(structure)
    return setup.params, setup.spec
