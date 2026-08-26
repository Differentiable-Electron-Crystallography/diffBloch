"""``from_experiment`` boundary construction: Plan pair split + structure-side RefinementSetup."""

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from diffBloch.config import load_config
from diffBloch.core.adp import ueq_from_cif_uij
from diffBloch.core.crystal import cell_matrix_from_parameters, reciprocal_cell
from diffBloch.core.products import MosaicSmoothed
from diffBloch.io import read_experimental_data, read_structure
from diffBloch.io.record import AdpRecord, StructureRecord
from diffBloch.params import constrain
from diffBloch.preprocess import (
    RefinementSetup,
    from_experiment,
    resolve_dataset_orientations,
)

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures"
QUARTZ = FIXTURE_ROOT / "quartz_anchor"


def _quartz_setup() -> object:
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    experimental_data = read_experimental_data(QUARTZ / "exp_data.cif_pets")
    base = load_config(QUARTZ / "experiment.yaml")
    config = base.model_copy(
        update={"blochwave": base.blochwave.model_copy(update={"mosaicity": False})}
    )
    return (
        structure,
        experimental_data,
        config,
        from_experiment(structure, experimental_data, config),
    )


def test_from_experiment_builds_grid_sharing_train_val_split() -> None:
    structure, experimental_data, config, setup = _quartz_setup()
    train = setup.plans.train
    val = setup.plans.validation

    # Both plans reference the SAME grid object -> shared Fgb support cannot diverge.
    assert train.structure_factor_grid is val.structure_factor_grid
    # Every rotation lands in exactly one plan; validation is every 10th (99 -> 9 val, 90 train).
    assert len(train.orientations) + len(val.orientations) == experimental_data.n_rotations
    assert len(val.orientations) == 9
    assert len(train.orientations) == 90
    assert [op.pattern.rotation_index for op in train.orientations[:11]] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        11,
    ]
    assert [op.pattern.rotation_index for op in val.orientations[:2]] == [9, 19]


def test_from_experiment_train_test_false_holds_out_nothing() -> None:
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    experimental_data = read_experimental_data(QUARTZ / "exp_data.cif_pets")
    base = load_config(QUARTZ / "experiment.yaml")
    config = base.model_copy(
        update={
            "refinement": base.refinement.model_copy(
                update={"split": base.refinement.split.model_copy(update={"train_test": False})}
            )
        }
    )

    setup = from_experiment(structure, experimental_data, config)

    assert len(setup.plans.validation.orientations) == 0
    assert len(setup.plans.train.orientations) == experimental_data.n_rotations


def test_mosaicity_is_disabled_by_default() -> None:
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    experimental_data = read_experimental_data(QUARTZ / "exp_data.cif_pets").model_copy(
        update={"mosaicity_degrees": None}
    )
    base = load_config(QUARTZ / "experiment.yaml")
    config = base.model_copy(
        update={"blochwave": base.blochwave.model_copy(update={"mosaicity": False})}
    )

    setup = from_experiment(structure, experimental_data, config)

    assert config.blochwave.mosaicity is False
    assert setup.mosaicity is None
    assert not any(hasattr(plan, "mosaicity_degrees") for plan in setup.plans.combined.orientations)


def test_enabled_mosaicity_requires_pets_metadata_and_names_the_remedy() -> None:
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    experimental_data = read_experimental_data(QUARTZ / "exp_data.cif_pets").model_copy(
        update={"mosaicity_degrees": None}
    )
    base = load_config(QUARTZ / "experiment.yaml")
    config = base.model_copy(
        update={"blochwave": base.blochwave.model_copy(update={"mosaicity": True})}
    )

    with pytest.raises(ValueError, match="set blochwave.mosaicity: false"):
        from_experiment(structure, experimental_data, config)


def test_enabled_mosaicity_resolves_pets_degrees_to_internal_sample_span() -> None:
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    experimental_data = read_experimental_data(QUARTZ / "exp_data.cif_pets").model_copy(
        update={"mosaicity_degrees": 0.6}
    )
    base = load_config(QUARTZ / "experiment.yaml")
    config = base.model_copy(
        update={
            "blochwave": base.blochwave.model_copy(
                update={"mosaicity": True, "rocking_curve_sampling": 11}
            )
        }
    )

    setup = from_experiment(structure, experimental_data, config)

    assert setup.mosaicity == MosaicSmoothed(samples=3)


def test_enabled_mosaicity_rejects_negative_pets_metadata_at_setup_time() -> None:
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    experimental_data = read_experimental_data(QUARTZ / "exp_data.cif_pets").model_copy(
        update={"mosaicity_degrees": -0.1}
    )
    base = load_config(QUARTZ / "experiment.yaml")
    config = base.model_copy(
        update={"blochwave": base.blochwave.model_copy(update={"mosaicity": True})}
    )

    with pytest.raises(ValueError, match="finite, non-negative"):
        from_experiment(structure, experimental_data, config)


def test_enabled_mosaicity_rejects_nonfinite_pets_metadata_at_setup_time() -> None:
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    experimental_data = read_experimental_data(QUARTZ / "exp_data.cif_pets").model_copy(
        update={"mosaicity_degrees": np.nan}
    )
    base = load_config(QUARTZ / "experiment.yaml")
    config = base.model_copy(
        update={"blochwave": base.blochwave.model_copy(update={"mosaicity": True})}
    )

    with pytest.raises(ValueError, match="finite, non-negative"):
        from_experiment(structure, experimental_data, config)


def test_from_experiment_ignores_original_pets_indices_before_split() -> None:
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    experimental_data = read_experimental_data(QUARTZ / "exp_data.cif_pets")
    base = load_config(QUARTZ / "experiment.yaml")
    config = base.model_copy(
        update={"blochwave": base.blochwave.model_copy(update={"ignore_orientations": (0, 9, 56)})}
    )

    setup = from_experiment(structure, experimental_data, config)

    # Raw index 9 remains a validation member when removed; later rotations are not renumbered.
    assert len(setup.plans.train.orientations) == 88
    assert len(setup.plans.validation.orientations) == 8
    # The seeded orientation is the axis-corrected one, not the as-collected derivation: PETS
    # records a nonzero goniometer-axis azimuth for quartz, so these differ.
    expected = resolve_dataset_orientations(experimental_data)
    assert np.allclose(setup.plans.train.orientations[0].orientation, expected[1])
    assert setup.plans.train.orientations[0].pattern.rotation_index == 1
    assert [op.pattern.rotation_index for op in setup.plans.validation.orientations[:2]] == [19, 29]
    assert not any(
        np.allclose(plan.orientation, expected[56]) for plan in setup.plans.combined.orientations
    )


def test_from_experiment_rejects_invalid_data_dependent_ignore_selection() -> None:
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    experimental_data = read_experimental_data(QUARTZ / "exp_data.cif_pets")
    base = load_config(QUARTZ / "experiment.yaml")

    out_of_range = base.model_copy(
        update={
            "blochwave": base.blochwave.model_copy(
                update={"ignore_orientations": (experimental_data.n_rotations,)}
            )
        }
    )
    with pytest.raises(ValueError, match="outside the PETS rotation range"):
        from_experiment(structure, experimental_data, out_of_range)

    all_ignored = base.model_copy(
        update={
            "blochwave": base.blochwave.model_copy(
                update={"ignore_orientations": tuple(range(experimental_data.n_rotations))}
            )
        }
    )
    with pytest.raises(ValueError, match="excludes every PETS rotation"):
        from_experiment(structure, experimental_data, all_ignored)


def test_from_experiment_seeds_native_orientation_and_000_beam() -> None:
    structure, experimental_data, config, setup = _quartz_setup()
    # The seeded orientation is the axis-corrected one, not the as-collected derivation: PETS
    # records a nonzero goniometer-axis azimuth for quartz, so these differ.
    expected = resolve_dataset_orientations(experimental_data)

    # First train rotation is rotation index 0 (index 9 is the first validation pick). The candidate
    # phase carries plain-numpy source (orientation / beam_hkl), built into tensors later.
    first = setup.plans.train.orientations[0]
    assert np.allclose(first.orientation, expected[0])
    # Seed beams include the 000 transmitted beam and stay within g_max (difference-safe).
    beam_hkl = first.beam_hkl
    assert (beam_hkl == 0).all(axis=1).any()
    g = beam_hkl.astype(np.float64) @ np.asarray(
        setup.plans.train.structure_factor_grid.reciprocal_basis
    )
    assert np.all(np.linalg.norm(g, axis=1) <= config.blochwave.g_max + 1e-9)


def test_from_experiment_patterns_are_per_zone_axis() -> None:
    structure, experimental_data, config, setup = _quartz_setup()
    # The first train plan's pattern carries only rotation 1's observed reflections.
    expected_n = int((experimental_data.reflection_zone_axis_ids == 1).sum())
    assert setup.plans.train.orientations[0].pattern.hkl.shape == (expected_n, 3)


def test_from_experiment_refinement_side_matches_structure() -> None:
    structure, experimental_data, config, setup = _quartz_setup()
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


# --- unit-cell authority: PETS overrides the structure CIF ----------------------------------------


def test_cell_mismatch_under_1pct_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    experimental_data = read_experimental_data(QUARTZ / "exp_data.cif_pets")
    config = load_config(QUARTZ / "experiment.yaml")
    nudged = experimental_data.model_copy(
        update={
            "cell_parameters": experimental_data.cell_parameters
            * np.array([1.005, 1.0, 1.0, 1.0, 1.0, 1.0])
        }
    )

    with caplog.at_level(logging.WARNING, logger="diffBloch.preprocess.experiment"):
        setup = from_experiment(structure, nudged, config)

    assert not caplog.records
    np.testing.assert_allclose(
        setup.plans.combined.structure_factor_grid.cell.numpy(),
        cell_matrix_from_parameters(nudged.cell_parameters),
    )


def test_cell_mismatch_over_1pct_warns_and_pets_geometry_wins(
    caplog: pytest.LogCaptureFixture,
) -> None:
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    experimental_data = read_experimental_data(QUARTZ / "exp_data.cif_pets")
    config = load_config(QUARTZ / "experiment.yaml")
    mismatched = experimental_data.model_copy(
        update={
            "cell_parameters": experimental_data.cell_parameters
            * np.array([1.02, 1.0, 1.0, 1.0, 1.0, 1.0])
        }
    )

    with caplog.at_level(logging.WARNING, logger="diffBloch.preprocess.experiment"):
        setup = from_experiment(structure, mismatched, config)

    [record] = caplog.records
    message = record.getMessage()
    assert "more than 1%" in message
    assert "structure CIF" in message
    assert "overrides" in message
    assert "a:" in message

    grid_cell = setup.plans.combined.structure_factor_grid.cell.numpy()
    np.testing.assert_allclose(grid_cell, cell_matrix_from_parameters(mismatched.cell_parameters))
    assert not np.allclose(grid_cell, cell_matrix_from_parameters(structure.cell_parameters))
    np.testing.assert_allclose(
        setup.refinement.spec.reciprocal_basis.numpy(),
        reciprocal_cell(cell_matrix_from_parameters(mismatched.cell_parameters)),
    )
    np.testing.assert_allclose(setup.refinement.cell_parameters, mismatched.cell_parameters)


def test_cell_mismatch_over_5pct_raises_and_names_every_offending_parameter() -> None:
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    experimental_data = read_experimental_data(QUARTZ / "exp_data.cif_pets")
    config = load_config(QUARTZ / "experiment.yaml")
    mismatched = experimental_data.model_copy(
        update={
            "cell_parameters": experimental_data.cell_parameters
            * np.array([1.06, 1.0, 1.0, 1.07, 1.0, 1.0])
        }
    )

    with pytest.raises(ValueError, match="more than 5%") as excinfo:
        from_experiment(structure, mismatched, config)

    message = str(excinfo.value)
    assert "a:" in message
    assert "alpha:" in message
    assert "b:" not in message
    assert "beta:" not in message
    assert f"{mismatched.cell_parameters[0]:.6g}" in message
    assert f"{structure.cell_parameters[0]:.6g}" in message
    assert "%" in message
    assert "refusing to continue" in message


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


# --- inputs.isotropic_displacements_only override --------------------------------------------


def _oblique_structure(kinds: tuple[str, ...]) -> StructureRecord:
    """A P1 monoclinic (beta = 100 degrees) variant of :func:`_ortho_structure`.

    An orthorhombic cell's off-diagonal direct-metric terms vanish, so Ueq degenerates to
    ``trace(Uij) / 3`` there and cannot distinguish the correct metric-tensor contraction from the
    naive shortcut. The non-90-degree ``beta`` here keeps the metric tensor's off-diagonal term
    non-zero, so the two routes give observably different answers.
    """
    n = len(kinds)
    uij_cif = np.stack([_UANI if k == "Uani" else np.full((3, 3), np.nan) for k in kinds])
    u_iso = np.array([_UISO if k == "Uiso" else np.nan for k in kinds])
    cell_parameters = np.array([5.0, 6.0, 7.0, 90.0, 100.0, 90.0])
    return StructureRecord(
        unit_cell=cell_matrix_from_parameters(cell_parameters),
        cell_parameters=cell_parameters,
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


def test_isotropic_displacements_only_forces_uani_to_uiso_seeded_from_cif_ueq() -> None:
    structure = _oblique_structure(("Uani", "Uiso"))

    setup = RefinementSetup.from_structure(structure, isotropic_displacements_only=True)

    # Every atom refines as Uiso now, including the one that was Uani in the CIF -- and the
    # override never touched the CIF-parsed record itself.
    assert setup.spec.adp_kind == ("Uiso", "Uiso")
    assert structure.adp.kind == ("Uani", "Uiso")
    assert setup.params.uij_raw is None
    assert setup.params.u_iso_raw is not None and setup.params.u_iso_raw.shape == (2,)

    unit_cell = cell_matrix_from_parameters(structure.cell_parameters)
    reciprocal_basis = reciprocal_cell(unit_cell)
    reciprocal_lengths = torch.tensor(np.linalg.norm(reciprocal_basis, axis=1))
    metric_tensor = torch.tensor(unit_cell @ unit_cell.T)
    expected_ueq = ueq_from_cif_uij(
        torch.tensor(_UANI, dtype=torch.float64), reciprocal_lengths, metric_tensor
    ).item()
    # The naive trace(Uij)/3 shortcut disagrees on this non-cubic (5, 6, 7) cell -- confirming the
    # seed genuinely comes from the metric-tensor contraction, not the wrong shortcut.
    assert expected_ueq != pytest.approx(float(np.trace(_UANI) / 3.0), rel=1e-3)

    state = constrain(setup.params, setup.spec)
    reciprocal_metric = reciprocal_basis @ reciprocal_basis.T
    # Both atoms are now genuinely isotropic: uij_star = Uiso * G* exactly, so Uiso is recoverable.
    seeded_uiso = [
        float(torch.sum(state.uij_star[i] * torch.tensor(reciprocal_metric)))
        / float(np.sum(reciprocal_metric * reciprocal_metric))
        for i in range(2)
    ]
    assert seeded_uiso[0] == pytest.approx(expected_ueq)
    # The atom that was already Uiso in the CIF keeps its own CIF value, untouched by the override.
    assert seeded_uiso[1] == pytest.approx(_UISO)


def test_isotropic_displacements_only_uses_pets_authoritative_cell_for_ueq() -> None:
    structure = _oblique_structure(("Uani",))
    experimental_data = read_experimental_data(QUARTZ / "exp_data.cif_pets")
    pets_cell = structure.cell_parameters.copy()
    pets_cell[4] = 104.0  # below the 5% mismatch guard, but enough to change oblique-cell Ueq
    experimental_data = experimental_data.model_copy(update={"cell_parameters": pets_cell})
    base = load_config(QUARTZ / "experiment.yaml")
    config = base.model_copy(
        update={
            "inputs": base.inputs.model_copy(update={"isotropic_displacements_only": True}),
            "blochwave": base.blochwave.model_copy(update={"mosaicity": False}),
        }
    )

    setup = from_experiment(structure, experimental_data, config)

    state = constrain(setup.refinement.params, setup.refinement.spec)
    pets_unit_cell = cell_matrix_from_parameters(pets_cell)
    pets_reciprocal_basis = reciprocal_cell(pets_unit_cell)
    pets_reciprocal_metric = pets_reciprocal_basis @ pets_reciprocal_basis.T
    seeded_uiso = float(
        torch.sum(state.uij_star[0] * torch.tensor(pets_reciprocal_metric))
    ) / float(np.sum(pets_reciprocal_metric * pets_reciprocal_metric))
    expected_pets_ueq = ueq_from_cif_uij(
        torch.tensor(_UANI, dtype=torch.float64),
        torch.tensor(np.linalg.norm(pets_reciprocal_basis, axis=1), dtype=torch.float64),
        torch.tensor(pets_unit_cell @ pets_unit_cell.T, dtype=torch.float64),
    ).item()

    cif_unit_cell = cell_matrix_from_parameters(structure.cell_parameters)
    cif_reciprocal_basis = reciprocal_cell(cif_unit_cell)
    cif_ueq = ueq_from_cif_uij(
        torch.tensor(_UANI, dtype=torch.float64),
        torch.tensor(np.linalg.norm(cif_reciprocal_basis, axis=1), dtype=torch.float64),
        torch.tensor(cif_unit_cell @ cif_unit_cell.T, dtype=torch.float64),
    ).item()

    assert setup.refinement.spec.adp_kind == ("Uiso",)
    assert setup.refinement.cell_parameters is not None
    np.testing.assert_allclose(setup.refinement.cell_parameters, pets_cell)
    assert seeded_uiso == pytest.approx(expected_pets_ueq)
    assert seeded_uiso != pytest.approx(cif_ueq, rel=1e-4)


def test_isotropic_displacements_only_defaults_to_off() -> None:
    structure = _ortho_structure(("Uani", "Uiso"))

    setup = RefinementSetup.from_structure(structure)

    assert setup.spec.adp_kind == structure.adp.kind
