from pathlib import Path

import gemmi
import numpy as np
import pytest

from diffBloch.io import (
    parse_cif_number,
    parse_observation_block,
    parse_structure_block,
    read_observations,
    read_structure,
    symmetry_constraints,
)

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"


def test_read_quartz_structure_fixture() -> None:
    record = read_structure(FIXTURE_ROOT / "enantiomer_1.cif")

    assert record.labels == ("Si1", "O1")
    assert record.numbers.tolist() == [14, 8]
    assert record.frac_positions.shape == (2, 3)
    assert np.isnan(record.frac_positions_su).all()
    assert record.occupancies.tolist() == [1.0, 1.0]
    assert np.isnan(record.occupancies_su).all()
    assert record.cell_parameters.tolist() == pytest.approx([4.9226, 4.9226, 5.4003, 90, 90, 120])
    assert np.isnan(record.cell_parameters_su).all()
    assert record.adp.kind == ("Uani", "Uani")
    assert record.uij_cif.shape == (2, 3, 3)
    assert np.linalg.eigvalsh(record.uij_cif).min() > 0.0
    assert np.isnan(record.adp.uij_cif_su).all()
    assert record.spacegroup_hm == "P3221"
    assert record.spacegroup_number == 154
    assert record.n_symops == 6

    constraints = symmetry_constraints(record)
    assert constraints.n_asymmetric_sites == 2
    assert constraints.n_symops == 6


def test_read_quartz_pets_fixture() -> None:
    record = read_observations(FIXTURE_ROOT / "exp_data.cif_pets")

    assert record.wavelength == 0.02510
    assert record.n_rotations == 99
    assert record.zone_axis_ids[0] == 1
    assert record.zone_axis_ids[-1] == 99
    assert record.zone_axes.shape == (99, 3)
    assert record.ub_matrix.shape == (3, 3)
    assert record.n_reflections == 6666
    assert record.hkl.shape == (6666, 3)
    assert record.intensities.shape == (6666,)
    assert record.sigmas.min() >= 0.0
    assert set(record.reflection_zone_axis_ids.tolist()) == set(range(1, 100))


@pytest.mark.parametrize(
    ("text", "value", "su"),
    [
        ("0.0144(8)", 0.0144, 0.0008),
        ("4.92260(15)", 4.92260, 0.00015),
        ("42(3)", 42.0, 3.0),
        ("1.23(45)", 1.23, 0.45),
        ("1.2e-3(4)", 1.2e-3, 0.4e-3),
    ],
)
def test_parse_cif_number_preserves_standard_uncertainty(
    text: str, value: float, su: float
) -> None:
    parsed = parse_cif_number(text)
    assert parsed.nominal == pytest.approx(value)
    assert parsed.su == pytest.approx(su)


def test_parse_cif_number_marks_absent_standard_uncertainty() -> None:
    parsed = parse_cif_number("1.5")
    assert parsed.nominal == 1.5
    assert np.isnan(parsed.su)


def test_parse_structure_block_keeps_uiso_separate_from_uij() -> None:
    block = gemmi.cif.read_string(
        """data_uiso
_cell_length_a 5.0(2)
_cell_length_b 6.0
_cell_length_c 7.0
_cell_angle_alpha 90
_cell_angle_beta 100
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_U_iso_or_equiv
_atom_site_thermal_displace_type
C1 C 0.10(2) 0.20 0.30 0.0144(8) Uiso
"""
    ).sole_block()

    record = parse_structure_block(block)

    assert record.adp.kind == ("Uiso",)
    assert record.adp.u_iso.tolist() == pytest.approx([0.0144])
    assert record.adp.u_iso_su.tolist() == pytest.approx([0.0008])
    assert np.isnan(record.adp.uij_cif).all()
    assert record.frac_positions[0, 0] == pytest.approx(0.10)
    assert record.frac_positions_su[0, 0] == pytest.approx(0.02)
    assert record.cell_parameters[0] == pytest.approx(5.0)
    assert record.cell_parameters_su[0] == pytest.approx(0.2)


def test_parse_structure_block_derives_all_symops_from_spacegroup_when_loop_missing() -> None:
    block = gemmi.cif.read_string(
        """data_symbol_only
_cell_length_a 5.0
_cell_length_b 6.0
_cell_length_c 7.0
_cell_angle_alpha 90
_cell_angle_beta 100
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'C 2/c'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.10 0.20 0.30
"""
    ).sole_block()

    record = parse_structure_block(block)

    assert record.n_symops == 8
    assert np.any(np.all(np.isclose(record.symops_t, [0.5, 0.5, 0.0]), axis=1))
    assert record.adp.kind == ("missing",)


def test_parse_observation_block_from_string() -> None:
    block = gemmi.cif.read_string(
        """data_pets
_cell_length_a 5.0
_cell_length_b 6.0
_cell_length_c 7.0
_cell_angle_alpha 90
_cell_angle_beta 100
_cell_angle_gamma 90
_diffrn_radiation_wavelength 0.0251
_diffrn_orient_matrix_UB_11 1
_diffrn_orient_matrix_UB_12 0
_diffrn_orient_matrix_UB_13 0
_diffrn_orient_matrix_UB_21 0
_diffrn_orient_matrix_UB_22 1
_diffrn_orient_matrix_UB_23 0
_diffrn_orient_matrix_UB_31 0
_diffrn_orient_matrix_UB_32 0
_diffrn_orient_matrix_UB_33 1
loop_
_diffrn_zone_axis_id
_diffrn_zone_axis_u
_diffrn_zone_axis_v
_diffrn_zone_axis_w
_diffrn_zone_axis_precession_angle
_diffrn_zone_axis_alpha
_diffrn_zone_axis_beta
_diffrn_zone_axis_omega
_diffrn_zone_axis_scale
1 0 0 1 0.5 10 20 30 1
loop_
_refln_index_h
_refln_index_k
_refln_index_l
_refln_intensity_meas
_refln_intensity_sigma
_refln_zone_axis_id
1 0 0 12.5 0.7 1
"""
    ).sole_block()

    record = parse_observation_block(block)

    assert record.source_path is None
    assert record.n_rotations == 1
    assert record.n_reflections == 1
    assert record.zone_axes.tolist() == [[0.0, 0.0, 1.0]]
    assert record.intensities.tolist() == [12.5]
    assert record.sigmas.tolist() == [0.7]
