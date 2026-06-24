from pathlib import Path

import numpy as np

from diffBloch.io import read_observations, read_structure, symmetry_constraints

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"


def test_read_quartz_structure_fixture() -> None:
    record = read_structure(FIXTURE_ROOT / "enantiomer_1.cif")

    assert record.labels == ("Si1", "O1")
    assert record.numbers.tolist() == [14, 8]
    assert record.frac_positions.shape == (2, 3)
    assert record.occupancies.tolist() == [1.0, 1.0]
    assert record.uij_cif.shape == (2, 3, 3)
    assert np.linalg.eigvalsh(record.uij_cif).min() > 0.0
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
