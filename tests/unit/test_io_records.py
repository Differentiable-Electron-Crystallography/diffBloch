import gemmi
import numpy as np
import pytest
from pydantic import ValidationError

from diffBloch.io import AdpRecord, ExperimentalRecord, StructureRecord
from diffBloch.io._cifio import loop_rows


def test_loop_rows_reads_every_row_of_a_multi_row_loop_in_order() -> None:
    # Guards the loop_rows parse: it binds loop.values once (the O(N^2)->O(N) fix a large PETS
    # reflection loop exposed) and must still return one correctly-keyed dict per row, in order.
    block = gemmi.cif.read_string("data_t\nloop_\n_a\n_b\n1 2\n3 4\n5 6\n").sole_block()
    assert loop_rows(block, "_a") == [
        {"_a": "1", "_b": "2"},
        {"_a": "3", "_b": "4"},
        {"_a": "5", "_b": "6"},
    ]


def test_structure_record_validates_shapes_and_physical_bounds() -> None:
    with pytest.raises(ValidationError, match="occupancies must be in"):
        StructureRecord(
            unit_cell=np.eye(3),
            spacegroup_hm="P1",
            symops_R=np.eye(3)[None, :, :],
            symops_t=np.zeros((1, 3)),
            labels=("Si1",),
            numbers=np.asarray([14]),
            frac_positions=np.zeros((1, 3)),
            frac_positions_su=np.full((1, 3), np.nan),
            occupancies=np.asarray([1.5]),
            occupancies_su=np.asarray([np.nan]),
            cell_parameters=np.asarray([1.0, 1.0, 1.0, 90.0, 90.0, 90.0]),
            cell_parameters_su=np.full((6,), np.nan),
            adp=AdpRecord(
                kind=("Uani",),
                u_iso=np.asarray([0.01]),
                u_iso_su=np.asarray([np.nan]),
                uij_cif=np.eye(3)[None, :, :] * 0.01,
                uij_cif_su=np.full((1, 3, 3), np.nan),
            ),
        )


def test_structure_record_rejects_non_psd_adp() -> None:
    with pytest.raises(ValidationError, match="positive semidefinite"):
        AdpRecord(
            kind=("Uani",),
            u_iso=np.asarray([0.01]),
            u_iso_su=np.asarray([np.nan]),
            uij_cif=-np.eye(3)[None, :, :],
            uij_cif_su=np.full((1, 3, 3), np.nan),
        )


def test_adp_record_validates_kind_specific_storage() -> None:
    with pytest.raises(ValidationError, match="finite u_iso"):
        AdpRecord(
            kind=("Uiso",),
            u_iso=np.asarray([np.nan]),
            u_iso_su=np.asarray([np.nan]),
            uij_cif=np.full((1, 3, 3), np.nan),
            uij_cif_su=np.full((1, 3, 3), np.nan),
        )


def test_adp_record_freezes_fields_and_arrays() -> None:
    record = AdpRecord(
        kind=("Uiso",),
        u_iso=np.asarray([0.01]),
        u_iso_su=np.asarray([np.nan]),
        uij_cif=np.full((1, 3, 3), np.nan),
        uij_cif_su=np.full((1, 3, 3), np.nan),
    )

    with pytest.raises(ValidationError, match="frozen"):
        record.u_iso = np.asarray([0.02])
    with pytest.raises(ValueError, match="read-only"):
        record.u_iso[0] = 0.02


def test_structure_record_rejects_negative_standard_uncertainties() -> None:
    with pytest.raises(ValidationError, match="frac_positions_su"):
        StructureRecord(
            unit_cell=np.eye(3),
            cell_parameters=np.asarray([1.0, 1.0, 1.0, 90.0, 90.0, 90.0]),
            cell_parameters_su=np.full((6,), np.nan),
            spacegroup_hm="P1",
            symops_R=np.eye(3)[None, :, :],
            symops_t=np.zeros((1, 3)),
            labels=("Si1",),
            numbers=np.asarray([14]),
            frac_positions=np.zeros((1, 3)),
            frac_positions_su=np.asarray([[-1.0, np.nan, np.nan]]),
            occupancies=np.asarray([1.0]),
            occupancies_su=np.asarray([np.nan]),
            adp=AdpRecord(
                kind=("Uiso",),
                u_iso=np.asarray([0.01]),
                u_iso_su=np.asarray([np.nan]),
                uij_cif=np.full((1, 3, 3), np.nan),
                uij_cif_su=np.full((1, 3, 3), np.nan),
            ),
        )


def test_observation_record_validates_sigmas_and_zone_ids() -> None:
    with pytest.raises(ValidationError, match="sigmas must be non-negative"):
        ExperimentalRecord(
            unit_cell=np.eye(3),
            cell_parameters=np.asarray([1.0, 1.0, 1.0, 90.0, 90.0, 90.0]),
            cell_parameters_su=np.full((6,), np.nan),
            wavelength=0.0251,
            ub_matrix=np.eye(3),
            zone_axis_ids=np.asarray([1]),
            zone_axes=np.zeros((1, 3)),
            precession_angles=np.asarray([1.0]),
            alphas=np.asarray([0.0]),
            betas=np.asarray([0.0]),
            omegas=np.asarray([0.0]),
            scales=np.asarray([1.0]),
            hkl=np.zeros((1, 3), dtype=np.int64),
            intensities=np.asarray([10.0]),
            sigmas=np.asarray([-1.0]),
            reflection_zone_axis_ids=np.asarray([1]),
        )


def test_observation_record_rejects_undeclared_reflection_zone_ids() -> None:
    with pytest.raises(ValidationError, match="not declared"):
        ExperimentalRecord(
            unit_cell=np.eye(3),
            cell_parameters=np.asarray([1.0, 1.0, 1.0, 90.0, 90.0, 90.0]),
            cell_parameters_su=np.full((6,), np.nan),
            wavelength=0.0251,
            ub_matrix=np.eye(3),
            zone_axis_ids=np.asarray([1]),
            zone_axes=np.zeros((1, 3)),
            precession_angles=np.asarray([1.0]),
            alphas=np.asarray([0.0]),
            betas=np.asarray([0.0]),
            omegas=np.asarray([0.0]),
            scales=np.asarray([1.0]),
            hkl=np.zeros((1, 3), dtype=np.int64),
            intensities=np.asarray([10.0]),
            sigmas=np.asarray([1.0]),
            reflection_zone_axis_ids=np.asarray([2]),
        )
