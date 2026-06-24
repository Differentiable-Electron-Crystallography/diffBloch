import numpy as np
import pytest
from pydantic import ValidationError

from diffBloch.io import ObservationRecord, StructureRecord


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
            occupancies=np.asarray([1.5]),
            uij_cif=np.eye(3)[None, :, :] * 0.01,
        )


def test_structure_record_rejects_non_psd_adp() -> None:
    with pytest.raises(ValidationError, match="positive semidefinite"):
        StructureRecord(
            unit_cell=np.eye(3),
            spacegroup_hm="P1",
            symops_R=np.eye(3)[None, :, :],
            symops_t=np.zeros((1, 3)),
            labels=("Si1",),
            numbers=np.asarray([14]),
            frac_positions=np.zeros((1, 3)),
            occupancies=np.asarray([1.0]),
            uij_cif=-np.eye(3)[None, :, :],
        )


def test_observation_record_validates_sigmas_and_zone_ids() -> None:
    with pytest.raises(ValidationError, match="sigmas must be non-negative"):
        ObservationRecord(
            unit_cell=np.eye(3),
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
        ObservationRecord(
            unit_cell=np.eye(3),
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
