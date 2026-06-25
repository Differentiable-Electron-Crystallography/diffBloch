"""PETS CIF-like observation reader."""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
from numpy.typing import NDArray

from diffBloch.io._cifio import as_float, loop_rows, required_float, unit_cell_matrix
from diffBloch.io.record import ObservationRecord


def read_observations(path: str | Path) -> ObservationRecord:
    """Read a PETS ``.cif_pets`` file into a validated :class:`ObservationRecord`."""
    source = Path(path)
    block = gemmi.cif.read_file(str(source)).sole_block()
    return parse_observation_block(block, source_path=source)


def parse_observation_block(
    block: gemmi.cif.Block, *, source_path: str | Path | None = None
) -> ObservationRecord:
    """Parse a Gemmi PETS CIF-like block into a validated :class:`ObservationRecord`."""
    zone_rows = loop_rows(block, "_diffrn_zone_axis_id")
    reflection_rows = loop_rows(block, "_refln_index_h")
    if not zone_rows:
        raise ValueError("PETS file does not contain a _diffrn_zone_axis loop")
    if not reflection_rows:
        raise ValueError("PETS file does not contain a _refln loop")

    return ObservationRecord(
        source_path=Path(source_path) if source_path is not None else None,
        unit_cell=unit_cell_matrix(block),
        wavelength=required_float(block, "_diffrn_radiation_wavelength"),
        ub_matrix=_ub_matrix(block),
        zone_axis_ids=np.asarray(
            [int(row["_diffrn_zone_axis_id"]) for row in zone_rows], dtype=np.int64
        ),
        zone_axes=np.asarray(
            [
                [
                    as_float(row["_diffrn_zone_axis_u"]),
                    as_float(row["_diffrn_zone_axis_v"]),
                    as_float(row["_diffrn_zone_axis_w"]),
                ]
                for row in zone_rows
            ],
            dtype=np.float64,
        ),
        precession_angles=np.asarray(
            [as_float(row["_diffrn_zone_axis_precession_angle"]) for row in zone_rows],
            dtype=np.float64,
        ),
        alphas=np.asarray(
            [as_float(row["_diffrn_zone_axis_alpha"]) for row in zone_rows], dtype=np.float64
        ),
        betas=np.asarray(
            [as_float(row["_diffrn_zone_axis_beta"]) for row in zone_rows], dtype=np.float64
        ),
        omegas=np.asarray(
            [as_float(row["_diffrn_zone_axis_omega"]) for row in zone_rows], dtype=np.float64
        ),
        scales=np.asarray(
            [as_float(row["_diffrn_zone_axis_scale"]) for row in zone_rows], dtype=np.float64
        ),
        hkl=np.asarray(
            [
                [
                    int(row["_refln_index_h"]),
                    int(row["_refln_index_k"]),
                    int(row["_refln_index_l"]),
                ]
                for row in reflection_rows
            ],
            dtype=np.int64,
        ),
        intensities=np.asarray(
            [as_float(row["_refln_intensity_meas"]) for row in reflection_rows], dtype=np.float64
        ),
        sigmas=np.asarray(
            [as_float(row["_refln_intensity_sigma"]) for row in reflection_rows], dtype=np.float64
        ),
        reflection_zone_axis_ids=np.asarray(
            [int(row["_refln_zone_axis_id"]) for row in reflection_rows], dtype=np.int64
        ),
    )


def _ub_matrix(block: gemmi.cif.Block) -> NDArray[np.float64]:
    return np.asarray(
        [
            [required_float(block, f"_diffrn_orient_matrix_UB_{row}{col}") for col in range(1, 4)]
            for row in range(1, 4)
        ],
        dtype=np.float64,
    )
