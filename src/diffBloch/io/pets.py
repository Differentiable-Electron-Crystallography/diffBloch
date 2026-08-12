"""PETS CIF-like experimental data reader."""

from __future__ import annotations

import re
from pathlib import Path

import gemmi
import numpy as np
from numpy.typing import NDArray

from diffBloch.core.crystal import cell_matrix_from_parameters
from diffBloch.io._cifio import as_float, cell_parameters, loop_rows, required_float
from diffBloch.io.record import ExperimentalRecord

_DSTAR_MAX = re.compile(r"dstarmax:\s*([\d.]+)", re.IGNORECASE)
_FLOAT_TEXT = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?|inf(?:inity)?|nan)"
_MOSAICITY = re.compile(rf"mosaicity:\s*({_FLOAT_TEXT})", re.IGNORECASE)


def read_experimental_data(path: str | Path) -> ExperimentalRecord:
    """Read a PETS ``.cif_pets`` file into a validated :class:`ExperimentalRecord`."""
    source = Path(path)
    block = gemmi.cif.read_file(str(source)).sole_block()
    return parse_experimental_block(block, source_path=source)


def parse_experimental_block(
    block: gemmi.cif.Block, *, source_path: str | Path | None = None
) -> ExperimentalRecord:
    """Parse a Gemmi PETS CIF-like block into a validated :class:`ExperimentalRecord`."""
    zone_rows = loop_rows(block, "_diffrn_zone_axis_id")
    reflection_rows = loop_rows(block, "_refln_index_h")
    if not zone_rows:
        raise ValueError("PETS file does not contain a _diffrn_zone_axis loop")
    if not reflection_rows:
        raise ValueError("PETS file does not contain a _refln loop")

    cellpar, cellpar_su = cell_parameters(block)
    return ExperimentalRecord(
        source_path=Path(source_path) if source_path is not None else None,
        unit_cell=cell_matrix_from_parameters(cellpar),
        cell_parameters=cellpar,
        cell_parameters_su=cellpar_su,
        wavelength=required_float(block, "_diffrn_radiation_wavelength"),
        dstar_max=_dstar_max(block),
        mosaicity_degrees=_mosaicity(block),
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


def _dstar_max(block: gemmi.cif.Block) -> float | None:
    """PETS2's processing-resolution cutoff (Å⁻¹) from the free-text ``_diffrn_measurement_details``.

    That tag is PETS2's own semicolon-delimited text block (``dstarmax:  1.800`` among other
    informal ``key: value`` lines), not a structured CIF field, so this greps rather than parses it
    as CIF. Returns ``None`` when the tag is absent or a PETS version that doesn't record
    ``dstarmax`` wrote the file.
    """
    text = block.find_value("_diffrn_measurement_details")
    if text is None:
        return None
    match = _DSTAR_MAX.search(str(text))
    return float(match.group(1)) if match else None


def _mosaicity(block: gemmi.cif.Block) -> float | None:
    """PETS2 apparent mosaicity in degrees from measurement details."""
    text = block.find_value("_diffrn_measurement_details")
    if text is None:
        return None
    match = _MOSAICITY.search(str(text))
    return float(match.group(1)) if match else None


def _ub_matrix(block: gemmi.cif.Block) -> NDArray[np.float64]:
    return np.asarray(
        [
            [required_float(block, f"_diffrn_orient_matrix_UB_{row}{col}") for col in range(1, 4)]
            for row in range(1, 4)
        ],
        dtype=np.float64,
    )
