"""PETS CIF-like experimental data reader."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import gemmi
import numpy as np
from numpy.typing import NDArray

from diffBloch.core.crystal import cell_matrix_from_parameters
from diffBloch.io._cifio import as_float, cell_parameters, loop_rows, read_document, required_float
from diffBloch.io.record import ExperimentalRecord

_DSTAR_MAX = re.compile(r"dstarmax:\s*([\d.]+)", re.IGNORECASE)
_FLOAT_TEXT = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?|inf(?:inity)?|nan)"
_MOSAICITY = re.compile(rf"mosaicity:\s*({_FLOAT_TEXT})", re.IGNORECASE)
_DATA_COLLECTION_GEOMETRY = re.compile(
    r"data\s+collection\s+geometry\s*:\s*([^\r\n;]+)", re.IGNORECASE
)
# PETS2 writes its informal `key: value` processing summary (data collection geometry, dstarmax,
# mosaicity, ...) as a semicolon-delimited text field, but different builds attach it to different
# CIF tags -- try each in order and use the first one present.
_MEASUREMENT_DETAILS_TAGS = ("_diffrn_measurement_details", "_diffrn_reflns_reduction_process")


def read_experimental_data(path: str | Path) -> ExperimentalRecord:
    """Read a PETS ``.cif_pets`` file into a validated :class:`ExperimentalRecord`."""
    source = Path(path)
    block = read_document(source).sole_block()
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
        data_collection_geometry=_data_collection_geometry(block),
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


def _measurement_details(block: gemmi.cif.Block) -> str | None:
    """Return PETS2's informal ``key: value`` processing summary text, wherever it was written."""
    for tag in _MEASUREMENT_DETAILS_TAGS:
        text = block.find_value(tag)
        if text is not None:
            return str(text)
    return None


def _dstar_max(block: gemmi.cif.Block) -> float | None:
    """PETS2's processing-resolution cutoff (Å⁻¹) from the free-text measurement-details block.

    That text is PETS2's own semicolon-delimited block (``dstarmax:  1.800`` among other informal
    ``key: value`` lines), not a structured CIF field, so this greps rather than parses it as CIF.
    Returns ``None`` when the tag is absent or a PETS version that doesn't record ``dstarmax``
    wrote the file.
    """
    text = _measurement_details(block)
    if text is None:
        return None
    match = _DSTAR_MAX.search(text)
    return float(match.group(1)) if match else None


def _mosaicity(block: gemmi.cif.Block) -> float | None:
    """PETS2 apparent mosaicity in degrees from measurement details."""
    text = _measurement_details(block)
    if text is None:
        return None
    match = _MOSAICITY.search(text)
    return float(match.group(1)) if match else None


def _data_collection_geometry(
    block: gemmi.cif.Block,
) -> Literal["continuous_rotation", "precession"]:
    """Return PETS2's acquisition geometry in the package's canonical spelling.

    PETS2 writes this value into an informal measurement-details text block rather than a
    structured CIF tag. Older files may omit it; those retain diffBloch's historical
    continuous-rotation default. An explicit unknown value fails at the I/O boundary rather than
    silently selecting scientifically different integration geometry.
    """
    text = _measurement_details(block)
    if text is None:
        return "continuous_rotation"
    match = _DATA_COLLECTION_GEOMETRY.search(text)
    if match is None:
        return "continuous_rotation"
    value = re.sub(r"[\s_-]+", "_", match.group(1).strip().lower())
    if value == "continuous_rotation":
        return "continuous_rotation"
    if value == "precession":
        return "precession"
    raise ValueError(
        "PETS data collection geometry must be 'continuous rotation' or 'precession'; "
        f"got {match.group(1).strip()!r}"
    )


def _ub_matrix(block: gemmi.cif.Block) -> NDArray[np.float64]:
    return np.asarray(
        [
            [required_float(block, f"_diffrn_orient_matrix_UB_{row}{col}") for col in range(1, 4)]
            for row in range(1, 4)
        ],
        dtype=np.float64,
    )
