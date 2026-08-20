"""PETS CIF-like experimental data reader."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, NamedTuple

import gemmi
import numpy as np
from numpy.typing import NDArray

from diffBloch.core.crystal import cell_matrix_from_parameters
from diffBloch.io._cifio import (
    as_float,
    cell_parameters,
    loop_rows,
    read_document_with_diagnostics,
    required_float,
)
from diffBloch.io.diagnostics import ParseDiagnostic, ParsedInput
from diffBloch.io.record import ExperimentalRecord

_DSTAR_MAX = re.compile(r"dstarmax:\s*([\d.]+)", re.IGNORECASE)
_FLOAT_TEXT = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?|inf(?:inity)?|nan)"
_MOSAICITY = re.compile(rf"mosaicity:\s*({_FLOAT_TEXT})", re.IGNORECASE)
_ROTATION_AXIS_POSITION = re.compile(rf"tilt\s+axis\s+position:\s*({_FLOAT_TEXT})", re.IGNORECASE)
_DATA_COLLECTION_GEOMETRY = re.compile(
    r"data\s+collection\s+geometry\s*:\s*([^\r\n;]+)", re.IGNORECASE
)
# PETS2 writes its informal `key: value` processing summary (data collection geometry, dstarmax,
# mosaicity, ...) as a semicolon-delimited text field, but different builds attach it to different
# CIF tags. For each key, try the tags in order and use the first text field containing that key.
_MEASUREMENT_DETAILS_TAGS = ("_diffrn_measurement_details", "_diffrn_reflns_reduction_process")


class _MeasurementDetails(NamedTuple):
    text: str
    tag: str


def read_experimental_data(path: str | Path) -> ExperimentalRecord:
    """Read a PETS ``.cif_pets`` file into a validated :class:`ExperimentalRecord`."""
    return read_experimental_data_with_diagnostics(path).record


def read_experimental_data_with_diagnostics(path: str | Path) -> ParsedInput[ExperimentalRecord]:
    """Read a PETS ``.cif_pets`` file and report non-fatal parser decisions."""
    source = Path(path)
    doc, diagnostics = read_document_with_diagnostics(source, input_kind="experimental_data")
    block = doc.sole_block()
    record = parse_experimental_block(block, source_path=source)
    return ParsedInput(
        record,
        diagnostics + _experimental_parse_diagnostics(block, source_path=source),
    )


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
        rotation_axis_position_degrees=_rotation_axis_position(block),
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


def _measurement_details(
    block: gemmi.cif.Block, pattern: re.Pattern[str]
) -> _MeasurementDetails | None:
    """Return the PETS2 summary text field containing ``pattern``, wherever it was written."""
    for tag in _MEASUREMENT_DETAILS_TAGS:
        text = block.find_value(tag)
        if text is not None and pattern.search(str(text)):
            return _MeasurementDetails(str(text), tag)
    return None


def _dstar_max(block: gemmi.cif.Block) -> float | None:
    """PETS2's processing-resolution cutoff (Å⁻¹) from the free-text measurement-details block.

    That text is PETS2's own semicolon-delimited block (``dstarmax:  1.800`` among other informal
    ``key: value`` lines), not a structured CIF field, so this greps rather than parses it as CIF.
    Returns ``None`` when the tag is absent or a PETS version that doesn't record ``dstarmax``
    wrote the file.
    """
    details = _measurement_details(block, _DSTAR_MAX)
    if details is None:
        return None
    match = _DSTAR_MAX.search(details.text)
    return float(match.group(1)) if match else None


def _mosaicity(block: gemmi.cif.Block) -> float | None:
    """PETS2 apparent mosaicity in degrees from measurement details."""
    details = _measurement_details(block, _MOSAICITY)
    if details is None:
        return None
    match = _MOSAICITY.search(details.text)
    return float(match.group(1)) if match else None


def _rotation_axis_position(block: gemmi.cif.Block) -> float | None:
    """PETS2's tilt-axis azimuthal offset (degrees) from measurement details.

    None when absent (an older PETS build that never wrote it) -- ``orientation_matrices`` treats
    that as coinciding with x, the goniometer axis convention every rotation/rocking-curve tilt in
    this package is expressed in, so no correction is applied.
    """
    details = _measurement_details(block, _ROTATION_AXIS_POSITION)
    if details is None:
        return None
    match = _ROTATION_AXIS_POSITION.search(details.text)
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
    details = _measurement_details(block, _DATA_COLLECTION_GEOMETRY)
    if details is None:
        return "continuous_rotation"
    match = _DATA_COLLECTION_GEOMETRY.search(details.text)
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


def _experimental_parse_diagnostics(
    block: gemmi.cif.Block, *, source_path: Path
) -> tuple[ParseDiagnostic, ...]:
    diagnostics: list[ParseDiagnostic] = []
    summary_fields = (
        ("data_collection_geometry", _DATA_COLLECTION_GEOMETRY),
        ("dstarmax", _DSTAR_MAX),
        ("mosaicity", _MOSAICITY),
        ("tilt axis position", _ROTATION_AXIS_POSITION),
    )
    used_by_tag: dict[str, list[str]] = {}
    absent_optional: list[str] = []
    for field, pattern in summary_fields:
        details = _measurement_details(block, pattern)
        if details is None:
            if field == "data_collection_geometry":
                diagnostics.append(
                    ParseDiagnostic(
                        code="pets_geometry_defaulted",
                        input_kind="experimental_data",
                        source_path=source_path,
                        message=(
                            "PETS data collection geometry absent; defaulted to continuous_rotation"
                        ),
                        details={"field": field, "default": "continuous_rotation"},
                    )
                )
            else:
                absent_optional.append(field)
            continue
        used_by_tag.setdefault(details.tag, []).append(field)

    for tag, fields in sorted(used_by_tag.items()):
        diagnostics.append(
            ParseDiagnostic(
                code="pets_summary_tag_used",
                input_kind="experimental_data",
                source_path=source_path,
                message=f"read PETS summary field(s) {', '.join(fields)} from {tag}",
                details={"tag": tag, "fields": ", ".join(fields)},
            )
        )
    if absent_optional:
        diagnostics.append(
            ParseDiagnostic(
                code="pets_optional_metadata_absent",
                input_kind="experimental_data",
                source_path=source_path,
                message=f"PETS optional metadata absent: {', '.join(absent_optional)}",
                details={"fields": ", ".join(absent_optional)},
            )
        )
    return tuple(diagnostics)


def _ub_matrix(block: gemmi.cif.Block) -> NDArray[np.float64]:
    return np.asarray(
        [
            [required_float(block, f"_diffrn_orient_matrix_UB_{row}{col}") for col in range(1, 4)]
            for row in range(1, 4)
        ],
        dtype=np.float64,
    )
