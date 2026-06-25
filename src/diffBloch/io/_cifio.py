"""Shared CIF parsing primitives for structure and PETS readers."""

from __future__ import annotations

import re
from typing import Any, NamedTuple

import gemmi
import numpy as np
from numpy.typing import NDArray

_NUMERIC_WITH_SU = re.compile(
    r"^(?P<nominal>[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?)"
    r"(?:\((?P<su>\d+)\))?$"
)


class CifNumber(NamedTuple):
    """CIF numeric value with optional standard uncertainty."""

    nominal: float
    su: float


def parse_cif_number(value: Any) -> CifNumber:
    """Parse a CIF number and optional standard uncertainty.

    A parenthesized SU is expressed in units of the final significant digit of the mantissa, so
    ``0.0144(8)`` has SU ``0.0008`` and ``42(3)`` has SU ``3``.
    """
    text = str(value).strip()
    if text in {".", "?"}:
        return CifNumber(np.nan, np.nan)
    match = _NUMERIC_WITH_SU.match(text)
    if match is None:
        return CifNumber(float(text), np.nan)
    nominal_text = match.group("nominal")
    su_digits = match.group("su")
    if su_digits is None:
        return CifNumber(float(nominal_text), np.nan)

    mantissa = nominal_text.lower().split("e", 1)[0]
    exponent = int(nominal_text.lower().split("e", 1)[1]) if "e" in nominal_text.lower() else 0
    decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
    su = int(su_digits) * 10.0 ** (exponent - decimals)
    return CifNumber(float(nominal_text), float(su))


def loop_rows(block: gemmi.cif.Block, first_tag: str) -> list[dict[str, str]]:
    """Return one dict per row for the loop containing ``first_tag``."""
    column = block.find_loop(first_tag)
    if not column:
        return []
    loop = column.get_loop()
    if loop is None:
        return []
    tags = [str(tag) for tag in loop.tags]
    width = int(loop.width())
    rows: list[dict[str, str]] = []
    for start in range(0, len(loop.values), width):
        values = [str(value) for value in loop.values[start : start + width]]
        rows.append(dict(zip(tags, values, strict=True)))
    return rows


def cell_parameters(block: gemmi.cif.Block) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return unit-cell parameters and shape-aligned SUs as ``(a, b, c, alpha, beta, gamma)``."""
    tags = (
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    )
    parsed = [parse_cif_number(required_value(block, tag)) for tag in tags]
    return (
        np.asarray([value.nominal for value in parsed], dtype=np.float64),
        np.asarray([value.su for value in parsed], dtype=np.float64),
    )


def unit_cell_matrix(block: gemmi.cif.Block) -> NDArray[np.float64]:
    """Return the fractional-to-Cartesian cell matrix for ``block``."""
    return unit_cell_matrix_from_parameters(cell_parameters(block)[0])


def unit_cell_matrix_from_parameters(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the fractional-to-Cartesian cell matrix for six cell parameters."""
    a, b, c, alpha_deg, beta_deg, gamma_deg = parameters
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    gamma = np.deg2rad(gamma_deg)

    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    volume_factor = np.sqrt(
        1.0 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2.0 * cos_alpha * cos_beta * cos_gamma
    )
    return np.asarray(
        [
            [a, 0.0, 0.0],
            [b * cos_gamma, b * sin_gamma, 0.0],
            [
                c * cos_beta,
                c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
                c * volume_factor / sin_gamma,
            ],
        ],
        dtype=np.float64,
    )


def required_value(block: gemmi.cif.Block, tag: str) -> str:
    """Return a required scalar CIF value with a clear missing-tag error."""
    value = block.find_value(tag)
    if value is None:
        raise ValueError(f"missing required CIF tag {tag}")
    return str(value)


def required_float(block: gemmi.cif.Block, tag: str) -> float:
    """Return a required scalar CIF value parsed through ``parse_cif_number``."""
    return parse_cif_number(required_value(block, tag)).nominal


def optional_int(value: Any) -> int | None:
    """Return ``None`` for absent CIF integer placeholders."""
    if value is None or str(value) in {".", "?"}:
        return None
    return int(str(value))


def as_float(value: Any) -> float:
    """Return the nominal value from a CIF numeric string."""
    return parse_cif_number(value).nominal


def unquote(value: str) -> str:
    """Strip CIF quote characters from a scalar string."""
    return value.strip().strip("'\"")
