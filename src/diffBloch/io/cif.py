"""CIF structure reader backed by gemmi."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, NamedTuple

import gemmi
import numpy as np
from numpy.typing import NDArray

from diffBloch.io.record import AdpRecord, StructureRecord

ANISO_TAGS = (
    "_atom_site_aniso_U_11",
    "_atom_site_aniso_U_22",
    "_atom_site_aniso_U_33",
    "_atom_site_aniso_U_23",
    "_atom_site_aniso_U_13",
    "_atom_site_aniso_U_12",
)
_NUMERIC_WITH_SU = re.compile(
    r"^(?P<nominal>[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?)"
    r"(?:\((?P<su>\d+)\))?$"
)


class CifNumber(NamedTuple):
    """CIF numeric value with optional standard uncertainty."""

    nominal: float
    su: float


class _AdpSite(NamedTuple):
    kind: Literal["Uiso", "Uani", "missing"]
    u_iso: float
    u_iso_su: float
    uij_cif: NDArray[np.float64]
    uij_cif_su: NDArray[np.float64]


def read_structure(
    path: str | Path, *, backend: str = "gemmi", load_hydrogens: bool = False
) -> StructureRecord:
    """Read a structure CIF into a validated :class:`StructureRecord`.

    Args:
        path: CIF path.
        backend: Parser backend. Stage 3 intentionally supports only ``"gemmi"``.
        load_hydrogens: Include hydrogen atom sites when present. The default mirrors electron
            diffraction refinement practice where H sites are usually excluded from this boundary.
    """
    if backend != "gemmi":
        raise ValueError(f"unsupported CIF backend: {backend}")

    source = Path(path)
    block = gemmi.cif.read_file(str(source)).sole_block()
    atom_rows = _loop_rows(block, "_atom_site_label")
    aniso_by_label = {
        str(row["_atom_site_aniso_label"]): row
        for row in _loop_rows(block, "_atom_site_aniso_label")
    }

    labels: list[str] = []
    numbers: list[int] = []
    frac_positions: list[list[float]] = []
    frac_positions_su: list[list[float]] = []
    occupancies: list[float] = []
    occupancies_su: list[float] = []
    adp_kind: list[Literal["Uiso", "Uani", "missing"]] = []
    u_iso: list[float] = []
    u_iso_su: list[float] = []
    uij_cif: list[NDArray[np.float64]] = []
    uij_cif_su: list[NDArray[np.float64]] = []
    for row in atom_rows:
        element = gemmi.Element(str(row["_atom_site_type_symbol"]))
        if not load_hydrogens and element.atomic_number == 1:
            continue
        label = str(row["_atom_site_label"])
        labels.append(label)
        numbers.append(int(element.atomic_number))
        frac_positions.append(
            [
                _as_float(row["_atom_site_fract_x"]),
                _as_float(row["_atom_site_fract_y"]),
                _as_float(row["_atom_site_fract_z"]),
            ]
        )
        frac_positions_su.append(
            [
                parse_cif_number(row["_atom_site_fract_x"]).su,
                parse_cif_number(row["_atom_site_fract_y"]).su,
                parse_cif_number(row["_atom_site_fract_z"]).su,
            ]
        )
        occupancy = parse_cif_number(row.get("_atom_site_occupancy", "1.0"))
        occupancies.append(occupancy.nominal)
        occupancies_su.append(occupancy.su)
        adp = _adp_for_site(label, row, aniso_by_label)
        adp_kind.append(adp.kind)
        u_iso.append(adp.u_iso)
        u_iso_su.append(adp.u_iso_su)
        uij_cif.append(adp.uij_cif)
        uij_cif_su.append(adp.uij_cif_su)

    symops_R, symops_t = _read_symops(block)
    cell_parameters, cell_parameters_su = _cell_parameters(block)
    return StructureRecord(
        source_path=source,
        unit_cell=_unit_cell_matrix_from_parameters(cell_parameters),
        cell_parameters=cell_parameters,
        cell_parameters_su=cell_parameters_su,
        spacegroup_hm=_unquote(
            block.find_value("_symmetry_space_group_name_H-M")
            or block.find_value("_space_group_name_H-M_alt")
            or ""
        ),
        spacegroup_number=_optional_int(
            block.find_value("_symmetry_Int_Tables_number")
            or block.find_value("_space_group_IT_number")
        ),
        symops_R=symops_R,
        symops_t=symops_t,
        labels=tuple(labels),
        numbers=np.asarray(numbers, dtype=np.int64),
        frac_positions=np.asarray(frac_positions, dtype=np.float64),
        frac_positions_su=np.asarray(frac_positions_su, dtype=np.float64),
        occupancies=np.asarray(occupancies, dtype=np.float64),
        occupancies_su=np.asarray(occupancies_su, dtype=np.float64),
        adp=AdpRecord(
            kind=tuple(adp_kind),
            u_iso=np.asarray(u_iso, dtype=np.float64),
            u_iso_su=np.asarray(u_iso_su, dtype=np.float64),
            uij_cif=np.asarray(uij_cif, dtype=np.float64),
            uij_cif_su=np.asarray(uij_cif_su, dtype=np.float64),
        ),
    )


def _loop_rows(block: gemmi.cif.Block, first_tag: str) -> list[dict[str, str]]:
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


def _cell_parameters(block: gemmi.cif.Block) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    tags = (
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    )
    parsed = [parse_cif_number(_required_value(block, tag)) for tag in tags]
    return (
        np.asarray([value.nominal for value in parsed], dtype=np.float64),
        np.asarray([value.su for value in parsed], dtype=np.float64),
    )


def _unit_cell_matrix(block: gemmi.cif.Block) -> NDArray[np.float64]:
    return _unit_cell_matrix_from_parameters(_cell_parameters(block)[0])


def _unit_cell_matrix_from_parameters(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
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


def _read_symops(block: gemmi.cif.Block) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rows = _loop_rows(block, "_symmetry_equiv_pos_as_xyz") or _loop_rows(
        block, "_space_group_symop_operation_xyz"
    )
    symops = [
        _unquote(
            str(row.get("_symmetry_equiv_pos_as_xyz") or row["_space_group_symop_operation_xyz"])
        )
        for row in rows
    ]
    if not symops:
        spacegroup = _spacegroup_for_block(block)
        if spacegroup is None:
            raise ValueError("CIF must provide symmetry operations or a space-group symbol/number")
        symops = [op.triplet() for op in spacegroup.operations()]

    rotations: list[list[list[float]]] = []
    translations: list[list[float]] = []
    for operation in symops:
        op = gemmi.Op(operation)
        rotations.append([[float(op.rot[i][j]) / op.DEN for j in range(3)] for i in range(3)])
        translations.append([float(op.tran[i]) / op.DEN for i in range(3)])
    return np.asarray(rotations, dtype=np.float64), np.asarray(translations, dtype=np.float64)


def _spacegroup_for_block(block: gemmi.cif.Block) -> gemmi.SpaceGroup | None:
    name = _unquote(
        block.find_value("_symmetry_space_group_name_H-M")
        or block.find_value("_space_group_name_H-M_alt")
        or ""
    )
    if name:
        spacegroup = gemmi.find_spacegroup_by_name(name)
        if spacegroup is not None:
            return spacegroup
    number = _optional_int(
        block.find_value("_symmetry_Int_Tables_number")
        or block.find_value("_space_group_IT_number")
    )
    if number is not None:
        return gemmi.find_spacegroup_by_number(number)
    return None


def _adp_for_site(
    label: str,
    atom_row: dict[str, str],
    aniso_by_label: dict[str, dict[str, str]],
) -> _AdpSite:
    u_iso_value = parse_cif_number(atom_row.get("_atom_site_U_iso_or_equiv", "."))
    if label in aniso_by_label:
        row = aniso_by_label[label]
        uij_values = _uij_matrix({tag: parse_cif_number(row[tag]).nominal for tag in ANISO_TAGS})
        uij_su = _uij_matrix({tag: parse_cif_number(row[tag]).su for tag in ANISO_TAGS})
        return _AdpSite(
            kind="Uani",
            u_iso=u_iso_value.nominal,
            u_iso_su=u_iso_value.su,
            uij_cif=uij_values,
            uij_cif_su=uij_su,
        )
    if np.isfinite(u_iso_value.nominal):
        return _AdpSite(
            kind="Uiso",
            u_iso=u_iso_value.nominal,
            u_iso_su=u_iso_value.su,
            uij_cif=np.full((3, 3), np.nan, dtype=np.float64),
            uij_cif_su=np.full((3, 3), np.nan, dtype=np.float64),
        )
    return _AdpSite(
        kind="missing",
        u_iso=np.nan,
        u_iso_su=np.nan,
        uij_cif=np.full((3, 3), np.nan, dtype=np.float64),
        uij_cif_su=np.full((3, 3), np.nan, dtype=np.float64),
    )


def _uij_matrix(values: dict[str, float]) -> NDArray[np.float64]:
    return np.asarray(
        [
            [
                values["_atom_site_aniso_U_11"],
                values["_atom_site_aniso_U_12"],
                values["_atom_site_aniso_U_13"],
            ],
            [
                values["_atom_site_aniso_U_12"],
                values["_atom_site_aniso_U_22"],
                values["_atom_site_aniso_U_23"],
            ],
            [
                values["_atom_site_aniso_U_13"],
                values["_atom_site_aniso_U_23"],
                values["_atom_site_aniso_U_33"],
            ],
        ],
        dtype=np.float64,
    )


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


def _required_value(block: gemmi.cif.Block, tag: str) -> str:
    value = block.find_value(tag)
    if value is None:
        raise ValueError(f"missing required CIF tag {tag}")
    return str(value)


def _required_float(block: gemmi.cif.Block, tag: str) -> float:
    return _as_float(_required_value(block, tag))


def _optional_int(value: Any) -> int | None:
    if value is None or str(value) in {".", "?"}:
        return None
    return int(str(value))


def _as_float(value: Any) -> float:
    return parse_cif_number(value).nominal


def _unquote(value: str) -> str:
    return value.strip().strip("'\"")
