"""CIF structure reader backed by gemmi."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gemmi
import numpy as np
from numpy.typing import NDArray

from diffBloch.io.record import StructureRecord


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
    occupancies: list[float] = []
    uij_cif: list[NDArray[np.float64]] = []
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
        occupancies.append(_as_float(row.get("_atom_site_occupancy", "1.0")))
        uij_cif.append(_uij_for_site(label, row, aniso_by_label))

    symops_R, symops_t = _read_symops(block)
    return StructureRecord(
        source_path=source,
        unit_cell=_unit_cell_matrix(block),
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
        occupancies=np.asarray(occupancies, dtype=np.float64),
        uij_cif=np.asarray(uij_cif, dtype=np.float64),
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


def _unit_cell_matrix(block: gemmi.cif.Block) -> NDArray[np.float64]:
    a = _required_float(block, "_cell_length_a")
    b = _required_float(block, "_cell_length_b")
    c = _required_float(block, "_cell_length_c")
    alpha = np.deg2rad(_required_float(block, "_cell_angle_alpha"))
    beta = np.deg2rad(_required_float(block, "_cell_angle_beta"))
    gamma = np.deg2rad(_required_float(block, "_cell_angle_gamma"))

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
        symops = ["x,y,z"]

    rotations: list[list[list[float]]] = []
    translations: list[list[float]] = []
    for operation in symops:
        op = gemmi.Op(operation)
        rotations.append([[float(op.rot[i][j]) / op.DEN for j in range(3)] for i in range(3)])
        translations.append([float(op.tran[i]) / op.DEN for i in range(3)])
    return np.asarray(rotations, dtype=np.float64), np.asarray(translations, dtype=np.float64)


def _uij_for_site(
    label: str,
    atom_row: dict[str, str],
    aniso_by_label: dict[str, dict[str, str]],
) -> NDArray[np.float64]:
    if label in aniso_by_label:
        row = aniso_by_label[label]
        return np.asarray(
            [
                [
                    _as_float(row["_atom_site_aniso_U_11"]),
                    _as_float(row["_atom_site_aniso_U_12"]),
                    _as_float(row["_atom_site_aniso_U_13"]),
                ],
                [
                    _as_float(row["_atom_site_aniso_U_12"]),
                    _as_float(row["_atom_site_aniso_U_22"]),
                    _as_float(row["_atom_site_aniso_U_23"]),
                ],
                [
                    _as_float(row["_atom_site_aniso_U_13"]),
                    _as_float(row["_atom_site_aniso_U_23"]),
                    _as_float(row["_atom_site_aniso_U_33"]),
                ],
            ],
            dtype=np.float64,
        )
    u_iso = _as_float(atom_row.get("_atom_site_U_iso_or_equiv", "0.0"))
    return np.eye(3, dtype=np.float64) * u_iso


def _required_float(block: gemmi.cif.Block, tag: str) -> float:
    value = block.find_value(tag)
    if value is None:
        raise ValueError(f"missing required CIF tag {tag}")
    return _as_float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or str(value) in {".", "?"}:
        return None
    return int(str(value))


def _as_float(value: Any) -> float:
    text = str(value).strip()
    if "(" in text:
        text = text.split("(", 1)[0]
    return float(text)


def _unquote(value: str) -> str:
    return value.strip().strip("'\"")
