"""CIF structure reader backed by gemmi."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple

import gemmi
import numpy as np
from numpy.typing import NDArray

from diffBloch.core.crystal import cell_matrix_from_parameters
from diffBloch.io._cifio import (
    as_float,
    loop_rows,
    optional_int,
    parse_cif_number,
    read_document,
    select_block,
    unquote,
)
from diffBloch.io._cifio import (
    cell_parameters as parse_cell_parameters,
)
from diffBloch.io.record import AdpRecord, StructureRecord

ANISO_TAGS = (
    "_atom_site_aniso_U_11",
    "_atom_site_aniso_U_22",
    "_atom_site_aniso_U_33",
    "_atom_site_aniso_U_23",
    "_atom_site_aniso_U_13",
    "_atom_site_aniso_U_12",
)


class _AdpSite(NamedTuple):
    kind: Literal["Uiso", "Uani", "missing"]
    u_iso: float
    u_iso_su: float
    uij_cif: NDArray[np.float64]
    uij_cif_su: NDArray[np.float64]


class _AtomSites(NamedTuple):
    labels: tuple[str, ...]
    numbers: NDArray[np.int64]
    frac_positions: NDArray[np.float64]
    frac_positions_su: NDArray[np.float64]
    occupancies: NDArray[np.float64]
    occupancies_su: NDArray[np.float64]
    adp: AdpRecord


def read_structure(path: str | Path, *, load_hydrogens: bool = False) -> StructureRecord:
    """Read a structure CIF into a validated :class:`StructureRecord`.

    Args:
        path: CIF path.
        load_hydrogens: Include hydrogen atom sites when present. The default mirrors electron
            diffraction refinement practice where H sites are usually excluded from this boundary.
    """
    source = Path(path)
    block = select_block(read_document(source), required_loop_tag="_atom_site_label")
    return parse_structure_block(block, source_path=source, load_hydrogens=load_hydrogens)


def parse_structure_block(
    block: gemmi.cif.Block,
    *,
    source_path: str | Path | None = None,
    load_hydrogens: bool = False,
) -> StructureRecord:
    """Parse a Gemmi CIF block into a validated :class:`StructureRecord`.

    Args:
        block: Parsed Gemmi CIF block.
        source_path: Optional source path to retain in the record.
        load_hydrogens: Include hydrogen atom sites when present. The default mirrors electron
            diffraction refinement practice where H sites are usually excluded from this boundary.
    """
    atom_sites = _read_atom_sites(block, load_hydrogens=load_hydrogens)
    symops_R, symops_t = _read_symops(block)
    cell_parameters, cell_parameters_su = parse_cell_parameters(block)
    return StructureRecord(
        source_path=Path(source_path) if source_path is not None else None,
        unit_cell=cell_matrix_from_parameters(cell_parameters),
        cell_parameters=cell_parameters,
        cell_parameters_su=cell_parameters_su,
        spacegroup_hm=unquote(
            block.find_value("_symmetry_space_group_name_H-M")
            or block.find_value("_space_group_name_H-M_alt")
            or ""
        ),
        spacegroup_number=optional_int(
            block.find_value("_symmetry_Int_Tables_number")
            or block.find_value("_space_group_IT_number")
        ),
        symops_R=symops_R,
        symops_t=symops_t,
        labels=atom_sites.labels,
        numbers=atom_sites.numbers,
        frac_positions=atom_sites.frac_positions,
        frac_positions_su=atom_sites.frac_positions_su,
        occupancies=atom_sites.occupancies,
        occupancies_su=atom_sites.occupancies_su,
        adp=atom_sites.adp,
    )


def _read_atom_sites(block: gemmi.cif.Block, *, load_hydrogens: bool) -> _AtomSites:
    atom_rows = loop_rows(block, "_atom_site_label")
    aniso_by_label = {
        str(row["_atom_site_aniso_label"]): row
        for row in loop_rows(block, "_atom_site_aniso_label")
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
                as_float(row["_atom_site_fract_x"]),
                as_float(row["_atom_site_fract_y"]),
                as_float(row["_atom_site_fract_z"]),
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

    return _AtomSites(
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


def _read_symops(block: gemmi.cif.Block) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rows = loop_rows(block, "_symmetry_equiv_pos_as_xyz") or loop_rows(
        block, "_space_group_symop_operation_xyz"
    )
    symops = [
        unquote(
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
    name = unquote(
        block.find_value("_symmetry_space_group_name_H-M")
        or block.find_value("_space_group_name_H-M_alt")
        or ""
    )
    if name:
        spacegroup = gemmi.find_spacegroup_by_name(name)
        if spacegroup is not None:
            return spacegroup
    number = optional_int(
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
