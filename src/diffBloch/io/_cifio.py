"""Shared CIF parsing primitives for structure and PETS readers."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import gemmi
import numpy as np
from numpy.typing import NDArray

from diffBloch.core.crystal import cell_matrix_from_parameters
from diffBloch.io.diagnostics import InputKind, ParseDetail, ParseDiagnostic

_TOP_LEVEL_TAG = re.compile(r"^(_\S+)(\s+\S.*)?$")

_NUMERIC_WITH_SU = re.compile(
    r"^(?P<nominal>[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?)"
    r"(?:\((?P<su>\d+)\))?$"
)


def read_document(path: str | Path) -> gemmi.cif.Document:
    """Read a CIF-like file into a :class:`gemmi.cif.Document`, tolerating duplicate scalar tags.

    Some PETS2 builds append a second, CIF-compliance-only section that restates a tag (e.g.
    ``_diffrn_radiation_wavelength``) already given earlier in the same block. That violates the
    CIF spec -- a data name may appear at most once per block -- so gemmi's strict parser rejects
    it outright. Drop later duplicates (keeping the first, higher-precision occurrence) before
    parsing.
    """
    text = Path(path).read_text()
    return gemmi.cif.read_string(_drop_duplicate_scalar_tags(text))


def read_document_with_diagnostics(
    path: str | Path, *, input_kind: InputKind
) -> tuple[gemmi.cif.Document, tuple[ParseDiagnostic, ...]]:
    """Read a CIF-like file and report non-fatal tolerance decisions made before parsing."""
    source = Path(path)
    text, dropped, kept_values = _drop_duplicate_scalar_tags_with_counts(source.read_text())
    diagnostics: list[ParseDiagnostic] = []
    for tag, count in sorted(dropped.items()):
        kept_value = kept_values.get(tag)
        if kept_value:
            message = f"dropped duplicate scalar CIF tag {tag}, kept first value {kept_value!r}"
            details: Mapping[str, ParseDetail] = {
                "tag": tag,
                "count": count,
                "kept_value": kept_value,
            }
        else:
            message = f"dropped duplicate scalar CIF tag {tag}, kept first value"
            details = {"tag": tag, "count": count}
        diagnostics.append(
            ParseDiagnostic(
                code="duplicate_scalar_tag_dropped",
                input_kind=input_kind,
                source_path=source,
                message=message,
                details=details,
            )
        )
    return gemmi.cif.read_string(text), tuple(diagnostics)


def _drop_duplicate_scalar_tags(text: str) -> str:
    return _drop_duplicate_scalar_tags_with_counts(text)[0]


def _drop_duplicate_scalar_tags_with_counts(
    text: str,
) -> tuple[str, Counter[str], dict[str, str]]:
    seen: set[str] = set()
    dropped: Counter[str] = Counter()
    kept_values: dict[str, str] = {}
    pending_kept_value_tag: str | None = None
    in_text_field = False
    in_loop_header = False
    skipping_duplicate_text_field = False
    skip_duplicate_value = False
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if skipping_duplicate_text_field:
            if stripped.startswith(";"):
                skipping_duplicate_text_field = False
            continue
        if skip_duplicate_value:
            if stripped.startswith(";"):
                skipping_duplicate_text_field = True
                skip_duplicate_value = False
            elif stripped and not stripped.startswith("#"):
                skip_duplicate_value = False
            continue
        if pending_kept_value_tag is not None:
            if stripped.startswith(";"):
                kept_values.setdefault(pending_kept_value_tag, "<text field>")
                pending_kept_value_tag = None
            elif stripped and not stripped.startswith("#"):
                kept_values.setdefault(pending_kept_value_tag, stripped)
                pending_kept_value_tag = None
        if stripped.startswith(";"):
            in_text_field = not in_text_field
            out.append(line)
            continue
        if in_text_field:
            out.append(line)
            continue
        if stripped.startswith("data_"):
            seen = set()
            pending_kept_value_tag = None
            in_loop_header = False
            out.append(line)
            continue
        if stripped == "loop_":
            in_loop_header = True
            out.append(line)
            continue
        match = _TOP_LEVEL_TAG.match(stripped)
        if in_loop_header:
            if not match:
                in_loop_header = False
            out.append(line)
            continue
        if match:
            tag = match.group(1)
            if tag in seen:
                dropped[tag] += 1
                if match.group(2) is None:
                    skip_duplicate_value = True
                continue
            seen.add(tag)
            if match.group(2) is not None:
                kept_values.setdefault(tag, match.group(2).strip())
            else:
                pending_kept_value_tag = tag
        out.append(line)
    return "\n".join(out), dropped, kept_values


def select_block(doc: gemmi.cif.Document, *, required_loop_tag: str) -> gemmi.cif.Block:
    """Return the sole block, or the one block carrying ``required_loop_tag`` among several.

    Software such as Jana2020 exports structure CIFs with a leading ``data_global`` block of blank
    journal-submission boilerplate ahead of the actual ``data_<name>`` block with the refined
    structure. ``gemmi``'s ``sole_block()`` rejects any file with more than one block, so pick the
    block that actually carries the data this reader needs instead.
    """
    if len(doc) == 1:
        return doc.sole_block()
    candidates = [block for block in doc if block.find_loop(required_loop_tag)]
    if len(candidates) == 1:
        return candidates[0]
    names = ", ".join(block.name for block in doc)
    raise ValueError(
        f"expected exactly one CIF block containing {required_loop_tag}, found "
        f"{len(candidates)} among {len(doc)} blocks ({names})"
    )


def select_block_with_diagnostics(
    doc: gemmi.cif.Document,
    *,
    required_loop_tag: str,
    source_path: str | Path | None,
    input_kind: InputKind,
) -> tuple[gemmi.cif.Block, tuple[ParseDiagnostic, ...]]:
    """Select a block and report when a data-bearing block is chosen from a multi-block CIF."""
    block = select_block(doc, required_loop_tag=required_loop_tag)
    if len(doc) == 1:
        return block, ()
    return block, (
        ParseDiagnostic(
            code="cif_block_selected",
            input_kind=input_kind,
            source_path=None if source_path is None else Path(source_path),
            message=(
                f"found {len(doc)} CIF blocks; using structure block {block.name!r} "
                f"(the only block with {required_loop_tag})"
            ),
            details={
                "block": block.name,
                "required_loop_tag": required_loop_tag,
                "n_blocks": len(doc),
            },
        ),
    )


class CifNumber(NamedTuple):
    """CIF numeric value with optional standard uncertainty.

    ``su`` is ``NaN`` when the source value has no parenthesized SU, matching the shape-aligned
    array convention used by IO records.
    """

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
    # Materialise loop.values once: it is a gemmi property that rebuilds the whole flat vector on
    # each access, so slicing per row would be O(rows^2) -- pathological on large reflection loops.
    flat = list(loop.values)
    rows: list[dict[str, str]] = []
    for start in range(0, len(flat), width):
        values = [str(value) for value in flat[start : start + width]]
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
    return cell_matrix_from_parameters(cell_parameters(block)[0])


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
