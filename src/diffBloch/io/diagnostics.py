"""Parse diagnostics for input-file tolerance and fallback decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Literal

type ParseDetail = str | int | float | bool
type InputKind = Literal["structure", "experimental_data"]
type ParseDiagnosticCode = Literal[
    "cif_block_selected",
    "duplicate_scalar_tag_dropped",
    "hydrogen_sites_filtered",
    "pets_geometry_defaulted",
    "pets_optional_metadata_absent",
    "pets_summary_tag_used",
    "symmetry_from_spacegroup",
]


@dataclass(frozen=True)
class ParseDiagnostic:
    """A non-fatal input parse decision worth surfacing at the app boundary."""

    channel: ClassVar[str] = "input parse"
    code: ParseDiagnosticCode
    input_kind: InputKind
    source_path: Path | None
    message: str
    details: Mapping[str, ParseDetail] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source = None if self.source_path is None else Path(self.source_path)
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @property
    def step(self) -> int | None:
        return None

    @property
    def measurements(self) -> Mapping[str, float]:
        return {}


@dataclass(frozen=True)
class ParsedInput[T]:
    """An IO record plus non-fatal diagnostics collected while parsing it."""

    record: T
    diagnostics: tuple[ParseDiagnostic, ...] = ()
