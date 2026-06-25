"""Symmetry-constraint setup seam."""

from __future__ import annotations

from dataclasses import dataclass

from diffBloch.io.record import StructureRecord


@dataclass(frozen=True)
class SymmetryConstraints:
    """Internal placeholder value object for future diffpy-backed parameter constraints."""

    n_asymmetric_sites: int
    n_symops: int


def symmetry_constraints(record: StructureRecord) -> SymmetryConstraints:
    """Return symmetry-constraint metadata for a validated structure record.

    The full diffpy-backed special-position and ADP constraint expansion lands with the constraints
    stage. Stage 3 exposes the seam without pretending that those constraints are implemented.
    """
    return SymmetryConstraints(n_asymmetric_sites=record.n_atoms, n_symops=record.n_symops)
