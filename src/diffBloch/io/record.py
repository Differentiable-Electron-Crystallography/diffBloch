"""Validated IO records.

The core simulation code should receive arrays with stable shapes and units, not parser objects.
These records are the trust boundary between file formats and differentiable kernels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]


class StructureRecord(BaseModel):
    """Asymmetric-unit structure data in CIF convention."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_path: Path | None = None
    unit_cell: FloatArray
    spacegroup_hm: str
    spacegroup_number: int | None = None
    symops_R: FloatArray
    symops_t: FloatArray
    labels: tuple[str, ...]
    numbers: IntArray
    frac_positions: FloatArray
    occupancies: FloatArray
    uij_cif: FloatArray

    @field_validator(
        "unit_cell",
        "symops_R",
        "symops_t",
        "frac_positions",
        "occupancies",
        "uij_cif",
        mode="before",
    )
    @classmethod
    def _as_float_array(cls, value: Any) -> FloatArray:
        return np.asarray(value, dtype=np.float64)

    @field_validator("numbers", mode="before")
    @classmethod
    def _as_int_array(cls, value: Any) -> IntArray:
        return np.asarray(value, dtype=np.int64)

    @model_validator(mode="after")
    def _validate_contract(self) -> Self:
        n_atoms = len(self.labels)
        if self.unit_cell.shape != (3, 3):
            raise ValueError("unit_cell must have shape (3, 3)")
        if self.symops_R.ndim != 3 or self.symops_R.shape[1:] != (3, 3):
            raise ValueError("symops_R must have shape (S, 3, 3)")
        if self.symops_t.shape != (self.symops_R.shape[0], 3):
            raise ValueError("symops_t must have shape (S, 3) matching symops_R")
        if self.numbers.shape != (n_atoms,):
            raise ValueError("numbers must have shape (N,) matching labels")
        if self.frac_positions.shape != (n_atoms, 3):
            raise ValueError("frac_positions must have shape (N, 3) matching labels")
        if self.occupancies.shape != (n_atoms,):
            raise ValueError("occupancies must have shape (N,) matching labels")
        if np.any((self.occupancies < 0.0) | (self.occupancies > 1.0)):
            raise ValueError("occupancies must be in [0, 1]")
        if self.uij_cif.shape != (n_atoms, 3, 3):
            raise ValueError("uij_cif must have shape (N, 3, 3) matching labels")
        if not np.allclose(self.uij_cif, np.swapaxes(self.uij_cif, 1, 2), atol=1e-12):
            raise ValueError("uij_cif matrices must be symmetric")
        eigvals = np.linalg.eigvalsh(self.uij_cif)
        if np.any(eigvals < -1e-12):
            raise ValueError("uij_cif matrices must be positive semidefinite")
        return self

    @property
    def n_atoms(self) -> int:
        """Number of asymmetric-unit atom sites."""
        return len(self.labels)

    @property
    def n_symops(self) -> int:
        """Number of symmetry operations provided by the source file."""
        return int(self.symops_R.shape[0])


class ObservationRecord(BaseModel):
    """Observed PETS reflection data keyed by rotation/zone-axis id."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_path: Path | None = None
    unit_cell: FloatArray
    wavelength: float
    ub_matrix: FloatArray
    zone_axis_ids: IntArray
    zone_axes: FloatArray
    precession_angles: FloatArray
    alphas: FloatArray
    betas: FloatArray
    omegas: FloatArray
    scales: FloatArray
    hkl: IntArray
    intensities: FloatArray
    sigmas: FloatArray
    reflection_zone_axis_ids: IntArray

    @field_validator(
        "unit_cell",
        "ub_matrix",
        "zone_axes",
        "precession_angles",
        "alphas",
        "betas",
        "omegas",
        "scales",
        "intensities",
        "sigmas",
        mode="before",
    )
    @classmethod
    def _as_float_array(cls, value: Any) -> FloatArray:
        return np.asarray(value, dtype=np.float64)

    @field_validator("zone_axis_ids", "hkl", "reflection_zone_axis_ids", mode="before")
    @classmethod
    def _as_int_array(cls, value: Any) -> IntArray:
        return np.asarray(value, dtype=np.int64)

    @model_validator(mode="after")
    def _validate_contract(self) -> Self:
        n_rotations = int(self.zone_axis_ids.shape[0])
        n_reflections = int(self.hkl.shape[0])
        if self.unit_cell.shape != (3, 3):
            raise ValueError("unit_cell must have shape (3, 3)")
        if self.wavelength <= 0.0:
            raise ValueError("wavelength must be positive")
        if self.ub_matrix.shape != (3, 3):
            raise ValueError("ub_matrix must have shape (3, 3)")
        if self.zone_axes.shape != (n_rotations, 3):
            raise ValueError("zone_axes must have shape (R, 3) matching zone_axis_ids")
        for name, value in (
            ("precession_angles", self.precession_angles),
            ("alphas", self.alphas),
            ("betas", self.betas),
            ("omegas", self.omegas),
            ("scales", self.scales),
        ):
            if value.shape != (n_rotations,):
                raise ValueError(f"{name} must have shape (R,) matching zone_axis_ids")
        if self.hkl.ndim != 2 or self.hkl.shape[1] != 3:
            raise ValueError("hkl must have shape (M, 3)")
        if self.intensities.shape != (n_reflections,):
            raise ValueError("intensities must have shape (M,) matching hkl")
        if self.sigmas.shape != (n_reflections,):
            raise ValueError("sigmas must have shape (M,) matching hkl")
        if self.reflection_zone_axis_ids.shape != (n_reflections,):
            raise ValueError("reflection_zone_axis_ids must have shape (M,) matching hkl")
        if np.any(self.sigmas < 0.0):
            raise ValueError("sigmas must be non-negative")
        known_zone_ids = set(int(zone_id) for zone_id in self.zone_axis_ids)
        used_zone_ids = set(int(zone_id) for zone_id in self.reflection_zone_axis_ids)
        unknown_zone_ids = used_zone_ids - known_zone_ids
        if unknown_zone_ids:
            raise ValueError(
                f"reflection zone-axis ids are not declared: {sorted(unknown_zone_ids)}"
            )
        return self

    @property
    def n_rotations(self) -> int:
        """Number of PETS zone-axis rows."""
        return int(self.zone_axis_ids.shape[0])

    @property
    def n_reflections(self) -> int:
        """Number of measured reflections."""
        return int(self.hkl.shape[0])
