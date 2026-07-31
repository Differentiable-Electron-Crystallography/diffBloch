"""Validated IO records.

The core simulation code should receive arrays with stable shapes and units, not parser objects.
These records are the trust boundary between file formats and differentiable kernels.

Record classes are pydantic models when they cross an IO/config boundary and need validation.
Parser-local helpers can stay as lighter Python value objects, but public records should keep this
validated boundary shape.

Standard uncertainties are stored as shape-aligned ``*_su`` arrays next to their nominal values,
with ``NaN`` meaning absent from the source file. Keep that parallel-array convention until an
uncertainty group needs behavior beyond validation, covariance storage, or multiple competing
representations. ADPs already use a grouped record because CIF Uiso, Uani, and missing ADPs have
different semantics.

``uij_cif`` names the CIF-source ADP convention at the IO boundary. Constrained tensors currently
reuse that field name for ASU ADPs produced from raw parameters; downstream scattering code must
explicitly convert to reciprocal-space ``U*`` if that is the representation it consumes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]


def _readonly_float_array(value: Any) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _readonly_int_array(value: Any) -> IntArray:
    array = np.array(value, dtype=np.int64, copy=True)
    array.setflags(write=False)
    return array


class AdpRecord(BaseModel):
    """Atomic displacement parameters, preserving CIF Uiso/Uani semantics and SUs."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    kind: tuple[Literal["Uiso", "Uani", "missing"], ...]
    u_iso: FloatArray
    u_iso_su: FloatArray
    uij_cif: FloatArray
    uij_cif_su: FloatArray

    @field_validator("u_iso", "u_iso_su", "uij_cif", "uij_cif_su", mode="before")
    @classmethod
    def _as_float_array(cls, value: Any) -> FloatArray:
        return _readonly_float_array(value)

    @model_validator(mode="after")
    def _validate_contract(self) -> Self:
        n_atoms = len(self.kind)
        if self.u_iso.shape != (n_atoms,):
            raise ValueError("adp.u_iso must have shape (N,) matching adp.kind")
        if self.u_iso_su.shape != (n_atoms,):
            raise ValueError("adp.u_iso_su must have shape (N,) matching adp.kind")
        if self.uij_cif.shape != (n_atoms, 3, 3):
            raise ValueError("adp.uij_cif must have shape (N, 3, 3) matching adp.kind")
        if self.uij_cif_su.shape != (n_atoms, 3, 3):
            raise ValueError("adp.uij_cif_su must have shape (N, 3, 3) matching adp.kind")
        if np.any(self.u_iso_su[np.isfinite(self.u_iso_su)] < 0.0):
            raise ValueError("adp.u_iso_su must be non-negative where present")
        if np.any(self.uij_cif_su[np.isfinite(self.uij_cif_su)] < 0.0):
            raise ValueError("adp.uij_cif_su must be non-negative where present")

        for index, kind in enumerate(self.kind):
            if kind == "Uiso" and not np.isfinite(self.u_iso[index]):
                raise ValueError("Uiso ADPs must have finite u_iso values")
            if kind == "Uani":
                uij = self.uij_cif[index]
                if not np.all(np.isfinite(uij)):
                    raise ValueError("Uani ADPs must have finite uij_cif matrices")
                if not np.allclose(uij, uij.T, atol=1e-12):
                    raise ValueError("Uani uij_cif matrices must be symmetric")
                if np.any(np.linalg.eigvalsh(uij) < -1e-12):
                    raise ValueError("Uani uij_cif matrices must be positive semidefinite")
        return self


class StructureRecord(BaseModel):
    """Asymmetric-unit structure data in CIF convention.

    Nominal arrays feed downstream kernels. The corresponding ``*_su`` arrays preserve CIF
    standard uncertainties as provenance metadata for reporting or later optional penalties.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    source_path: Path | None = None
    unit_cell: FloatArray
    cell_parameters: FloatArray
    cell_parameters_su: FloatArray
    spacegroup_hm: str
    spacegroup_number: int | None = None
    symops_R: FloatArray
    symops_t: FloatArray
    labels: tuple[str, ...]
    numbers: IntArray
    frac_positions: FloatArray
    frac_positions_su: FloatArray
    occupancies: FloatArray
    occupancies_su: FloatArray
    adp: AdpRecord

    @field_validator(
        "unit_cell",
        "cell_parameters",
        "cell_parameters_su",
        "symops_R",
        "symops_t",
        "frac_positions",
        "frac_positions_su",
        "occupancies",
        "occupancies_su",
        mode="before",
    )
    @classmethod
    def _as_float_array(cls, value: Any) -> FloatArray:
        return _readonly_float_array(value)

    @field_validator("numbers", mode="before")
    @classmethod
    def _as_int_array(cls, value: Any) -> IntArray:
        return _readonly_int_array(value)

    @model_validator(mode="after")
    def _validate_contract(self) -> Self:
        n_atoms = len(self.labels)
        if self.unit_cell.shape != (3, 3):
            raise ValueError("unit_cell must have shape (3, 3)")
        if self.cell_parameters.shape != (6,):
            raise ValueError("cell_parameters must have shape (6,)")
        if self.cell_parameters_su.shape != (6,):
            raise ValueError("cell_parameters_su must have shape (6,)")
        if np.any(self.cell_parameters_su[np.isfinite(self.cell_parameters_su)] < 0.0):
            raise ValueError("cell_parameters_su must be non-negative where present")
        if self.symops_R.ndim != 3 or self.symops_R.shape[1:] != (3, 3):
            raise ValueError("symops_R must have shape (S, 3, 3)")
        if self.symops_t.shape != (self.symops_R.shape[0], 3):
            raise ValueError("symops_t must have shape (S, 3) matching symops_R")
        if self.numbers.shape != (n_atoms,):
            raise ValueError("numbers must have shape (N,) matching labels")
        if self.frac_positions.shape != (n_atoms, 3):
            raise ValueError("frac_positions must have shape (N, 3) matching labels")
        if self.frac_positions_su.shape != (n_atoms, 3):
            raise ValueError("frac_positions_su must have shape (N, 3) matching labels")
        if np.any(self.frac_positions_su[np.isfinite(self.frac_positions_su)] < 0.0):
            raise ValueError("frac_positions_su must be non-negative where present")
        if self.occupancies.shape != (n_atoms,):
            raise ValueError("occupancies must have shape (N,) matching labels")
        if self.occupancies_su.shape != (n_atoms,):
            raise ValueError("occupancies_su must have shape (N,) matching labels")
        if np.any(self.occupancies_su[np.isfinite(self.occupancies_su)] < 0.0):
            raise ValueError("occupancies_su must be non-negative where present")
        if np.any((self.occupancies < 0.0) | (self.occupancies > 1.0)):
            raise ValueError("occupancies must be in [0, 1]")
        if len(self.adp.kind) != n_atoms:
            raise ValueError("adp must have one entry per atom label")
        return self

    @property
    def n_atoms(self) -> int:
        """Number of asymmetric-unit atom sites."""
        return len(self.labels)

    @property
    def n_symops(self) -> int:
        """Number of symmetry operations provided by the source file."""
        return int(self.symops_R.shape[0])

    @property
    def uij_cif(self) -> FloatArray:
        """Anisotropic ADP matrices; non-Uani rows are NaN."""
        return self.adp.uij_cif


class ExperimentalRecord(BaseModel):
    """Experimental PETS reflection data keyed by rotation/zone-axis id."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    source_path: Path | None = None
    unit_cell: FloatArray
    cell_parameters: FloatArray
    cell_parameters_su: FloatArray
    wavelength: float
    # PETS2's own processing-resolution cutoff (Angstrom^-1), parsed from the free-text
    # _diffrn_measurement_details block when present; None if that field is absent or the PETS
    # version that wrote this file doesn't record it (a manual/config g_max is then the only option).
    dstar_max: float | None = None
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
        "cell_parameters",
        "cell_parameters_su",
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
        return _readonly_float_array(value)

    @field_validator("zone_axis_ids", "hkl", "reflection_zone_axis_ids", mode="before")
    @classmethod
    def _as_int_array(cls, value: Any) -> IntArray:
        return _readonly_int_array(value)

    @model_validator(mode="after")
    def _validate_contract(self) -> Self:
        n_rotations = int(self.zone_axis_ids.shape[0])
        n_reflections = int(self.hkl.shape[0])
        if self.unit_cell.shape != (3, 3):
            raise ValueError("unit_cell must have shape (3, 3)")
        if self.cell_parameters.shape != (6,):
            raise ValueError("cell_parameters must have shape (6,)")
        if self.cell_parameters_su.shape != (6,):
            raise ValueError("cell_parameters_su must have shape (6,)")
        if np.any(self.cell_parameters_su[np.isfinite(self.cell_parameters_su)] < 0.0):
            raise ValueError("cell_parameters_su must be non-negative where present")
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
        if np.any(self.precession_angles <= 0.0):
            raise ValueError("PETS precession angles must be positive")
        if not np.allclose(self.precession_angles, self.precession_angles[0], rtol=0.0, atol=1e-9):
            raise ValueError("PETS precession angles must be constant across virtual frames")
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
    def integration_semiangle(self) -> float:
        """Angular integration semi-angle recorded consistently across the PETS virtual frames."""
        return float(self.precession_angles[0])

    @property
    def n_rotations(self) -> int:
        """Number of PETS zone-axis rows."""
        return int(self.zone_axis_ids.shape[0])

    @property
    def n_reflections(self) -> int:
        """Number of measured reflections."""
        return int(self.hkl.shape[0])
