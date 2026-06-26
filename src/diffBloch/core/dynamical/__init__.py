"""Dynamical-diffraction core: electron-optics primitives + structure-matrix assembly.

``primitives`` holds the NumPy electron-optics constants (setup geometry, not refined); ``assembly``
holds the torch differentiable structure-matrix path built on them. Re-exported here so
``from diffBloch.core.dynamical import ...`` stays stable across the package split.
"""

from diffBloch.core.dynamical.assembly import (
    BeamPlan,
    BlochSystem,
    StructureFactorGather,
    build_beam_plan,
    build_bloch_system,
    build_structure_factor_gather,
    gather_structure_factors,
    structure_matrix,
)
from diffBloch.core.dynamical.primitives import (
    energy2sigma,
    energy2wavelength,
    excitation_errors,
    kappa,
    m_factors,
    structure_matrix_prefactor,
    wavevector_magnitude,
)

__all__ = [
    "BeamPlan",
    "BlochSystem",
    "StructureFactorGather",
    "build_beam_plan",
    "build_bloch_system",
    "build_structure_factor_gather",
    "energy2sigma",
    "energy2wavelength",
    "excitation_errors",
    "gather_structure_factors",
    "kappa",
    "m_factors",
    "structure_matrix",
    "structure_matrix_prefactor",
    "wavevector_magnitude",
]
