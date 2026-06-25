"""Relativistic electron-optics primitives for the dynamical-diffraction path.

Stage-8 foundation. Native ports of the electron-optics helpers the private predecessor
(``diffBloch_private/diffBloch/dynamical.py``) imported directly from abTEM
(``abtem.core.energy.energy2wavelength`` / ``energy2sigma``; ``abtem.core.constants.kappa``) plus
the Spence & Zuo excitation-error convention
(``diffBloch_private/diffBloch/utils.py::excitation_errors``). See ``REFERENCES.md`` — abTEM and
Spence & Zuo (1992) are credited; abTEM is not a runtime dependency. The ``energy2sigma``/``kappa``
ports reproduce abTEM's values to ~1e-8 (CODATA-2018 vs ASE constants).

These are setup constants on the geometry/numerics plan — beam energy is an experimental constant
and ``g`` is fixed geometry, neither is refined — so they live on the NumPy planning path like
``core.reciprocal`` (the differentiable structure-factor path stays in ``core.scattering``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]

# CODATA 2018 (SI). Kept local so core/ carries no physical-constants dependency.
_PLANCK = 6.62607015e-34  # J s
_ELECTRON_MASS = 9.1093837015e-31  # kg
_ELEMENTARY_CHARGE = 1.602176634e-19  # C
_SPEED_OF_LIGHT = 299792458.0  # m s^-1
_VACUUM_PERMITTIVITY = 8.8541878128e-12  # F m^-1
_BOHR_RADIUS = 0.529177210903e-10  # m

# Conversion from the unitless (Lobato) potential parametrization to potential units, used in the
# structure-matrix prefactor. Dimensionless in this convention (Å/eV potential units), ~0.0209.
# Native form of abTEM's abtem.core.constants.kappa (4 pi eps0 / (2 pi a0 e) in ASE units), here the
# equivalent CODATA-2018 closed form 2 eps0 / (a0 e) * 1e-20; reproduces abTEM's exact value
# 0.0208865737082965 to ~1e-8 (the residual is CODATA-2018 vs ASE's constants). See REFERENCES.md.
kappa: float = 2.0 * _VACUUM_PERMITTIVITY / (_BOHR_RADIUS * _ELEMENTARY_CHARGE) * 1e-20


def energy2wavelength(energy: float) -> float:
    """Relativistic electron wavelength in angstrom for a beam ``energy`` in eV.

    ``lambda = h c / sqrt(E e (2 m_e c^2 + E e))``, the relativistic de Broglie wavelength. This is
    algebraically identical to abTEM's ``energy2wavelength`` (see ``REFERENCES.md``); reproduces the
    textbook values 0.03701 / 0.02508 / 0.01969 Å at 100 / 200 / 300 keV.
    """
    if energy <= 0.0:
        raise ValueError("energy must be positive")
    charge_energy = energy * _ELEMENTARY_CHARGE
    rest = 2.0 * _ELECTRON_MASS * _SPEED_OF_LIGHT**2
    metres = _PLANCK * _SPEED_OF_LIGHT / np.sqrt(charge_energy * (rest + charge_energy))
    return float(metres * 1e10)


def energy2sigma(energy: float) -> float:
    """Electron interaction parameter ``sigma`` in 1/(angstrom*eV) for a beam ``energy`` in eV.

    ``sigma = 2 pi m e lambda / h^2`` with the relativistic mass ``m = (1 + E e / (m_e c^2)) m_e``.
    Native form of abTEM's ``energy2sigma`` (see ``REFERENCES.md``); reproduces its values
    9.2440e-4 / 7.2884e-4 / 6.5262e-4 at 100 / 200 / 300 keV to ~1e-8.
    """
    if energy <= 0.0:
        raise ValueError("energy must be positive")
    relativistic_mass = (
        1.0 + _ELEMENTARY_CHARGE * energy / (_ELECTRON_MASS * _SPEED_OF_LIGHT**2)
    ) * (_ELECTRON_MASS)
    wavelength_metres = energy2wavelength(energy) * 1e-10
    sigma_si = 2.0 * np.pi * relativistic_mass * _ELEMENTARY_CHARGE * wavelength_metres / _PLANCK**2
    return float(sigma_si * 1e-10)


def structure_matrix_prefactor(energy: float) -> float:
    """Off-diagonal structure-matrix prefactor ``sigma / (kappa * lambda * pi)``.

    Scales structure factors into the Bloch structure matrix ``A``; ported from the repeated private
    ``energy2sigma(energy) / (kappa * energy2wavelength(energy) * np.pi)``
    (``diffBloch_private`` utils.py:772/878, dynamical.py:1001/1059).
    """
    return energy2sigma(energy) / (kappa * energy2wavelength(energy) * np.pi)


def wavevector_magnitude(energy: float, *, u0: float = 0.0) -> float:
    """Corrected wavevector magnitude ``K_n = sqrt(1/lambda^2 + U0)`` in Å^-1.

    ``u0`` is the mean-inner-potential correction term (added to ``1/lambda^2`` in Å^-2, as the
    private code treats it); ``u0=0`` gives the vacuum wavevector ``1/lambda``.
    """
    k0 = 1.0 / energy2wavelength(energy)
    radicand = k0**2 + u0
    if radicand <= 0.0:
        raise ValueError("u0 must keep 1/lambda^2 + u0 positive")
    return float(np.sqrt(radicand))


def excitation_errors(g: FloatArray, energy: float, *, u0: float = 0.0) -> FloatArray:
    """Excitation errors ``Sg`` (Å^-1) for reciprocal vectors ``g`` (Spence & Zuo method).

    ``Sg = (|K|^2 - |K + g|^2) / (2 |K|)`` with the beam ``K`` along ``-z`` and magnitude
    ``wavevector_magnitude(energy, u0=u0)``. Measures each reflection's distance from the Ewald
    sphere; ``Sg = 0`` exactly at ``g = 0``. ``g`` is ``(N, 3)`` in Å^-1; returns ``(N,)``.
    """
    g_array = np.asarray(g, dtype=np.float64)
    if g_array.ndim != 2 or g_array.shape[1] != 3:
        raise ValueError("g must have shape (N, 3)")
    k_mag = wavevector_magnitude(energy, u0=u0)
    k_vector = np.array([0.0, 0.0, -k_mag], dtype=np.float64)
    return (k_mag**2 - np.linalg.norm(k_vector + g_array, axis=1) ** 2) / (2.0 * k_mag)


def m_factors(g: FloatArray, energy: float, *, u0: float = 0.0) -> FloatArray:
    """Diagonal ``Mii`` factors that symmetrise the Bloch structure matrix.

    ``Mii = 1 / sqrt(1 - g_z / K_n)`` with ``K_n = wavevector_magnitude(energy, u0=u0)``; port of
    ``diffBloch_private`` ``dynamical.py::calculate_M_matrix``. The structure matrix uses them on
    both axes off-diagonal (``Mii_i Mii_j``) and once on the diagonal. ``Mii = 1`` at ``g = 0``;
    ``g`` is ``(N, 3)`` in Å^-1, returns ``(N,)``.
    """
    g_array = np.asarray(g, dtype=np.float64)
    if g_array.ndim != 2 or g_array.shape[1] != 3:
        raise ValueError("g must have shape (N, 3)")
    k_n = wavevector_magnitude(energy, u0=u0)
    radicand = 1.0 - g_array[:, 2] / k_n
    if np.any(radicand <= 0.0):
        raise ValueError("g_z must satisfy g_z < K_n for every reflection (1 - g_z/K_n > 0)")
    return 1.0 / np.sqrt(radicand)
