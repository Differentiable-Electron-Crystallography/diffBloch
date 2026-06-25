"""Relativistic electron-optics primitives for the dynamical-diffraction path.

Stage-8 foundation. Native ports of the electron-optics helpers the private predecessor
(``diffBloch_private/diffBloch/dynamical.py``) imported directly from abTEM
(``abtem.core.energy.energy2wavelength``) plus the Spence & Zuo excitation-error convention
(``diffBloch_private/diffBloch/utils.py::excitation_errors``). See ``REFERENCES.md`` — abTEM and
Spence & Zuo (1992) are credited; abTEM is not a runtime dependency.

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
