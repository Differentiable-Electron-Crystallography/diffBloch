"""Electron-optics primitives: relativistic wavelength, wavevector, excitation errors.

Golden wavelength values are the textbook relativistic de Broglie values (also abTEM's
``energy2wavelength``); the excitation-error convention follows Spence & Zuo (see REFERENCES.md).
"""

import numpy as np
import pytest

from diffBloch.core.dynamical import (
    energy2sigma,
    energy2wavelength,
    excitation_errors,
    kappa,
    structure_matrix_prefactor,
    wavevector_magnitude,
)


def test_energy2wavelength_matches_textbook() -> None:
    # Relativistic electron wavelength (Å) at common accelerating voltages.
    assert energy2wavelength(100e3) == pytest.approx(0.03701, abs=1e-5)
    assert energy2wavelength(200e3) == pytest.approx(0.02508, abs=1e-5)
    assert energy2wavelength(300e3) == pytest.approx(0.01969, abs=1e-5)


def test_energy2wavelength_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="energy must be positive"):
        energy2wavelength(0.0)


def test_wavevector_magnitude_is_inverse_wavelength_without_correction() -> None:
    energy = 200e3
    assert wavevector_magnitude(energy) == pytest.approx(1.0 / energy2wavelength(energy))
    # A positive mean-inner-potential term raises the in-crystal wavevector.
    assert wavevector_magnitude(energy, u0=0.1) > wavevector_magnitude(energy)


def test_excitation_errors_zero_at_origin_and_shaped() -> None:
    g = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.05]])
    sg = excitation_errors(g, 200e3)
    assert sg.shape == (3,)
    assert sg[0] == 0.0  # g = 0 sits exactly on the Ewald sphere


def test_excitation_errors_match_reference_formula() -> None:
    g = np.array([[0.2, 0.1, -0.3]])
    energy, u0 = 200e3, 0.0
    k = 1.0 / energy2wavelength(energy)
    expected = (k**2 - float(np.sum((np.array([0.0, 0.0, -k]) + g[0]) ** 2))) / (2.0 * k)
    assert excitation_errors(g, energy, u0=u0)[0] == pytest.approx(expected)


def test_excitation_errors_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="g must have shape"):
        excitation_errors(np.zeros((3,)), 200e3)


def test_energy2sigma_matches_abtem_oracle() -> None:
    # abTEM energy2sigma values [1/(Å·eV)]; native CODATA-2018 form reproduces them to ~1e-8.
    assert energy2sigma(100e3) == pytest.approx(9.24395822e-04, rel=1e-6)
    assert energy2sigma(200e3) == pytest.approx(7.28840109e-04, rel=1e-6)
    assert energy2sigma(300e3) == pytest.approx(6.52616146e-04, rel=1e-6)


def test_energy2sigma_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="energy must be positive"):
        energy2sigma(-1.0)


def test_kappa_matches_abtem_oracle() -> None:
    # abTEM abtem.core.constants.kappa; native form matches to ~1e-8.
    assert kappa == pytest.approx(0.0208865737082965, rel=1e-6)


def test_structure_matrix_prefactor_composes_sigma_kappa_wavelength() -> None:
    energy = 200e3
    expected = energy2sigma(energy) / (kappa * energy2wavelength(energy) * np.pi)
    assert structure_matrix_prefactor(energy) == pytest.approx(expected)
