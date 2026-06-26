"""Electron-optics primitives: relativistic wavelength, wavevector, excitation errors.

Golden wavelength values are the textbook relativistic de Broglie values (also abTEM's
``energy2wavelength``); the excitation-error convention follows Spence & Zuo (see REFERENCES.md).
"""

import numpy as np
import pytest
import torch

from diffBloch.core.dynamical import (
    build_structure_factor_gather,
    energy2sigma,
    energy2wavelength,
    excitation_errors,
    gather_structure_factors,
    kappa,
    m_factors,
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


def test_m_factors_unity_at_origin_and_match_reference() -> None:
    energy, u0 = 200e3, 0.0
    k_n = wavevector_magnitude(energy, u0=u0)
    g = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.05], [0.0, 0.2, -0.05]])
    mii = m_factors(g, energy, u0=u0)
    assert mii.shape == (3,)
    assert mii[0] == pytest.approx(1.0)  # g = 0 -> Mii = 1
    expected = 1.0 / np.sqrt(1.0 - g[:, 2] / k_n)
    assert np.allclose(mii, expected)
    assert mii[1] > 1.0  # g_z > 0 -> Mii > 1
    assert mii[2] < 1.0  # g_z < 0 -> Mii < 1


def test_m_factors_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="g must have shape"):
        m_factors(np.zeros((3,)), 200e3)


def test_structure_matrix_prefactor_composes_sigma_kappa_wavelength() -> None:
    energy = 200e3
    # Independent oracle: abTEM-exact sigma / (kappa * lambda * pi) at 200 keV (via ase units).
    assert structure_matrix_prefactor(energy) == pytest.approx(0.4428932687947089, rel=1e-6)
    # Wiring check: composes the three helpers as documented.
    expected = energy2sigma(energy) / (kappa * energy2wavelength(energy) * np.pi)
    assert structure_matrix_prefactor(energy) == pytest.approx(expected)


# --- structure-factor gather (off-diagonal index plan) ---------------------------------------

# Three beams spanning a small in-plane patch; their pairwise differences hkl_i - hkl_j span the
# 7 distinct cells below. gpts=(3,3,1) is the smallest box covering h,k in [-1,1], l=0.
_BEAM_HKL = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int64)
_GRID_HKL = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [-1, 1, 0], [0, -1, 0], [1, -1, 0]],
    dtype=np.int64,
)
_GPTS = (3, 3, 1)


def _encoded_factors(grid_hkl: np.ndarray) -> torch.Tensor:
    # Each grid cell gets a unique recoverable value F(h,k,l) = h + 1j*(10k + l).
    real = grid_hkl[:, 0].astype(np.float64)
    imag = 10.0 * grid_hkl[:, 1] + grid_hkl[:, 2]
    return torch.tensor(real + 1j * imag, dtype=torch.complex128)


def test_gather_round_trip_recovers_difference_factors() -> None:
    gather = build_structure_factor_gather(_GRID_HKL, _BEAM_HKL, _GPTS)
    out = gather_structure_factors(gather, _encoded_factors(_GRID_HKL))
    assert out.shape == (3, 3)

    lookup = {tuple(hkl): _encoded_factors(_GRID_HKL)[i] for i, hkl in enumerate(_GRID_HKL)}
    for i in range(3):
        for j in range(3):
            difference = tuple(_BEAM_HKL[j] - _BEAM_HKL[i])  # private ordering: beam_j - beam_i
            assert out[i, j] == lookup[difference]


def test_gather_uses_beam_j_minus_beam_i_sign_convention() -> None:
    gather = build_structure_factor_gather(_GRID_HKL, _BEAM_HKL, _GPTS)
    out = gather_structure_factors(gather, _encoded_factors(_GRID_HKL))
    lookup = {tuple(hkl): _encoded_factors(_GRID_HKL)[i] for i, hkl in enumerate(_GRID_HKL)}
    # out[0, 1] gathers beam_1 - beam_0 = (1, 0, 0), not beam_0 - beam_1 = (-1, 0, 0).
    assert out[0, 1] == lookup[(1, 0, 0)]
    assert out[1, 0] == lookup[(-1, 0, 0)]


def test_gather_is_differentiable_and_grad_counts_pair_multiplicity() -> None:
    gather = build_structure_factor_gather(_GRID_HKL, _BEAM_HKL, _GPTS)
    factors = torch.arange(1.0, _GRID_HKL.shape[0] + 1.0, dtype=torch.float64, requires_grad=True)
    gather_structure_factors(gather, factors).sum().backward()

    # Linear scatter: d(sum out)/dF[m] = #(i,j) pairs with beam_j - beam_i = grid[m].
    differences = (_BEAM_HKL[None] - _BEAM_HKL[:, None]).reshape(-1, 3)
    expected = np.array(
        [int((differences == hkl).all(axis=1).sum()) for hkl in _GRID_HKL], dtype=np.float64
    )
    assert factors.grad is not None
    assert torch.allclose(factors.grad, torch.tensor(expected))


def test_build_gather_rejects_grid_not_covering_differences() -> None:
    # Grid holds only (0,0,0); the (1,0,0) difference is off-grid and must raise, not gather zero.
    with pytest.raises(ValueError, match="must cover every beam difference"):
        build_structure_factor_gather(np.array([[0, 0, 0]], dtype=np.int64), _BEAM_HKL, _GPTS)


def test_build_gather_rejects_duplicate_grid_indices() -> None:
    duplicated = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="must not contain duplicate"):
        build_structure_factor_gather(duplicated, _BEAM_HKL, _GPTS)


def test_build_gather_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="grid_hkl must have shape"):
        build_structure_factor_gather(np.zeros(3, dtype=np.int64), _BEAM_HKL, _GPTS)
    with pytest.raises(ValueError, match="beam_hkl must have shape"):
        build_structure_factor_gather(_GRID_HKL, np.zeros((2, 2), dtype=np.int64), _GPTS)


def test_gather_rejects_mismatched_factor_length() -> None:
    gather = build_structure_factor_gather(_GRID_HKL, _BEAM_HKL, _GPTS)
    with pytest.raises(ValueError, match="structure_factors must have shape"):
        gather_structure_factors(gather, torch.ones(_GRID_HKL.shape[0] + 1, dtype=torch.complex128))
