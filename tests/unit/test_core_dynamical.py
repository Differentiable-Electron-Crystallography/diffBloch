"""Electron-optics primitives: relativistic wavelength, wavevector, excitation errors.

Golden wavelength values are the textbook relativistic de Broglie values (also abTEM's
``energy2wavelength``); the excitation-error convention follows Spence & Zuo (see REFERENCES.md).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from diffBloch.core.dynamical import (
    BeamPlan,
    build_beam_plan,
    build_bloch_system,
    build_structure_factor_gather,
    energy2sigma,
    energy2wavelength,
    excitation_errors,
    gather_structure_factors,
    kappa,
    m_factors,
    structure_matrix,
    structure_matrix_prefactor,
    wavevector_magnitude,
)
from diffBloch.core.reciprocal import g_vectors


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
    # In-box but off-grid: (1,0,0) fits the (3,3,1) box yet is absent from the single-cell grid.
    with pytest.raises(ValueError, match="must cover every beam difference"):
        build_structure_factor_gather(np.array([[0, 0, 0]], dtype=np.int64), _BEAM_HKL, _GPTS)


def test_build_gather_rejects_gpts_too_small_for_differences() -> None:
    # Realistic failure: gpts sized to the beams, not 2x -- the (5,0,0) difference falls outside the
    # (3,3,1) box. Must give a clear gpts message, not numpy's "invalid entry in coordinates array".
    beams = np.array([[0, 0, 0], [5, 0, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="gpts is too small"):
        build_structure_factor_gather(np.array([[0, 0, 0]], dtype=np.int64), beams, _GPTS)


def test_build_gather_rejects_fractional_hkl() -> None:
    # Genuinely fractional indices must raise, not silently truncate (0.5 -> 0).
    fractional_beams = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="must contain integer Miller indices"):
        build_structure_factor_gather(_GRID_HKL, fractional_beams, _GPTS)
    with pytest.raises(ValueError, match="must contain integer Miller indices"):
        build_structure_factor_gather(_GRID_HKL.astype(np.float64) + 0.5, _BEAM_HKL, _GPTS)


def test_build_gather_accepts_integer_valued_float_hkl() -> None:
    # 1.0 is a valid Miller index; only genuinely fractional values are rejected.
    from_float = build_structure_factor_gather(
        _GRID_HKL.astype(np.float64), _BEAM_HKL.astype(np.float64), _GPTS
    )
    from_int = build_structure_factor_gather(_GRID_HKL, _BEAM_HKL, _GPTS)
    assert torch.equal(from_float.destination_indices, from_int.destination_indices)


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


# --- structure-matrix assembly (A = scale ⊙ gather(F), diagonal replaced) ---------------------

# A tilted reciprocal basis so l=0 beams still get distinct g_z (hence non-trivial, varying Mii):
# beam (1,0,0) → g_z>0 (Mii>1), (0,1,0) → g_z<0 (Mii<1), (0,0,0) → g_z=0 (Mii=1, Sg=0).
_RECIP_BASIS = np.array([[0.20, 0.0, 0.03], [0.0, 0.25, -0.02], [0.0, 0.0, 0.18]])
_ENERGY = 200e3


def _small_plan() -> BeamPlan:
    return build_beam_plan(_BEAM_HKL, _GRID_HKL, _RECIP_BASIS, energy=_ENERGY, gpts=_GPTS)


def test_structure_matrix_decomposes_into_scale_and_diagonal() -> None:
    plan = _small_plan()
    factors = _encoded_factors(_GRID_HKL)
    a = structure_matrix(plan, factors)
    assert a.shape == (3, 3)

    g = g_vectors(_BEAM_HKL, _RECIP_BASIS)
    mii = m_factors(g, _ENERGY)
    sg = excitation_errors(g, _ENERGY)
    k_n = wavevector_magnitude(_ENERGY)
    prefactor = structure_matrix_prefactor(_ENERGY)
    lookup = {tuple(hkl): complex(factors[i].item()) for i, hkl in enumerate(_GRID_HKL)}

    for i in range(3):
        assert a[i, i].item() == pytest.approx(complex(2.0 * k_n * sg[i] * mii[i]))  # diagonal
        for j in range(3):
            if i == j:
                continue
            difference = tuple(_BEAM_HKL[j] - _BEAM_HKL[i])  # F(beam_j - beam_i)
            expected = prefactor * float(mii[i]) * float(mii[j]) * lookup[difference]
            assert a[i, j].item() == pytest.approx(expected)


def test_structure_matrix_replaces_diagonal_at_origin_beam() -> None:
    plan = _small_plan()
    a = structure_matrix(plan, _encoded_factors(_GRID_HKL))
    # _BEAM_HKL[0] == (0,0,0): Sg = 0 there, so the (replaced) diagonal is exactly 0...
    assert a[0, 0].item() == 0
    # ...while that beam's off-diagonal (gather + scale) is not.
    assert a[0, 1].item() != 0


def test_beam_plan_is_reusable_across_factors() -> None:
    plan = _small_plan()
    a1 = structure_matrix(plan, _encoded_factors(_GRID_HKL))
    a2 = structure_matrix(plan, _encoded_factors(_GRID_HKL) * 2.0 + 1.0)
    # The diagonal is geometry (F-independent) → identical; the off-diagonal tracks F → differs.
    assert torch.allclose(a1.diagonal(), a2.diagonal())
    assert a1[0, 1].item() != a2[0, 1].item()


def test_structure_matrix_is_differentiable_and_diagonal_carries_no_factor_grad() -> None:
    plan = _small_plan()
    factors = torch.arange(1.0, _GRID_HKL.shape[0] + 1.0, dtype=torch.float64, requires_grad=True)
    structure_matrix(plan, factors).sum().backward()

    assert factors.grad is not None
    # F(0,0,0) only ever lands on the diagonal (beam_j - beam_i = 0 ⇔ i = j), which is replaced,
    # so its gradient is exactly 0; the other beams' factors do flow.
    zero_index = next(i for i, hkl in enumerate(_GRID_HKL) if tuple(hkl) == (0, 0, 0))
    assert factors.grad[zero_index] == 0
    assert factors.grad.abs().sum() > 0


def test_structure_matrix_rejects_mismatched_factor_length() -> None:
    plan = _small_plan()
    with pytest.raises(ValueError, match="structure_factors must have shape"):
        structure_matrix(plan, torch.ones(_GRID_HKL.shape[0] + 1, dtype=torch.complex128))


def test_build_beam_plan_carries_propagation_fields() -> None:
    plan = _small_plan()

    assert plan.k_n == pytest.approx(wavevector_magnitude(_ENERGY))
    assert plan.psi0.dtype == torch.complex128
    assert plan.psi0.shape == (_BEAM_HKL.shape[0],)
    assert torch.equal(plan.psi0, torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex128))
    assert plan.mask.dtype == torch.bool
    assert torch.equal(plan.mask, torch.ones(_BEAM_HKL.shape[0], dtype=torch.bool))


def test_build_bloch_system_wires_structure_matrix_and_plan_fields() -> None:
    plan = _small_plan()
    factors = _encoded_factors(_GRID_HKL)
    system = build_bloch_system(plan, factors)

    assert system.a.shape == (_BEAM_HKL.shape[0], _BEAM_HKL.shape[0])
    assert system.a.dtype == torch.complex128
    assert system.mii.shape == (_BEAM_HKL.shape[0],)
    assert system.mii.dtype == torch.float64
    assert system.psi0.shape == (_BEAM_HKL.shape[0],)
    assert system.psi0.dtype == torch.complex128
    assert system.mask.shape == (_BEAM_HKL.shape[0],)
    assert system.mask.dtype == torch.bool
    assert torch.allclose(system.a, structure_matrix(plan, factors))
    assert torch.equal(system.mii, plan.mii)
    assert torch.equal(system.psi0, plan.psi0)
    assert system.k_n == plan.k_n
    assert torch.equal(system.mask, plan.mask)


_ORACLE_NPZ = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "structure_matrix_oracle"
    / "structure_matrix_oracle.npz"
)


def test_structure_matrix_matches_private_oracle() -> None:
    # Independent oracle: A from diffBloch_private's verbatim calculate_structure_matrix on a small
    # alpha-quartz case (see notebooks/iain/stage8_structure_matrix_oracle.ipynb + provenance.json).
    data = np.load(_ORACLE_NPZ)
    # Geometry sanity first, so a failure localizes to the convention vs the assembly.
    assert np.allclose(g_vectors(data["beam_hkl"], data["reciprocal_basis"]), data["g"])
    plan = build_beam_plan(
        data["beam_hkl"],
        data["grid_hkl"],
        data["reciprocal_basis"],
        energy=float(data["energy"]),
        gpts=tuple(int(point) for point in data["gpts"]),
        u0=float(data["u0"]),
    )
    a_ours = structure_matrix(plan, torch.tensor(data["structure_factor"]))
    assert torch.allclose(a_ours, torch.tensor(data["A"]), rtol=1e-10, atol=1e-12)
