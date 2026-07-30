"""Structure-factor helpers: Lobato form factors, Debye-Waller, vectorised Fgb.

The form-factor golden values are the Lobato-Van Dyck (2014) parametrization as tabulated by abTEM
(see REFERENCES.md); our native functional form must reproduce them.
"""

import pytest
import torch

from diffBloch.core.absorption import absorptive_form_factors
from diffBloch.core.scattering import (
    debye_waller_factor,
    lobato_form_factors,
    structure_factor_cutoff,
    structure_factors,
)
from diffBloch.specs import Absorption

# abTEM oracle for f_e(g**2) at g = [0, 0.5, 1, 2] (cross-checked against diffsims).
_REFERENCE = {
    8: [2.029092, 1.231758, 0.542695, 0.158564],  # O
    14: [5.836000, 1.967708, 0.742909, 0.270509],  # Si
}
_G = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64)


def test_absorptive_form_factors_match_legacy_paper_oracle() -> None:
    factors = absorptive_form_factors(
        torch.tensor([1, 14, 55, 103]),
        torch.tensor([0.0, 0.25, 0.75], dtype=torch.float64),
        torch.tensor([0.1, 0.8, 1.7, 4.0], dtype=torch.float64),
        energy=200_000.0,
    )
    expected = torch.tensor(
        [
            [0.000115650270652303, 0.00010488221302328136, 0.00006743805816084504],
            [0.05028876677584426, 0.04484278559044578, 0.027551326431258846],
            [0.758564306580739, 0.6834398173724344, 0.3728305399278465],
            [2.5639759769437473, 2.3220945980764203, 0.9222574807272947],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(factors, expected, rtol=1e-12, atol=1e-14)


def test_lobato_form_factors_match_reference() -> None:
    numbers = torch.tensor([8, 14])
    factors = lobato_form_factors(numbers, _G)
    assert torch.allclose(factors[0], torch.tensor(_REFERENCE[8], dtype=torch.float64), atol=1e-4)
    assert torch.allclose(factors[1], torch.tensor(_REFERENCE[14], dtype=torch.float64), atol=1e-4)


def test_form_factors_vectorise_over_unique_z() -> None:
    # Repeated Z must give identical rows (the unique-Z grouping is order-preserving).
    numbers = torch.tensor([8, 8, 14, 8])
    factors = lobato_form_factors(numbers, _G)
    assert torch.allclose(factors[0], factors[1])
    assert torch.allclose(factors[0], factors[3])
    assert not torch.allclose(factors[0], factors[2])


def test_debye_waller_zero_adp_is_unity_and_decays() -> None:
    hkl = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    zero = debye_waller_factor(hkl, torch.zeros(1, 3, 3, dtype=torch.float64))
    assert torch.allclose(zero, torch.ones(1, 3, dtype=torch.float64))

    uij = (torch.eye(3, dtype=torch.float64) * 0.01).unsqueeze(0)
    dwf = debye_waller_factor(hkl, uij)[0]
    assert dwf[0] == 1.0  # |g| = 0 unaffected
    assert dwf[1] < 1.0 and dwf[2] < dwf[1]  # higher |g| damped more


def test_resolution_cutoff_hard_and_taper() -> None:
    g = torch.tensor([0.0, 1.0, 1.6, 2.0], dtype=torch.float64)
    hard = structure_factor_cutoff(g, 1.6, mode="hard")
    assert hard.tolist() == [1.0, 1.0, 1.0, 0.0]
    taper = structure_factor_cutoff(g, 1.6, mode="taper")
    assert taper[0] > taper[-1]  # monotone falloff toward g_max
    assert (taper >= 0.0).all() and (taper <= 1.0).all()


def test_structure_factors_gradients_flow_and_extinction() -> None:
    positions = torch.rand(2, 3, dtype=torch.float64, requires_grad=True)
    numbers = torch.tensor([8, 14])
    occupancies = torch.ones(2, dtype=torch.float64, requires_grad=True)
    uij = (torch.eye(3, dtype=torch.float64) * 0.01).repeat(2, 1, 1).clone().requires_grad_(True)
    hkl = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]])
    reciprocal_basis = torch.eye(3, dtype=torch.float64)  # |g| = |hkl|: [0, 1, sqrt(2), 2]

    fgb = structure_factors(
        positions,
        numbers,
        occupancies,
        uij,
        hkl=hkl,
        reciprocal_basis=reciprocal_basis,
        cell_volume=113.3,
        g_max=1.6,
    )
    assert fgb.shape == (4,) and fgb.dtype == torch.complex128
    assert fgb[-1] == 0  # |g| = 2 > g_max -> hard cutoff zeroes it

    fgb.abs().sum().backward()
    assert positions.grad.abs().sum() > 0
    assert uij.grad.abs().sum() > 0
    assert occupancies.grad.abs().sum() > 0


def test_structure_factors_rejects_mismatched_adp_count() -> None:
    # A single ADP matrix for 3 atoms must error, not silently broadcast to wrong physics.
    positions = torch.rand(3, 3, dtype=torch.float64)
    numbers = torch.tensor([8, 14, 8])
    occupancies = torch.ones(3, dtype=torch.float64)
    uij = (torch.eye(3, dtype=torch.float64) * 0.01).unsqueeze(0)  # (1, 3, 3), not (3, 3, 3)
    hkl = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    reciprocal_basis = torch.eye(3, dtype=torch.float64)

    with pytest.raises(ValueError, match="uij_star must have shape"):
        structure_factors(
            positions,
            numbers,
            occupancies,
            uij,
            hkl=hkl,
            reciprocal_basis=reciprocal_basis,
            cell_volume=113.3,
            g_max=2.0,
        )


def test_parameterized_absorption_is_complex_and_differentiable() -> None:
    positions = torch.tensor([[0.13, 0.27, 0.31]], dtype=torch.float64, requires_grad=True)
    numbers = torch.tensor([14])
    occupancies = torch.ones(1, dtype=torch.float64, requires_grad=True)
    uij = (torch.eye(3, dtype=torch.float64) * 0.01).unsqueeze(0).clone().requires_grad_(True)
    factors = structure_factors(
        positions,
        numbers,
        occupancies,
        uij,
        hkl=torch.tensor([[0, 0, 0], [1, 0, 0]]),
        reciprocal_basis=torch.eye(3, dtype=torch.float64),
        cell_volume=100.0,
        g_max=2.0,
        absorption=Absorption(enabled=True),
        energy=200_000.0,
    )
    assert factors[0].imag > 0.0
    factors.abs().sum().backward()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert occupancies.grad is not None and torch.isfinite(occupancies.grad).all()
    assert uij.grad is not None and torch.isfinite(uij.grad).all()
