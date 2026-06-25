"""Structure-factor helpers: Lobato form factors, Debye-Waller, vectorised Fgb.

The form-factor golden values are the Lobato-Van Dyck (2014) parametrization as tabulated by abTEM
(see REFERENCES.md); our native functional form must reproduce them.
"""

import torch

from diffBloch.core.scattering import (
    debye_waller_factor,
    lobato_form_factors,
    resolution_cutoff,
    structure_factors,
)

# abTEM oracle for f_e(g**2) at g = [0, 0.5, 1, 2] (cross-checked against diffsims).
_REFERENCE = {
    8: [2.029092, 1.231758, 0.542695, 0.158564],  # O
    14: [5.836000, 1.967708, 0.742909, 0.270509],  # Si
}
_G = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64)


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
    hard = resolution_cutoff(g, 1.6, mode="hard")
    assert hard.tolist() == [1.0, 1.0, 1.0, 0.0]
    taper = resolution_cutoff(g, 1.6, mode="taper")
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
