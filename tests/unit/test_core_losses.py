"""Intensity observable and intensity-space losses (``core.products`` / ``core.losses``).

Loss oracles inline the verbatim ``diffBloch_private/diffBloch/metrics.py`` bodies (pure torch, no
``ase``) so the port is checked against the private source, not a paraphrase.
"""

from __future__ import annotations

import pytest
import torch

from diffBloch.core.losses import l1, mse, rbragg, w_rbragg, weighted_mse
from diffBloch.core.products import intensities


# --- private metrics.py bodies, verbatim (renamed args only) ------------------------------------
def _private_mse(sim, exp, sigmas=None):
    return torch.mean((sim - exp) ** 2, dim=-1)


def _private_l1(sim, exp, sigmas=None):
    return torch.mean(torch.abs(sim - exp), dim=-1)


def _private_weighted_mse(sim, exp, sigmas):
    weight = 1 / sigmas**2
    return torch.sum(weight * (sim - exp) ** 2, dim=-1)


def _private_rbragg_abs(sim, exp, sigmas):
    mask = exp > 3 * sigmas
    sqrt_exp = exp.sqrt()
    sqrt_sim = sim.sqrt()
    num = torch.sum(torch.abs(sqrt_exp - sqrt_sim) * mask, dim=-1)
    denom = torch.sum(sqrt_exp * mask, dim=-1)
    return num / denom


def _private_wrbragg(sim, exp, sigmas, mu=0.01):
    eps = 1e-12
    sqrt_I = torch.sqrt(torch.clamp(exp, min=eps))
    weak_mask = exp < (0.01 * sigmas)
    sigma_sqrt_I = torch.empty_like(sigmas)
    sigma_sqrt_I[weak_mask] = 5 * torch.sqrt(sigmas[weak_mask])
    sigma_sqrt_I[~weak_mask] = 0.5 * sigmas[~weak_mask] / sqrt_I[~weak_mask]
    w = 1.0 / torch.sqrt(sigma_sqrt_I**2 + (mu * sqrt_I) ** 2)
    numerator = torch.sum((w * (sim - exp)) ** 2, dim=-1)
    denominator = torch.sum((w * exp) ** 2, dim=-1)
    return torch.sqrt(numerator / denominator)


# --- intensities -------------------------------------------------------------------------------
def test_intensities_match_modulus_squared() -> None:
    psi = torch.tensor([[3 + 4j, 0 + 0j], [1 + 0j, 0 + 1j]], dtype=torch.complex128)
    expected = torch.tensor([[25.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

    out = intensities(psi)
    assert torch.allclose(out, expected)
    assert out.dtype == torch.float64 and out.shape == psi.shape


def test_intensities_is_differentiable() -> None:
    psi = torch.tensor([1 + 2j, 3 - 1j], dtype=torch.complex128, requires_grad=True)
    intensities(psi).sum().backward()
    assert psi.grad is not None and psi.grad.abs().sum() > 0


# --- losses vs private oracle -------------------------------------------------------------------
@pytest.fixture
def _intensity_pair():
    g = torch.Generator().manual_seed(11)
    calc = torch.rand((4, 7), generator=g, dtype=torch.float64)
    # observed intensities sit clear of the 3*sigma cut so rbragg's mask keeps every reflection
    # (avoids 0/0 rows); the mask behaviour itself is covered by test_rbragg_masks_weak_reflections.
    obs = 0.3 + torch.rand((4, 7), generator=g, dtype=torch.float64)
    sigmas = 0.01 + 0.03 * torch.rand((4, 7), generator=g, dtype=torch.float64)
    return calc, obs, sigmas


def test_mse_matches_private(_intensity_pair) -> None:
    calc, obs, _ = _intensity_pair
    assert torch.allclose(mse(calc, obs), _private_mse(calc, obs))


def test_l1_matches_private(_intensity_pair) -> None:
    calc, obs, _ = _intensity_pair
    assert torch.allclose(l1(calc, obs), _private_l1(calc, obs))


def test_weighted_mse_matches_private(_intensity_pair) -> None:
    calc, obs, sigmas = _intensity_pair
    assert torch.allclose(weighted_mse(calc, obs, sigmas), _private_weighted_mse(calc, obs, sigmas))


def test_rbragg_matches_private(_intensity_pair) -> None:
    calc, obs, sigmas = _intensity_pair
    assert torch.allclose(rbragg(calc, obs, sigmas), _private_rbragg_abs(calc, obs, sigmas))


def test_rbragg_masks_weak_reflections() -> None:
    # Two reflections; the second is below the I > 3*sigma cut, so a large calc mismatch there must
    # not affect R(obs).
    obs = torch.tensor([[1.0, 0.01]], dtype=torch.float64)
    sigmas = torch.tensor([[0.05, 0.10]], dtype=torch.float64)  # 3*sigma = 0.15, 0.30
    calc_match = torch.tensor([[1.0, 0.01]], dtype=torch.float64)
    calc_off = torch.tensor([[1.0, 9.0]], dtype=torch.float64)
    assert torch.allclose(rbragg(calc_match, obs, sigmas), rbragg(calc_off, obs, sigmas))
    assert torch.allclose(rbragg(calc_match, obs, sigmas), torch.zeros(1, dtype=torch.float64))


def test_rbragg_is_nan_safe_for_negative_masked_reflections() -> None:
    # Experimental intensities can be negative (background-subtracted). A negative reflection is
    # always below the I > 3*sigma cut, so it must be excluded -- and must not poison the sum with a
    # NaN from sqrt(negative). The result must equal rbragg over the observed subset alone.
    obs = torch.tensor([[1.0, 4.0, -0.5]], dtype=torch.float64)
    sigmas = torch.tensor([[0.05, 0.10, 0.10]], dtype=torch.float64)
    calc = torch.tensor([[1.2, 3.6, 9.0]], dtype=torch.float64)
    full = rbragg(calc, obs, sigmas)
    subset = rbragg(calc[:, :2], obs[:, :2], sigmas[:, :2])
    assert torch.isfinite(full).all()
    assert torch.allclose(full, subset)


def test_w_rbragg_matches_private(_intensity_pair) -> None:
    calc, obs, sigmas = _intensity_pair
    assert torch.allclose(w_rbragg(calc, obs, sigmas), _private_wrbragg(calc, obs, sigmas))


# --- properties --------------------------------------------------------------------------------
def test_losses_vanish_for_perfect_agreement(_intensity_pair) -> None:
    _, obs, sigmas = _intensity_pair
    assert torch.allclose(mse(obs, obs), torch.zeros(4, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(l1(obs, obs), torch.zeros(4, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(rbragg(obs, obs, sigmas), torch.zeros(4, dtype=torch.float64), atol=1e-12)


def test_losses_reduce_reflection_axis(_intensity_pair) -> None:
    calc, obs, sigmas = _intensity_pair
    assert mse(calc, obs).shape == (4,)
    assert w_rbragg(calc, obs, sigmas).shape == (4,)


def test_losses_are_differentiable(_intensity_pair) -> None:
    calc, obs, sigmas = _intensity_pair
    calc = calc.clone().requires_grad_(True)
    rbragg(calc, obs, sigmas).sum().backward()
    assert calc.grad is not None and calc.grad.abs().sum() > 0


def test_loss_shape_mismatch_raises(_intensity_pair) -> None:
    calc, obs, sigmas = _intensity_pair
    with pytest.raises(ValueError, match="same shape"):
        mse(calc, obs[:, :3])
    with pytest.raises(ValueError, match="sigmas must have shape"):
        weighted_mse(calc, obs, sigmas[:, :3])
