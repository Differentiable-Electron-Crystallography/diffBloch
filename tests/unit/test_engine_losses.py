"""Engine ``LossFn`` builders (``engine/losses.py``) -- focus on ``wr2_loss``.

The refinement objective must reconcile calculated intensities (arbitrary dynamical structure-factor
scale) with observed (PETS intensity scale) before wR2, or the loss is flat and gradient-free
(``w_rbragg_loss`` on raw calc-vs-obs parks near ~1 with a vanishing gradient).
``wr2_loss`` refits the same scale grid used by orientation scoring; these pin its defining
properties: global-scale
invariance, correcting a pure scale mismatch, a live gradient on a *shape* mismatch, and finiteness
when calculated is degenerate.
"""

from __future__ import annotations

import pytest
import torch

from diffBloch.core.products import AlignedIntensities
from diffBloch.engine import (
    rbragg_loss,
    robs_scores,
    w_rbragg_loss,
    wr2_loss,
    wr2_scores,
)


def _aligned(calc: torch.Tensor, obs: torch.Tensor, sigmas: torch.Tensor) -> AlignedIntensities:
    return AlignedIntensities(calculated=calc, observed=obs, sigmas=sigmas)


@pytest.fixture
def _shapes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    obs = 0.3 + torch.rand((2, 7), generator=g, dtype=torch.float64)  # (T, K), clear of 3*sigma
    sigmas = 0.01 + 0.03 * torch.rand((2, 7), generator=g, dtype=torch.float64)
    factor = 0.5 + torch.rand((2, 7), generator=g, dtype=torch.float64)  # non-constant per hkl
    return obs, sigmas, factor


def test_scaled_loss_is_invariant_to_a_global_rescale(_shapes) -> None:
    # The defining property: renormalising calc->obs removes any overall scale, so multiplying
    # calculated by any constant leaves the loss unchanged (it sees only the relative distribution).
    obs, sigmas, factor = _shapes
    calc = obs * factor  # a relative-shape mismatch, NOT a pure scalar multiple of obs
    base = wr2_loss(_aligned(calc, obs, sigmas))
    rescaled = wr2_loss(_aligned(17.0 * calc, obs, sigmas))
    assert torch.allclose(base, rescaled, atol=1e-10)


def test_scaled_loss_fixes_a_pure_scale_mismatch(_shapes) -> None:
    # calc differs from obs by only a global factor -> after scaling, calc == obs -> wR2 == 0,
    # whereas the raw (unscaled) loss stays large (the bug the scaling fixes).
    obs, sigmas, _ = _shapes
    calc = 10.0 * obs
    assert float(wr2_loss(_aligned(calc, obs, sigmas))) < 1e-6
    assert float(w_rbragg_loss(_aligned(calc, obs, sigmas))) > 0.5  # raw stays large


def test_scaled_loss_gradient_alive_on_shape_mismatch(_shapes) -> None:
    # A genuine (non-scale) mismatch must produce a usable gradient -- what the raw loss lacked.
    obs, sigmas, factor = _shapes
    calc = (obs * factor).clone().requires_grad_(True)
    wr2_loss(_aligned(calc, obs, sigmas)).backward()
    assert calc.grad is not None and calc.grad.abs().sum() > 0


def test_scaled_loss_is_finite_for_degenerate_calculated(_shapes) -> None:
    # A near-zero calculated total would explode the scale (and its backward) without the
    # clamp_min(eps) guard -- pin both the value and the gradient stay finite.
    obs, sigmas, _ = _shapes
    calc = torch.full_like(obs, 1e-20).requires_grad_(True)
    value = wr2_loss(_aligned(calc, obs, sigmas))
    assert torch.isfinite(value)
    value.backward()
    assert calc.grad is not None and torch.isfinite(calc.grad).all()


def test_wr2_loss_is_wr2_scores_summed_over_thickness(_shapes) -> None:
    obs, sigmas, factor = _shapes
    aligned = _aligned(obs * factor, obs, sigmas)
    assert torch.allclose(wr2_loss(aligned), wr2_scores(aligned).sum())
    assert wr2_scores(aligned).shape == (obs.shape[0],)


def test_robs_scores_fixes_a_pure_scale_mismatch(_shapes) -> None:
    # Same scale-fit property as wr2_scores, but against the rbragg (R_obs) metric: a pure scale
    # mismatch resolves to ~0 after fitting, matching how R_obs is reported everywhere else.
    obs, sigmas, _ = _shapes
    calc = 10.0 * obs
    assert float(rbragg_loss(_aligned(calc, obs, sigmas))) < 1e-6
    assert torch.allclose(
        rbragg_loss(_aligned(calc, obs, sigmas)), robs_scores(_aligned(calc, obs, sigmas)).sum()
    )


def test_robs_scores_gradient_alive_on_shape_mismatch(_shapes) -> None:
    obs, sigmas, factor = _shapes
    calc = (obs * factor).clone().requires_grad_(True)
    rbragg_loss(_aligned(calc, obs, sigmas)).backward()
    assert calc.grad is not None and calc.grad.abs().sum() > 0
