"""Engine ``LossFn``/``ScoresFn`` builders (``engine/losses.py``): ``wr`` and ``wr2``.

The refinement objective must reconcile calculated intensities (arbitrary dynamical structure-factor
scale) with observed (PETS intensity scale) before a weighted R, or the loss is flat and
gradient-free (the raw metric on calc-vs-obs parks near ~1 with a vanishing gradient). Both
``wr_loss`` (on ``|F|``) and ``wr2_loss`` (on intensities) refit the same scale grid used by
orientation scoring; these pin their defining properties: global-scale invariance, correcting a pure
scale mismatch, a live gradient on a *shape* mismatch, and finiteness when calculated is degenerate.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from diffBloch.core.losses import weighted_r_amplitude, weighted_r_intensity
from diffBloch.core.products import AlignedIntensities
from diffBloch.engine import (
    rbragg_loss,
    robs_scores,
    wr2_loss,
    wr2_scores,
    wr_loss,
    wr_scores,
)

_LOSSES = pytest.mark.parametrize(
    ("loss", "scores", "raw"),
    [
        (wr_loss, wr_scores, weighted_r_amplitude),
        (wr2_loss, wr2_scores, weighted_r_intensity),
    ],
    ids=["wr", "wr2"],
)
type _Fn = Callable[[AlignedIntensities], Tensor]
type _Raw = Callable[[Tensor, Tensor, Tensor], Tensor]


def _aligned(calc: torch.Tensor, obs: torch.Tensor, sigmas: torch.Tensor) -> AlignedIntensities:
    return AlignedIntensities(calculated=calc, observed=obs, sigmas=sigmas)


@pytest.fixture
def _shapes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    obs = 0.3 + torch.rand((2, 7), generator=g, dtype=torch.float64)  # (T, K), clear of 3*sigma
    sigmas = 0.01 + 0.03 * torch.rand((2, 7), generator=g, dtype=torch.float64)
    factor = 0.5 + torch.rand((2, 7), generator=g, dtype=torch.float64)  # non-constant per hkl
    return obs, sigmas, factor


@_LOSSES
def test_scaled_loss_is_invariant_to_a_global_rescale(
    _shapes, loss: _Fn, scores: _Fn, raw: _Raw
) -> None:
    # The defining property: renormalising calc->obs removes any overall scale, so multiplying
    # calculated by any constant leaves the loss unchanged (it sees only the relative distribution).
    obs, sigmas, factor = _shapes
    calc = obs * factor  # a relative-shape mismatch, NOT a pure scalar multiple of obs
    base = loss(_aligned(calc, obs, sigmas))
    rescaled = loss(_aligned(17.0 * calc, obs, sigmas))
    assert torch.allclose(base, rescaled, atol=1e-10)


@_LOSSES
def test_scaled_loss_fixes_a_pure_scale_mismatch(
    _shapes, loss: _Fn, scores: _Fn, raw: _Raw
) -> None:
    # calc differs from obs by only a global factor -> after scaling, calc == obs -> 0, whereas
    # the raw (unscaled) metric stays large (the bug the scaling fixes).
    obs, sigmas, _ = _shapes
    calc = 10.0 * obs
    assert float(loss(_aligned(calc, obs, sigmas))) < 1e-6
    assert float(raw(calc, obs, sigmas).sum()) > 0.5  # raw stays large


@_LOSSES
def test_scaled_loss_gradient_alive_on_shape_mismatch(
    _shapes, loss: _Fn, scores: _Fn, raw: _Raw
) -> None:
    # A genuine (non-scale) mismatch must produce a usable gradient -- what the raw loss lacked.
    obs, sigmas, factor = _shapes
    calc = (obs * factor).clone().requires_grad_(True)
    loss(_aligned(calc, obs, sigmas)).backward()
    assert calc.grad is not None and calc.grad.abs().sum() > 0


@_LOSSES
def test_scaled_loss_is_finite_for_degenerate_calculated(
    _shapes, loss: _Fn, scores: _Fn, raw: _Raw
) -> None:
    # A near-zero calculated total would explode the scale (and its backward) without the
    # clamp_min(eps) guard -- pin both the value and the gradient stay finite.
    obs, sigmas, _ = _shapes
    calc = torch.full_like(obs, 1e-20).requires_grad_(True)
    value = loss(_aligned(calc, obs, sigmas))
    assert torch.isfinite(value)
    value.backward()
    assert calc.grad is not None and torch.isfinite(calc.grad).all()


@_LOSSES
def test_loss_is_scores_summed_over_thickness(_shapes, loss: _Fn, scores: _Fn, raw: _Raw) -> None:
    obs, sigmas, factor = _shapes
    aligned = _aligned(obs * factor, obs, sigmas)
    assert torch.allclose(loss(aligned), scores(aligned).sum())
    assert scores(aligned).shape == (obs.shape[0],)


def test_robs_scores_fixes_a_pure_scale_mismatch(_shapes) -> None:
    # Same scale-fit property as wr_scores, but against the rbragg (R_obs) metric: a pure scale
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
