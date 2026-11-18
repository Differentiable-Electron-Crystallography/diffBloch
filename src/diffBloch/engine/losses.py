"""Named ``LossFn``/``ScoresFn`` builders: the objective terms ``ExperimentConfig.loss_metrics`` picks.

Each adapts a pure :mod:`diffBloch.core.losses` intensity comparison (which reduces over the
reflection axis, yielding a ``(T,)`` per-thickness loss) into two shapes: a ``ScoresFn``
(``AlignedIntensities -> (T,) Tensor``, one score per thickness) and the corresponding ``LossFn``
(``AlignedIntensities -> scalar``, the ``ScoresFn`` summed over thickness). ``RefinementEngine``
uses both off the *same* underlying metric -- ``scores`` for the orientation/thickness
preprocessing search (``score_orientation``/``score_orientation_per_thickness``, which need a
per-thickness vector to argmin over) and ``loss`` for the differentiable gradient-refinement
objective -- so picking one residual genuinely drives the whole pipeline, not just the gradient
stage.

Three residuals, each re-fitting the multiplicative intensity scale per thickness through
:func:`~diffBloch.core.losses.optimal_scale` (the calculated intensities come off the dynamical
solve on an arbitrary scale; compared raw, a weighted R is denominator-dominated and parks near ~1
with a vanishing gradient):

- ``wr`` -- :func:`~diffBloch.core.losses.weighted_r_amplitude`, Jana2020's ``wR`` on ``|F|``
  (the default);
- ``wr2`` -- :func:`~diffBloch.core.losses.weighted_r_intensity`, ``wR2`` on intensities;
- ``robs`` -- :func:`~diffBloch.core.losses.rbragg`, ``R(obs)`` on ``|F|``.

The selected grid branch remains differentiable in the calculated intensities (``torch.min``
routes the gradient through the winning candidate); only a boundary where the winning grid point
changes is piecewise-smooth.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from diffBloch.core.losses import (
    l1,
    mse,
    optimal_scale,
    rbragg,
    weighted_r_amplitude,
    weighted_r_intensity,
)
from diffBloch.core.products import AlignedIntensities

__all__ = [
    "l1_loss",
    "mse_loss",
    "rbragg_loss",
    "robs_scores",
    "wr2_loss",
    "wr2_scores",
    "wr_loss",
    "wr_scores",
]


def mse_loss(aligned: AlignedIntensities) -> Tensor:
    """Per-orientation MSE loss term, summed over thicknesses to a scalar."""
    return mse(aligned.calculated, aligned.observed).sum()


def l1_loss(aligned: AlignedIntensities) -> Tensor:
    """Per-orientation L1 loss term, summed over thicknesses to a scalar."""
    return l1(aligned.calculated, aligned.observed).sum()


def _scaled_scores(
    aligned: AlignedIntensities, metric: Callable[[Tensor, Tensor, Tensor], Tensor]
) -> Tensor:
    calc, obs = aligned.calculated, aligned.observed
    return torch.stack(
        [
            optimal_scale(calc[t], obs[t], aligned.sigmas[t], metric=metric)[1]
            for t in range(calc.shape[0])
        ]
    )


def wr_scores(aligned: AlignedIntensities) -> Tensor:
    """Per-thickness scaling-optimised ``wR`` on amplitudes (shape ``(T,)``)."""
    return _scaled_scores(aligned, weighted_r_amplitude)


def wr_loss(aligned: AlignedIntensities) -> Tensor:
    """Scaling-optimised ``wR`` on amplitudes -- the default refinement and search objective."""
    return wr_scores(aligned).sum()


def wr2_scores(aligned: AlignedIntensities) -> Tensor:
    """Per-thickness scaling-optimised ``wR2`` on intensities (shape ``(T,)``)."""
    return _scaled_scores(aligned, weighted_r_intensity)


def wr2_loss(aligned: AlignedIntensities) -> Tensor:
    """Scaling-optimised ``wR2`` on intensities. Sums :func:`wr2_scores` over thickness."""
    return wr2_scores(aligned).sum()


def robs_scores(aligned: AlignedIntensities) -> Tensor:
    """Per-thickness scaling-optimised Bragg ``R(obs)`` (shape ``(T,)``), the ``R_obs`` reported."""
    return _scaled_scores(aligned, rbragg)


def rbragg_loss(aligned: AlignedIntensities) -> Tensor:
    """Scaling-optimised Bragg ``R(obs)`` objective. Sums :func:`robs_scores` over thickness."""
    return robs_scores(aligned).sum()
