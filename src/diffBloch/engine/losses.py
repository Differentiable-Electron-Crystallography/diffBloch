"""Named ``LossFn`` builders: per-orientation loss terms you plug into ``RefinementEngine.loss``.

Each adapts a pure :mod:`diffBloch.core.losses` intensity comparison (which reduces over the
reflection axis, yielding a ``(T,)`` per-thickness loss) into the engine's per-orientation
``LossFn`` contract: ``AlignedIntensities -> scalar`` (summed over thickness). They save every
caller from rewriting ``lambda a: mse(a.calculated, a.observed).sum()`` and give the loss a named,
importable home -- e.g. ``RefinementEngine(loss=weighted_mse_loss, ...)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from diffBloch.core.losses import l1, mse, optimal_scale, rbragg, w_rbragg, weighted_mse
from diffBloch.core.products import AlignedIntensities

__all__ = [
    "l1_loss",
    "mse_loss",
    "rbragg_loss",
    "wr2_loss",
    "w_rbragg_loss",
    "weighted_mse_loss",
]


def mse_loss(aligned: AlignedIntensities) -> Tensor:
    """Per-orientation MSE loss term, summed over thicknesses to a scalar."""
    return mse(aligned.calculated, aligned.observed).sum()


def l1_loss(aligned: AlignedIntensities) -> Tensor:
    """Per-orientation L1 loss term, summed over thicknesses to a scalar."""
    return l1(aligned.calculated, aligned.observed).sum()


def weighted_mse_loss(aligned: AlignedIntensities) -> Tensor:
    """Per-orientation inverse-variance weighted MSE term, summed over thicknesses to a scalar."""
    return weighted_mse(aligned.calculated, aligned.observed, aligned.sigmas).sum()


def rbragg_loss(aligned: AlignedIntensities) -> Tensor:
    """Per-orientation Bragg R(obs) term, summed over thicknesses to a scalar."""
    return rbragg(aligned.calculated, aligned.observed, aligned.sigmas).sum()


def w_rbragg_loss(aligned: AlignedIntensities) -> Tensor:
    """Per-orientation weighted-R2 term (default ``mu``), summed over thicknesses to a scalar.

    Raw: no calc<->obs scaling. Correct only where the caller has already put calculated on the
    observed scale; for the refinement objective use :func:`wr2_loss`, which is the
    :func:`~diffBloch.preprocess.scoring.build_engine` default.
    """
    return w_rbragg(aligned.calculated, aligned.observed, aligned.sigmas).sum()


def wr2_loss(aligned: AlignedIntensities) -> Tensor:
    """Scaling-optimised weighted-R2 -- the refinement and orientation-search objective.

    The calculated intensities come off the dynamical solve on an arbitrary structure-factor scale,
    while the observed are PETS intensities on their own scale. Compared raw
    (:func:`w_rbragg_loss`), wR2 is denominator-dominated and parks near ~1 with a vanishing
    gradient, so a gradient refinement cannot descend it. Every call therefore re-fits the
    multiplicative intensity scale independently for every thickness through
    :func:`~diffBloch.core.losses.optimal_scale`, exactly as orientation preprocessing does. The
    selected grid branch remains differentiable in its calculated intensities (``torch.min`` routes
    the gradient through the winning candidate); only a boundary where the winning grid point
    changes is piecewise-smooth. Summed over the thickness axis, like the sibling losses.
    """
    calc, obs = aligned.calculated, aligned.observed
    return torch.stack(
        [optimal_scale(calc[t], obs[t], aligned.sigmas[t])[1] for t in range(calc.shape[0])]
    ).sum()
