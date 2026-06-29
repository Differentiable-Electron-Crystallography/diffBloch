"""Named ``LossFn`` builders: per-orientation loss terms you plug into ``RefinementEngine.loss``.

Each adapts a pure :mod:`diffBloch.core.losses` intensity comparison (which reduces over the
reflection axis, yielding a ``(T,)`` per-thickness loss) into the engine's per-orientation
``LossFn`` contract: ``AlignedIntensities -> scalar`` (summed over thickness). They save every
caller from rewriting ``lambda a: mse(a.calculated, a.observed).sum()`` and give the loss a named,
importable home -- e.g. ``RefinementEngine(loss=weighted_mse_loss, ...)``.
"""

from __future__ import annotations

from torch import Tensor

from diffBloch.core.losses import l1, mse, rbragg, w_rbragg, weighted_mse
from diffBloch.core.products import AlignedIntensities

__all__ = [
    "l1_loss",
    "mse_loss",
    "rbragg_loss",
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
    """Per-orientation weighted-R2 term (default ``mu``), summed over thicknesses to a scalar."""
    return w_rbragg(aligned.calculated, aligned.observed, aligned.sigmas).sum()
