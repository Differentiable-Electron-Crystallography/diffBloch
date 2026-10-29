"""Intensity-space loss/metric functions for refinement.

The intensity comparisons used to score and refine a structure. (Position-space metrics belong to
the engine/eval layer and are out of scope here.) All functions compare a ``calculated`` intensity
tensor against an ``observed`` one and reduce over the final (reflection) axis, so a ``(T, N)``
thickness/orientation batch yields a ``(T,)`` loss.

- ``mse`` / ``l1`` -- generic regression losses.
- ``rbragg`` -- the crystallographic Bragg R(obs) factor over reflections with ``I_obs > 3*sigma``.
- ``w_rbragg`` -- the weighted R2 of Klar et al. 2023.

Differentiable in ``calculated``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

__all__ = ["l1", "mse", "optimal_scale", "rbragg", "w_rbragg"]


def _check_pair(calculated: Tensor, observed: Tensor) -> None:
    if calculated.shape != observed.shape:
        raise ValueError(
            f"calculated and observed must have the same shape, got {tuple(calculated.shape)} "
            f"and {tuple(observed.shape)}"
        )


def _check_sigmas(sigmas: Tensor, observed: Tensor) -> None:
    if sigmas.shape != observed.shape:
        raise ValueError(
            f"sigmas must have shape {tuple(observed.shape)}, got {tuple(sigmas.shape)}"
        )


def mse(calculated: Tensor, observed: Tensor) -> Tensor:
    """Mean squared error over the reflection axis."""
    _check_pair(calculated, observed)
    return ((calculated - observed) ** 2).mean(dim=-1)


def l1(calculated: Tensor, observed: Tensor) -> Tensor:
    """Mean absolute error over the reflection axis."""
    _check_pair(calculated, observed)
    return (calculated - observed).abs().mean(dim=-1)


def rbragg(calculated: Tensor, observed: Tensor, sigmas: Tensor) -> Tensor:
    """Bragg R(obs) factor ``sum|sqrt(I_obs) - sqrt(I_calc)| / sum sqrt(I_obs)``.

    Restricted to observed reflections (``I_obs > 3*sigma``), the standard crystallographic
    significance cut. Reduces over the reflection axis.

    The ``I_obs > 3*sigma`` cut is applied by *selection* (``torch.where``), not by multiplying a
    0/1 mask: experimental intensities can be negative (background-subtracted), so ``sqrt`` of an
    excluded reflection is ``NaN``, and ``NaN * 0`` would poison the sum. Masked-in reflections have
    ``I_obs > 3*sigma > 0`` (and calculated ``|psi|^2 >= 0``), so their square roots are always
    finite; the clamps guard only against numerical noise. A 0/1 multiply-mask would be
    ``NaN``-unsafe on the negative excluded intensities, which is why selection is used instead.
    """
    _check_pair(calculated, observed)
    _check_sigmas(sigmas, observed)
    mask = observed > 3 * sigmas
    sqrt_obs = observed.clamp(min=0.0).sqrt()
    sqrt_calc = calculated.clamp(min=0.0).sqrt()
    zero = torch.zeros_like(observed)
    numerator = torch.where(mask, (sqrt_obs - sqrt_calc).abs(), zero).sum(dim=-1)
    denominator = torch.where(mask, sqrt_obs, zero).sum(dim=-1)
    return numerator / denominator


def w_rbragg(calculated: Tensor, observed: Tensor, sigmas: Tensor, *, mu: float = 0.01) -> Tensor:
    """Weighted R2 ``sqrt( sum w*(I_calc - I_obs)^2 / sum (w*I_obs)^2 )`` (Klar et al. 2023).

    ``w = 1 / sqrt( sigma(sqrt(I_obs))^2 + (mu*sqrt(I_obs))^2 )`` with ``mu`` the instability
    factor;
    weak reflections (``I_obs < 0.01*sigma``) use the ``5*sqrt(sigma)`` floor from the Klar et al.
    2023 supplementary information.

    Reflections with ``I_obs < 0`` (background-subtracted noise, not real signal) are excluded by
    *selection* (``torch.where``), the same NaN-safe pattern :func:`rbragg` uses -- they take no
    part in either sum rather than being fit against.
    """
    _check_pair(calculated, observed)
    _check_sigmas(sigmas, observed)
    eps = 1e-12
    mask = observed >= 0.0
    sqrt_obs = torch.sqrt(torch.clamp(observed, min=eps))
    weak = observed < (0.01 * sigmas)
    sigma_sqrt = torch.where(weak, 5 * torch.sqrt(sigmas), 0.5 * sigmas / sqrt_obs)
    w = 1.0 / torch.sqrt(sigma_sqrt**2 + (mu * sqrt_obs) ** 2)
    zero = torch.zeros_like(observed)
    numerator = torch.where(mask, (w * (calculated - observed)) ** 2, zero).sum(dim=-1)
    denominator = torch.where(mask, (w * observed) ** 2, zero).sum(dim=-1)
    return torch.sqrt(numerator / denominator)


def optimal_scale(
    calculated: Tensor,
    observed: Tensor,
    sigmas: Tensor,
    *,
    metric: Callable[[Tensor, Tensor, Tensor], Tensor] = w_rbragg,
    num_points: int = 100,
    lo: float = 0.02,
    hi: float = 2.0,
) -> tuple[Tensor, Tensor]:
    """Grid-search the multiplicative scale on ``calculated`` that minimises ``metric``.

    The search runs ``num_points``
    factors in ``[lo, hi]`` *relative to* the total ratio ``sum(observed)/sum(calculated)`` (so it
    is centred near scale 1), evaluates ``metric(scaled, observed, sigmas)`` at each, and returns
    the **absolute** scale applied to ``calculated`` and the minimum metric value. ``metric``
    defaults to :func:`w_rbragg` (the wR2 used to score orientations); it must reduce over the final
    reflection axis so a ``(num_points, N)`` batch yields ``(num_points,)``.
    """
    _check_pair(calculated, observed)
    _check_sigmas(sigmas, observed)
    grid = torch.linspace(lo, hi, num_points, dtype=calculated.dtype, device=calculated.device)
    ratio = observed.sum() / calculated.sum()
    scaled = grid.view(-1, 1) * ratio * calculated  # (num_points, N)
    values = metric(scaled, observed.expand_as(scaled), sigmas.expand_as(scaled))
    min_value, index = torch.min(values, dim=0)
    return grid[index] * ratio, min_value
