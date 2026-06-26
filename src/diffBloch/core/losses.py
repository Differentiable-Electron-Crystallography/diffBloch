"""Intensity-space loss/metric functions for refinement.

Pure ports of the intensity comparisons in ``diffBloch_private/diffBloch/metrics.py`` (the
``model``/``ase`` position metrics there belong to the engine/eval layer and are out of scope here).
All functions compare a ``calculated`` intensity tensor against an ``observed`` one and reduce over
the final (reflection) axis, so a ``(T, N)`` thickness/orientation batch yields a ``(T,)`` loss.

- ``mse`` / ``l1`` / ``weighted_mse`` -- generic regression losses (private ``mse_loss`` /
  ``l1_loss`` / ``weighted_mse_loss``).
- ``rbragg`` -- the crystallographic Bragg R(obs) factor over reflections with ``I_obs > 3*sigma``
  (private ``rbragg_abs``).
- ``w_rbragg`` -- the weighted R2 of Klar et al. 2023 (private ``wRbragg``).

See ``REFERENCES.md`` for the R-factor sources. Differentiable in ``calculated``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["l1", "mse", "rbragg", "w_rbragg", "weighted_mse"]


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


def weighted_mse(calculated: Tensor, observed: Tensor, sigmas: Tensor) -> Tensor:
    """Inverse-variance (``1/sigma^2``) weighted squared error, summed over the reflection axis."""
    _check_pair(calculated, observed)
    _check_sigmas(sigmas, observed)
    return ((calculated - observed) ** 2 / sigmas**2).sum(dim=-1)


def rbragg(calculated: Tensor, observed: Tensor, sigmas: Tensor) -> Tensor:
    """Bragg R(obs) factor ``sum|sqrt(I_obs) - sqrt(I_calc)| / sum sqrt(I_obs)``.

    Restricted to observed reflections (``I_obs > 3*sigma``), the standard crystallographic
    significance cut. Reduces over the reflection axis.
    """
    _check_pair(calculated, observed)
    _check_sigmas(sigmas, observed)
    mask = observed > 3 * sigmas
    sqrt_obs, sqrt_calc = observed.sqrt(), calculated.sqrt()
    numerator = ((sqrt_obs - sqrt_calc).abs() * mask).sum(dim=-1)
    denominator = (sqrt_obs * mask).sum(dim=-1)
    return numerator / denominator


def w_rbragg(calculated: Tensor, observed: Tensor, sigmas: Tensor, *, mu: float = 0.01) -> Tensor:
    """Weighted R2 ``sqrt( sum w*(I_calc - I_obs)^2 / sum (w*I_obs)^2 )`` (Klar et al. 2023).

    ``w = 1 / sqrt( sigma(sqrt(I_obs))^2 + (mu*sqrt(I_obs))^2 )`` with ``mu`` the instability
    factor;
    weak reflections (``I_obs < 0.01*sigma``) use the ``5*sqrt(sigma)`` floor from the reference
    SI.
    """
    _check_pair(calculated, observed)
    _check_sigmas(sigmas, observed)
    eps = 1e-12
    sqrt_obs = torch.sqrt(torch.clamp(observed, min=eps))
    weak = observed < (0.01 * sigmas)
    sigma_sqrt = torch.where(weak, 5 * torch.sqrt(sigmas), 0.5 * sigmas / sqrt_obs)
    w = 1.0 / torch.sqrt(sigma_sqrt**2 + (mu * sqrt_obs) ** 2)
    numerator = ((w * (calculated - observed)) ** 2).sum(dim=-1)
    denominator = ((w * observed) ** 2).sum(dim=-1)
    return torch.sqrt(numerator / denominator)
