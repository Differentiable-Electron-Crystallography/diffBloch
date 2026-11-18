"""Intensity-space loss/metric functions for refinement.

The intensity comparisons used to score and refine a structure. (Position-space metrics belong to
the engine/eval layer and are out of scope here.) All functions compare a ``calculated`` intensity
tensor against an ``observed`` one and reduce over the final (reflection) axis, so a ``(T, N)``
thickness/orientation batch yields a ``(T,)`` loss.

- ``mse`` / ``l1`` -- generic regression losses.
- ``rbragg`` -- the crystallographic Bragg R(obs) factor over reflections with ``I_obs > 3*sigma``.
- ``weighted_r_amplitude`` -- Jana2020's weighted ``wR`` on amplitudes ``|F|``.
- ``weighted_r_intensity`` -- the weighted ``wR2`` on intensities (``wR(F^2)``).

Differentiable in ``calculated``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

__all__ = [
    "amplitude_sigma",
    "l1",
    "mse",
    "optimal_scale",
    "rbragg",
    "weighted_r_amplitude",
    "weighted_r_intensity",
]


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


def amplitude_sigma(observed: Tensor, sigmas: Tensor) -> Tensor:
    """Standard uncertainty of the amplitude ``|F| = sqrt(I)`` from the intensity uncertainty.

    ``sigma(|F|) = sigma(I) / (2 sqrt(I))`` for measurable reflections; weak ones
    (``I < 0.01*sigma(I)``, including negative background-subtracted intensities) take the floor
    ``5*sqrt(sigma(I))``, which equals the same formula evaluated at ``I = 0.01*sigma(I)`` (Klar et
    al. 2023 SI). This is exactly the ``sigma(F)`` Jana2020 uses for its dynamical refinement on
    ``|F|``: recomputing Jana's own listed ``sigma(F)`` from its ``I``/``sigma(I)`` reproduces it.
    """
    _check_sigmas(sigmas, observed)
    sqrt_obs = torch.sqrt(torch.clamp(observed, min=1e-12))
    weak = observed < (0.01 * sigmas)
    return torch.where(weak, 5 * torch.sqrt(sigmas), 0.5 * sigmas / sqrt_obs)


def weighted_r_amplitude(
    calculated: Tensor, observed: Tensor, sigmas: Tensor, *, mu: float = 0.01
) -> Tensor:
    """Weighted R on amplitudes, ``wR = sqrt( sum w (Fo - Fc)^2 / sum w Fo^2 )`` (Jana2020's ``wR``).

    ``Fo = sqrt(max(I_obs, 0))`` and ``Fc = sqrt(I_calc)``; ``w = 1/(sigma(F)^2 + (mu*Fo)^2)`` with
    ``sigma(F)`` from :func:`amplitude_sigma` and ``mu`` the instability factor. This is the ``wR``
    Jana2020 prints for a refinement on ``|F|``, and the ``wR(all)`` of Klar et al. (2023): their SI
    prints the residual without the square roots, but their published values reproduce only with
    them. Every reflection takes part; a negative (background-subtracted) intensity enters at
    ``Fo = 0`` with the weak-reflection ``sigma(F)`` floor, as in Jana2020.

    ``I_calc`` is floored at ``1e-12`` before the square root so the gradient stays finite at a
    vanishing calculated intensity.
    """
    _check_pair(calculated, observed)
    _check_sigmas(sigmas, observed)
    fo = observed.clamp(min=0.0).sqrt()
    fc = calculated.clamp(min=1e-12).sqrt()
    w = 1.0 / (amplitude_sigma(observed, sigmas) ** 2 + (mu * fo) ** 2)
    numerator = (w * (fo - fc) ** 2).sum(dim=-1)
    denominator = (w * fo**2).sum(dim=-1)
    return torch.sqrt(numerator / denominator)


def weighted_r_intensity(
    calculated: Tensor, observed: Tensor, sigmas: Tensor, *, mu: float = 0.01
) -> Tensor:
    """Weighted R on intensities, ``wR2 = sqrt( sum w (Io - Ic)^2 / sum w Io^2 )`` (``wR(F^2)``).

    ``w = 1/(sigma(I)^2 + (2*mu*I_obs)^2)``: the same instability factor as
    :func:`weighted_r_amplitude`, carried over to intensities by propagation
    (``sigma(I) = 2 F sigma(F)``, so ``mu*F`` becomes ``2*mu*I``). Every reflection takes part;
    negative intensities need no special handling on this scale and enter as measured.
    """
    _check_pair(calculated, observed)
    _check_sigmas(sigmas, observed)
    w = 1.0 / (sigmas**2 + (2.0 * mu * observed) ** 2)
    numerator = (w * (observed - calculated) ** 2).sum(dim=-1)
    denominator = (w * observed**2).sum(dim=-1)
    return torch.sqrt(numerator / denominator)


def optimal_scale(
    calculated: Tensor,
    observed: Tensor,
    sigmas: Tensor,
    *,
    metric: Callable[[Tensor, Tensor, Tensor], Tensor] = weighted_r_amplitude,
    num_points: int = 100,
    lo: float = 0.02,
    hi: float = 2.0,
) -> tuple[Tensor, Tensor]:
    """Grid-search the multiplicative scale on ``calculated`` that minimises ``metric``.

    The search runs ``num_points``
    factors in ``[lo, hi]`` *relative to* the total ratio ``sum(observed)/sum(calculated)`` (so it
    is centred near scale 1), evaluates ``metric(scaled, observed, sigmas)`` at each, and returns
    the **absolute** scale applied to ``calculated`` and the minimum metric value. ``metric``
    defaults to :func:`weighted_r_amplitude` (the default objective, ``wr``); it must reduce over the final
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
