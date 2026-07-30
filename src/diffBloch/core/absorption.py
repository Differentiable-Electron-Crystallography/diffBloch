"""Differentiable absorptive electron-scattering factors for Bloch-wave refinement."""

from __future__ import annotations

import torch
from torch import Tensor

from diffBloch.core.absorptive_parameters import ABSORPTIVE_B_VALUES, ABSORPTIVE_PARAMETERS

__all__ = ["absorptive_form_factors", "equivalent_isotropic_b"]

_ELECTRON_REST_ENERGY_EV = 510_998.95


def equivalent_isotropic_b(uij_star: Tensor, reciprocal_basis: Tensor) -> Tensor:
    """Convert reciprocal-basis ``U*`` tensors to equivalent isotropic ``B`` in Angstrom²."""
    basis = reciprocal_basis.to(device=uij_star.device, dtype=uij_star.dtype)
    inverse = torch.linalg.inv(basis)
    u_cart = torch.einsum("ij,ajk,lk->ail", inverse, uij_star, inverse)
    return 8.0 * torch.pi**2 * torch.diagonal(u_cart, dim1=-2, dim2=-1).mean(dim=-1)


def _curves(s: Tensor, parameters: Tensor) -> Tensor:
    """Evaluate the four-Gaussian-plus-constant fits, returning ``(A, B, G)``."""
    s2 = s.square()[None, None, :]
    amplitudes = parameters[..., 0:8:2, None]
    widths = parameters[..., 1:8:2, None].abs()
    return (amplitudes * torch.exp(-widths * s2[:, :, None, :])).sum(dim=2) + parameters[
        ..., 8, None
    ]


def absorptive_form_factors(
    numbers: Tensor,
    s: Tensor,
    b_iso: Tensor,
    *,
    energy: float,
) -> Tensor:
    """Legacy fitted absorptive atomic factors ``f'`` for atoms × scattering-vector lengths.

    ``s`` is ``|g| / 2`` in inverse Angstrom, ``b_iso`` is one value per atom, and energy is in eV.
    The fitted range is clamped to ``0.1 <= B <= 4.0`` exactly as in the paper implementation.
    Atomic numbers 1--103 are supported. The interpolation remains differentiable in ``b_iso``.
    """
    if energy <= 0.0:
        raise ValueError("energy must be positive")
    if numbers.ndim != 1 or b_iso.shape != numbers.shape:
        raise ValueError("numbers and b_iso must be matching one-dimensional tensors")
    if s.ndim != 1 or bool(torch.any(s < 0.0)):
        raise ValueError("s must be a non-negative one-dimensional tensor")
    if bool(torch.any((numbers < 1) | (numbers > 103))):
        raise ValueError("parameterized absorption supports atomic numbers 1 through 103")

    dtype, device = s.dtype, s.device
    knots = ABSORPTIVE_B_VALUES.to(device=device, dtype=dtype)
    table = ABSORPTIVE_PARAMETERS.to(device=device, dtype=dtype)[numbers.to(device).long() - 1]
    values = _curves(s, table)

    spacing = torch.diff(knots)
    slopes = torch.diff(values, dim=1) / spacing[None, :, None]
    weights = torch.abs(slopes[:, 1:] - slopes[:, :-1]) + 1e-10
    interior = (weights * slopes[:, :-1] + weights.flip(1) * slopes[:, 1:]) / (
        weights + weights.flip(1)
    )
    derivatives = torch.cat((slopes[:, :1], interior, slopes[:, -1:]), dim=1)

    b = b_iso.to(device=device, dtype=dtype).clamp(0.1, 4.0)
    upper = torch.searchsorted(knots, b, right=False).clamp(1, knots.numel() - 1)
    lower = upper - 1
    atom = torch.arange(numbers.numel(), device=device)
    width = spacing[lower]
    t = (b - knots[lower]) / width
    t2, t3 = t.square(), t.square() * t
    y0, y1 = values[atom, lower], values[atom, upper]
    d0, d1 = derivatives[atom, lower], derivatives[atom, upper]
    interpolated = (
        (2 * t3 - 3 * t2 + 1)[:, None] * y0
        + ((t3 - 2 * t2 + t) * width)[:, None] * d0
        + (-2 * t3 + 3 * t2)[:, None] * y1
        + ((t3 - t2) * width)[:, None] * d1
    )

    gamma = 1.0 + energy / _ELECTRON_REST_ENERGY_EV
    c_over_v = 1.0 / (1.0 - gamma**-2) ** 0.5
    result: Tensor = c_over_v * interpolated.clamp_min(0.0)
    return result
