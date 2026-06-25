"""Pure tensor constraints and bijectors for raw refinement parameters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def unit_interval(raw: Tensor) -> Tensor:
    """Map unconstrained values to the open interval ``(0, 1)``."""
    return torch.sigmoid(raw)


def positive(raw: Tensor) -> Tensor:
    """Map unconstrained values to positive values."""
    return torch.nn.functional.softplus(raw)


@dataclass(frozen=True)
class SymmetryMask:
    """Freeze symmetry-constrained degrees of freedom while preserving free gradients."""

    mask: Tensor
    fixed: Tensor

    def forward(self, raw: Tensor) -> Tensor:
        """Apply ``raw * mask + fixed.detach() * (1 - mask)`` without mutating ``raw``."""
        if raw.shape != self.mask.shape or raw.shape != self.fixed.shape:
            raise ValueError("raw, mask, and fixed tensors must have matching shapes")
        mask = self.mask.to(dtype=raw.dtype, device=raw.device)
        fixed = self.fixed.to(dtype=raw.dtype, device=raw.device)
        return raw * mask + fixed.detach() * (1.0 - mask)


def apply_symmetry_mask(raw: Tensor, *, mask: Tensor, fixed: Tensor) -> Tensor:
    """Functional wrapper for :class:`SymmetryMask`."""
    return SymmetryMask(mask=mask, fixed=fixed).forward(raw)
