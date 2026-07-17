"""Soft refinement restraints evaluated on the bounded physical ASU state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor

from diffBloch.params import PhysicalState

__all__ = ["BondRestraints"]


@dataclass(frozen=True)
class BondRestraints:
    """Bond-length soft restraints in Cartesian Angstrom units.

    The current ASU positions are fractional coordinates, so the restraint owns the invariant
    fractional-to-Cartesian cell matrix. ``pairs`` indexes ASU atom rows; ``target_angstrom`` and
    ``sigma_angstrom`` carry the restraint target and tolerance for each pair. The raw loss is the
    mean squared normalized bond-distance residual by default. ``flat_bottom_l1`` is the
    private/abiraterone-style robust criterion: zero inside the sigma tolerance and linear outside.
    """

    pairs: Tensor
    target_angstrom: Tensor
    sigma_angstrom: Tensor
    frac_to_cart: Tensor
    weight: float = 1.0
    name: str = "bond"
    criterion: Literal["mse", "flat_bottom_l1"] = "mse"

    def __post_init__(self) -> None:
        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2:
            raise ValueError("bond pairs must have shape (N, 2)")
        n_bonds = int(self.pairs.shape[0])
        if self.target_angstrom.shape != (n_bonds,):
            raise ValueError("bond targets must have shape (N,)")
        if self.sigma_angstrom.shape != (n_bonds,):
            raise ValueError("bond sigmas must have shape (N,)")
        if self.frac_to_cart.shape != (3, 3):
            raise ValueError("frac_to_cart must have shape (3, 3)")
        if bool((self.sigma_angstrom <= 0).any()):
            raise ValueError("bond sigmas must be positive")
        if self.weight < 0:
            raise ValueError("bond restraint weight must be non-negative")
        if self.criterion not in {"mse", "flat_bottom_l1"}:
            raise ValueError("bond restraint criterion must be 'mse' or 'flat_bottom_l1'")

    def loss(self, state: PhysicalState) -> Tensor:
        """Return the raw mean squared normalized bond-distance residual."""
        device = state.positions.device
        dtype = state.positions.dtype
        pairs = self.pairs.to(device=device, dtype=torch.long)
        targets = self.target_angstrom.to(device=device, dtype=dtype)
        sigmas = self.sigma_angstrom.to(device=device, dtype=dtype)
        frac_to_cart = self.frac_to_cart.to(device=device, dtype=dtype)
        positions_cart = state.positions @ frac_to_cart
        vectors = positions_cart[pairs[:, 1]] - positions_cart[pairs[:, 0]]
        distances = torch.linalg.norm(vectors, dim=1)
        deviations = distances - targets
        if self.criterion == "mse":
            residuals = deviations / sigmas
            return cast(Tensor, residuals.square().mean())
        return torch.clamp(deviations.abs() - sigmas, min=0.0).mean()
