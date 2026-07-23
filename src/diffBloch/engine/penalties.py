"""Soft refinement penalties evaluated on the bounded physical ASU state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import torch
from torch import Tensor

from diffBloch.engine.chemistry import covalent_radius
from diffBloch.io.record import StructureRecord
from diffBloch.params import PhysicalState

__all__ = ["BondLengthPenalty", "perceive_bond_length_penalty"]


@dataclass(frozen=True)
class BondLengthPenalty:
    """Bond-length soft penalties in Cartesian Angstrom units.

    The current ASU positions are fractional coordinates, so the penalty owns the invariant
    fractional-to-Cartesian cell matrix. ``pairs`` indexes ASU atom rows; ``target_angstrom`` and
    ``sigma_angstrom`` carry the penalty target and tolerance for each pair. The raw loss is the
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
            raise ValueError("bond penalty weight must be non-negative")
        if self.criterion not in {"mse", "flat_bottom_l1"}:
            raise ValueError("bond penalty criterion must be 'mse' or 'flat_bottom_l1'")

    def value(self, state: PhysicalState) -> Tensor:
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


def perceive_bond_length_penalty(
    structure: StructureRecord,
    *,
    include_hydrogen: bool = False,
    sigma_angstrom: float = 0.02,
    cutoff_scale: float = 1.2,
    cutoff_margin_angstrom: float = 0.1,
    weight: float = 1.0,
    criterion: Literal["mse", "flat_bottom_l1"] = "mse",
) -> BondLengthPenalty:
    """Perceive ASU-contiguous bonds and tether them to the starting distances.

    This is an explicit source/builder layer, separate from the pure penalty value. It assumes
    the bonded molecule is contiguous in the ASU and deliberately does **not** do minimum-image
    wrapping, matching the private bond loss. The perceived target for each bond is the current
    Cartesian distance in the input structure; Mogul/CIF penalty sources can later provide
    literature targets and sigmas.
    """
    if sigma_angstrom <= 0:
        raise ValueError("bond-penalty sigma must be positive")
    if cutoff_scale <= 0:
        raise ValueError("bond perception cutoff_scale must be positive")
    if cutoff_margin_angstrom < 0:
        raise ValueError("bond perception cutoff_margin_angstrom must be non-negative")

    positions_cart = np.asarray(structure.frac_positions @ structure.unit_cell, dtype=np.float64)
    numbers = np.asarray(structure.numbers, dtype=np.int64)
    pairs: list[tuple[int, int]] = []
    distances: list[float] = []
    for i in range(structure.n_atoms):
        if not include_hydrogen and numbers[i] == 1:
            continue
        radius_i = covalent_radius(numbers[i])
        for j in range(i + 1, structure.n_atoms):
            if not include_hydrogen and numbers[j] == 1:
                continue
            radius_j = covalent_radius(numbers[j])
            distance = float(np.linalg.norm(positions_cart[j] - positions_cart[i]))
            cutoff = cutoff_scale * (radius_i + radius_j) + cutoff_margin_angstrom
            if 1e-8 < distance <= cutoff:
                pairs.append((i, j))
                distances.append(distance)
    if not pairs:
        raise ValueError("bond perception found no bonds")
    return BondLengthPenalty(
        pairs=torch.tensor(pairs, dtype=torch.int64),
        target_angstrom=torch.tensor(distances, dtype=torch.float64),
        sigma_angstrom=torch.full((len(pairs),), sigma_angstrom, dtype=torch.float64),
        frac_to_cart=torch.tensor(structure.unit_cell, dtype=torch.float64),
        weight=weight,
        criterion=criterion,
    )
