"""Symmetry expansion with precomputed ASU membership."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

type DuplicatePolicy = Literal["error", "warn", "keep", "replace"]


@dataclass(frozen=True)
class AsuExpansionPlan:
    """Precomputed ASU membership for differentiable symmetry expansion."""

    asu_indices: Tensor
    symop_indices: Tensor
    rotations: Tensor
    translations: Tensor
    n_asu_sites: int

    @property
    def n_expanded_sites(self) -> int:
        """Number of unique expanded sites in the plan."""
        return int(self.asu_indices.shape[0])


@dataclass(frozen=True)
class ExpandedAsu:
    """Expanded ASU tensors and the membership that produced them."""

    positions: Tensor
    asu_indices: Tensor
    symop_indices: Tensor
    numbers: Tensor | None = None
    uij: Tensor | None = None
    occupancies: Tensor | None = None


def build_asu_expansion_plan(
    frac_positions: NDArray[np.float64],
    symops_R: NDArray[np.float64],
    symops_t: NDArray[np.float64],
    *,
    symprec: float = 1e-3,
    onduplicates: DuplicatePolicy = "error",
) -> AsuExpansionPlan:
    """Precompute unique ASU/symop memberships for later torch expansion.

    This mirrors the private atom-major, symop-minor membership order while moving duplicate
    detection out of the differentiable path.
    """
    if symprec <= 0.0:
        raise ValueError("symprec must be positive")
    if onduplicates not in {"error", "warn", "keep", "replace"}:
        raise ValueError(f"unsupported duplicate policy: {onduplicates!r}")

    positions = np.asarray(frac_positions, dtype=np.float64)
    rotations = np.asarray(symops_R, dtype=np.float64)
    translations = np.asarray(symops_t, dtype=np.float64)
    _validate_plan_inputs(positions, rotations, translations)

    sites: list[NDArray[np.float64]] = []
    asu_indices: list[int] = []
    symop_indices: list[int] = []
    for asu_index, position in enumerate(positions):
        for symop_index, (rotation, translation) in enumerate(
            zip(rotations, translations, strict=True)
        ):
            site = np.remainder(rotation @ position + translation, 1.0)
            duplicate_indices = _duplicate_indices(site, sites, symprec=symprec)
            if not duplicate_indices:
                sites.append(site)
                asu_indices.append(asu_index)
                symop_indices.append(symop_index)
                continue

            for duplicate_index in duplicate_indices:
                if asu_indices[duplicate_index] == asu_index:
                    continue
                if onduplicates == "keep":
                    continue
                if onduplicates == "warn":
                    warnings.warn(
                        f"scaled_positions {asu_indices[duplicate_index]} and {asu_index} "
                        "are equivalent",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                if onduplicates == "replace":
                    sites[duplicate_index] = site
                    asu_indices[duplicate_index] = asu_index
                    symop_indices[duplicate_index] = symop_index
                    continue
                raise ValueError(
                    f"scaled_positions {asu_indices[duplicate_index]} and {asu_index} "
                    "are equivalent"
                )

    return AsuExpansionPlan(
        asu_indices=torch.tensor(asu_indices, dtype=torch.long),
        symop_indices=torch.tensor(symop_indices, dtype=torch.long),
        rotations=torch.tensor(rotations, dtype=torch.float64),
        translations=torch.tensor(translations, dtype=torch.float64),
        n_asu_sites=int(positions.shape[0]),
    )


def expand_asu(
    plan: AsuExpansionPlan,
    positions: Tensor,
    *,
    numbers: Tensor | None = None,
    uij: Tensor | None = None,
    occupancies: Tensor | None = None,
) -> ExpandedAsu:
    """Expand ASU tensors using a precomputed membership plan."""
    if positions.ndim != 2 or tuple(positions.shape[1:]) != (3,):
        raise ValueError("positions must have shape (N, 3)")
    if int(positions.shape[0]) != plan.n_asu_sites:
        raise ValueError("positions atom count must match plan.n_asu_sites")

    asu_indices = plan.asu_indices.to(device=positions.device)
    symop_indices = plan.symop_indices.to(device=positions.device)
    rotations = plan.rotations.to(device=positions.device, dtype=positions.dtype)[symop_indices]
    translations = plan.translations.to(device=positions.device, dtype=positions.dtype)[
        symop_indices
    ]

    expanded_positions = torch.remainder(
        torch.einsum("mij,mj->mi", rotations, positions[asu_indices]) + translations,
        1.0,
    )
    return ExpandedAsu(
        positions=expanded_positions,
        asu_indices=asu_indices,
        symop_indices=symop_indices,
        numbers=_expand_optional_vector(numbers, asu_indices, plan=plan, name="numbers"),
        uij=_expand_optional_uij(uij, asu_indices, rotations, plan=plan),
        occupancies=_expand_optional_vector(
            occupancies,
            asu_indices,
            plan=plan,
            name="occupancies",
        ),
    )


def _duplicate_indices(
    site: NDArray[np.float64],
    sites: list[NDArray[np.float64]],
    *,
    symprec: float,
) -> list[int]:
    if not sites:
        return []
    diff = site - np.asarray(sites, dtype=np.float64)
    duplicate_mask = np.all(
        (np.abs(diff) < symprec) | (np.abs(np.abs(diff) - 1.0) < symprec),
        axis=1,
    )
    return [int(index) for index in np.argwhere(duplicate_mask).flatten()]


def _expand_optional_vector(
    values: Tensor | None,
    asu_indices: Tensor,
    *,
    plan: AsuExpansionPlan,
    name: str,
) -> Tensor | None:
    if values is None:
        return None
    if values.ndim != 1 or int(values.shape[0]) != plan.n_asu_sites:
        raise ValueError(f"{name} must have shape (N,) matching plan.n_asu_sites")
    return values[asu_indices.to(device=values.device)]


def _expand_optional_uij(
    uij: Tensor | None,
    asu_indices: Tensor,
    rotations: Tensor,
    *,
    plan: AsuExpansionPlan,
) -> Tensor | None:
    if uij is None:
        return None
    if uij.ndim != 3 or tuple(uij.shape[1:]) != (3, 3) or int(uij.shape[0]) != plan.n_asu_sites:
        raise ValueError("uij must have shape (N, 3, 3) matching plan.n_asu_sites")
    selected = uij[asu_indices.to(device=uij.device)]
    uij_rotations = rotations.to(device=uij.device, dtype=uij.dtype)
    return uij_rotations @ selected @ uij_rotations.transpose(-1, -2)


def _validate_plan_inputs(
    positions: NDArray[np.float64],
    rotations: NDArray[np.float64],
    translations: NDArray[np.float64],
) -> None:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("frac_positions must have shape (N, 3)")
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("symops_R must have shape (S, 3, 3)")
    if translations.shape != (rotations.shape[0], 3):
        raise ValueError("symops_t must have shape (S, 3) matching symops_R")
