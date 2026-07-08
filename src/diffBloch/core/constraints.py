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
class _SymmetryProjection:
    """Project raw coordinates onto the site-symmetry-allowed subspace, holding the fixed part."""

    projection: Tensor  # (N, 3, 3) per-atom site-symmetry projector P
    offset: Tensor  # (N, 3) on-site offset (I - P) @ x0

    def forward(self, raw: Tensor) -> Tensor:
        """Apply ``P @ raw + offset.detach()`` per atom without mutating ``raw``.

        With ``offset = (I - P) @ x0`` this equals ``x0 + P @ (raw - x0)``, so the constrained
        position stays on the special-position manifold ``x0 + image(P)`` for any ``raw``: gradients
        reach only ``image(P)`` (the free degrees of freedom), and off-site directions are projected
        out.
        """
        n_atoms = int(raw.shape[0])
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError("raw must have shape (N, 3)")
        if self.projection.shape != (n_atoms, 3, 3):
            raise ValueError("projection must have shape (N, 3, 3)")
        if self.offset.shape != raw.shape:
            raise ValueError("offset must have shape (N, 3) matching raw")
        projection = self.projection.to(dtype=raw.dtype, device=raw.device)
        offset = self.offset.to(dtype=raw.dtype, device=raw.device)
        return torch.einsum("nij,nj->ni", projection, raw) + offset.detach()


def apply_symmetry_projection(raw: Tensor, *, projection: Tensor, offset: Tensor) -> Tensor:
    """Project raw coordinates onto the site-symmetry-allowed subspace.

    This is the public API for the operation; the tiny callable object is kept private so there is
    one advertised way to apply the projector.
    """
    return _SymmetryProjection(projection=projection, offset=offset).forward(raw)


type AdpConstraint = tuple[int, int, int, int, float]
type AdpConstraints = tuple[tuple[AdpConstraint, ...], ...]


def apply_adp_constraints(uij: Tensor, constraints: AdpConstraints) -> Tensor:
    """Enforce site-symmetry ADP equalities ``Uij[i,j] = coeff * Uij[src_i, src_j]`` per atom.

    ``uij`` is the ``(N, 3, 3)`` symmetric ADP tensor in the CIF frame; ``constraints`` holds, per
    atom (one entry each), the tuples ``(i, j, src_i, src_j, coeff)`` -- meaning
    ``Uij[i,j] = coeff * Uij[src_i, src_j]`` -- extracted from the space-group ``Ueqns`` (see
    :func:`diffBloch.io.symmetry_setup.symmetry_constraints`). Both are aligned by
    construction -- the caller (:func:`diffBloch.params.constrain`, validated once at its boundary)
    guarantees ``len(constraints) == N`` -- so this is a pure transform, not a validator. Each
    right-hand side reads the *original* (unconstrained) component, so the order of application does
    not matter, and both ``[i,j]`` and ``[j,i]`` are set to keep the tensor symmetric. Atoms with no
    constraints pass through unchanged; dependent raw components simply stop affecting the output
    (their gradient vanishes), which is the ADP analogue of freezing an over-parameterized
    coordinate.
    """
    result = uij.clone()
    for atom, atom_constraints in enumerate(constraints):
        for i, j, src_i, src_j, coeff in atom_constraints:
            value = coeff * uij[atom, src_i, src_j]
            result[atom, i, j] = value
            result[atom, j, i] = value
    return result


def diagonal_projection(mask: Tensor, fixed: Tensor) -> tuple[Tensor, Tensor]:
    """Build ``(projection, offset)`` for the axis-aligned special case from a 0/1 mask.

    The diagonal projector ``P = diag(mask)`` with ``offset = fixed * (1 - mask)`` reproduces the
    per-coordinate freeze ``raw * mask + fixed * (1 - mask)`` -- the special case of
    :func:`apply_symmetry_projection` in which every constrained coordinate is fixed to a constant
    (``1 = free``, ``0 = held fixed``). Coupled degrees of freedom (``x = y``) need the full
    off-diagonal projector and cannot be expressed this way.
    """
    if mask.shape != fixed.shape:
        raise ValueError("mask and fixed tensors must have matching shapes")
    return torch.diag_embed(mask), fixed * (1.0 - mask)
