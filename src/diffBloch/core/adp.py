"""ADP transforms used at the raw-parameter constraint boundary."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor


def cholesky_adp(raw_factor: Tensor) -> Tensor:
    """Map raw 3x3 factors to symmetric positive-semidefinite ADP matrices.

    The private implementation stores anisotropic ADPs as Cholesky factors and expands them as
    ``L @ L.T``. Only the lower triangle is used, giving the six degrees of freedom of a symmetric
    PSD matrix and avoiding gauge-redundant upper-triangular parameters.
    """
    _require_trailing_matrix(raw_factor, name="raw_factor")
    lower = torch.tril(raw_factor)
    return lower @ lower.transpose(-1, -2)


def cholesky_raw_from_adp(uij: Tensor) -> Tensor:
    """Return a Cholesky factor suitable for initializing ``cholesky_adp``.

    This initializer requires positive-definite ADPs, matching ``torch.linalg.cholesky``. Singular
    positive-semidefinite matrices should be regularized before initialization.
    """
    _require_trailing_matrix(uij, name="uij")
    return cast(Tensor, torch.linalg.cholesky(uij))


def isotropic_adp(u_iso: Tensor) -> Tensor:
    """Expand isotropic ``Uiso`` values to 3x3 Cartesian ADP matrices."""
    eye = torch.eye(3, dtype=u_iso.dtype, device=u_iso.device)
    return u_iso[..., None, None] * eye


def cif_adp_to_star(uij_cif: Tensor, reciprocal_lengths: Tensor) -> Tensor:
    """Convert CIF-frame anisotropic ADPs ``Uij_cif`` to the reciprocal ``U*`` frame.

    ``U*_ij = d*_i d*_j Uij_cif`` with ``d* = (|a*|, |b*|, |c*|)`` (``reciprocal_lengths``, shape
    ``(3,)``). This is the private CIF->Cartesian->star chain (``diffBloch_private`` ``Uij_layer``,
    atoms.py:169-171) with the orthogonalization matrix ``A`` cancelled algebraically
    (``A^-1 A D* U D* A^T A^-T = D* U D*``), so it depends only on ``reciprocal_cell`` and is
    exactly faithful to the private result. Differentiable in ``uij_cif``; supports a batch axis.
    """
    _require_trailing_matrix(uij_cif, name="uij_cif")
    if reciprocal_lengths.shape != (3,):
        raise ValueError("reciprocal_lengths must have shape (3,)")
    outer = reciprocal_lengths[:, None] * reciprocal_lengths[None, :]
    return uij_cif * outer


def cartesian_adp_to_star(uij_cart: Tensor, reciprocal_basis: Tensor) -> Tensor:
    """Convert Cartesian-frame ADPs ``Uij_cart`` to the reciprocal ``U*`` frame.

    ``U* = B Uij_cart B^T`` with ``B = reciprocal_cell`` (rows ``a*, b*, c*``). For an isotropic
    Cartesian displacement ``Uij_cart = Uiso I`` this reduces to the textbook ``U* = Uiso G*``
    (``G* = B B^T`` the reciprocal metric), so ``DWF = exp(-2 pi^2 Uiso |g|^2)``. This replaces the
    private ``Uij_layer`` ``A^-1 (Uiso I) A^-T`` path, whose ``A`` (Trueblood eq. 50) is built from
    a ``c_star`` using ``cross(c, b) = |a*|`` rather than ``cross(a, b) = |c*|`` -- a private bug
    that mislabels ``|a*|`` as ``c*`` and only affects anisotropic cells (see REFERENCES.md).
    Differentiable in ``uij_cart``; supports a leading batch axis.
    """
    _require_trailing_matrix(uij_cart, name="uij_cart")
    if reciprocal_basis.shape != (3, 3):
        raise ValueError("reciprocal_basis must have shape (3, 3)")
    return torch.einsum("ij,...jk,lk->...il", reciprocal_basis, uij_cart, reciprocal_basis)


def equivalent_isotropic_adp(uij: Tensor) -> Tensor:
    """Return the trace-equivalent isotropic ADP for 3x3 matrices."""
    _require_trailing_matrix(uij, name="uij")
    return torch.diagonal(uij, dim1=-2, dim2=-1).sum(dim=-1) / 3.0


def _require_trailing_matrix(value: Tensor, *, name: str) -> None:
    if value.ndim < 2 or tuple(value.shape[-2:]) != (3, 3):
        raise ValueError(f"{name} must have trailing shape (3, 3)")
