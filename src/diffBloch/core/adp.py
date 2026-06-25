"""ADP transforms used at the raw-parameter constraint boundary."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor


def cholesky_adp(raw_factor: Tensor) -> Tensor:
    """Map raw 3x3 factors to symmetric positive-semidefinite ADP matrices.

    The private implementation stores anisotropic ADPs as Cholesky factors and expands them as
    ``L @ L.T``. This public helper keeps the same PSD guarantee without mutating the input tensor.
    """
    _require_trailing_matrix(raw_factor, name="raw_factor")
    return raw_factor @ raw_factor.transpose(-1, -2)


def cholesky_raw_from_adp(uij: Tensor) -> Tensor:
    """Return a Cholesky factor suitable for initializing ``cholesky_adp``."""
    _require_trailing_matrix(uij, name="uij")
    return cast(Tensor, torch.linalg.cholesky(uij))


def isotropic_adp(u_iso: Tensor) -> Tensor:
    """Expand isotropic ``Uiso`` values to 3x3 Cartesian ADP matrices."""
    eye = torch.eye(3, dtype=u_iso.dtype, device=u_iso.device)
    return u_iso[..., None, None] * eye


def equivalent_isotropic_adp(uij: Tensor) -> Tensor:
    """Return the trace-equivalent isotropic ADP for 3x3 matrices."""
    _require_trailing_matrix(uij, name="uij")
    return torch.diagonal(uij, dim1=-2, dim2=-1).sum(dim=-1) / 3.0


def _require_trailing_matrix(value: Tensor, *, name: str) -> None:
    if value.ndim < 2 or tuple(value.shape[-2:]) != (3, 3):
        raise ValueError(f"{name} must have trailing shape (3, 3)")
