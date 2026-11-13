"""Pair-algebra ``matrix_exp`` (core._matrix_exp_pair) vs. torch.matrix_exp's own autograd."""

import pytest
import torch

from diffBloch.core._matrix_exp_pair import matrix_exp as pair_matrix_exp


def _random_operator(
    dtype: torch.dtype, n: int, batch: int, seed: int, scale: float
) -> torch.Tensor:
    torch.manual_seed(seed)
    shape = (batch, n, n) if batch else (n, n)
    if dtype.is_complex:
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        a = (torch.randn(*shape, dtype=real_dtype) + 1j * torch.randn(*shape, dtype=real_dtype)).to(
            dtype
        )
    else:
        a = torch.randn(*shape, dtype=dtype)
    return a * scale


@pytest.mark.parametrize(
    ("dtype", "n", "batch", "scale"),
    [
        (torch.float64, 8, 0, 1.0),
        (torch.complex128, 8, 0, 1.0),
        (torch.complex64, 8, 0, 1.0),
        (torch.complex64, 64, 5, 1.0),
        # Large operator norm: exercises the scaling-and-squaring loop, not just the base case.
        (torch.complex64, 20, 0, 5.0),
    ],
)
def test_pair_matrix_exp_matches_torch_matrix_exp(
    dtype: torch.dtype, n: int, batch: int, scale: float
) -> None:
    a_ref = _random_operator(dtype, n, batch, seed=0, scale=scale).requires_grad_(True)
    a_new = a_ref.detach().clone().requires_grad_(True)

    y_ref = torch.matrix_exp(a_ref)
    y_new = pair_matrix_exp(a_new)
    assert torch.equal(y_ref.detach(), y_new.detach())  # same kernel call, bit-identical forward

    torch.manual_seed(1)
    if dtype.is_complex:
        g = torch.randn_like(y_ref.real).to(dtype) + 1j * torch.randn_like(y_ref.real).to(dtype)
    else:
        g = torch.randn_like(y_ref)
    y_ref.backward(g)  # type: ignore[no-untyped-call]
    y_new.backward(g)  # type: ignore[no-untyped-call]

    assert a_ref.grad is not None
    assert a_new.grad is not None
    grad_scale = a_ref.grad.abs().max().item()
    rel_diff = (a_ref.grad - a_new.grad).abs().max().item() / max(grad_scale, 1e-300)
    tol = 1e-4 if dtype in (torch.float32, torch.complex64) else 1e-9
    assert rel_diff < tol
