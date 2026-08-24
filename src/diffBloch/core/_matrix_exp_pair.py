"""``matrix_exp`` with a pair-algebra backward (Al-Mohy & Higham, 2009), not a new solver.

The forward pass is untouched -- ``torch.matrix_exp(A)``, byte-identical to what every caller
already gets -- so nothing here changes an already-settled ``psi(t)``. Only the *backward* is
reimplemented. PyTorch's own ``matrix_exp`` backward computes the Frechet-derivative adjoint by
embedding the operator into a ``(2N, 2N)`` block matrix ``[[A, E], [0, A]]`` and running the same
generic dense scaling-and-squaring algorithm on it, which costs ``(2N)^3 = 8*N^3`` per matrix
product even though three of its four blocks are zero or redundant.

The pair-algebra form instead represents ``(A, E)`` as a matrix dual number ``A + E*eps``
(``eps^2 = 0`` -- exactly the algebraic content of that zero block) and propagates the pair through
the scaling-and-squaring algorithm's own arithmetic:

- pair-multiply ``(X,X').(Y,Y') = (XY, XY' + X'Y)`` costs 3 real matmuls, not one ``(2N)x(2N)``
  matmul (8 matmul-equivalents) -- the 8/3 ~ 2.67x reduction the construction is named for;
- the degree-13 diagonal Pade base case (Higham, 2005) needs 6 pair-multiplies total, independent
  of matrix size, in place of PyTorch's generic dense exponentiation of the doubled matrix;
- the Pade step's linear solve differentiates via the standard matrix-inverse derivative
  (``Y=D^-1`` => ``Y'=-Y D' Y``), reusing one LU factorization for both the value and its pair.

Validated against ``torch.matrix_exp``'s own autograd gradient (real/complex, float32-complex64
through float64-complex128, small and large operator norms, batched) to within each dtype's own
floating-point precision floor, and against the production ``core.solver.propagate`` path on a real
Bloch operator: forward bit-identical, ~1.6x faster, ~56% less peak memory, gradients within 2e-5
relative of PyTorch's own backward (see the PR this landed in for the measurements).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

# Higham (2005), "The Scaling and Squaring Method for the Matrix Exponential Revisited": the
# degree-13 diagonal Pade numerator/denominator coefficients (one shared coefficient set for both,
# since the diagonal Pade approximant of exp satisfies q(A) = p(-A)).
_PADE13_B: tuple[float, ...] = (
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
)
# Largest 1-norm for which degree-13 Pade is accurate to double-precision unit roundoff without
# further scaling (Higham 2005, Table 2.3). Used unconditionally, even at complex64/float32, which
# only needs a looser bound -- the extra squaring this costs beyond what single precision strictly
# needs is cheap insurance against under-scaling, not a correctness gap the way too loose a bound
# would be.
_THETA_13 = 5.371920351148152


def _pair_matmul(x: Tensor, xp: Tensor, y: Tensor, yp: Tensor) -> tuple[Tensor, Tensor]:
    """``(X, X') . (Y, Y')`` under ``X + X'eps``, ``eps^2 = 0`` -> ``(XY, XY' + X'Y)``. 3 matmuls."""
    return x @ y, x @ yp + xp @ y


def _pair_power_chain(
    b: Tensor, bp: Tensor
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """``(B,B')^2, ^4, ^6`` as pairs -- 3 pair-multiplies (9 matmuls), shared by ``p(B)``/``q(B)``."""
    b2, b2p = _pair_matmul(b, bp, b, bp)
    b4, b4p = _pair_matmul(b2, b2p, b2, b2p)
    b6, b6p = _pair_matmul(b2, b2p, b4, b4p)
    return (b2, b2p), (b4, b4p), (b6, b6p)


def _pade13_pair(b: Tensor, bp: Tensor) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """Pair-propagated ``(numerator, denominator)`` of the degree-13 diagonal Pade approximant.

    Returns ``((p, p'), (q, q'))`` with ``exp(B) ~ solve(q, p)`` (the standard Higham/scipy
    ``expm_pade13`` evaluation scheme) and its Frechet-derivative pair alongside it, at a fixed
    cost of 6 pair-multiplies (18 matmuls) -- independent of ``B``'s size or the accuracy target,
    unlike a truncated Taylor series. Every intermediate is dropped (``del``) the moment its last
    use passes, so peak memory is set by the widest *point* in this evaluation, not the sum of
    everything it ever touches.
    """
    n = b.shape[-1]
    eye = torch.eye(n, dtype=b.dtype, device=b.device).expand_as(b)
    c = _PADE13_B
    (b2, b2p), (b4, b4p), (b6, b6p) = _pair_power_chain(b, bp)

    # V = b6.(c12 b6 + c10 b4 + c8 b2) + c6 b6 + c4 b4 + c2 b2 + c0 I
    inner_v = c[12] * b6 + c[10] * b4 + c[8] * b2
    inner_vp = c[12] * b6p + c[10] * b4p + c[8] * b2p
    b6_innerv, b6_innervp = _pair_matmul(b6, b6p, inner_v, inner_vp)
    del inner_v, inner_vp
    v = b6_innerv + c[6] * b6 + c[4] * b4 + c[2] * b2 + c[0] * eye
    vp = b6_innervp + c[6] * b6p + c[4] * b4p + c[2] * b2p
    del b6_innerv, b6_innervp

    # U = B . (b6.(c13 b6 + c11 b4 + c9 b2) + c7 b6 + c5 b4 + c3 b2 + c1 I)
    inner_u = c[13] * b6 + c[11] * b4 + c[9] * b2
    inner_up = c[13] * b6p + c[11] * b4p + c[9] * b2p
    b6_inneru, b6_inneru_p = _pair_matmul(b6, b6p, inner_u, inner_up)
    del inner_u, inner_up
    inner_full = b6_inneru + c[7] * b6 + c[5] * b4 + c[3] * b2 + c[1] * eye
    inner_full_p = b6_inneru_p + c[7] * b6p + c[5] * b4p + c[3] * b2p
    del b6_inneru, b6_inneru_p, b2, b2p, b4, b4p, b6, b6p, eye
    u, up = _pair_matmul(b, bp, inner_full, inner_full_p)
    del inner_full, inner_full_p

    p, pp = u + v, up + vp
    q, qp = v - u, vp - up
    del u, up, v, vp
    return (p, pp), (q, qp)


def frechet_expm(a: Tensor, e: Tensor) -> Tensor:
    """``L(A, E)``: the Frechet derivative of ``matrix_exp`` at ``A`` in direction ``E``.

    Scaling-and-squaring with a degree-13 diagonal Pade base case, propagated through pair
    arithmetic (``A + E*eps``, ``eps^2 = 0``). Batched over leading dims.
    """
    norm = torch.linalg.matrix_norm(a, ord=1)
    s = torch.clamp(torch.ceil(torch.log2(norm.clamp(min=1e-300) / _THETA_13)), min=0).to(
        torch.int64
    )
    s_max = int(s.max().item()) if s.numel() else 0
    s_float = s.to(a.dtype).real if a.is_complex() else s.to(a.dtype)
    scale = (2.0**-s_float).to(a.dtype)
    b = a * scale[..., None, None]
    bp = e * scale[..., None, None]

    (p, pp), (q, qp) = _pade13_pair(b, bp)

    # exp(B) = solve(q, p); differentiate the implicit solve: q X = p => q X' + q' X = p'
    # => X' = solve(q, p' - q' X). Reuses q's LU factorization for both solves.
    lu, pivots = torch.linalg.lu_factor(q)
    x = torch.linalg.lu_solve(lu, pivots, p)
    rhs = pp - qp @ x
    xp = torch.linalg.lu_solve(lu, pivots, rhs)
    del p, q, pp, qp, lu, pivots, rhs

    for i in range(s_max):
        active = (s > i)[..., None, None]
        new_x, new_xp = _pair_matmul(x, xp, x, xp)
        x = torch.where(active, new_x, x)
        xp = torch.where(active, new_xp, xp)
    return cast(Tensor, xp)


class _PairMatrixExp(torch.autograd.Function):
    """``torch.matrix_exp`` with the pair-algebra backward; forward is the unmodified builtin."""

    @staticmethod
    def forward(ctx, a: Tensor) -> Tensor:  # type: ignore[no-untyped-def]
        ctx.save_for_backward(a)
        return torch.matrix_exp(a)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:  # type: ignore[no-untyped-def]
        (a,) = ctx.saved_tensors
        # Adjoint of the Frechet derivative under the Wirtinger convention PyTorch's complex
        # autograd uses: dL/dA = L(A^H, G) (A^T for real A, where .mH and .mT coincide).
        a_adj = a.mH if a.is_complex() else a.mT
        return frechet_expm(a_adj, grad_output)


def matrix_exp(a: Tensor) -> Tensor:
    """Drop-in replacement for ``torch.matrix_exp`` with a cheaper, leaner backward.

    Forward result is byte-identical to ``torch.matrix_exp(a)``. See the module docstring for what
    changes in backward and why.
    """
    return cast(Tensor, _PairMatrixExp.apply(a))  # type: ignore[no-untyped-call]
