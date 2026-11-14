"""Bloch-wave propagators: integrate a :class:`BlochSystem` to the exit wavefunction.

Two first-class methods, selected by a ``SolverMethod`` *value* (strategy-as-value, not a stateful class):

- ``matrix_exp`` -- the refine default. ``psi(t) = matrix_exp(A * i pi t / k_n) @ psi0``; a single
  dense matrix exponential with stable autograd.
- ``bloch_eigen`` -- eval-only. Diagonalise ``A`` once, then every thickness is a cheap phase
  multiply -- fast for many thicknesses, but ``eigh``'s backward is ill-conditioned near degenerate
  eigenvalues (which symmetric crystals routinely produce), so it is not the refine default.

Both are first-class and swappable off the *same* ``BlochSystem`` (no geometry/energy/hkl needed --
the system is the closed problem). They differ in *what* they return -- symmetrised vs physical
amplitudes -- coinciding only at ``Mii == 1``; that distinction is a feature to experiment with, not
a bug. The no-absorption path assumes ``A`` is Hermitian.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from diffBloch.core.dynamical.assembly import BlochSystem, _fill_diagonal

type SolverMethod = Literal["matrix_exp", "bloch_eigen"]
type Thicknesses = float | Sequence[float] | Tensor
# What is known about A ahead of the solve: "real" (centrosymmetric about the origin, no absorption
# -- real symmetric), "hermitian" (no absorption), "general" (absorptive). Decided once per engine
# (see RefinementEngine.operator_kind); None makes each solve inspect A itself, which syncs the host.
type OperatorKind = Literal["real", "hermitian", "general"]

# `matrix_exp` scaling-and-squaring holds several N*N temporaries per matrix (Pade numerator/
# denominator, the linear solve). This conservative multiplier estimates the live footprint of one
# (blk, N, N) block so `memory_safe_max_batch` can keep the peak well under accelerator memory.
_MATRIX_EXP_LIVE_COPIES = 8
# The default per-block budget the engine bounds a `matrix_exp` call to when no explicit `max_batch`
# is given. A guardrail, not a tuned value: ~2 GiB is a no-op for ordinary solves (their whole
# operator stack is far smaller) yet caps the wide-segment / many-thickness stacks that would
# otherwise materialize tens of GiB at once (the adaptive-union optimize_thickness OOM). Override
# `max_batch` explicitly for a specific device budget.
DEFAULT_MATRIX_EXP_BUDGET_BYTES = 2 * 1024**3


def propagate(
    system: BlochSystem,
    thicknesses: Thicknesses,
    *,
    method: SolverMethod = "matrix_exp",
    max_batch: int | None = None,
    operator_kind: OperatorKind | None = None,
) -> Tensor:
    """Propagate ``system.psi0`` to each thickness, returning the exit wavefunction.

    ``thicknesses`` is a scalar or 1-D sequence/tensor (Å). Rank-polymorphic in the operator: a
    single system (``a`` ``(N, N)``) returns ``(T, N)``; a batched system (``a`` ``(B, N, N)``, e.g.
    an orientation's rocking-curve tilts stacked by :func:`core.dynamical.build_bloch_systems`)
    returns ``(B, T, N)`` -- one batched ``eigh`` / ``matrix_exp`` over all tilts. The single-system
    path is exactly the un-batched computation (the batch axis is simply absent).
    Differentiable in ``A`` (hence in ``Fgb``). ``method`` picks the propagator: ``matrix_exp``
    (refine default, stable autograd) or ``bloch_eigen`` (eval-only). The no-absorption path
    assumes ``A`` is Hermitian. The whole propagation runs at float32/complex64.

    ``max_batch`` (``matrix_exp`` only) caps how many ``(N, N)`` operators are exponentiated in one
    ``torch.matrix_exp`` call. ``None`` (the default) builds the whole ``(..., T, N, N)`` transfer
    at once. A positive integer instead streams the flattened
    ``(B*T, N, N)`` operator stack in row-blocks, so that transfer is never materialized -- the peak
    drops from ``~K*B*T*N**2`` to ``~K*max_batch*N**2``. ``torch.matrix_exp`` shares one
    scaling-and-squaring count across a batch (from its max norm), so regrouping shifts rounding by
    ~1 ulp: the result matches the unbounded solve *to machine precision*, never in accuracy. A
    memory knob, not a result knob. ``bloch_eigen`` ignores it (it diagonalises once, no
    ``(B, T, N, N)`` intermediate).
    """
    if max_batch is not None and max_batch < 1:
        raise ValueError(f"max_batch must be a positive integer or None, got {max_batch}")
    t = torch.as_tensor(thicknesses, dtype=torch.float32, device=system.a.device)
    if t.ndim == 0:
        t = t.reshape(1)
    if t.ndim != 1:
        raise ValueError("thicknesses must be a scalar or 1-D sequence")

    if method == "matrix_exp":
        # Real A (centrosymmetric, no absorption) takes the real-arithmetic path; anything else
        # falls back to the complex torch.matrix_exp inside it.
        return _propagate_matrix_exp_real(system, t, max_batch=max_batch, kind=operator_kind)
    if method == "bloch_eigen":
        return _propagate_bloch_eigen(system, t)
    raise ValueError(f"method must be 'matrix_exp' or 'bloch_eigen', got {method!r}")


def memory_safe_max_batch(
    n_beams: int,
    *,
    budget_bytes: int = DEFAULT_MATRIX_EXP_BUDGET_BYTES,
) -> int:
    """The largest ``max_batch`` keeping one ``(max_batch, N, N)`` matrix_exp block under budget.

    ``N`` (``n_beams``) is the solve's beam count, so the bound *adapts to cell size* -- a
    large-cell compound (bigger ``N``, cubically heavier propagator) gets a proportionally smaller
    block, where a fixed block *count* would still blow up. Returns at least 1 (a single matrix is
    always attempted, even if it alone exceeds the budget). This is the safe default the engine
    applies when a caller does not pin ``max_batch`` itself; the result matches an unbounded solve
    to machine precision (a rounding-level ~1 ulp shift, see :func:`propagate`), never in accuracy.
    """
    per_matrix = _MATRIX_EXP_LIVE_COPIES * n_beams * n_beams * 8
    return max(1, budget_bytes // per_matrix)


def _propagate_matrix_exp(
    system: BlochSystem,
    thicknesses: Tensor,
    *,
    max_batch: int | None = None,
) -> Tensor:
    a = _complex_operator(system.a).to(torch.complex64)  # (..., N, N)
    # Co-locate the geometry-plan tensors onto the operator device (they may be built CPU-side while
    # A is parameter-derived on an accelerator); thicknesses is already on a.device, k_n is a float.
    psi0 = system.psi0.to(dtype=a.dtype, device=a.device)
    # i pi t / k_n: the dynamical-diffraction propagation scaling. For a Hermitian A this scalar is
    # pure-imaginary, so matrix_exp(A * scalar) is unitary (flux-conserving).
    scalars = (1j * torch.pi * thicknesses / system.k_n).to(a.dtype)  # (T,)
    if max_batch is None:
        # a.unsqueeze(-3) inserts the thickness axis before (N, N): a single (N, N) becomes
        # (1, N, N) broadcasting to (T, N, N); a batched (B, N, N) becomes (B, 1, N, N) broadcasting
        # to (B, T, N, N) -- one matrix_exp over the whole tilt/thickness grid.
        transfer = torch.matrix_exp(a.unsqueeze(-3) * scalars[:, None, None])  # (..., T, N, N)
        return (transfer @ psi0.unsqueeze(-1)).squeeze(-1)  # (..., T, N)
    # Bounded-memory path: exponentiate the flattened (B*T, N, N) operator stack in row-blocks of
    # `max_batch`, applying psi0 per block, so the full (..., T, N, N) transfer is never
    # materialized. Matches the unbounded solve to machine precision (torch.matrix_exp shares one
    # squaring count per batch, so regrouping shifts rounding by ~1 ulp, never accuracy); this
    # generalizes a single-thickness cover-batch chunk to the multi-thickness grid.
    n = a.shape[-1]
    a_flat = a.reshape(-1, n, n)  # (B, N, N); B == 1 for a single (N, N) system
    n_batch, n_thick = a_flat.shape[0], scalars.shape[0]
    total = n_batch * n_thick  # flat index f = b * T + t, so a contiguous block cats back in order
    # Not itself checkpointed: every caller (RefinementEngine._solve / _solve_segmented) already
    # wraps its whole per-orientation/per-segment solve (this loop included) in one outer
    # checkpoint. Nesting a second checkpoint per block here would make the outer boundary's
    # backward recompute *this* loop, which would then re-defer (and re-recompute) each block a
    # second time -- ~3x forward-equivalent matrix_exp cost instead of 2x, for no extra memory
    # saved (the outer boundary already discards everything down to this call).
    amplitudes = []
    for start in range(0, total, max_batch):
        flat = torch.arange(start, min(total, start + max_batch), device=a.device)
        block = a_flat[flat // n_thick] * scalars[flat % n_thick][:, None, None]  # (blk, N, N)
        amplitudes.append((torch.matrix_exp(block) @ psi0.unsqueeze(-1)).squeeze(-1))
    return torch.cat(amplitudes, dim=0).reshape(*a.shape[:-2], n_thick, n)  # (..., T, N)


# A complex operator counts as real when its imaginary part is below this fraction of its real
# part: float32 structure factors of a centrosymmetric crystal carry ~1e-7 relative imaginary noise.
_REAL_OPERATOR_RTOL = 1e-6
# Taylor coefficients of cos / sin in Z = Y^2 (|Y| <= 1 after scaling): cos(Y) = sum c_k Z^k,
# sin(Y) = Y sum s_k Z^k, k = 0..6 -- truncation below 1/14! ~ 1e-11, far under float32 rounding.
_COS_TAYLOR = tuple((-1) ** k / math.factorial(2 * k) for k in range(7))
_SIN_TAYLOR = tuple((-1) ** k / math.factorial(2 * k + 1) for k in range(7))


def _operator_kind(a: Tensor) -> OperatorKind:
    """Inspect ``A`` (host-syncing): the fallback when the caller has not decided the kind."""
    if not a.is_complex():
        return "real"
    a = a.detach()
    if float(a.imag.abs().max()) <= _REAL_OPERATOR_RTOL * max(float(a.real.abs().max()), 1e-30):
        return "real"
    return "hermitian" if torch.allclose(a, a.mH) else "general"


def _propagate_matrix_exp_real(
    system: BlochSystem,
    thicknesses: Tensor,
    *,
    max_batch: int | None = None,
    kind: OperatorKind | None = None,
) -> Tensor:
    """``matrix_exp`` for a real symmetric ``A`` in real arithmetic: ``exp(i theta A) = cos(theta A)
    + i sin(theta A)``.

    A centrosymmetric crystal (origin on the inversion centre) without absorption has real
    structure factors, so ``A`` is real symmetric. ``cos`` / ``sin`` are then evaluated with real
    matmuls only -- a scaled degree-6 Taylor series in ``Y^2`` (Paterson-Stockmeyer, 6 matmuls),
    then ``s`` double-angle steps ``S <- 2 S C``, ``C <- C^2 - S^2`` (3 matmuls each) -- where a
    complex matmul costs 4 real ones. Differentiable by autograd through those real ops. Falls back
    to :func:`_propagate_matrix_exp` when ``A`` is not real (non-centrosymmetric or absorptive).
    """
    a = system.a
    if (kind or _operator_kind(a)) != "real":
        return _propagate_matrix_exp(system, thicknesses, max_batch=max_batch)
    a = (a.real if a.is_complex() else a).to(torch.float32)
    n = a.shape[-1]
    psi0 = system.psi0
    psi0 = (psi0.real if psi0.is_complex() else psi0).to(device=a.device, dtype=torch.float32)
    theta = (torch.pi * thicknesses / system.k_n).to(torch.float32)  # (T,)
    a_flat = a.reshape(-1, n, n)
    n_batch, n_thick = a_flat.shape[0], theta.shape[0]
    total = n_batch * n_thick
    block = total if max_batch is None else max_batch
    real_parts, imag_parts = [], []
    for start in range(0, total, block):
        flat = torch.arange(start, min(total, start + block), device=a.device)
        x = a_flat[flat // n_thick] * theta[flat % n_thick][:, None, None]  # (blk, N, N)
        cos_x, sin_x = _cos_sin(x)
        column = psi0.unsqueeze(-1)
        real_parts.append((cos_x @ column).squeeze(-1))
        imag_parts.append((sin_x @ column).squeeze(-1))
    amplitudes = torch.complex(torch.cat(real_parts), torch.cat(imag_parts))
    return amplitudes.reshape(*a.shape[:-2], n_thick, n)  # (..., T, N)


def _cos_sin(x: Tensor) -> tuple[Tensor, Tensor]:
    """``(cos(X), sin(X))`` of a real ``(B, N, N)`` stack by scaling and double-angle squaring."""
    with torch.no_grad():
        norm = float(x.abs().sum(dim=-2).amax()) if x.numel() else 0.0  # max 1-norm
        squarings = max(0, math.ceil(math.log2(norm))) if norm > 1.0 else 0
    y = x / (2.0**squarings)
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device).expand_as(x)
    z = y @ y
    z2 = z @ z
    z3 = z2 @ z

    def series(c: tuple[float, ...]) -> Tensor:
        low = c[0] * eye + c[1] * z + c[2] * z2
        high = c[3] * eye + c[4] * z + c[5] * z2 + c[6] * z3
        return low + z3 @ high

    cos_y = series(_COS_TAYLOR)
    sin_y = y @ series(_SIN_TAYLOR)
    for _ in range(squarings):
        # C^2 - S^2, not 2 C^2 - I: the latter amplifies rounding 4x per step (it loses the small
        # 1 - C part to cancellation), this one 2x -- the same as squaring the complex exponential.
        cos_y, sin_y = cos_y @ cos_y - sin_y @ sin_y, 2.0 * (sin_y @ cos_y)
    return cos_y, sin_y


def _propagate_bloch_eigen(system: BlochSystem, thicknesses: Tensor) -> Tensor:
    a = _complex_operator(system.a).to(torch.complex64)  # (..., N, N)
    if not torch.allclose(a, a.mH):
        raise ValueError(
            "bloch_eigen requires a Hermitian structure matrix; use matrix_exp with absorption"
        )
    # Co-locate the geometry-plan tensors onto the operator device (see _propagate_matrix_exp).
    mii = system.mii.to(device=a.device)  # (..., N)
    psi0 = system.psi0.to(device=a.device)  # (N,), shared across the batch
    # Hermitian eigendecomposition (no-absorption path); v are the Bloch-wave excitations. Batched
    # over any leading dims: one eigh solves every tilt at once.
    v, eigvecs = torch.linalg.eigh(a)  # v (..., N), eigvecs (..., N, N)
    gamma = v / (2.0 * system.k_n)
    # Un-symmetrise: A was Mii-symmetrised to be Hermitian, so divide the eigenvectors' diagonal
    # back to recover the physical Bloch coefficients.
    physical_diag = torch.diagonal(eigvecs, dim1=-2, dim2=-1) / mii.to(eigvecs.dtype)
    c = _fill_diagonal(eigvecs, physical_diag)
    alpha = torch.conj(c.mT) @ psi0.to(c.dtype)  # decompose psi0 onto the Bloch waves, (..., N)
    phase = torch.exp(2.0j * torch.pi * thicknesses[:, None] * gamma.unsqueeze(-2))  # (..., T, N)
    return (phase * alpha[..., None, :]) @ c.mT  # psi(t) = C @ (phase ⊙ alpha), (..., T, N)


def _complex_operator(operator: Tensor) -> Tensor:
    """Return ``operator`` in a complex dtype, preserving complex inputs exactly."""
    if operator.is_complex():
        return operator
    dtype = torch.complex64 if operator.dtype == torch.float32 else torch.complex128
    return operator.to(dtype)
