"""Bloch-wave propagators: integrate a :class:`BlochSystem` to the exit wavefunction.

Two first-class methods, selected by a ``Method`` *value* (strategy-as-value, not a stateful class):

- ``matrix_exp`` -- the refine default. ``psi(t) = matrix_exp(A * i pi t / k_n) @ psi0``; a single
  dense matrix exponential with stable autograd.
- ``bloch_eigen`` -- eval-only. Diagonalise ``A`` once, then every thickness is a cheap phase
  multiply -- fast for many thicknesses, but ``eigh``'s backward is ill-conditioned near degenerate
  eigenvalues (which symmetric crystals routinely produce), so it is not the refine default.

Both are first-class and swappable off the *same* ``BlochSystem`` (no geometry/energy/hkl needed --
the system is the closed problem). They differ in *what* they return -- symmetrised vs physical
amplitudes -- coinciding only at ``Mii == 1``; that distinction is a feature to experiment with, not
a bug. Ports the no-absorption path of ``diffBloch_private``
``dynamical.py::calculate_dynamical_scattering_batched`` (here single-system).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from diffBloch.core.dynamical.assembly import BlochSystem, _fill_diagonal

type Method = Literal["matrix_exp", "bloch_eigen"]
# The solve's numeric format, orthogonal to Method (the algorithm). "fp64" = float64 + complex128
# (the exact, reproducible field); "fp32" = float32 + complex64. It is deliberately a *coarse*
# knob -- the coupled orientation fit's O(N^3) eigensolve scales with the beam count (~cell volume),
# so on a large cell "fp32" ~halves that dominant cost, trading a basin-sensitive search
# (non-determinism across platforms) for speed. The terminal Plan->Result (run_inference scoring,
# refine) stays "fp64": the fit is re-scored there, so the pinned result stays exact even when the
# search was coarse (design/decisions/combinators-and-recipe-identity.md: precision==checkpoint
# boundary). The selecting param stays named `precision` (the role); the type is FloatFormat so the
# type and its members compose (FloatFormat/fp32), avoiding the redundant Precision.FP32.
type FloatFormat = Literal["fp32", "fp64"]
type Thicknesses = float | Sequence[float] | Tensor


def precision_dtypes(fmt: FloatFormat) -> tuple[torch.dtype, torch.dtype]:
    """The ``(real, complex)`` torch dtypes for a float format -- the single source of the mapping.

    ``"fp32" -> (float32, complex64)``; ``"fp64" -> (float64, complex128)``. The three places that
    need it -- ``propagate`` (thickness / real dtype), ``_at_precision`` (operator complex dtype),
    and the engine's segmented solve (its curve-buffer dtype) -- all read it here, so they cannot
    drift (the drift that let the fp32 coupled path scatter float32 into a float64 buffer).
    """
    return (torch.float32, torch.complex64) if fmt == "fp32" else (torch.float64, torch.complex128)


def propagate(
    system: BlochSystem,
    thicknesses: Thicknesses,
    *,
    method: Method = "matrix_exp",
    precision: FloatFormat = "fp64",
) -> Tensor:
    """Propagate ``system.psi0`` to each thickness, returning the exit wavefunction.

    ``thicknesses`` is a scalar or 1-D sequence/tensor (Å). Rank-polymorphic in the operator: a
    single system (``a`` ``(N, N)``) returns ``(T, N)``; a batched system (``a`` ``(B, N, N)``, e.g.
    an orientation's rocking-curve tilts stacked by :func:`core.dynamical.build_bloch_systems`)
    returns ``(B, T, N)`` -- one batched ``eigh`` / ``matrix_exp`` over all tilts. The single-system
    path is byte-identical to the un-batched computation (the batch axis is simply absent).
    Differentiable in ``A`` (hence in ``Fgb``). ``method`` picks the propagator: ``matrix_exp``
    (refine default, stable autograd) or ``bloch_eigen`` (eval-only). The no-absorption path
    assumes ``A`` is Hermitian.

    ``precision`` selects the numeric field of the eigensolve/matrix-exponential -- orthogonal to
    ``method``. ``"fp64"`` (the default) runs the whole propagation in complex128/float64 and is a
    pure identity on today's path (byte-identical). ``"fp32"`` downcasts the operator to complex64
    and builds ``thicknesses`` at float32, roughly halving the O(N^3) solve's time and memory -- a
    *search-time* knob for the preprocess fits (the O(N^3) cost scales with the beam count, i.e. the
    cell volume). The terminal estimators (``run_inference`` scoring, ``refine``) never enable it:
    fp32 perturbs the fit's basin, so the reproducible pinned result stays fp64. Ports
    ``diffBloch_private`` ``dynamical.py``'s complex64 eigensolve (#127/#133).
    """
    real_dtype, _ = precision_dtypes(precision)
    t = torch.as_tensor(thicknesses, dtype=real_dtype, device=system.a.device)
    if t.ndim == 0:
        t = t.reshape(1)
    if t.ndim != 1:
        raise ValueError("thicknesses must be a scalar or 1-D sequence")

    if method == "matrix_exp":
        return _propagate_matrix_exp(system, t, precision)
    if method == "bloch_eigen":
        return _propagate_bloch_eigen(system, t, precision)
    raise ValueError(f"method must be 'matrix_exp' or 'bloch_eigen', got {method!r}")


def _at_precision(operator: Tensor, precision: FloatFormat) -> Tensor:
    """Downcast a complex operator to complex64 for ``"fp32"``; identity for ``"fp64"``.

    The one place precision enters the solve. Differentiable (``.to`` casts the incoming gradient
    back on the backward pass), no ``.detach``/in-place -- so ``"fp64"`` is a pure identity and
    ``"fp32"`` preserves the gradient to ``Fgb``.
    """
    return operator.to(precision_dtypes(precision)[1])


def _propagate_matrix_exp(
    system: BlochSystem, thicknesses: Tensor, precision: FloatFormat
) -> Tensor:
    a = _at_precision(_complex_operator(system.a), precision)  # (..., N, N)
    # Co-locate the geometry-plan tensors onto the operator device (they may be built CPU-side while
    # A is parameter-derived on an accelerator); thicknesses is already on a.device, k_n is a float.
    psi0 = system.psi0.to(dtype=a.dtype, device=a.device)
    # i pi t / k_n: the dynamical-diffraction propagation scaling. For a Hermitian A this scalar is
    # pure-imaginary, so matrix_exp(A * scalar) is unitary (flux-conserving).
    scalars = (1j * torch.pi * thicknesses / system.k_n).to(a.dtype)
    # a.unsqueeze(-3) inserts the thickness axis before (N, N): a single (N, N) becomes (1, N, N)
    # broadcasting to (T, N, N); a batched (B, N, N) becomes (B, 1, N, N) broadcasting to
    # (B, T, N, N) -- one matrix_exp over the whole tilt/thickness grid.
    transfer = torch.matrix_exp(a.unsqueeze(-3) * scalars[:, None, None])  # (..., T, N, N)
    return (transfer @ psi0.unsqueeze(-1)).squeeze(-1)  # (..., T, N)


def _propagate_bloch_eigen(
    system: BlochSystem, thicknesses: Tensor, precision: FloatFormat
) -> Tensor:
    a = _at_precision(_complex_operator(system.a), precision)  # (..., N, N)
    # Co-locate the geometry-plan tensors onto the operator device (see _propagate_matrix_exp).
    mii = system.mii.to(device=a.device)  # (..., N)
    psi0 = system.psi0.to(device=a.device)  # (N,), shared across the batch
    # Hermitian eigendecomposition (no-absorption path); v are the Bloch-wave excitations. Batched
    # over any leading dims: one eigh solves every tilt at once.
    v, eigvecs = torch.linalg.eigh(a)  # v (..., N), eigvecs (..., N, N)
    gamma = v / (2.0 * system.k_n)
    # Un-symmetrise: A was Mii-symmetrised to be Hermitian, so divide the eigenvectors' diagonal
    # back to recover the physical Bloch coefficients (private dynamical.py:877).
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
