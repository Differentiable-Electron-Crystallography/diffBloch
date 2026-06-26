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
a bug. Rationale, evidence, and deferred work: ``design/decisions/stage8-bloch-propagators.md`` and
``scripts/stage8_propagator_experiment.py``. Ports the no-absorption path of ``diffBloch_private``
``dynamical.py::calculate_dynamical_scattering_batched`` (here single-system).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from diffBloch.core.dynamical.assembly import BlochSystem, _fill_diagonal

type Method = Literal["matrix_exp", "bloch_eigen"]
type Thicknesses = float | Sequence[float] | Tensor


def propagate(
    system: BlochSystem,
    thicknesses: Thicknesses,
    *,
    method: Method = "matrix_exp",
) -> Tensor:
    """Propagate ``system.psi0`` to each thickness, returning the exit wavefunction.

    ``thicknesses`` is a scalar or 1-D sequence/tensor (Å); returns a complex ``(T, N)`` tensor of
    exit-wave amplitudes (intensities ``|psi|^2`` are a downstream concern). Differentiable in ``A``
    (hence in ``Fgb``). ``method`` picks the propagator: ``matrix_exp`` (refine default, stable
    autograd) or ``bloch_eigen`` (eval-only). The no-absorption path assumes ``A`` is Hermitian.
    """
    t = torch.as_tensor(thicknesses, dtype=torch.float64, device=system.a.device)
    if t.ndim == 0:
        t = t.reshape(1)
    if t.ndim != 1:
        raise ValueError("thicknesses must be a scalar or 1-D sequence")

    if method == "matrix_exp":
        return _propagate_matrix_exp(system, t)
    if method == "bloch_eigen":
        return _propagate_bloch_eigen(system, t)
    raise ValueError(f"method must be 'matrix_exp' or 'bloch_eigen', got {method!r}")


def _propagate_matrix_exp(system: BlochSystem, thicknesses: Tensor) -> Tensor:
    a = _complex_operator(system.a)
    # Co-locate the geometry-plan tensors onto the operator device (they may be built CPU-side while
    # A is parameter-derived on an accelerator); thicknesses is already on a.device, k_n is a float.
    psi0 = system.psi0.to(dtype=a.dtype, device=a.device)
    # i pi t / k_n: the dynamical-diffraction propagation scaling. For a Hermitian A this scalar is
    # pure-imaginary, so matrix_exp(A * scalar) is unitary (flux-conserving).
    scalars = (1j * torch.pi * thicknesses / system.k_n).to(a.dtype)
    transfer = torch.matrix_exp(a[None] * scalars[:, None, None])  # (T, N, N)
    return (transfer @ psi0.unsqueeze(-1)).squeeze(-1)  # (T, N)


def _propagate_bloch_eigen(system: BlochSystem, thicknesses: Tensor) -> Tensor:
    a = _complex_operator(system.a)
    # Co-locate the geometry-plan tensors onto the operator device (see _propagate_matrix_exp).
    mii = system.mii.to(device=a.device)
    psi0 = system.psi0.to(device=a.device)
    # Hermitian eigendecomposition (no-absorption path); v are the Bloch-wave excitations.
    v, eigvecs = torch.linalg.eigh(a)
    gamma = v / (2.0 * system.k_n)
    # Un-symmetrise: A was Mii-symmetrised to be Hermitian, so divide the eigenvectors' diagonal
    # back to recover the physical Bloch coefficients (private dynamical.py:877).
    physical_diag = torch.diagonal(eigvecs) / mii.to(eigvecs.dtype)
    c = _fill_diagonal(eigvecs, physical_diag)
    alpha = torch.conj(c.mT) @ psi0.to(c.dtype)  # decompose psi0 onto the Bloch waves
    phase = torch.exp(2.0j * torch.pi * thicknesses[:, None] * gamma[None, :])  # (T, N)
    return (phase * alpha[None, :]) @ c.mT  # recombine: psi(t) = C @ (phase ⊙ alpha), (T, N)


def _complex_operator(operator: Tensor) -> Tensor:
    """Return ``operator`` in a complex dtype, preserving complex inputs exactly."""
    if operator.is_complex():
        return operator
    dtype = torch.complex64 if operator.dtype == torch.float32 else torch.complex128
    return operator.to(dtype)
