"""Electron structure factors (Lobato parametrization), vectorised.

The elastic structure-factor path -- form factors, Debye-Waller factors, and the phase sum --
vectorised over unique-Z groups into a single batched phase sum.

Form factors use one of two vendored parametrizations, selected per call by ``scattering_factors``
(:type:`ScatteringFactorModel`): the 2026 element-adaptive Dirac–Pade basis (Lobato, Zhang, Van
Aert & Kirkland, 2026; ``core/data/lobato2026.json``, the default) or the original Lobato–Van Dyck
(2014) fixed basis (``core/data/lobato.json``). Either way no external scattering library is needed
at runtime. The form factor is a setup constant — only ``positions``, ``uij_star`` and
``occupancies`` carry gradients.

Absorption (the imaginary ``U0'`` path) is intentionally deferred (runs set ``absorption: false``).
``structure_factors`` consumes ADPs already in the U* (reciprocal) frame; the Cartesian→U*
conversion belongs to the ADP/spec layer that wires this into the engine.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Literal

import torch
from torch import Tensor

from diffBloch.core.absorption import absorptive_form_factors, equivalent_isotropic_b
from diffBloch.specs import NO_ABSORPTION, Absorption

type StructureFactorCutoff = Literal["hard", "taper"]
# "lobato2014": Lobato & Van Dyck (2014), fixed 5-term-or-fewer basis, all terms the same
# functional form (vendored in core/data/lobato.json). "lobato2026": Lobato, Zhang, Van Aert &
# Kirkland (2026), element-adaptive basis (2-15 terms) fitted against updated relativistic
# Dirac-Fock reference densities, extended to 36 A^-1 and ending in one charge-carrying
# Dirac-Pade term with its own functional form (vendored in core/data/lobato2026.json).
type ScatteringFactorModel = Literal["lobato2014", "lobato2026"]


def _g_vector_lengths(hkl: Tensor, reciprocal_basis: Tensor) -> Tensor:
    """``|g|`` per reflection from Miller indices and a reciprocal basis (rows = a*, b*, c*).

    The torch counterpart of the NumPy ``core.reciprocal.g_vector_lengths`` (which serves the
    planning path); kept here so the differentiable structure-factor path stays in torch.
    """
    if hkl.ndim != 2 or hkl.shape[1] != 3:
        raise ValueError("hkl must have shape (M, 3)")
    if reciprocal_basis.shape != (3, 3):
        raise ValueError("reciprocal_basis must have shape (3, 3)")
    g_vectors = hkl.to(reciprocal_basis.dtype) @ reciprocal_basis
    lengths: Tensor = torch.linalg.vector_norm(g_vectors, dim=1)
    return lengths


@lru_cache(maxsize=2)
def _lobato_table(
    model: ScatteringFactorModel,
) -> dict[int, tuple[tuple[float, ...], tuple[float, ...], tuple[bool, ...]]]:
    """Load one vendored Lobato table: ``{Z: ((a_i), (b_i), (is_dirac_pade_i))}``.

    The table is keyed by Z, not element symbol, so the core needs no symbol authority — element
    identity comes from the parsed atomic ``numbers`` plus the vendored data alone, keeping
    ``core/`` free of any parser/periodic-table dependency. ``lobato2014`` has no Dirac-Pade term
    (every entry in ``is_dirac_pade`` is ``False``); ``lobato2026`` marks its final per-element term
    ``True``.
    """
    filename = "lobato.json" if model == "lobato2014" else "lobato2026.json"
    text = (resources.files("diffBloch.core") / "data" / filename).read_text()
    raw = json.loads(text)
    if model == "lobato2014":
        return {
            int(z): (tuple(ab[0]), tuple(ab[1]), tuple(False for _ in ab[0]))
            for z, ab in raw.items()
        }
    return {
        int(z): (
            tuple(entry["a"]),
            tuple(entry["b"]),
            tuple(t == "DP" for t in entry["type"]),
        )
        for z, entry in raw.items()
    }


def lobato_form_factors(
    numbers: Tensor, g: Tensor, *, model: ScatteringFactorModel = "lobato2026"
) -> Tensor:
    """Electron scattering factor ``f_e(Z, |g|)`` for each atom, vectorised over unique Z.

    Every term is ``f_e(s) = a_i (2 + b_i s^2) / (1 + b_i s^2)^2`` with ``s^2 = |g|^2``, except
    ``lobato2026``'s single trailing Dirac-Pade term per element, which is instead
    ``f_e(s) = a_DP (3 + 3 u + u^2) / (1 + u)^3`` with ``u = b_DP s^2`` (Lobato, Zhang, Van Aert &
    Kirkland, 2026, eq. 37). Returns a real ``(N_atoms, N_g)`` tensor (in ``g``'s dtype); a constant
    with respect to the refinement (depends only on Z, ``model``, and the fixed geometry).
    """
    if numbers.ndim != 1:
        raise ValueError("numbers must have shape (N,)")
    if g.ndim != 1:
        raise ValueError("g must have shape (M,)")
    table = _lobato_table(model)
    g2 = g**2
    factors = torch.zeros((numbers.shape[0], g.shape[0]), dtype=g.dtype, device=g.device)
    for z in torch.unique(numbers).tolist():
        a_coeffs, b_coeffs, is_dp = table[int(z)]
        a = torch.tensor(a_coeffs, dtype=g.dtype, device=g.device)[:, None]
        b = torch.tensor(b_coeffs, dtype=g.dtype, device=g.device)[:, None]
        if any(is_dp):
            u = b * g2[None, :]
            denom = 1.0 + u
            nr_term = a * (2.0 + u) / denom**2
            dp_term = a * (3.0 + 3.0 * u + u * u) / denom**3
            dp = torch.tensor(is_dp, dtype=torch.bool, device=g.device)[:, None]
            f = torch.where(dp, dp_term, nr_term).sum(dim=0)
        else:
            # No Dirac-Pade term (lobato2014, or any all-NR table): the plain sum, bit-identical
            # to the pre-lobato2026 implementation -- skips the unused dp_term computation and the
            # torch.where selection, neither of which a pure-NR table needs.
            f = (a * (2.0 + b * g2[None, :]) / (1.0 + b * g2[None, :]) ** 2).sum(dim=0)
        factors[numbers == z] = f
    return factors


def debye_waller_factor(hkl: Tensor, uij_star: Tensor) -> Tensor:
    """Anisotropic Debye–Waller factor ``exp(-2 pi^2 h^T U* h)`` per (atom, reflection).

    ``hkl`` is ``(M, 3)``; ``uij_star`` is ``(N, 3, 3)`` in the U* (reciprocal) frame. Returns
    ``(N, M)``, differentiable in ``uij_star``.
    """
    if hkl.ndim != 2 or hkl.shape[1] != 3:
        raise ValueError("hkl must have shape (M, 3)")
    if uij_star.ndim != 3 or tuple(uij_star.shape[1:]) != (3, 3):
        raise ValueError("uij_star must have shape (N, 3, 3)")
    h = hkl.to(uij_star.dtype)
    quadratic = torch.einsum("rx,axy,ry->ar", h, uij_star, h)
    return torch.exp(-2.0 * torch.pi**2 * quadratic)


def structure_factor_cutoff(
    g: Tensor, g_max: float, *, mode: StructureFactorCutoff = "hard"
) -> Tensor:
    """Reflection resolution cutoff: a hard ``|g| <= g_max`` mask or a logistic taper window."""
    if g_max <= 0.0:
        raise ValueError("g_max must be positive")
    if mode == "hard":
        return (g <= g_max).to(g.dtype)
    if mode == "taper":
        # Logistic taper window: roll-off centred at TAPER_ALPHA * g_max with
        # logistic width TAPER_WIDTH. TAPER_ALPHA = 1 - 0.05 starts the roll-off ~5% below g_max.
        taper_width, taper_alpha = 0.005, 1.0 - 0.05
        return 1.0 / (1.0 + torch.exp((g / g_max - taper_alpha) / taper_width))
    raise ValueError(f"unsupported cutoff mode: {mode!r}")


def structure_factors(
    positions: Tensor,
    numbers: Tensor,
    occupancies: Tensor,
    uij_star: Tensor,
    hkl: Tensor,
    reciprocal_basis: Tensor,
    cell_volume: float,
    *,
    g_max: float,
    cutoff: StructureFactorCutoff = "hard",
    zero_threshold: float = 1e-12,
    absorption: Absorption = NO_ABSORPTION,
    energy: float | None = None,
    scattering_factors: ScatteringFactorModel = "lobato2026",
) -> Tensor:
    """Vectorised electron structure factors ``Fgb``, optionally including absorption.

    ``Fgb(h) = (1/V) sum_atoms f_e * DWF * occ * cutoff * exp(2 pi i r . h)``. ``|g|`` is derived
    internally from ``hkl`` and ``reciprocal_basis`` (no separate ``g`` argument to keep in sync).
    Differentiable in ``positions``, ``uij_star``, ``occupancies``; ``f_e`` is a constant form
    factor. Symmetry-related atoms can sum a component to exactly zero mathematically (systematic
    absences); floating-point roundoff lands near but not at zero, so components below
    ``zero_threshold`` are snapped to a clean ``0.0``. Returns a complex ``(M,)`` tensor.
    """
    n_atoms = positions.shape[0]
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if numbers.shape != (n_atoms,) or occupancies.shape != (n_atoms,):
        raise ValueError("numbers and occupancies must have shape (N,) matching positions")
    if uij_star.shape != (n_atoms, 3, 3):
        raise ValueError("uij_star must have shape (N, 3, 3) matching positions")
    if cell_volume <= 0.0:
        raise ValueError("cell_volume must be positive")

    g = _g_vector_lengths(hkl, reciprocal_basis)
    form_factors = lobato_form_factors(numbers, g, model=scattering_factors)
    dwf = debye_waller_factor(hkl, uij_star)
    cutoff_window = structure_factor_cutoff(g, g_max, mode=cutoff)
    atomic_factors = form_factors
    if absorption.enabled:
        if energy is None:
            raise ValueError("energy is required for parameterized absorption")
        b_iso = equivalent_isotropic_b(uij_star, reciprocal_basis)
        atomic_factors = torch.complex(
            form_factors,
            absorptive_form_factors(numbers, g / 2.0, b_iso, energy=energy),
        )
    per_atom = (atomic_factors * dwf * occupancies[:, None]) * cutoff_window[None, :]

    phase = torch.exp(2.0j * torch.pi * (positions @ hkl.to(positions.dtype).transpose(0, 1)))
    unmasked = (per_atom.to(phase.dtype) * phase).sum(dim=0)
    real = torch.where(
        unmasked.real.abs() >= zero_threshold, unmasked.real, torch.zeros_like(unmasked.real)
    )
    imag = torch.where(
        unmasked.imag.abs() >= zero_threshold, unmasked.imag, torch.zeros_like(unmasked.imag)
    )
    return torch.complex(real, imag) / cell_volume
