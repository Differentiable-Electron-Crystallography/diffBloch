"""Electron structure factors (Lobato parametrization), vectorised.

Pure port of the elastic path in ``diffBloch_private/diffBloch/dynamical.py``:
``StructureFactorNet.calculate_scattering_factors`` / ``calculate_dwf_factor`` / ``forward``. The
private per-atom Python loops are vectorised here over unique-Z groups and a single batched phase
sum.

Form factors use the native Lobato–Van Dyck (2014) parametrization (coefficients vendored in
``core/data/lobato.json``; see ``REFERENCES.md``), so abTEM is not a runtime dependency. The form
factor is a setup constant — only ``positions``, ``uij_star`` and ``occupancies`` carry gradients.

Absorption (the imaginary ``U0'`` path) is intentionally deferred; the quartz anchor runs with
``absorption: false``. ``structure_factors`` consumes ADPs already in the U* (reciprocal) frame; the
Cartesian→U* conversion belongs to the ADP/spec layer that wires this into the engine.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Literal

import torch
from torch import Tensor

type Cutoff = Literal["hard", "taper"]

# Atomic-number -> symbol for indexing the symbol-keyed Lobato table (Z = index + 1).
_ELEMENTS: tuple[str, ...] = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
)


@lru_cache(maxsize=1)
def _lobato_table() -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    """Load the vendored Lobato coefficients ``{symbol: ((a1..a5), (b1..b5))}``."""
    text = (resources.files("diffBloch.core") / "data" / "lobato.json").read_text()
    raw = json.loads(text)
    return {symbol: (tuple(ab[0]), tuple(ab[1])) for symbol, ab in raw.items()}


def lobato_form_factors(numbers: Tensor, g: Tensor) -> Tensor:
    """Electron scattering factor ``f_e(Z, |g|)`` for each atom, vectorised over unique Z.

    ``f_e(s) = sum_i a_i (2 + b_i s^2) / (1 + b_i s^2)^2`` with ``s^2 = |g|^2`` (matching the
    private ``scattering_factor(g**2)`` call). Returns a real ``(N_atoms, N_g)`` tensor; a constant
    with respect to the refinement (depends only on Z and the fixed geometry).
    """
    if numbers.ndim != 1:
        raise ValueError("numbers must have shape (N,)")
    if g.ndim != 1:
        raise ValueError("g must have shape (M,)")
    table = _lobato_table()
    g2 = (g.to(torch.float64)) ** 2
    factors = torch.zeros((numbers.shape[0], g.shape[0]), dtype=torch.float64, device=g.device)
    for z in torch.unique(numbers).tolist():
        symbol = _ELEMENTS[int(z) - 1]
        a_coeffs, b_coeffs = table[symbol]
        a = torch.tensor(a_coeffs, dtype=torch.float64, device=g.device)[:, None]
        b = torch.tensor(b_coeffs, dtype=torch.float64, device=g.device)[:, None]
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


def resolution_cutoff(g: Tensor, g_max: float, *, mode: Cutoff = "hard") -> Tensor:
    """Reflection resolution cutoff: a hard ``|g| <= g_max`` mask or the private taper window."""
    if g_max <= 0.0:
        raise ValueError("g_max must be positive")
    if mode == "hard":
        return (g <= g_max).to(g.dtype)
    if mode == "taper":
        temperature, alpha = 0.005, 1.0 - 0.05
        return 1.0 / (1.0 + torch.exp((g / g_max - alpha) / temperature))
    raise ValueError(f"unsupported cutoff mode: {mode!r}")


def structure_factors(
    positions: Tensor,
    numbers: Tensor,
    occupancies: Tensor,
    uij_star: Tensor,
    hkl: Tensor,
    g: Tensor,
    cell_volume: float,
    *,
    g_max: float,
    cutoff: Cutoff = "hard",
    extinction_threshold: float = 1e-12,
) -> Tensor:
    """Vectorised electron structure factors ``Fgb`` (elastic).

    ``Fgb(h) = (1/V) sum_atoms f_e * DWF * occ * cutoff * exp(2 pi i r . h)``, ported from
    ``StructureFactorNet.forward``. Differentiable in ``positions``, ``uij_star``, ``occupancies``;
    ``f_e`` is a constant form factor. Components below ``extinction_threshold`` are zeroed to match
    the private symmetry-extinction cleanup. Returns a complex ``(M,)`` tensor.
    """
    n_atoms = positions.shape[0]
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if numbers.shape != (n_atoms,) or occupancies.shape != (n_atoms,):
        raise ValueError("numbers and occupancies must have shape (N,) matching positions")
    if cell_volume <= 0.0:
        raise ValueError("cell_volume must be positive")

    form_factors = lobato_form_factors(numbers, g)
    dwf = debye_waller_factor(hkl, uij_star)
    cutoff_window = resolution_cutoff(g, g_max, mode=cutoff)
    per_atom = (form_factors * dwf * occupancies[:, None]) * cutoff_window[None, :]

    phase = torch.exp(2.0j * torch.pi * (positions @ hkl.to(positions.dtype).transpose(0, 1)))
    unmasked = (per_atom.to(phase.dtype) * phase).sum(dim=0)

    real = torch.where(
        unmasked.real.abs() >= extinction_threshold, unmasked.real, torch.zeros_like(unmasked.real)
    )
    imag = torch.where(
        unmasked.imag.abs() >= extinction_threshold, unmasked.imag, torch.zeros_like(unmasked.imag)
    )
    return torch.complex(real, imag) / cell_volume
