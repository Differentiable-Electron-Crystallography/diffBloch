"""Stage 8 closer: characterise the two Bloch-wave propagators so a scientist can choose.

Both ``matrix_exp`` and ``bloch_eigen`` are first-class and swappable off the *same*
``BlochSystem`` (``propagate(system, T, method=...)``). This script is the auditable evidence behind
``design/decisions/stage8-bloch-propagators.md``; it does NOT pick a winner -- it reports the
trade-offs and confirms the regime where the two coincide.

It characterises, on the committed zone-axis (Mii==1) and oblique (Mii!=1) alpha-quartz fixtures,
for T in {1, 8, 42, 500} Angstrom:

* wall-time per method (warmup + repeats, CPU) -- bloch_eigen amortises one eig over many T;
* forward agreement  max|psi_matrix_exp - psi_bloch_eigen|;
* flux per method -- matrix_exp is unitary (symmetrised, flux==1 for Hermitian A); bloch_eigen
  returns *physical* amplitudes (flux != 1 once Mii != 1).

The methods agree to machine precision only at Mii==1; off zone axis they intentionally differ at
O(g_z/K_n) -- the symmetrised-vs-physical distinction, which is a feature to experiment with, not a
bug. Gradient behaviour is left to the scientist via the swappable API (a numerically airtight
gradient-stability study, and an R-factor/intensity oracle, are deferred -- see the decision note).

Pure-native (no abtem/scipy). Run:

    uv run python scripts/stage8_propagator_experiment.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from diffBloch.core.dynamical import build_beam_plan, build_bloch_system
from diffBloch.core.solver import Method, propagate

_FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "structure_matrix_oracle"
_CASES = (
    ("zone-axis (Mii == 1)", "structure_matrix_oracle.npz"),
    ("oblique   (Mii != 1)", "structure_matrix_oracle_oblique.npz"),
)
_THICKNESSES = (1.0, 8.0, 42.0, 500.0)
_REPEATS = 50
_WARMUP = 5


def _system(data: np.lib.npyio.NpzFile):
    plan = build_beam_plan(
        data["beam_hkl"],
        data["grid_hkl"],
        data["reciprocal_basis"],
        energy=float(data["energy"]),
        gpts=tuple(int(point) for point in data["gpts"]),
        u0=float(data["u0"]),
    )
    return build_bloch_system(plan, torch.tensor(data["structure_factor"], dtype=torch.complex128))


def _time(system, t: torch.Tensor, method: Method) -> float:
    for _ in range(_WARMUP):
        propagate(system, t, method=method)
    start = time.perf_counter()
    for _ in range(_REPEATS):
        propagate(system, t, method=method)
    return (time.perf_counter() - start) / _REPEATS * 1e6  # microseconds/call


def main() -> None:
    print(f"timing: {_REPEATS} repeats after {_WARMUP} warmup, CPU\n")
    header = (
        f"{'T (A)':>7} | {'me (us)':>9} {'be (us)':>9} {'be/me':>6} "
        f"| {'fwd max|d|':>11} | {'flux_me':>9} {'flux_be':>9}"
    )
    for label, fixture in _CASES:
        data = np.load(_FIXTURES / fixture)
        system = _system(data)
        beams, kev = data["beam_hkl"].shape[0], float(data["energy"]) / 1e3
        print(f"{label}  --  {beams} beams, {kev:.0f} keV")
        print(header)
        print("-" * len(header))
        for thickness in _THICKNESSES:
            t = torch.tensor([thickness], dtype=torch.float64)
            t_me, t_be = _time(system, t, "matrix_exp"), _time(system, t, "bloch_eigen")
            psi_me = propagate(system, t, method="matrix_exp")
            psi_be = propagate(system, t, method="bloch_eigen")
            fwd = (psi_me - psi_be).abs().max().item()
            flux_me = psi_me.abs().square().sum().item()
            flux_be = psi_be.abs().square().sum().item()
            print(
                f"{thickness:7.0f} | {t_me:9.1f} {t_be:9.1f} {t_be / t_me:6.2f} "
                f"| {fwd:11.2e} | {flux_me:9.6f} {flux_be:9.6f}"
            )
        print()

    print(
        "Reading: at Mii == 1 the methods coincide (~1e-15) and both conserve flux. At Mii != 1\n"
        "they differ at O(g_z/K_n): matrix_exp stays unitary (symmetrised), bloch_eigen flux\n"
        "departs from 1 (physical amplitudes). Default = matrix_exp (single autograd-stable\n"
        "primitive); bloch_eigen is first-class and swappable for fast eval / physical amplitudes."
    )


if __name__ == "__main__":
    main()
