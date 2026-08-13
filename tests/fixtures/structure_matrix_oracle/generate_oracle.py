"""Regenerate the stage-8 structure-matrix / propagator oracle fixtures (in place).

Writes ``structure_matrix_oracle.npz`` (zone-axis, ``Mii == 1``),
``structure_matrix_oracle_oblique.npz`` (oblique, ``Mii != 1``) and ``provenance.json`` next to
this script, then cross-checks that the *native* ``diffBloch.core`` implementation reproduces the
freshly written goldens.

Self-contained: the oracle bodies below are **verbatim reference implementations** (pure
numpy/torch, timing/CUDA logging elided), vendored here so regeneration needs no external
checkout. Run with this repo's venv::

    uv run python tests/fixtures/structure_matrix_oracle/generate_oracle.py

Documented deviation: the source propagator internally casts to ``complex64``; this oracle keeps
``complex128`` throughout for tight regression tolerances. ``Fgb`` is synthetic but
Friedel-symmetric (``F(-g) = conj(F(g))``, ``F(000)`` real), as a real no-absorption potential
requires, so the ``Mii``-symmetrised ``A`` is Hermitian and the propagator goldens are physical.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import torch

from diffBloch.core import build_beam_plan, build_bloch_system, g_vectors, propagate
from diffBloch.core import structure_matrix as native_structure_matrix

OUT_DIR = Path(__file__).resolve().parent

# abTEM constants as CODATA-2018 closed-forms -- identical to core.dynamical.primitives;
# reproduce abTEM's energy2wavelength / energy2sigma / kappa to ~1e-8 (see REFERENCES.md).
_PLANCK = 6.62607015e-34
_ELECTRON_MASS = 9.1093837015e-31
_ELEMENTARY_CHARGE = 1.602176634e-19
_SPEED_OF_LIGHT = 299792458.0
_VACUUM_PERMITTIVITY = 8.8541878128e-12
_BOHR_RADIUS = 0.529177210903e-10

kappa = 2.0 * _VACUUM_PERMITTIVITY / (_BOHR_RADIUS * _ELEMENTARY_CHARGE) * 1e-20


def energy2wavelength(energy):
    charge_energy = energy * _ELEMENTARY_CHARGE
    rest = 2.0 * _ELECTRON_MASS * _SPEED_OF_LIGHT**2
    metres = _PLANCK * _SPEED_OF_LIGHT / np.sqrt(charge_energy * (rest + charge_energy))
    return float(metres * 1e10)


def energy2sigma(energy):
    relativistic_mass = (
        1.0 + _ELEMENTARY_CHARGE * energy / (_ELECTRON_MASS * _SPEED_OF_LIGHT**2)
    ) * _ELECTRON_MASS
    wavelength_metres = energy2wavelength(energy) * 1e-10
    sigma_si = 2.0 * np.pi * relativistic_mass * _ELEMENTARY_CHARGE * wavelength_metres / _PLANCK**2
    return float(sigma_si * 1e-10)


# Verbatim reference bodies. Pure numpy/torch; copied unchanged except timing/CUDA logging is
# elided. Sources: utils.py::reciprocal_cell (35-58), calculate_g_vec
# (61-79), excitation_errors (261-300), ravel_hkl (520-548), raveled_hkl_to_hkl_torch (453-517),
# fill_diagonal_torch (564-602), fill_diagonal_torch_batched (605-624);
# dynamical.py::calculate_M_matrix (1128-1143), calculate_structure_matrix (975-1025),
# calculate_dynamical_scattering_batched (820-972).


def reciprocal_cell(cell):
    return np.linalg.pinv(cell).transpose()


def calculate_g_vec(hkl, cell):
    return hkl @ reciprocal_cell(cell)


def excitation_errors(g, energy, U0=0.0):
    l = energy2wavelength(energy)  # noqa: E741 (verbatim oracle body)
    K0 = 1 / l
    Kmag = np.sqrt(K0**2 + U0)
    K = np.array([0.0, 0.0, -Kmag])
    Sg = (np.linalg.norm(K) ** 2 - np.linalg.norm(K + g, axis=1) ** 2) / (2 * Kmag)
    return Sg


def calculate_M_matrix(hkl, cell, energy, U0=0.0):
    rc = reciprocal_cell(cell)
    g = hkl @ rc
    k0 = -1 / energy2wavelength(energy)
    Kn = np.sqrt(k0**2 + U0)
    Mii = 1 / np.sqrt(1 + g[:, 2] / -Kn)
    return Mii


def ravel_hkl(hkl, gpts):
    hkl = np.asarray(hkl)
    shift = np.array((gpts[0] // 2, gpts[1] // 2, gpts[2] // 2))
    hkl = hkl + shift
    multi_index = (hkl[..., 0], hkl[..., 1], hkl[..., 2])
    return np.ravel_multi_index(multi_index, gpts)


def raveled_hkl_to_hkl_torch(
    array,
    hkl_source,
    hkl_destination,
    gpts,
    hkl_source_raveled=None,
    hkl_destination_raveled=None,
    sparse_prebuilt=None,
):
    if hkl_source_raveled is None:
        hkl_source_raveled = ravel_hkl(hkl_source, gpts)
    if hkl_destination_raveled is None:
        hkl_destination_raveled = ravel_hkl(hkl_destination, gpts)
    hkl_source_raveled = torch.tensor(hkl_source_raveled, dtype=torch.long, device=array.device)
    if torch.is_tensor(hkl_destination_raveled):
        hkl_destination_raveled = hkl_destination_raveled.to(device=array.device, dtype=torch.long)
    else:
        hkl_destination_raveled = torch.tensor(
            hkl_destination_raveled, dtype=torch.long, device=array.device
        )
    max_index = hkl_source_raveled.max() + 1
    if sparse_prebuilt is not None:
        sparse_array = sparse_prebuilt
    else:
        sparse_array = torch.zeros(max_index, dtype=array.dtype, device=array.device)
        sparse_array.index_add_(0, hkl_source_raveled, array)
    output_array = sparse_array[hkl_destination_raveled]
    return output_array


def fill_diagonal_torch(A, diag):
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    assert len(diag) == A.shape[0], "diag must have the same length as the number of rows in A"
    A_copy = A.clone()
    indices = torch.arange(A.shape[0], device=A.device)
    A_copy[indices, indices] = diag
    return A_copy


def fill_diagonal_torch_batched(A, diag):
    A_copy = A.clone()
    idx = torch.arange(A.shape[1], device=A.device)
    A_copy[:, idx, idx] = diag
    return A_copy


def calculate_structure_matrix(
    structure_factor,
    hkl,
    hkl_selected,
    cell,
    energy,
    U_0_prime,
    gpts,
    device="cpu",
    absorption=False,
    U0=0.0,
):
    g = np.asarray(calculate_g_vec(hkl_selected, cell))
    Mii = calculate_M_matrix(hkl_selected, cell, energy, U0=U0)
    hkl_selected = np.asarray(hkl_selected)

    gmh = hkl_selected[None] - hkl_selected[:, None]
    gmh = gmh.reshape(-1, 3)
    A = raveled_hkl_to_hkl_torch(structure_factor, hkl, gmh, gpts)
    A = A.reshape((len(hkl_selected),) * 2)

    prefactor = energy2sigma(energy) / (kappa * 1 * energy2wavelength(energy) * np.pi)

    Mii = torch.tensor(Mii, device=device)
    A *= prefactor * Mii[None] * Mii[:, None]

    sg = np.asarray(excitation_errors(g, energy, U0=U0))
    diag = 2 * np.sqrt((1 / energy2wavelength(energy)) ** 2 + U0) * sg
    diag = torch.tensor(diag, device=device)

    if absorption:
        diag = diag.to(torch.complex128)
        if U_0_prime is not None:
            diag += 1j * U_0_prime
        else:
            max_imaginary_off_diag = (
                A.imag[~torch.eye(A.size(0), dtype=bool, device=A.device)].abs().max().item()
            )
            diag += 1j * 1.7 * max_imaginary_off_diag

    diag *= Mii
    diag = diag.to(A.dtype)

    A_filled = fill_diagonal_torch(A, diag)
    return A_filled


def calculate_dynamical_scattering_batched(
    structure_matrices,
    hkl,
    cells,
    energy,
    thicknesses,
    device="cpu",
    absorption=False,
    U0=0.0,
    method="matrix_exp",
):
    B, N, _ = structure_matrices.shape
    thicknesses_t = torch.as_tensor(thicknesses, dtype=torch.float64, device=device)

    if method == "bloch_eigen":
        Mii = torch.empty(B, N, dtype=torch.float64, device=device)
        for b in range(B):
            Mii[b] = torch.tensor(
                calculate_M_matrix(hkl, cells[b], energy, U0=U0),
                device=device,
            )

        A = structure_matrices.to(torch.complex128)
        if absorption:
            v, C_temp = torch.linalg.eig(A)
        else:
            v, C_temp = torch.linalg.eigh(A)

        Kn = np.sqrt(1 / energy2wavelength(energy) ** 2 + U0)
        gamma = v / (Kn * 2.0)

        idx = torch.arange(N, device=device)
        diag = C_temp[:, idx, idx] / Mii.to(C_temp.dtype)
        diag = diag.to(C_temp.dtype)
        C = fill_diagonal_torch_batched(C_temp, diag)

        if absorption:  # noqa: SIM108 (verbatim oracle body)
            C_inv = torch.linalg.inv(C_temp)
        else:
            C_inv = torch.conj(C.mT)

        initial = np.all(hkl == [0, 0, 0], axis=1).astype(complex)
        initial = torch.tensor(initial, device=device).to(C_inv.dtype)

        alpha = torch.matmul(C_inv, initial)
        phase = torch.exp(2.0j * np.pi * thicknesses_t[None, :, None] * gamma[:, None, :])
        return torch.matmul(phase * alpha[:, None, :], C.mT)

    if method == "matrix_exp":
        k_n = np.sqrt((1 / energy2wavelength(energy)) ** 2 + U0)
        A = structure_matrices.to(torch.complex128)
        initial = np.all(hkl == [0, 0, 0], axis=1).astype(complex)
        psi0 = torch.tensor(initial, dtype=A.dtype, device=device)
        scalars = (np.pi * 1j * thicknesses_t / k_n).to(A.dtype)
        M = A.unsqueeze(0) * scalars[:, None, None, None]
        M_flat = M.reshape(-1, A.shape[1], A.shape[2])
        S_flat = torch.matrix_exp(M_flat)
        S = S_flat.reshape(thicknesses_t.numel(), A.shape[0], A.shape[1], A.shape[2])
        psi0_b = psi0.expand(A.shape[0], -1)
        psi0_tb = psi0_b.unsqueeze(0).expand(thicknesses_t.numel(), -1, -1)
        result = torch.matmul(S, psi0_tb.unsqueeze(-1)).squeeze(-1)
        return result.permute(1, 0, 2).contiguous()

    raise ValueError("method must be 'matrix_exp', 'bloch_eigen'")


# Small alpha-quartz case: a=b=4.913 A, c=5.405 A, gamma=120 deg. Zone beams: |h|,|k| <= 1, l=0
# (9 beams) over the full |h|,|k| <= 2, l=0 grid (25 cells) -- covers every pairwise difference.
# The assembly is linear in F, so synthetic (Friedel-symmetric) input fully exercises
# gather + scale + diagonal.
CELL_A = 4.913
CELL_C = 5.405
_GAMMA = np.deg2rad(120.0)
CELL = np.array(
    [
        [CELL_A, 0.0, 0.0],
        [CELL_A * np.cos(_GAMMA), CELL_A * np.sin(_GAMMA), 0.0],
        [0.0, 0.0, CELL_C],
    ]
)
RECIPROCAL_BASIS = reciprocal_cell(CELL)
ENERGY = 200e3
U0 = 0.0
THICKNESSES = np.array([1.0, 8.0, 42.0, 500.0], dtype=np.float64)


def friedel_symmetric_factors(grid, seed):
    """Synthetic Fgb obeying Friedel's law F(-g) = conj(F(g)) (F(000) real).

    A real, no-absorption potential has Friedel-symmetric structure factors -- exactly what makes
    the Mii-symmetrised structure matrix A Hermitian. Plain random complex Fgb does NOT, leaving A
    non-Hermitian so matrix_exp is non-unitary (flux blows up) and eigh silently uses only half of
    A; the two propagators then disagree by orders of magnitude. Enforcing Friedel keeps the goldens
    physical and the propagators comparable.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(len(grid)) + 1j * rng.standard_normal(len(grid))
    idx = {tuple(int(x) for x in h): i for i, h in enumerate(grid)}
    factors = np.empty(len(grid), dtype=complex)
    for h in grid:
        i = idx[tuple(int(x) for x in h)]
        j = idx[tuple(-int(x) for x in h)]
        if i < j:
            factors[i] = raw[i]
            factors[j] = np.conj(raw[i])
        elif i == j:
            factors[i] = raw[i].real
    return torch.tensor(factors, dtype=torch.complex128)


def propagate_case(beam_hkl, structure_factor_hkl, gpts, seed):
    structure_factor = friedel_symmetric_factors(structure_factor_hkl, seed)
    A = calculate_structure_matrix(
        structure_factor,
        structure_factor_hkl,
        beam_hkl,
        CELL,
        ENERGY,
        None,
        gpts,
        device="cpu",
        absorption=False,
        U0=U0,
    )
    cells = np.asarray([CELL])
    psi_matrix_exp = calculate_dynamical_scattering_batched(
        A.unsqueeze(0), beam_hkl, cells, ENERGY, THICKNESSES, U0=U0, method="matrix_exp"
    )[0]
    psi_bloch_eigen = calculate_dynamical_scattering_batched(
        A.unsqueeze(0), beam_hkl, cells, ENERGY, THICKNESSES, U0=U0, method="bloch_eigen"
    )[0]
    g = calculate_g_vec(beam_hkl, CELL)
    hermitian = bool(torch.allclose(A, A.mH, atol=1e-9))
    return dict(
        A=A.numpy(),
        psi_matrix_exp=psi_matrix_exp.numpy(),
        psi_bloch_eigen=psi_bloch_eigen.numpy(),
        thicknesses=THICKNESSES,
        structure_factor=structure_factor.numpy(),
        structure_factor_hkl=structure_factor_hkl,
        beam_hkl=beam_hkl,
        reciprocal_basis=RECIPROCAL_BASIS,
        g=g,
        gpts=np.array(gpts, dtype=np.int64),
        energy=np.float64(ENERGY),
        u0=np.float64(U0),
        _hermitian=hermitian,
    )


_FIXTURE_KEYS = (
    "A",
    "psi_matrix_exp",
    "psi_bloch_eigen",
    "thicknesses",
    "structure_factor",
    "structure_factor_hkl",
    "beam_hkl",
    "reciprocal_basis",
    "g",
    "gpts",
    "energy",
    "u0",
)


def cross_check(name, case):
    """Confirm the native diffBloch.core implementation reproduces the just-written golden.

    The same assertion ``test_core_dynamical.py::test_structure_matrix_matches_oracle``
    makes -- run here so regeneration is self-validating.
    """
    plan = build_beam_plan(
        case["beam_hkl"],
        case["structure_factor_hkl"],
        case["reciprocal_basis"],
        energy=float(case["energy"]),
        gpts=tuple(int(x) for x in case["gpts"]),
        u0=float(case["u0"]),
    )
    assert np.allclose(g_vectors(case["beam_hkl"], case["reciprocal_basis"]), case["g"])
    sf = torch.tensor(case["structure_factor"])
    system = build_bloch_system(plan, sf)

    a_diff = (native_structure_matrix(plan, sf) - torch.tensor(case["A"])).abs().max().item()
    me = propagate(system, torch.tensor(case["thicknesses"]), method="matrix_exp")
    be = propagate(system, torch.tensor(case["thicknesses"]), method="bloch_eigen")
    me_diff = (me - torch.tensor(case["psi_matrix_exp"])).abs().max().item()
    be_diff = (be - torch.tensor(case["psi_bloch_eigen"])).abs().max().item()
    assert a_diff < 1e-9 and me_diff < 1e-9 and be_diff < 1e-9, (name, a_diff, me_diff, be_diff)
    print(
        f"{name:8s} | native vs oracle  A {a_diff:.1e}  psi_me {me_diff:.1e}  psi_be {be_diff:.1e}"
    )


def main() -> None:
    # Zone-axis case: in-plane beams (l = 0) -> g_z = 0 -> Mii == 1; A Hermitian (Friedel Fgb).
    beam_hkl = np.array([[h, k, 0] for h in (-1, 0, 1) for k in (-1, 0, 1)], dtype=np.int64)
    structure_factor_hkl = np.array(
        [[h, k, 0] for h in (-2, -1, 0, 1, 2) for k in (-2, -1, 0, 1, 2)], dtype=np.int64
    )
    zone = propagate_case(beam_hkl, structure_factor_hkl, (5, 5, 1), seed=8)

    # Oblique case: out-of-plane beams (l != 0) -> g_z != 0 -> Mii != 1, exercising the
    # symmetrisation/un-symmetrisation the zone-axis case leaves trivial. Still Friedel-Hermitian.
    beam_hkl_ob = np.array(
        [[h, k, l] for h in (-1, 0, 1) for k in (-1, 0, 1) for l in (-1, 0, 1)],  # noqa: E741
        dtype=np.int64,
    )
    grid_hkl_ob = np.array(
        [[h, k, l] for h in range(-2, 3) for k in range(-2, 3) for l in range(-2, 3)],  # noqa: E741
        dtype=np.int64,
    )
    oblique = propagate_case(beam_hkl_ob, grid_hkl_ob, (5, 5, 5), seed=9)

    assert zone["_hermitian"] and oblique["_hermitian"]
    np.savez(OUT_DIR / "structure_matrix_oracle.npz", **{k: zone[k] for k in _FIXTURE_KEYS})
    np.savez(
        OUT_DIR / "structure_matrix_oracle_oblique.npz", **{k: oblique[k] for k in _FIXTURE_KEYS}
    )

    provenance = {
        "fixtures": {
            "structure_matrix_oracle.npz": "zone-axis (l=0 beams) -> g_z=0 -> Mii==1; A Hermitian",
            "structure_matrix_oracle_oblique.npz": (
                "oblique (l!=0 beams) -> g_z!=0 -> Mii!=1; A Hermitian"
            ),
        },
        "generated_by": "generate_oracle.py (co-located; self-contained verbatim oracle bodies)",
        "oracle_functions": {
            "calculate_structure_matrix": (
                "dynamical.py:975-1025 (no-absorption; timing/cuda elided)"
            ),
            "calculate_dynamical_scattering_batched": (
                "dynamical.py:820-972 (no-absorption, B=1; timing/cuda elided; "
                "complex128 oracle deviation)"
            ),
            "calculate_M_matrix": "dynamical.py:1128-1143",
            "reciprocal_cell": "utils.py:35-58",
            "calculate_g_vec": "utils.py:61-79",
            "excitation_errors": "utils.py:261-300",
            "ravel_hkl": "utils.py:520-548",
            "raveled_hkl_to_hkl_torch": "utils.py:453-517",
            "fill_diagonal_torch": "utils.py:564-602",
            "fill_diagonal_torch_batched": "utils.py:605-624",
        },
        "constants": (
            "energy2wavelength/energy2sigma/kappa as CODATA-2018 closed-forms "
            "(abTEM-pinned ~1e-8, stage 8.1-8.3)"
        ),
        "dtype_note": (
            "The source propagator casts to complex64 internally; this oracle keeps complex128 "
            "throughout for tight regression tolerances."
        ),
        "fgb_note": (
            "Fgb is synthetic but Friedel-symmetric F(-g)=conj(F(g)) (F(000) real), as a real "
            "no-absorption potential requires, so the Mii-symmetrised A is Hermitian and the "
            "propagator goldens are physical."
        ),
        "propagator_note": (
            "matrix_exp returns the symmetrised wavefunction (unitary, flux==1); bloch_eigen "
            "un-symmetrises to physical amplitudes. They agree to machine precision only at "
            "Mii==1 (zone axis) and differ at O(g_z/K_n) for the oblique case."
        ),
        "case": {
            "material": "alpha-quartz",
            "cell_a": CELL_A,
            "cell_c": CELL_C,
            "gamma_deg": 120.0,
            "energy_eV": float(ENERGY),
            "u0": U0,
            "absorption": False,
            "thicknesses_A": THICKNESSES.tolist(),
            "zone": {
                "n_beams": int(len(zone["beam_hkl"])),
                "n_grid": int(len(zone["structure_factor_hkl"])),
                "gpts": zone["gpts"].tolist(),
                "seed": 8,
            },
            "oblique": {
                "n_beams": int(len(oblique["beam_hkl"])),
                "n_grid": int(len(oblique["structure_factor_hkl"])),
                "gpts": oblique["gpts"].tolist(),
                "seed": 9,
            },
        },
        "date": str(date.today()),
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    cross_check("zone", zone)
    cross_check("oblique", oblique)
    print(f"PASS: native core reproduces both goldens; wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
