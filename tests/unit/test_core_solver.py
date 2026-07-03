"""Bloch-wave propagators for closed ``BlochSystem`` values."""

from pathlib import Path

import numpy as np
import pytest
import torch

from diffBloch.core.dynamical import (
    build_beam_plan,
    build_bloch_system,
    build_bloch_systems,
    m_factors,
    stack_beam_plans,
)
from diffBloch.core.reciprocal import g_vectors
from diffBloch.core.solver import propagate

_BEAM_HKL = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
_GRID_HKL = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)
_RECIP_BASIS = np.eye(3)
_ENERGY = 200e3
_GPTS = (3, 1, 1)


def _system(factors: torch.Tensor | None = None):
    if factors is None:
        factors = torch.tensor([0.0, 1.0, 1.0], dtype=torch.complex128)
    plan = build_beam_plan(_BEAM_HKL, _GRID_HKL, _RECIP_BASIS, energy=_ENERGY, gpts=_GPTS)
    return build_bloch_system(plan, factors)


@pytest.mark.parametrize("method", ["matrix_exp", "bloch_eigen"])
def test_propagate_zero_thickness_returns_initial_wavefunction(method: str) -> None:
    system = _system()
    psi = propagate(system, [0.0], method=method)

    assert psi.shape == (1, _BEAM_HKL.shape[0])
    assert torch.allclose(psi[0], system.psi0.to(psi.dtype))


@pytest.mark.parametrize("method", ["matrix_exp", "bloch_eigen"])
def test_propagate_conserves_flux_for_hermitian_system(method: str) -> None:
    psi = propagate(_system(), torch.tensor([0.0, 1.0, 8.0, 42.0]), method=method)
    flux = psi.abs().square().sum(dim=1)

    assert torch.allclose(flux, torch.ones_like(flux), atol=1e-12, rtol=1e-12)


def test_matrix_exp_and_bloch_eigen_agree_for_hermitian_system() -> None:
    system = _system()
    thicknesses = torch.tensor([0.0, 1.0, 8.0, 42.0], dtype=torch.float64)

    assert torch.allclose(
        propagate(system, thicknesses, method="matrix_exp"),
        propagate(system, thicknesses, method="bloch_eigen"),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("method", ["matrix_exp", "bloch_eigen"])
def test_batched_propagate_matches_the_per_tilt_loop(method: str) -> None:
    """A batched (B, N, N) solve equals stacking B single solves -- the tilt-batching perf path.

    The rocking-curve tilts share one beam set (same gather / energy), differing only in the
    per-tilt geometry (here a slight reciprocal-basis scale standing in for the tilt). The batched
    operator must reproduce the loop bit-for-bit (``matrix_exp``) or to machine precision (eigh).
    """
    factors = torch.tensor([0.0, 1.0, 1.0], dtype=torch.complex128)
    plans = [
        build_beam_plan(_BEAM_HKL, _GRID_HKL, _RECIP_BASIS * s, energy=_ENERGY, gpts=_GPTS)
        for s in (1.0, 1.002, 0.998)
    ]
    thicknesses = torch.tensor([0.0, 12.0, 137.0], dtype=torch.float64)

    looped = torch.stack(
        [propagate(build_bloch_system(p, factors), thicknesses, method=method) for p in plans]
    )  # (B, T, N)
    batched = propagate(
        build_bloch_systems(stack_beam_plans(plans), factors), thicknesses, method=method
    )

    assert batched.shape == looped.shape
    assert batched.shape == (len(plans), thicknesses.shape[0], _BEAM_HKL.shape[0])
    assert torch.allclose(batched, looped, atol=1e-13, rtol=0.0)


def test_stack_beam_plans_rejects_unrelated_beam_sets() -> None:
    shared = build_beam_plan(_BEAM_HKL, _GRID_HKL, _RECIP_BASIS, energy=_ENERGY, gpts=_GPTS)
    other_beams = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int64)
    other_grid = np.array([[i, j, 0] for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=np.int64)
    unrelated = build_beam_plan(other_beams, other_grid, np.eye(3), energy=_ENERGY, gpts=(3, 3, 1))
    with pytest.raises(ValueError, match="sharing one beam set"):
        stack_beam_plans([shared, unrelated])


@pytest.mark.parametrize("method", ["matrix_exp", "bloch_eigen"])
def test_propagate_returns_on_the_operator_device(method: str) -> None:
    # Contract: the geometry fields (k_n/psi0/mii) are co-located onto a.device at the use site, so
    # the exit wavefunction lands on the operator device even when the plan was built CPU-side. A
    # CPU no-op here; meaningful on CUDA/MPS (see KNOWN_ISSUES.md / build_bloch_system).
    system = _system()
    psi = propagate(system, [1.0, 8.0], method=method)
    assert psi.device == system.a.device


@pytest.mark.parametrize("method", ["matrix_exp", "bloch_eigen"])
def test_propagate_is_differentiable_in_structure_factors(method: str) -> None:
    factors = torch.tensor([0.0, 1.0, 1.0], dtype=torch.complex128, requires_grad=True)
    psi = propagate(_system(factors), [1.0, 8.0], method=method)
    loss = psi[:, 1].abs().square().sum()
    loss.backward()

    assert factors.grad is not None
    assert factors.grad.abs().sum() > 0


def test_propagate_rejects_bad_method() -> None:
    with pytest.raises(ValueError, match="method must be"):
        propagate(_system(), [1.0], method="not-a-method")  # type: ignore[arg-type]


def test_propagate_rejects_bad_thickness_shape() -> None:
    with pytest.raises(ValueError, match="thicknesses must be"):
        propagate(_system(), torch.zeros((1, 1)))


_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "structure_matrix_oracle"
_ORACLE_ZONE = _FIXTURES / "structure_matrix_oracle.npz"  # zone axis: g_z = 0 -> Mii == 1
_ORACLE_OBLIQUE = _FIXTURES / "structure_matrix_oracle_oblique.npz"  # g_z != 0 -> Mii != 1


def _system_from(npz: Path):
    data = np.load(npz)
    plan = build_beam_plan(
        data["beam_hkl"],
        data["grid_hkl"],
        data["reciprocal_basis"],
        energy=float(data["energy"]),
        gpts=tuple(int(point) for point in data["gpts"]),
        u0=float(data["u0"]),
    )
    return build_bloch_system(plan, torch.tensor(data["structure_factor"])), data


def test_propagate_matches_private_oracle_zone() -> None:
    # Zone axis (Mii == 1), Friedel-symmetric Fgb -> A Hermitian. The two propagators agree to
    # machine precision here, and matrix_exp conserves flux.
    system, data = _system_from(_ORACLE_ZONE)
    g = g_vectors(data["beam_hkl"], data["reciprocal_basis"])
    assert np.allclose(m_factors(g, float(data["energy"]), u0=float(data["u0"])), 1.0)

    thicknesses = torch.tensor(data["thicknesses"])
    psi_me = propagate(system, thicknesses, method="matrix_exp")
    psi_be = propagate(system, thicknesses, method="bloch_eigen")
    assert torch.allclose(psi_me, torch.tensor(data["psi_matrix_exp"]), rtol=1e-10, atol=1e-12)
    assert torch.allclose(psi_be, torch.tensor(data["psi_bloch_eigen"]), rtol=1e-10, atol=1e-12)
    # Hermitian + Mii == 1: matrix_exp is unitary and the methods coincide.
    flux = psi_me.abs().square().sum(1)
    assert torch.allclose(flux, torch.ones_like(flux), atol=1e-12)
    assert torch.allclose(psi_me, psi_be, atol=1e-10)


def test_propagate_matches_private_oracle_oblique() -> None:
    # Off zone axis (Mii != 1), still Friedel -> A Hermitian. matrix_exp returns the *symmetrised*
    # wavefunction (unitary); bloch_eigen un-symmetrises to *physical* amplitudes. Both reproduce
    # their private goldens, but agree only to O(g_z/K_n) -- not bit-for-bit -- the obliquity
    # difference this case exists to pin (see provenance.json propagator_note).
    system, data = _system_from(_ORACLE_OBLIQUE)
    g = g_vectors(data["beam_hkl"], data["reciprocal_basis"])
    assert np.any(np.abs(m_factors(g, float(data["energy"]), u0=float(data["u0"])) - 1.0) > 1e-4)

    thicknesses = torch.tensor(data["thicknesses"])
    psi_me = propagate(system, thicknesses, method="matrix_exp")
    psi_be = propagate(system, thicknesses, method="bloch_eigen")
    assert torch.allclose(psi_me, torch.tensor(data["psi_matrix_exp"]), rtol=1e-10, atol=1e-12)
    assert torch.allclose(psi_be, torch.tensor(data["psi_bloch_eigen"]), rtol=1e-10, atol=1e-12)
    # matrix_exp stays unitary; the two methods are close but NOT equal off zone axis.
    flux = psi_me.abs().square().sum(1)
    assert torch.allclose(flux, torch.ones_like(flux), atol=1e-12)
    gap = (psi_me - psi_be).abs().max().item()
    assert 1e-9 < gap < 1e-2  # obliquity-scale disagreement, not machine noise
