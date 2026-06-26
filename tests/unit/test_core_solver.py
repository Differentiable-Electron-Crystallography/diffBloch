"""Bloch-wave propagators for closed ``BlochSystem`` values."""

from pathlib import Path

import numpy as np
import pytest
import torch

from diffBloch.core.dynamical import build_beam_plan, build_bloch_system
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


_ORACLE_NPZ = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "structure_matrix_oracle"
    / "structure_matrix_oracle.npz"
)


def test_propagate_matches_private_oracle() -> None:
    data = np.load(_ORACLE_NPZ)
    plan = build_beam_plan(
        data["beam_hkl"],
        data["grid_hkl"],
        data["reciprocal_basis"],
        energy=float(data["energy"]),
        gpts=tuple(int(point) for point in data["gpts"]),
        u0=float(data["u0"]),
    )
    system = build_bloch_system(plan, torch.tensor(data["structure_factor"]))

    assert torch.allclose(
        propagate(system, torch.tensor(data["thicknesses"]), method="matrix_exp"),
        torch.tensor(data["psi_matrix_exp"]),
        rtol=1e-10,
        atol=1e-12,
    )
    assert torch.allclose(
        propagate(system, torch.tensor(data["thicknesses"]), method="bloch_eigen"),
        torch.tensor(data["psi_bloch_eigen"]),
        rtol=1e-10,
        atol=1e-12,
    )
