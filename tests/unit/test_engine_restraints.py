"""Soft refinement restraint terms."""

from __future__ import annotations

import pytest
import torch

from diffBloch.engine import BondRestraints
from diffBloch.params import PhysicalState


def _state(positions: torch.Tensor) -> PhysicalState:
    n_atoms = int(positions.shape[0])
    return PhysicalState(
        positions=positions,
        uij_star=torch.eye(3, dtype=positions.dtype).expand(n_atoms, 3, 3),
        occupancies=torch.ones(n_atoms, dtype=positions.dtype),
    )


def _bond(
    *, distance_fractional: float, target: float = 2.0, sigma: float = 0.1
) -> tuple[BondRestraints, PhysicalState]:
    cell = torch.eye(3, dtype=torch.float64) * 10.0
    restraint = BondRestraints(
        pairs=torch.tensor([[0, 1]], dtype=torch.int64),
        target_angstrom=torch.tensor([target], dtype=torch.float64),
        sigma_angstrom=torch.tensor([sigma], dtype=torch.float64),
        frac_to_cart=cell,
    )
    state = _state(
        torch.tensor(
            [[0.0, 0.0, 0.0], [distance_fractional, 0.0, 0.0]],
            dtype=torch.float64,
        )
    )
    return restraint, state


def test_bond_restraint_is_zero_at_target_distance() -> None:
    restraint, state = _bond(distance_fractional=0.2)  # 0.2 * 10 A = 2 A

    assert torch.equal(restraint.loss(state), torch.zeros((), dtype=torch.float64))


def test_bond_restraint_penalizes_stretched_bond_with_mse_default() -> None:
    restraint, state = _bond(distance_fractional=0.3)  # 3 A vs target 2 A, sigma 0.1 A

    assert torch.equal(restraint.loss(state), torch.tensor(100.0, dtype=torch.float64))


def test_bond_restraint_flat_bottom_l1_is_zero_inside_tolerance() -> None:
    restraint, state = _bond(distance_fractional=0.205, sigma=0.1)  # 2.05 A vs 2.0 A
    restraint = BondRestraints(
        pairs=restraint.pairs,
        target_angstrom=restraint.target_angstrom,
        sigma_angstrom=restraint.sigma_angstrom,
        frac_to_cart=restraint.frac_to_cart,
        criterion="flat_bottom_l1",
    )

    assert torch.equal(restraint.loss(state), torch.zeros((), dtype=torch.float64))


def test_bond_restraint_flat_bottom_l1_is_linear_outside_tolerance() -> None:
    restraint, state = _bond(distance_fractional=0.3, sigma=0.1)  # |3 - 2| - 0.1 = 0.9 A
    restraint = BondRestraints(
        pairs=restraint.pairs,
        target_angstrom=restraint.target_angstrom,
        sigma_angstrom=restraint.sigma_angstrom,
        frac_to_cart=restraint.frac_to_cart,
        criterion="flat_bottom_l1",
    )

    assert torch.equal(restraint.loss(state), torch.tensor(0.9, dtype=torch.float64))


def test_bond_restraint_gradient_pulls_stretched_bond_shorter() -> None:
    restraint, state = _bond(distance_fractional=0.3)
    positions = state.positions.detach().clone().requires_grad_(True)

    loss = restraint.loss(_state(positions))
    loss.backward()  # type: ignore[no-untyped-call]

    assert positions.grad is not None
    assert positions.grad[0, 0] < 0.0
    assert positions.grad[1, 0] > 0.0


def test_bond_restraint_is_translation_invariant() -> None:
    restraint, state = _bond(distance_fractional=0.3)
    translated = _state(state.positions + torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64))

    assert torch.equal(restraint.loss(translated), restraint.loss(state))


def test_bond_restraint_validates_shapes_and_sigmas() -> None:
    cell = torch.eye(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="pairs"):
        BondRestraints(
            pairs=torch.tensor([0, 1]),
            target_angstrom=torch.tensor([1.0]),
            sigma_angstrom=torch.tensor([0.1]),
            frac_to_cart=cell,
        )
    with pytest.raises(ValueError, match="sigmas must be positive"):
        BondRestraints(
            pairs=torch.tensor([[0, 1]]),
            target_angstrom=torch.tensor([1.0]),
            sigma_angstrom=torch.tensor([0.0]),
            frac_to_cart=cell,
        )
    with pytest.raises(ValueError, match="criterion"):
        BondRestraints(
            pairs=torch.tensor([[0, 1]]),
            target_angstrom=torch.tensor([1.0]),
            sigma_angstrom=torch.tensor([0.1]),
            frac_to_cart=cell,
            criterion="not-a-criterion",  # type: ignore[arg-type]
        )
