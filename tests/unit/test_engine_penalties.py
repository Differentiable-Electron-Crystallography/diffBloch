"""Soft refinement penalty terms."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diffBloch.engine import BondLengthPenalty, perceive_bond_length_penalty
from diffBloch.io.record import AdpRecord, StructureRecord
from diffBloch.params import PhysicalState


def _structure(
    *, positions: list[list[float]], numbers: list[int], cell_scale: float = 10.0
) -> StructureRecord:
    n_atoms = len(numbers)
    return StructureRecord(
        unit_cell=np.eye(3, dtype=np.float64) * cell_scale,
        cell_parameters=np.array([cell_scale, cell_scale, cell_scale, 90.0, 90.0, 90.0]),
        cell_parameters_su=np.full(6, np.nan),
        spacegroup_hm="P 1",
        spacegroup_number=1,
        symops_R=np.eye(3, dtype=np.float64)[None],
        symops_t=np.zeros((1, 3), dtype=np.float64),
        labels=tuple(f"A{i}" for i in range(n_atoms)),
        numbers=np.array(numbers, dtype=np.int64),
        frac_positions=np.array(positions, dtype=np.float64),
        frac_positions_su=np.full((n_atoms, 3), np.nan),
        occupancies=np.ones(n_atoms, dtype=np.float64),
        occupancies_su=np.full(n_atoms, np.nan),
        adp=AdpRecord(
            kind=tuple("Uiso" for _ in range(n_atoms)),
            u_iso=np.zeros(n_atoms, dtype=np.float64),
            u_iso_su=np.full(n_atoms, np.nan),
            uij_cif=np.full((n_atoms, 3, 3), np.nan),
            uij_cif_su=np.full((n_atoms, 3, 3), np.nan),
        ),
    )


def _state(positions: torch.Tensor) -> PhysicalState:
    n_atoms = int(positions.shape[0])
    return PhysicalState(
        positions=positions,
        uij_star=torch.eye(3, dtype=positions.dtype).expand(n_atoms, 3, 3),
        occupancies=torch.ones(n_atoms, dtype=positions.dtype),
    )


def _bond(
    *, distance_fractional: float, target: float = 2.0, sigma: float = 0.1
) -> tuple[BondLengthPenalty, PhysicalState]:
    cell = torch.eye(3, dtype=torch.float64) * 10.0
    penalty = BondLengthPenalty(
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
    return penalty, state


def test_bond_penalty_is_zero_at_target_distance() -> None:
    penalty, state = _bond(distance_fractional=0.2)  # 0.2 * 10 A = 2 A

    assert torch.equal(penalty.value(state), torch.zeros((), dtype=torch.float64))


def test_bond_penalty_penalizes_stretched_bond_with_mse_default() -> None:
    penalty, state = _bond(distance_fractional=0.3)  # 3 A vs target 2 A, sigma 0.1 A

    assert torch.equal(penalty.value(state), torch.tensor(100.0, dtype=torch.float64))


def test_bond_penalty_flat_bottom_l1_is_zero_inside_tolerance() -> None:
    penalty, state = _bond(distance_fractional=0.205, sigma=0.1)  # 2.05 A vs 2.0 A
    penalty = BondLengthPenalty(
        pairs=penalty.pairs,
        target_angstrom=penalty.target_angstrom,
        sigma_angstrom=penalty.sigma_angstrom,
        frac_to_cart=penalty.frac_to_cart,
        criterion="flat_bottom_l1",
    )

    assert torch.equal(penalty.value(state), torch.zeros((), dtype=torch.float64))


def test_bond_penalty_flat_bottom_l1_is_linear_outside_tolerance() -> None:
    penalty, state = _bond(distance_fractional=0.3, sigma=0.1)  # |3 - 2| - 0.1 = 0.9 A
    penalty = BondLengthPenalty(
        pairs=penalty.pairs,
        target_angstrom=penalty.target_angstrom,
        sigma_angstrom=penalty.sigma_angstrom,
        frac_to_cart=penalty.frac_to_cart,
        criterion="flat_bottom_l1",
    )

    assert torch.equal(penalty.value(state), torch.tensor(0.9, dtype=torch.float64))


def test_bond_penalty_gradient_pulls_stretched_bond_shorter() -> None:
    penalty, state = _bond(distance_fractional=0.3)
    positions = state.positions.detach().clone().requires_grad_(True)

    loss = penalty.value(_state(positions))
    loss.backward()  # type: ignore[no-untyped-call]

    assert positions.grad is not None
    assert positions.grad[0, 0] < 0.0
    assert positions.grad[1, 0] > 0.0


def test_bond_penalty_is_translation_invariant() -> None:
    penalty, state = _bond(distance_fractional=0.3)
    translated = _state(state.positions + torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64))

    assert torch.equal(penalty.value(translated), penalty.value(state))


def test_perceive_bond_length_penalty_uses_current_heavy_atom_distances_as_targets() -> None:
    structure = _structure(
        positions=[[0.0, 0.0, 0.0], [0.14, 0.0, 0.0], [0.5, 0.0, 0.0]],
        numbers=[6, 6, 6],
    )

    penalties = perceive_bond_length_penalty(
        structure, sigma_angstrom=0.03, criterion="flat_bottom_l1"
    )

    assert torch.equal(penalties.pairs, torch.tensor([[0, 1]], dtype=torch.int64))
    assert torch.allclose(penalties.target_angstrom, torch.tensor([1.4], dtype=torch.float64))
    assert torch.equal(penalties.sigma_angstrom, torch.tensor([0.03], dtype=torch.float64))
    assert penalties.criterion == "flat_bottom_l1"
    assert torch.equal(penalties.frac_to_cart, torch.eye(3, dtype=torch.float64) * 10.0)


def test_perceive_bond_length_penalty_excludes_hydrogen_by_default() -> None:
    structure = _structure(positions=[[0.0, 0.0, 0.0], [0.109, 0.0, 0.0]], numbers=[6, 1])

    with pytest.raises(ValueError, match="found no bonds"):
        perceive_bond_length_penalty(structure)
    penalties = perceive_bond_length_penalty(structure, include_hydrogen=True)
    assert torch.equal(penalties.pairs, torch.tensor([[0, 1]], dtype=torch.int64))


def test_perceive_bond_length_penalty_reports_missing_covalent_radius() -> None:
    structure = _structure(positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], numbers=[14, 6])

    with pytest.raises(ValueError, match="no covalent radius"):
        perceive_bond_length_penalty(structure)


def test_bond_penalty_validates_shapes_and_sigmas() -> None:
    cell = torch.eye(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="pairs"):
        BondLengthPenalty(
            pairs=torch.tensor([0, 1]),
            target_angstrom=torch.tensor([1.0]),
            sigma_angstrom=torch.tensor([0.1]),
            frac_to_cart=cell,
        )
    with pytest.raises(ValueError, match="sigmas must be positive"):
        BondLengthPenalty(
            pairs=torch.tensor([[0, 1]]),
            target_angstrom=torch.tensor([1.0]),
            sigma_angstrom=torch.tensor([0.0]),
            frac_to_cart=cell,
        )
    with pytest.raises(ValueError, match="criterion"):
        BondLengthPenalty(
            pairs=torch.tensor([[0, 1]]),
            target_angstrom=torch.tensor([1.0]),
            sigma_angstrom=torch.tensor([0.1]),
            frac_to_cart=cell,
            criterion="not-a-criterion",  # type: ignore[arg-type]
        )
