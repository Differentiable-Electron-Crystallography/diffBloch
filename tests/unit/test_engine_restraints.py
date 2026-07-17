"""Soft refinement restraint terms."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diffBloch.engine import BondRestraints, perceive_bond_restraints
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


def test_perceive_bond_restraints_uses_current_heavy_atom_distances_as_targets() -> None:
    structure = _structure(
        positions=[[0.0, 0.0, 0.0], [0.14, 0.0, 0.0], [0.5, 0.0, 0.0]],
        numbers=[6, 6, 6],
    )

    restraints = perceive_bond_restraints(
        structure, sigma_angstrom=0.03, criterion="flat_bottom_l1"
    )

    assert torch.equal(restraints.pairs, torch.tensor([[0, 1]], dtype=torch.int64))
    assert torch.allclose(restraints.target_angstrom, torch.tensor([1.4], dtype=torch.float64))
    assert torch.equal(restraints.sigma_angstrom, torch.tensor([0.03], dtype=torch.float64))
    assert restraints.criterion == "flat_bottom_l1"
    assert torch.equal(restraints.frac_to_cart, torch.eye(3, dtype=torch.float64) * 10.0)


def test_perceive_bond_restraints_excludes_hydrogen_by_default() -> None:
    structure = _structure(positions=[[0.0, 0.0, 0.0], [0.109, 0.0, 0.0]], numbers=[6, 1])

    with pytest.raises(ValueError, match="found no bonds"):
        perceive_bond_restraints(structure)
    restraints = perceive_bond_restraints(structure, include_hydrogen=True)
    assert torch.equal(restraints.pairs, torch.tensor([[0, 1]], dtype=torch.int64))


def test_perceive_bond_restraints_reports_missing_covalent_radius() -> None:
    structure = _structure(positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], numbers=[14, 6])

    with pytest.raises(ValueError, match="no covalent radius"):
        perceive_bond_restraints(structure)


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
