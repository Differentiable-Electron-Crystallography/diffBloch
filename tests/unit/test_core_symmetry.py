from pathlib import Path

import numpy as np
import pytest
import torch

from diffBloch.core import build_asu_expansion_plan, expand_asu
from diffBloch.io import read_structure

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"


def test_quartz_expansion_plan_pins_membership_order() -> None:
    record = read_structure(FIXTURE_ROOT / "enantiomer_1.cif")

    plan = build_asu_expansion_plan(record.frac_positions, record.symops_R, record.symops_t)

    assert plan.n_asu_sites == 2
    assert plan.n_expanded_sites == 9
    assert plan.asu_indices.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert plan.symop_indices.tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 5]


def test_expand_asu_matches_quartz_numpy_symops_and_preserves_gradients() -> None:
    record = read_structure(FIXTURE_ROOT / "enantiomer_1.cif")
    plan = build_asu_expansion_plan(record.frac_positions, record.symops_R, record.symops_t)
    positions = torch.tensor(record.frac_positions, dtype=torch.float64, requires_grad=True)
    uij = torch.tensor(record.uij_cif, dtype=torch.float64, requires_grad=True)
    numbers = torch.tensor(record.numbers, dtype=torch.int64)
    occupancies = torch.tensor(record.occupancies, dtype=torch.float64)

    expanded = expand_asu(plan, positions, numbers=numbers, uij=uij, occupancies=occupancies)
    assert expanded.uij is not None
    loss = expanded.positions.sum() + expanded.uij.sum()
    loss.backward()

    expected_positions = np.remainder(
        np.einsum(
            "mij,mj->mi",
            record.symops_R[plan.symop_indices.numpy()],
            record.frac_positions[plan.asu_indices.numpy()],
        )
        + record.symops_t[plan.symop_indices.numpy()],
        1.0,
    )
    assert torch.allclose(
        expanded.positions,
        torch.tensor(expected_positions, dtype=torch.float64),
        atol=1e-12,
    )
    assert expanded.numbers is not None
    assert expanded.numbers.tolist() == [14, 14, 14, 8, 8, 8, 8, 8, 8]
    assert expanded.occupancies is not None
    assert expanded.occupancies.tolist() == pytest.approx([1.0] * 9)
    assert expanded.uij is not None
    assert expanded.uij.shape == (9, 3, 3)
    assert positions.grad is not None
    assert torch.any(positions.grad != 0)
    assert uij.grad is not None
    assert torch.any(uij.grad != 0)


def test_expand_asu_rotates_uij_matrices() -> None:
    plan = build_asu_expansion_plan(
        np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        np.asarray(
            [
                np.eye(3),
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float64,
        ),
        np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64),
    )
    uij = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]])

    expanded = expand_asu(plan, torch.tensor([[0.1, 0.2, 0.3]]), uij=uij)

    assert expanded.uij is not None
    assert torch.allclose(expanded.uij[0], uij[0])
    assert torch.allclose(
        expanded.uij[1],
        torch.tensor([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]]),
    )


def test_build_asu_expansion_plan_rejects_duplicate_atoms_by_default() -> None:
    positions = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    rotations = np.eye(3, dtype=np.float64)[None, :, :]
    translations = np.zeros((1, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="equivalent"):
        build_asu_expansion_plan(positions, rotations, translations)

    plan = build_asu_expansion_plan(positions, rotations, translations, onduplicates="keep")
    assert plan.asu_indices.tolist() == [0]


def test_expand_asu_validates_input_shapes() -> None:
    plan = build_asu_expansion_plan(
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        np.eye(3, dtype=np.float64)[None, :, :],
        np.zeros((1, 3), dtype=np.float64),
    )

    with pytest.raises(ValueError, match="positions"):
        expand_asu(plan, torch.zeros((1, 2), dtype=torch.float64))
    with pytest.raises(ValueError, match="numbers"):
        expand_asu(plan, torch.zeros((1, 3), dtype=torch.float64), numbers=torch.ones(2))
