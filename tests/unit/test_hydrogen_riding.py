"""Hydrogen riding: the constant-offset ConstraintTransform and its perception builder."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tests.unit.test_engine import _engine, _params
from tests.unit.test_engine_penalties import _state, _structure

from diffBloch.engine import (
    AtomSelection,
    RefinementProblem,
    TrainableSpec,
    build_refinement_model,
    run_refinement_model,
)
from diffBloch.engine.constraints import HydrogenRiding, perceive_hydrogen_riding


def _riding(offset: list[list[float]], scale: float = 1.2) -> HydrogenRiding:
    """A one-hydrogen riding: row 1 rides parent row 0 with the given fractional offset."""
    return HydrogenRiding(
        name="hydrogen_riding",
        h_index=torch.tensor([1], dtype=torch.int64),
        parent_index=torch.tensor([0], dtype=torch.int64),
        offset=torch.tensor(offset, dtype=torch.float64),
        u_iso_scale=torch.tensor([scale], dtype=torch.float64),
    )


# --- the transform ---


def test_riding_derives_hydrogen_from_parent_plus_offset() -> None:
    state = _state(torch.tensor([[0.1, 0.0, 0.0], [0.9, 0.9, 0.9]], dtype=torch.float64))
    out = _riding([[0.4, 0.5, 0.5]]).apply(state)
    assert torch.allclose(out.positions[1], torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64))
    assert torch.allclose(out.positions[0], state.positions[0])  # parent (heavy) untouched


def test_riding_tracks_parent_translation() -> None:
    riding = _riding([[0.4, 0.0, 0.0]])
    base = riding.apply(
        _state(torch.tensor([[0.1, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64))
    )
    moved = riding.apply(
        _state(torch.tensor([[0.3, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64))
    )
    # parent shifted +0.2 in x; the hydrogen follows by the same amount
    assert torch.allclose(
        moved.positions[1] - base.positions[1], torch.tensor([0.2, 0.0, 0.0], dtype=torch.float64)
    )


def test_riding_gradient_reaches_parent_not_hydrogen() -> None:
    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    out = _riding([[0.4, 0.0, 0.0]]).apply(_state(positions))
    out.positions[1].sum().backward()
    assert positions.grad is not None
    assert positions.grad[0].abs().sum() > 0.0  # gradient flows to the parent
    assert torch.all(positions.grad[1] == 0.0)  # the overwritten hydrogen row carries none


def test_riding_scales_hydrogen_uiso_from_parent() -> None:
    state = _state(torch.tensor([[0.1, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64))
    out = _riding([[0.4, 0.0, 0.0]], scale=1.2).apply(state)
    assert torch.allclose(out.uij_star[1], 1.2 * state.uij_star[0])


def test_riding_leaves_heavy_rows_untouched() -> None:
    state = _state(
        torch.tensor([[0.1, 0.0, 0.0], [0.5, 0.0, 0.0], [0.7, 0.2, 0.0]], dtype=torch.float64)
    )
    # H is row 1 riding parent row 0; row 2 is another heavy atom
    riding = HydrogenRiding(
        name="hydrogen_riding",
        h_index=torch.tensor([1], dtype=torch.int64),
        parent_index=torch.tensor([0], dtype=torch.int64),
        offset=torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float64),
        u_iso_scale=torch.tensor([1.2], dtype=torch.float64),
    )
    out = riding.apply(state)
    assert torch.equal(out.positions[2], state.positions[2])
    assert torch.equal(out.uij_star[2], state.uij_star[2])


def test_riding_rejects_hydrogen_that_is_its_own_parent() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        HydrogenRiding(
            name="bad",
            h_index=torch.tensor([0], dtype=torch.int64),
            parent_index=torch.tensor([0], dtype=torch.int64),
            offset=torch.zeros((1, 3), dtype=torch.float64),
            u_iso_scale=torch.tensor([1.2], dtype=torch.float64),
        )


# --- perception ---


def test_perceive_identifies_bonded_heavy_parent() -> None:
    # C at origin, H ~1.09 A away, a decoy O out of H's covalent range
    structure = _structure(
        positions=[[0.0, 0.0, 0.0], [0.109, 0.0, 0.0], [0.3, 0.0, 0.0]],
        numbers=[6, 1, 8],
        cell_scale=10.0,
    )
    riding = perceive_hydrogen_riding(structure)
    assert riding is not None
    assert riding.h_index.tolist() == [1]
    assert riding.parent_index.tolist() == [0]
    assert torch.allclose(riding.u_iso_scale, torch.tensor([1.2], dtype=torch.float64))


def test_perceive_offset_is_the_fractional_parent_to_h_vector() -> None:
    structure = _structure(
        positions=[[0.0, 0.0, 0.0], [0.109, 0.0, 0.0]], numbers=[6, 1], cell_scale=10.0
    )
    riding = perceive_hydrogen_riding(structure)
    assert riding is not None
    assert torch.allclose(riding.offset[0], torch.tensor([0.109, 0.0, 0.0], dtype=torch.float64))


def test_perceive_picks_nearest_of_two_heavy_neighbours() -> None:
    # H (row 2) at 0.8 A from C0 and 0.7 A from C1 -- both bonded, C1 is nearer
    structure = _structure(
        positions=[[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.08, 0.0, 0.0]],
        numbers=[6, 6, 1],
        cell_scale=10.0,
    )
    riding = perceive_hydrogen_riding(structure)
    assert riding is not None
    assert riding.parent_index.tolist() == [1]


def test_perceive_raises_on_isolated_hydrogen() -> None:
    structure = _structure(
        positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], numbers=[6, 1], cell_scale=10.0
    )  # H is 5 A from the only heavy atom -- far outside the covalent cutoff
    with pytest.raises(ValueError, match="no covalently-bonded heavy"):
        perceive_hydrogen_riding(structure)


def test_perceive_returns_none_without_hydrogen() -> None:
    structure = _structure(positions=[[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]], numbers=[6, 8])
    assert perceive_hydrogen_riding(structure) is None


def test_perceive_rejects_special_position_hydrogen() -> None:
    # inversion symmetry fixes the origin, so an H at (0,0,0) is on a special position
    base = _structure(positions=[[0.0, 0.0, 0.0], [0.109, 0.0, 0.0]], numbers=[1, 6])
    special = base.model_copy(
        update={
            "symops_R": np.stack([np.eye(3), -np.eye(3)]),
            "symops_t": np.zeros((2, 3), dtype=np.float64),
        }
    )
    with pytest.raises(ValueError, match="special position"):
        perceive_hydrogen_riding(special)


# --- index validation (hand-constructed specs must fail at construction) ---


def _valid_kwargs() -> dict:
    return {
        "name": "hydrogen_riding",
        "h_index": torch.tensor([1], dtype=torch.int64),
        "parent_index": torch.tensor([0], dtype=torch.int64),
        "offset": torch.zeros((1, 3), dtype=torch.float64),
        "u_iso_scale": torch.tensor([1.2], dtype=torch.float64),
    }


def test_rejects_non_integer_index() -> None:
    with pytest.raises(ValueError, match="int64"):
        HydrogenRiding(**{**_valid_kwargs(), "h_index": torch.tensor([1.0])})


def test_rejects_duplicate_hydrogen_index() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        HydrogenRiding(
            **{
                **_valid_kwargs(),
                "h_index": torch.tensor([1, 1], dtype=torch.int64),
                "parent_index": torch.tensor([0, 2], dtype=torch.int64),
                "offset": torch.zeros((2, 3), dtype=torch.float64),
                "u_iso_scale": torch.tensor([1.2, 1.2], dtype=torch.float64),
            }
        )


def test_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        HydrogenRiding(**{**_valid_kwargs(), "parent_index": torch.tensor([-1], dtype=torch.int64)})


def test_riding_runs_through_run_refinement_model() -> None:
    # a 2-atom C + H engine: riding must thread through the real optimizer shell without error
    positions = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float64)
    engine = _engine(asu_positions=positions, numbers=torch.tensor([6, 1], dtype=torch.int64))
    model = build_refinement_model(
        initial=_params(asu_positions=torch.tensor(positions)),
        constraints=(_riding([[0.2, 0.0, 0.0]]),),
    )
    trainable = TrainableSpec(
        positions=AtomSelection.exclude_elements("H"), adp=AtomSelection.exclude_elements("H")
    )
    problem = RefinementProblem()
    result = run_refinement_model(
        engine, model, problem, trainable=trainable, steps=2, optimizer="adam", lr=1e-3
    )
    assert result.losses.shape == (2,)
    assert torch.isfinite(result.losses).all()
