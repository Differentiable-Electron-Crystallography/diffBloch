"""Hard constraint transforms (:class:`ConstraintTransform`) in the refinement objective."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from tests.unit.test_engine import _engine, _params

from diffBloch.engine import (
    RefinementProblem,
    TrainableSpec,
    build_refinement_model,
    run_refinement_model,
)
from diffBloch.params import PhysicalState


@dataclasses.dataclass(frozen=True)
class _ShiftX:
    """A dummy hard constraint: add a fixed fractional amount to every atom's x coordinate."""

    name: str = "shift_x"
    dx: float = 0.1

    def apply(self, state: PhysicalState) -> PhysicalState:
        shifted = state.positions.clone()
        shifted[:, 0] = shifted[:, 0] + self.dx
        return dataclasses.replace(state, positions=shifted)


@dataclasses.dataclass(frozen=True)
class _ScaleX:
    """A dummy hard constraint: multiply every atom's x coordinate by a fixed factor.

    Non-commuting with :class:`_ShiftX` -- composing add-then-scale differs from scale-then-add,
    so it distinguishes tuple *order* from mere composition.
    """

    name: str = "scale_x"
    factor: float = 2.0

    def apply(self, state: PhysicalState) -> PhysicalState:
        scaled = state.positions.clone()
        scaled[:, 0] = scaled[:, 0] * self.factor
        return dataclasses.replace(state, positions=scaled)


@dataclasses.dataclass(frozen=True)
class _ProbeX:
    """A penalty whose raw loss is the current x-position sum -- reports the state it is handed."""

    name: str = "probe"
    weight: float = 1.0

    def loss(self, state: PhysicalState) -> torch.Tensor:
        return state.positions[:, 0].sum()


def test_constraint_transforms_state_before_diffraction() -> None:
    engine = _engine()
    params = _params(asu_positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64))
    base = engine.objective_value(params).total
    shifted = engine.objective_value(params, constraints=(_ShiftX(dx=0.3),)).total
    # the shift moves the atom, changing the structure factors -> the diffraction term differs
    assert not torch.equal(base, shifted)


def test_penalty_sees_the_constrained_state() -> None:
    engine = _engine()
    params = _params(asu_positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64))
    result = engine.objective_value(params, penalties=(_ProbeX(),), constraints=(_ShiftX(dx=0.25),))
    # the probe reads x AFTER the constraint runs: 0.0 (constrain, identity projector) + 0.25
    assert torch.allclose(result.components["probe"].raw, torch.tensor(0.25, dtype=torch.float64))


def test_constraints_apply_in_tuple_order() -> None:
    engine = _engine()
    params = _params(asu_positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64))

    def probed_x(*constraints: object) -> torch.Tensor:
        return (
            engine.objective_value(params, penalties=(_ProbeX(),), constraints=constraints)
            .components["probe"]
            .raw
        )

    # non-commuting transforms: (x + 1) * 2 = 2  vs  (x * 2) + 1 = 1 at x = 0
    add_then_scale = probed_x(_ShiftX(name="add", dx=1.0), _ScaleX(name="scale", factor=2.0))
    scale_then_add = probed_x(_ScaleX(name="scale", factor=2.0), _ShiftX(name="add", dx=1.0))
    assert torch.allclose(add_then_scale, torch.tensor(2.0, dtype=torch.float64))
    assert torch.allclose(scale_then_add, torch.tensor(1.0, dtype=torch.float64))


def test_duplicate_constraint_names_rejected() -> None:
    engine = _engine()
    with pytest.raises(ValueError, match="duplicate constraint name"):
        engine.objective_value(_params(), constraints=(_ShiftX(name="dup"), _ShiftX(name="dup")))


def test_no_constraints_is_a_noop() -> None:
    engine = _engine()
    params = _params(asu_positions=torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float64))
    assert torch.equal(
        engine.objective_value(params).total,
        engine.objective_value(params, constraints=()).total,
    )


def test_physical_structure_records_constraints() -> None:
    constraint = _ShiftX()
    model = build_refinement_model(initial=_params(), constraints=(constraint,))
    assert model.structure.constraints == (constraint,)


def test_run_refinement_model_threads_constraints_into_the_objective() -> None:
    engine = _engine()
    applied: list[str] = []

    @dataclasses.dataclass(frozen=True)
    class _Spy:
        name: str = "spy"

        def apply(self, state: PhysicalState) -> PhysicalState:
            applied.append(self.name)
            return state

    model = build_refinement_model(initial=_params(), constraints=(_Spy(),))
    problem = RefinementProblem()
    run_refinement_model(
        engine,
        model,
        problem,
        trainable=TrainableSpec.positions_and_adp(),
        steps=1,
        optimizer="adam",
        lr=1e-3,
    )
    assert applied  # the executor invoked the constraint's apply during optimization
