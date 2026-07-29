"""JAX-style refinement model wrapper parity.

Phase 1 introduced ``RefinementModel`` / ``StructureComponent`` as a no-behavior-change wrapper.
These checks pin that the wrapper delegates to the existing physical structure path, including a
quartz fixture objective comparison so refactors cannot silently alter the underlying physics model.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
from tests.unit.test_engine import _DummyPenalty, _engine, _params

from diffBloch.config import load_experiment
from diffBloch.engine import (
    HydrogenRiding,
    RefinementModel,
    StructureComponent,
    build_refinement_model,
)
from diffBloch.io import read_observations, read_structure
from diffBloch.preprocess import from_experiment, read_plan
from diffBloch.preprocess.scoring import build_engine

_QUARTZ_ROOT = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"


def test_physical_structure_wraps_existing_refinement_inputs() -> None:
    params = _params()
    structure = StructureComponent(initial=params)
    model = RefinementModel(structure=structure)

    assert model.structure.initial is params
    assert model.structure.constraints == ()
    assert model.components == ()


def test_build_refinement_model_is_structure_only_adapter() -> None:
    params = _params()

    model = build_refinement_model(initial=params)

    assert model == RefinementModel(structure=StructureComponent(initial=params))


def test_objective_value_model_matches_legacy_objective_with_penalty() -> None:
    engine = _engine()
    params = _params(asu_positions=torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64))
    penalty = _DummyPenalty()

    legacy = engine.objective_value(params, penalties=(penalty,))
    model = build_refinement_model(initial=params)
    wrapped = engine.objective_value_model(model, penalties=(penalty,))

    assert torch.equal(wrapped.total, legacy.total)
    assert tuple(wrapped.components) == tuple(legacy.components)
    for name in legacy.components:
        assert torch.equal(wrapped.components[name].raw, legacy.components[name].raw)
        assert wrapped.components[name].weight == legacy.components[name].weight


def test_refinement_model_path_matches_legacy_hydrogen_riding_objective() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=torch.float64)
    engine = _engine(
        asu_positions=positions.numpy(), numbers=torch.tensor([6, 1], dtype=torch.int64)
    )
    params = _params(asu_positions=positions)
    riding = HydrogenRiding(
        name="hydrogen_riding",
        h_index=torch.tensor([1], dtype=torch.int64),
        parent_index=torch.tensor([0], dtype=torch.int64),
        offset=torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float64),
        u_iso_scale=torch.tensor([1.2], dtype=torch.float64),
    )

    legacy = engine.objective_value(params, constraints=(riding,))
    wrapped = engine.objective_value_model(
        build_refinement_model(initial=params, constraints=(riding,))
    )

    assert torch.allclose(wrapped.total, legacy.total, rtol=0.0, atol=0.0)
    assert torch.allclose(
        wrapped.components["diffraction"].raw,
        legacy.components["diffraction"].raw,
        rtol=0.0,
        atol=0.0,
    )


def test_quartz_refinement_model_path_matches_legacy_objective() -> None:
    """The model wrapper must not perturb the quartz Bloch physics path.

    Use the committed quartz anchor ``plan.npz`` and the first settled orientation as a fast unit
    guard. Both paths consume the same engine/static context and the same raw parameters; any drift
    here indicates the refactor changed objective plumbing rather than only API shape.
    """
    cfg, _lock = load_experiment(_QUARTZ_ROOT)
    structure = read_structure(_QUARTZ_ROOT / cfg.inputs.structure)
    observations = read_observations(_QUARTZ_ROOT / cfg.inputs.exp_data)
    setup = from_experiment(structure, observations, cfg)
    plan = read_plan(_QUARTZ_ROOT / "plan.npz")
    engine = build_engine(
        plan,
        setup.refinement,
        loss=cfg.refinement.objective.to_loss(),
        method=cfg.blochwave.solver.refine,
    )
    engine = dataclasses.replace(engine, orientations=engine.orientations[:1])
    params = setup.refinement.params

    legacy = engine.objective_value(params)
    wrapped = engine.objective_value_model(build_refinement_model(initial=params))

    assert torch.allclose(wrapped.total, legacy.total, rtol=0.0, atol=0.0)
    assert torch.allclose(
        wrapped.components["diffraction"].raw,
        legacy.components["diffraction"].raw,
        rtol=0.0,
        atol=0.0,
    )
