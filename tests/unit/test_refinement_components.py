"""Refinement model component vocabulary and validation."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import pytest
import torch
from tests.unit.test_engine import _engine, _observed_pattern, _params
from torch import Tensor

from diffBloch.engine import (
    ApparentThicknessNN,
    AtomSelection,
    ForwardContext,
    PerOrientationThickness,
    QuadraticThicknessProfile,
    RefinementProblem,
    ThicknessBounds,
    TrainableSpec,
    build_refinement_model,
    mean_plan_thickness,
    run_refinement_model,
)
from diffBloch.engine.plan import OrientationPlanLike


@dataclasses.dataclass(frozen=True)
class _DummyComponent:
    key: str = "dummy"

    def initial_params(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Mapping[str, Tensor]:
        return {"value": torch.zeros((), dtype=dtype, device=device)}

    def forward_context(
        self,
        params: Mapping[str, Tensor],
        *,
        rotation_index: int,
        orientation: OrientationPlanLike,
    ) -> ForwardContext:
        _ = rotation_index, orientation
        return ForwardContext(thickness=params["value"].reshape(1))


@dataclasses.dataclass(frozen=True)
class _NoopComponent:
    key: str = "noop"

    def initial_params(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Mapping[str, Tensor]:
        _ = dtype, device
        return {}

    def forward_context(
        self,
        params: Mapping[str, Tensor],
        *,
        rotation_index: int,
        orientation: OrientationPlanLike,
    ) -> ForwardContext:
        _ = params, rotation_index, orientation
        return ForwardContext()


def test_forward_context_is_thickness_only_for_now() -> None:
    context = ForwardContext(thickness=torch.tensor([300.0], dtype=torch.float64))

    assert torch.equal(context.thickness, torch.tensor([300.0], dtype=torch.float64))
    assert not hasattr(context, "scale")
    assert not hasattr(context, "background")


def test_refinement_model_rejects_duplicate_component_keys() -> None:
    with pytest.raises(ValueError, match="duplicate refinement component key"):
        build_refinement_model(
            initial=_params(), components=(_DummyComponent("dup"), _DummyComponent("dup"))
        )


def test_refinement_model_rejects_component_params_without_component() -> None:
    with pytest.raises(ValueError, match="component_params has no matching component"):
        build_refinement_model(
            initial=_params(),
            component_params={"missing": {"value": torch.zeros((), dtype=torch.float64)}},
        )


def test_refinement_model_records_component_params_as_read_only_mapping() -> None:
    value = torch.tensor(1.0, dtype=torch.float64)
    model = build_refinement_model(
        initial=_params(),
        components=(_DummyComponent(),),
        component_params={"dummy": {"value": value}},
    )

    assert model.components[0].key == "dummy"
    assert model.component_params["dummy"]["value"] is value
    with pytest.raises(TypeError):
        model.component_params["dummy"]["other"] = value  # type: ignore[index]
    with pytest.raises(TypeError):
        model.component_params["other"] = {"value": value}  # type: ignore[index]


def test_apparent_thickness_nn_validates_legacy_settings() -> None:
    with pytest.raises(ValueError, match="num_samples"):
        ApparentThicknessNN(
            bounds=ThicknessBounds(400.0, 1100.0), normalized_alphas=(0.0,), num_samples=0
        )
    with pytest.raises(ValueError, match="form"):
        ApparentThicknessNN(
            bounds=ThicknessBounds(400.0, 1100.0),
            normalized_alphas=(0.0,),
            form="quadratic",  # type: ignore[arg-type]
        )


def test_apparent_thickness_nn_can_be_scoped_to_one_dataset() -> None:
    engine = _engine()
    component = ApparentThicknessNN(
        bounds=ThicknessBounds(200.0, 800.0),
        normalized_alphas=(-1.0, 1.0),
        rotation_range=(5, 7),
        key="apparent_thickness[a.cif_pets]",
        label="a.cif_pets",
    )
    params = component.initial_params(dtype=torch.float64, device=torch.device("cpu"))
    inside = dataclasses.replace(
        engine.orientations[0],
        pattern=dataclasses.replace(engine.orientations[0].pattern, rotation_index=5),
    )
    outside = dataclasses.replace(
        engine.orientations[0],
        pattern=dataclasses.replace(engine.orientations[0].pattern, rotation_index=2),
    )

    assert (
        component.forward_context(params, rotation_index=5, orientation=inside).thickness
        is not None
    )
    assert (
        component.forward_context(params, rotation_index=2, orientation=outside).thickness is None
    )


def test_apparent_thickness_nn_rejects_mismatched_dataset_range() -> None:
    with pytest.raises(ValueError, match="rotation_range width"):
        ApparentThicknessNN(
            bounds=ThicknessBounds(200.0, 800.0),
            normalized_alphas=(0.0,),
            rotation_range=(5, 7),
        )


def test_apparent_thickness_nn_legacy_mean_is_differentiable() -> None:
    engine = _engine()
    component = ApparentThicknessNN(bounds=ThicknessBounds(200.0, 800.0), normalized_alphas=(0.0,))
    params = component.initial_params(
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    leaves = {name: value.detach().clone().requires_grad_(True) for name, value in params.items()}

    context = component.forward_context(
        leaves, rotation_index=0, orientation=engine.orientations[0]
    )
    assert context.thickness is not None
    context.thickness.sum().backward()

    for name in leaves:
        assert leaves[name].grad is not None
        assert torch.isfinite(leaves[name].grad).all()
    assert leaves["layer2.bias"].grad[0].abs() > 0
    assert leaves["layer2.bias"].grad[1] == 0


def test_apparent_thickness_nn_legacy_gaussian_sampling_is_positive_and_deterministic() -> None:
    engine = _engine()
    component = ApparentThicknessNN(
        bounds=ThicknessBounds(100.0, 3500.0),
        normalized_alphas=(0.25,),
        sample_thickness=True,
        num_samples=7,
        init_seed=3,
    )
    params = component.initial_params(dtype=torch.float64, device=torch.device("cpu"))
    leaves = {name: value.detach().clone().requires_grad_(True) for name, value in params.items()}

    first = component.forward_context(
        leaves, rotation_index=0, orientation=engine.orientations[0]
    ).thickness
    second = component.forward_context(
        leaves, rotation_index=0, orientation=engine.orientations[0]
    ).thickness

    assert first is not None and second is not None
    assert first.shape == (7,)
    assert bool((first > 0.0).all())
    assert torch.equal(first, second)
    first.sum().backward()
    assert leaves["layer2.bias"].grad is not None
    assert bool((leaves["layer2.bias"].grad.abs() > 0.0).all())


def test_run_refinement_model_optimizes_apparent_thickness_nn_params() -> None:
    engine = _engine()
    structure_params = _params()
    component = ApparentThicknessNN(bounds=ThicknessBounds(200.0, 800.0), normalized_alphas=(0.0,))
    component_params = component.initial_params(
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    model = build_refinement_model(
        initial=structure_params,
        components=(component,),
        component_params={component.key: component_params},
    )

    result = run_refinement_model(
        engine,
        model,
        RefinementProblem(),
        trainable=TrainableSpec.positions_and_adp(),
        steps=2,
        optimizer="adam",
        lr=1e-3,
    )

    final_bias = result.model.component_params[component.key]["layer2.bias"]
    assert not final_bias.requires_grad
    assert not torch.equal(final_bias, component_params["layer2.bias"])


def test_thickness_bounds_reject_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="min_angstrom must be positive"):
        ThicknessBounds(0.0, 100.0)
    with pytest.raises(ValueError, match="max_angstrom"):
        ThicknessBounds(100.0, 100.0)


def test_thickness_bounds_transform_is_bounded() -> None:
    bounds = ThicknessBounds(400.0, 1100.0)
    unconstrained = torch.tensor([-100.0, 0.0, 100.0], dtype=torch.float64)

    thickness = bounds.transform(unconstrained)

    assert bool((thickness >= 400.0).all())
    assert bool((thickness <= 1100.0).all())
    assert torch.allclose(thickness[1], torch.tensor(750.0, dtype=torch.float64))


def test_thickness_bounds_inverse_round_trips_interior_values() -> None:
    bounds = ThicknessBounds(400.0, 1100.0)
    thickness = torch.tensor([500.0, 750.0, 1000.0], dtype=torch.float64)

    unconstrained = bounds.inverse(thickness)

    assert torch.allclose(bounds.transform(unconstrained), thickness)


def test_mean_plan_thickness_reads_settled_orientation_thickness() -> None:
    engine = _engine()

    assert torch.allclose(
        mean_plan_thickness(engine.orientations), engine.orientations[0].thickness.mean()
    )


def test_quadratic_thickness_profile_seeds_from_initial_thickness_and_is_differentiable() -> None:
    engine = _engine()
    component = QuadraticThicknessProfile(bounds=ThicknessBounds(200.0, 800.0))
    params = component.initial_params(
        dtype=torch.float64,
        device=torch.device("cpu"),
        initial_thickness=mean_plan_thickness(engine.orientations),
    )
    coefficients = params["coefficients"].detach().clone().requires_grad_(True)

    context = component.forward_context(
        {"coefficients": coefficients}, rotation_index=0, orientation=engine.orientations[0]
    )
    assert context.thickness is not None
    assert torch.allclose(context.thickness, engine.orientations[0].thickness)
    context.thickness.sum().backward()

    assert bool((context.thickness >= 200.0).all())
    assert bool((context.thickness <= 800.0).all())
    assert coefficients.grad is not None
    assert torch.isfinite(coefficients.grad).all()


def test_bounded_quadratic_component_can_seed_denovo_at_bounds_midpoint() -> None:
    engine = _engine()
    bounds = ThicknessBounds(200.0, 800.0)

    profile = QuadraticThicknessProfile(bounds=bounds)
    profile_params = profile.initial_params(dtype=torch.float64, device=torch.device("cpu"))
    profile_context = profile.forward_context(
        profile_params, rotation_index=0, orientation=engine.orientations[0]
    )

    assert profile_context.thickness is not None
    assert torch.allclose(profile_context.thickness, torch.tensor([500.0], dtype=torch.float64))


def test_bounded_quadratic_component_rejects_initial_thickness_outside_bounds() -> None:
    engine = _engine()
    initial_thickness = mean_plan_thickness(engine.orientations)
    profile = QuadraticThicknessProfile(bounds=ThicknessBounds(400.0, 800.0))
    with pytest.raises(ValueError, match="strictly inside ThicknessBounds"):
        profile.initial_params(
            dtype=torch.float64,
            device=torch.device("cpu"),
            initial_thickness=initial_thickness,
        )


def test_per_orientation_thickness_seeds_from_plan() -> None:
    engine = _engine()
    component = PerOrientationThickness()

    params = component.initial_params(
        plan=engine.orientations, dtype=torch.float64, device=torch.device("cpu")
    )
    context = component.forward_context(
        params,
        rotation_index=0,
        orientation=engine.orientations[0],
    )

    assert context.thickness is not None
    assert torch.allclose(context.thickness, engine.orientations[0].thickness)


def test_per_orientation_thickness_overrides_plan_thickness_in_model_objective() -> None:
    engine = _engine()
    structure_params = _params()
    component = PerOrientationThickness()
    params = component.initial_params(
        plan=engine.orientations, dtype=torch.float64, device=torch.device("cpu")
    )
    changed = {component.key: {"unconstrained": params["unconstrained"].clone()}}
    changed[component.key]["unconstrained"][0, 0] = torch.log(
        torch.expm1(torch.tensor(500.0, dtype=torch.float64))
    )

    baseline = engine.objective_value(structure_params)
    model = build_refinement_model(
        initial=structure_params, components=(component,), component_params=changed
    )
    overridden = engine.objective_value_model(model)

    assert not torch.equal(overridden.total, baseline.total)


def test_per_orientation_thickness_component_receives_gradients() -> None:
    engine = _engine()
    structure_params = _params()
    component = PerOrientationThickness()
    params = component.initial_params(
        plan=engine.orientations, dtype=torch.float64, device=torch.device("cpu")
    )
    leaf = params["unconstrained"].detach().clone().requires_grad_(True)
    model = build_refinement_model(
        initial=structure_params,
        components=(component,),
        component_params={component.key: {"unconstrained": leaf}},
    )

    objective = engine.objective_value_model(model)
    objective.total.backward()

    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad).all()
    assert leaf.grad.abs().sum() > 0


def test_run_refinement_model_returns_optimized_component_params() -> None:
    true_params = _params(asu_positions=torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float64))
    engine = _engine(pattern=_observed_pattern(true_params))
    structure_params = _params(asu_positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64))
    component = PerOrientationThickness()
    params = component.initial_params(
        plan=engine.orientations, dtype=torch.float64, device=torch.device("cpu")
    )
    params["unconstrained"][0, 0] = torch.log(torch.expm1(torch.tensor(450.0, dtype=torch.float64)))
    caller_params_before = params["unconstrained"].clone()
    model = build_refinement_model(
        initial=structure_params, components=(component,), component_params={component.key: params}
    )

    result = run_refinement_model(
        engine,
        model,
        RefinementProblem(),
        trainable=TrainableSpec(positions=AtomSelection.all()),
        steps=3,
        optimizer="adam",
        lr=0.1,
    )

    assert result.losses[-1] < result.losses[0]
    assert result.model is not model
    assert component.key in result.model.component_params
    final_params = result.model.component_params[component.key]["unconstrained"]
    assert not final_params.requires_grad
    assert not torch.equal(final_params, params["unconstrained"])
    assert torch.equal(params["unconstrained"], caller_params_before)


def test_structure_only_model_result_keeps_params_compatibility_accessors() -> None:
    engine = _engine()
    params = _params()
    result = run_refinement_model(
        engine,
        build_refinement_model(initial=params),
        RefinementProblem(),
        trainable=TrainableSpec.positions_and_adp(),
        steps=1,
        optimizer="adam",
        lr=1e-3,
    )

    assert result.params is result.model.structure.initial
    assert result.best_params is result.best_model.structure.initial
    assert result.best_loss == float(result.losses[result.best_step])


def test_component_model_with_no_forward_values_matches_legacy_objective_exactly() -> None:
    engine = _engine()
    params = _params()
    model = build_refinement_model(initial=params, components=(_NoopComponent(),))

    legacy = engine.objective_value(params)
    wrapped = engine.objective_value_model(model)

    assert torch.equal(wrapped.total, legacy.total)
    assert torch.equal(wrapped.components["diffraction"].raw, legacy.components["diffraction"].raw)


def test_structure_only_model_keeps_quartz_objective_exactly_unchanged() -> None:
    engine = _engine()
    params = _params()

    legacy = engine.objective_value(params)
    wrapped = engine.objective_value_model(build_refinement_model(initial=params))

    assert torch.allclose(wrapped.total, legacy.total, rtol=0.0, atol=0.0)
