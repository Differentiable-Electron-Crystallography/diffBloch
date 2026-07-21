# %% [markdown]
# # Malik et al. 2026 — quartz ThicknessNN public API example
#
# A compact public `diffBloch` script: preprocess quartz, build a refinement engine, compose the
# structure component plus `ApparentThicknessNN`, and run a CPU-safe refinement evaluation.

# %%
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from diffBloch.app.program import preprocess_experiment
from diffBloch.config import ExperimentConfig, load_experiment
from diffBloch.engine import (
    ApparentThicknessNN,
    AtomSelection,
    ModelRefinementResult,
    RefinementEngine,
    RefinementModel,
    RefinementProblem,
    ThicknessBounds,
    TrainableSpec,
    build_refinement_model,
    build_refinement_problem,
    mean_plan_thickness,
    run_refinement_model,
)
from diffBloch.io import read_observations, read_structure
from diffBloch.preprocess import Plan, RefinementSetup, from_experiment
from diffBloch.preprocess.scoring import build_engine


def run_quartz_thickness_nn_example(
    experiment_dir: Path,
    *,
    output_dir: Path,
    device: torch.device,
    steps: int,
    lr: float,
    thickness_bounds: ThicknessBounds,
) -> dict[str, Any]:
    cfg, _lock = load_experiment(experiment_dir)
    structure = read_structure(
        experiment_dir / cfg.inputs.structure,
        load_hydrogens=cfg.inputs.load_hydrogens,
    )
    observations = read_observations(experiment_dir / cfg.inputs.observations)
    refinement = from_experiment(structure, observations, cfg).refinement
    plan = preprocess_experiment(experiment_dir, checkpoint=True, refresh=False, device=device)
    plan = _finite_loss_plan(plan, refinement, cfg)
    engine = build_engine(
        plan,
        refinement,
        loss=cfg.refinement.objective.to_loss(),
        method=cfg.solver.refine,
    )
    model, problem, trainable, thickness_nn = _build_model(
        refinement,
        engine,
        device=device,
        thickness_bounds=thickness_bounds,
    )
    result = run_refinement_model(
        engine,
        model,
        problem,
        trainable=trainable,
        steps=steps,
        optimizer="adam",
        lr=lr,
    )
    summary = _summary(
        cfg=cfg,
        n_atoms=structure.n_atoms,
        n_observed_rotations=observations.n_rotations,
        engine=engine,
        model=model,
        problem=problem,
        result=result,
        thickness_nn=thickness_nn,
        steps=steps,
    )
    _write_outputs(output_dir, result, thickness_nn, engine, summary)
    return summary


def _finite_loss_plan(plan: Plan, refinement: RefinementSetup, cfg: ExperimentConfig) -> Plan:
    finite = []
    for orientation in plan.orientations:
        one = replace(plan, orientations=(orientation,))
        engine = build_engine(
            one,
            refinement,
            loss=cfg.refinement.objective.to_loss(),
            method=cfg.solver.refine,
        )
        if torch.isfinite(engine.objective_value(refinement.params).total):
            finite.append(orientation)
    if not finite:
        raise RuntimeError("quartz preprocess produced no finite-loss orientations")
    return replace(plan, orientations=tuple(finite))


def _build_model(
    refinement: RefinementSetup,
    engine: RefinementEngine,
    *,
    device: torch.device,
    thickness_bounds: ThicknessBounds,
) -> tuple[RefinementModel, RefinementProblem, TrainableSpec, ApparentThicknessNN]:
    initial = refinement.params.to(device)
    thickness_nn = ApparentThicknessNN(bounds=thickness_bounds)
    component_params = {
        thickness_nn.key: thickness_nn.initial_params(
            dtype=initial.asu_positions.dtype,
            device=device,
            initial_thickness=mean_plan_thickness(engine.orientations),
        )
    }
    model = build_refinement_model(
        initial=initial,
        components=(thickness_nn,),
        component_params=component_params,
    )
    problem = build_refinement_problem()
    trainable = TrainableSpec(positions=AtomSelection.all())
    return model, problem, trainable, thickness_nn


def _summary(
    *,
    cfg: ExperimentConfig,
    n_atoms: int,
    n_observed_rotations: int,
    engine: RefinementEngine,
    model: RefinementModel,
    problem: RefinementProblem,
    result: ModelRefinementResult,
    thickness_nn: ApparentThicknessNN,
    steps: int,
) -> dict[str, Any]:
    initial_params = model.component_params[thickness_nn.key]
    final_params = result.model.component_params[thickness_nn.key]
    changed = any(
        not torch.equal(final_params[name].cpu(), initial_params[name].cpu())
        for name in initial_params
    )
    return {
        "experiment": cfg.name,
        "steps": steps,
        "initial_loss": float(result.losses[0]),
        "final_loss": float(result.losses[-1]),
        "best_step": result.best_step,
        "component_params_changed": changed,
        "n_atoms": n_atoms,
        "n_observed_rotations": n_observed_rotations,
        "n_refinement_orientations": len(engine.orientations),
        "mean_plan_thickness_angstrom": float(mean_plan_thickness(engine.orientations)),
        "structure_component": type(model.structure).__name__,
        "components": [component.key for component in model.components],
        "constraints": [constraint.name for constraint in model.structure.constraints],
        "penalties": [penalty.name for penalty in problem.penalties],
    }


def _write_outputs(
    output_dir: Path,
    result: ModelRefinementResult,
    thickness_nn: ApparentThicknessNN,
    engine: RefinementEngine,
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "thickness_nn_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "thickness_nn_profile.json").write_text(
        json.dumps(_thickness_profile(result, thickness_nn, engine), indent=2, sort_keys=True)
        + "\n"
    )


def _thickness_profile(
    result: ModelRefinementResult,
    thickness_nn: ApparentThicknessNN,
    engine: RefinementEngine,
) -> list[dict[str, float]]:
    params = result.model.component_params[thickness_nn.key]
    rows = []
    for i, orientation in enumerate(engine.orientations):
        context = thickness_nn.forward_context(params, orientation_index=i, orientation=orientation)
        assert context.thickness is not None
        rows.append(
            {"orientation_index": float(i), "thickness_angstrom": float(context.thickness[0])}
        )
    return rows


# %%
if __name__ == "__main__" or "get_ipython" in globals():
    root = Path("examples/papers/malik-2026-hybrid-physics-ml").resolve()
    summary = run_quartz_thickness_nn_example(
        root / "experiments" / "quartz-synthetic",
        output_dir=root / "outputs" / "quartz-synthetic",
        device=torch.device("cpu"),
        steps=1,
        lr=1e-5,
        thickness_bounds=ThicknessBounds(100.0, 2000.0),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
