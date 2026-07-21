# %% [markdown]
# # Malik et al. 2026 — quartz ThicknessNN public API example
#
# A compact public `diffBloch` script: preprocess quartz, build a refinement engine, compose the
# structure component plus `ApparentThicknessNN`, and run a CPU-safe refinement evaluation.

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from diffBloch.config import ExperimentConfig, load_experiment
from diffBloch.engine import (
    ApparentThicknessNN,
    AtomSelection,
    ModelRefinementResult,
    RefinementEngine,
    ThicknessBounds,
    TrainableSpec,
    build_refinement_model,
    build_refinement_problem,
    run_refinement_model,
)
from diffBloch.engine.plan import mean_plan_thickness
from diffBloch.io import read_observations, read_structure
from diffBloch.preprocess import (
    build_orientation_plans,
    fit_orientation,
    from_experiment,
    integrate_rocking_curve,
    mosaicity,
    pipeline,
    select_beams,
    select_finite_loss_frames,
)
from diffBloch.preprocess.scoring import build_engine
from diffBloch.specs import (
    BeamSelection,
    HexagonalSearch,
    IntegrationGeometry,
    Mosaicity,
    RockingCurve,
    ScoredSelection,
    TiltSegmentUnion,
    TrialCoupling,
)


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
    setup = from_experiment(structure, observations, cfg)
    refinement = setup.refinement
    integration = IntegrationGeometry(semiangle=1.0)
    beam_selection = BeamSelection(rsg=0.9, dsg=0.0015, integration=integration)
    orientation_search = HexagonalSearch(
        max_search_angle=0.4,
        min_search_angle=0.001,
        n_steps=6,
        max_iterations=2000,
    )
    coupling = TrialCoupling(
        policy=TiltSegmentUnion(n_splits=12, g_max=1.0, cap_margin=0.2, sg_max=0.01),
        scored=ScoredSelection(klar=beam_selection, g_max=1.0),
    )
    plan = pipeline(
        [
            select_beams(beam_selection),
            build_orientation_plans(),
            integrate_rocking_curve(RockingCurve(sampling=42, integration=integration)),
            mosaicity(Mosaicity(window=5)),
            fit_orientation(
                refinement,
                orientation_search,
                method=cfg.solver.refine,
                coupling=coupling,
                device=device,
            ),
            select_finite_loss_frames(
                refinement,
                loss=cfg.refinement.objective.to_loss(),
                method=cfg.solver.refine,
            ),
        ]
    )(setup.plans.combined)
    engine = build_engine(
        plan,
        refinement,
        loss=cfg.refinement.objective.to_loss(),
        method=cfg.solver.refine,
    )
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


def _summary(
    *,
    cfg: ExperimentConfig,
    n_atoms: int,
    n_observed_rotations: int,
    engine: RefinementEngine,
    model: Any,
    problem: Any,
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
