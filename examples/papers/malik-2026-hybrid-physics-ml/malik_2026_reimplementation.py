# %% [markdown]
# # Malik et al. 2026 — a hybrid physics + learned-thickness refinement on quartz
#
# Paper: *Hybrid physics-machine learning models for quantitative electron diffraction refinements*,
# Nature Communications (2026), DOI: https://doi.org/10.1038/s41467-026-71673-9. Data:
# https://doi.org/10.5281/zenodo.18281349.
#
# This notebook is a linear, top-to-bottom demonstration of the public `diffBloch` API. It composes
# the paper's core idea from ordinary Python values: the differentiable **Bloch physics** (the
# structure) plus a small **learned per-orientation thickness** model (`ApparentThicknessNN`),
# refined *jointly* against the observed intensities. Read it as a story: load → preprocess → build
# the engine → compose the model → refine → inspect.

# %%
from pathlib import Path

import torch

from diffBloch.config import load_experiment
from diffBloch.engine import (
    ApparentThicknessNN,
    AtomSelection,
    ThicknessBounds,
    TrainableSpec,
    build_refinement_model,
    build_refinement_problem,
    mean_plan_thickness,
    run_refinement_model,
    weighted_mse_loss,
)
from diffBloch.io import read_observations, read_structure
from diffBloch.preprocess import (
    build_engine,
    build_orientation_plans,
    fit_orientation,
    from_experiment,
    integrate_rocking_curve,
    mosaicity,
    pipeline,
    select_beams,
    select_finite_loss_frames,
)
from diffBloch.specs import ScoredHklSelection, TrialCoupling

# Anchored to this file, not the CWD: as a script __file__ resolves it; as a notebook the
# Jupyter kernel's CWD is this directory (regardless of where `jupyter lab` was launched, so a
# repo-root-relative path would break there).
HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
EXPERIMENT_DIR = HERE / "experiments/quartz-synthetic"
# CUDA when available, else CPU. Apple's MPS backend is deliberately excluded: the default
# solve format is fp64 (float64/complex128 -- see core/solver.py) and MPS has no float64.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## 1. Load the experiment
#
# `load_experiment` parses (and lock-verifies) `experiment.yaml`; then read the structure and the
# observed reflections, and assemble the static refinement inputs with `from_experiment`.

# %%
cfg, _lock = load_experiment(EXPERIMENT_DIR)
structure = read_structure(
    EXPERIMENT_DIR / cfg.inputs.structure, load_hydrogens=cfg.inputs.load_hydrogens
)
observations = read_observations(EXPERIMENT_DIR / cfg.inputs.observations)
setup = from_experiment(structure, observations, cfg)
refinement = setup.refinement

print(f"{cfg.name}: {structure.n_atoms} atoms, {observations.n_rotations} observed rotations")

# %% [markdown]
# ## 2. Preprocess — compose the pipeline explicitly
#
# diffBloch's preprocess is a composable pipeline of pure `Plan -> Plan` steps. We compose it here
# by hand so every stage is visible, drawing each step's parameters from the config's value-types
# (`cfg.numerics.to_beam_selection()`, `cfg.preprocess.orientation.to_search()`, ...): select the
# beams, build the per-orientation plans, integrate the rocking curve, apply mosaicity, fit the
# orientations under per-trial beam coupling, and drop frames with no finite refinement loss.
#
# Two deliberate choices: we omit `fit_thickness` — the `ApparentThicknessNN` below *learns*
# thickness instead — and we keep `select_finite_loss_frames` because this synthetic dataset has
# degenerate frames the faithful recipe does not drop (a recipe gap worth closing upstream).

# %%
assert cfg.preprocess.coupling is not None, "this experiment must declare preprocess.coupling"
coupling = TrialCoupling(
    policy=cfg.preprocess.coupling.to_policy(),
    scored=ScoredHklSelection(
        klar=cfg.numerics.to_beam_selection(), g_max=cfg.numerics.g_max_refine
    ),
)
plan = pipeline(
    [
        select_beams(cfg.numerics.to_beam_selection()),
        build_orientation_plans(),
        integrate_rocking_curve(cfg.numerics.to_rocking_curve()),
        mosaicity(cfg.numerics.mosaicity),
        fit_orientation(
            refinement,
            cfg.preprocess.orientation.to_search(),
            method=cfg.solver.refine,
            coupling=coupling,
            device=DEVICE,
        ),
        select_finite_loss_frames(
            refinement, loss=cfg.refinement.objective.to_loss(), method=cfg.solver.refine
        ),
    ]
)(setup.plans.combined)
print(f"orientations after preprocess: {len(plan.orientations)}")

# %% [markdown]
# ## 3. Build the refinement engine
#
# The engine pairs the settled geometry (`plan`) with the static structure context and a data-term
# loss, exposing a differentiable objective over the refinable parameters. We use the least-squares
# term (`weighted_mse_loss`) for this demo: the faithful weighted-R term currently has an unstable
# gradient through the learned-thickness path (a known limitation, being revisited), so MSE gives a
# stable joint refinement. The MSE magnitude is unnormalized — read the *relative* change and the
# learned thickness profile, not the absolute value.

# %%
engine = build_engine(plan, refinement, loss=weighted_mse_loss, method=cfg.solver.refine)

# %% [markdown]
# ## 4. Compose the refinement model — structure + a learned thickness
#
# diffBloch separates three kinds of thing, and refinement is composed from them as typed Python
# values (not config):
#
# - **components** provide forward-model values (here, `ApparentThicknessNN` predicts a per-
#   orientation apparent thickness from a small MLP, bounded by `ThicknessBounds`);
# - **constraints** enforce hard invariants on the structure (none here);
# - **penalties** add soft objective terms (none here).
#
# `build_refinement_model` bundles the initial structure parameters with the components; the NN's
# parameters are seeded around the mean plan thickness so it starts near the physics baseline.

# %%
thickness_nn = ApparentThicknessNN(bounds=ThicknessBounds(100.0, 2000.0))
initial = refinement.params.to(DEVICE)
model = build_refinement_model(
    initial=initial,
    components=(thickness_nn,),
    component_params={
        thickness_nn.key: thickness_nn.initial_params(
            dtype=initial.asu_positions.dtype,
            device=DEVICE,
            initial_thickness=mean_plan_thickness(engine.orientations),
        )
    },
)
problem = build_refinement_problem()  # no penalties for this demo
# Refine the atomic displacement parameters (ADPs) as the structure lever alongside the thickness NN
# -- a stable joint structure+thickness refinement (free-position refinement on this synthetic data
# is ill-conditioned).
trainable = TrainableSpec(adp=AtomSelection.all())

# %% [markdown]
# ## 5. Refine jointly
#
# `run_refinement_model` optimizes the trainable structure parameters (ADPs) **and** the
# thickness-NN parameters together through the one differentiable objective (Adam), returning the
# optimized model plus the per-step loss trajectory and the best snapshot. A short run here (this is
# a linear demo, not a converged refinement); the learned thickness profile below is the point of
# interest.

# %%
STEPS = 40
LR = 1e-3
result = run_refinement_model(
    engine, model, problem, trainable=trainable, steps=STEPS, optimizer="adam", lr=LR
)
print(f"loss: {float(result.losses[0]):.4f} -> {float(result.losses[-1]):.4f}")
print(f"best {result.best_loss:.4f} at step {result.best_step}")

# %% [markdown]
# ## 6. Inspect the learned per-orientation thickness
#
# Evaluate the optimized NN for each orientation to read off the thickness profile it learned. (A
# plot if matplotlib is available; otherwise a compact summary.)

# %%
final_params = result.model.component_params[thickness_nn.key]
per_orientation = []
for i, orientation in enumerate(engine.orientations):
    context = thickness_nn.forward_context(
        final_params, orientation_index=i, orientation=orientation
    )
    assert context.thickness is not None  # ApparentThicknessNN always provides a thickness
    per_orientation.append(context.thickness[0])
thickness = torch.stack(per_orientation).detach()
print(
    f"learned thickness (Å): min {float(thickness.min()):.1f}, "
    f"mean {float(thickness.mean()):.1f}, max {float(thickness.max()):.1f}  "
    f"(seed {float(mean_plan_thickness(engine.orientations)):.1f})"
)

try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 3))
    plt.plot(thickness.cpu().numpy(), marker=".")
    plt.xlabel("orientation index")
    plt.ylabel("apparent thickness (Å)")
    plt.title(f"{cfg.name}: learned per-orientation thickness")
    plt.tight_layout()
    plt.show()
except ModuleNotFoundError:
    print("(install matplotlib to plot the profile)")
