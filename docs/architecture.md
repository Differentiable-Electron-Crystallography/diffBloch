# Architecture

Electrons interact with matter far more strongly than X-rays, so a crystal only a few tens to a few
hundred nanometres thick already scatters a diffracted beam strongly enough to re-scatter again
before it exits. That multiple-scattering — dynamical diffraction — means the intensity of a
reflection is not simply proportional to \(|F_{hkl}|^2\) the way it is in the kinematical
(single-scattering) approximation used for X-ray and neutron work. diffBloch simulates the full
multiple-scattering problem with a Bloch-wave calculation, and makes that simulation differentiable
so a structure can be refined against it directly, the same way X-ray refinement fits against a
kinematical model.

## What gets computed, in order

1. A starting structure (`.cif`) and reduced rocking-curve diffraction data (`.cif_pets`) are read in.
2. Because the Bloch-wave calculation is extremely sensitive to the exact crystal orientation and
   specimen thickness at each rotation, both are fitted against the data before the structure is
   touched — this is what diffBloch calls preprocessing, and its result is stored as a `Plan`.
3. Structure factors \(F_{hkl}\) are computed from the current atomic positions, ADPs, and
   occupancies.
4. For each rotation, the beam set within the chosen resolution is propagated through the crystal
   thickness — an eigenvalue problem for a static tilt, or a matrix exponential when refining, since
   it stays stable under backpropagation — and the tilts spanning that rotation's rocking curve are
   summed to give one simulated intensity per reflection.
5. Simulated and observed intensities are compared with a scaling-optimized weighted R-factor.
6. Gradients of that R-factor flow back through the whole calculation to the atomic parameters, and
   an ordinary PyTorch optimizer (Adam or L-BFGS) updates them. Steps 3–6 repeat until the structure
   converges.

Preprocessing (step 2) and refinement (steps 3–6) are deliberately separate: a `Plan` is expensive to
compute but describes only geometry, not the structure, so it can be checkpointed once and reused
across many refinements of the same dataset. See [Preprocessing](preprocessing.md) for how orientation
and thickness are fitted, and [Refinement](refinement.md) for the optimization loop itself.

## Where each piece lives

| Physical role | Code | Guide |
|---|---|---|
| Parse `.cif` / `.cif_pets` into validated records | [`io`](api/io.md) | [Inputs](inputs.md) |
| Validate `experiment.yaml`, pin input identity | [`config`](api/config.md) | [Inputs](inputs.md), [Reproducibility](reproducibility.md) |
| Fit orientation and thickness, build the `Plan` | [`preprocess`](api/preprocess.md) | [Preprocessing](preprocessing.md) |
| Structure factors and the Bloch-wave propagator | [`core`](api/core.md) | [Refinement](refinement.md) |
| Atomic positions, ADPs, occupancies, and their crystallographic constraints | [`params`](api/params.md) | [Refinement](refinement.md) |
| Simulate, score, and run the optimization loop | [`engine`](api/engine.md) | [Refinement](refinement.md) |
| Report per-rotation R-factors and progress | [`observability`](api/observability.md) | [Observability](observability-guide.md) |
| CLI and default recipes | [`app`](api/app.md) | [Examples](examples.md) |

## Running the whole calculation from Python

The CLI (`diffbloch run refine ...`) is a thin wrapper over the same public functions shown below:
read the inputs, fit orientation/thickness into a `Plan`, then simulate and refine against it.

```python
from pathlib import Path

from diffBloch.app import CSVLogger, ConsoleLogger
from diffBloch.config import load_experiment
from diffBloch.engine import build_refinement_model, build_refinement_problem, run_refinement_model
from diffBloch.io import read_observations, read_structure
from diffBloch.observability import MultiLogger
from diffBloch.preprocess import (
    build_engine,
    build_orientation_plans,
    fit_orientation,
    fit_thickness,
    from_experiment,
    pipeline,
    run_inference,
)
from diffBloch.specs import ScoredHklSelection, TrialCoupling

root = Path("examples/experiments/quartz-checkpoint")

# Read the starting structure and the rocking-curve observations.
cfg, experiment_lock = load_experiment(root)
structure = read_structure(root / cfg.inputs.structure, load_hydrogens=cfg.inputs.load_hydrogens)
observations = read_observations(root / cfg.inputs.exp_data)

# Progress (per-rotation wR2/R_obs, orientation-search steps, ...) streams to these loggers.
logger = MultiLogger((ConsoleLogger(), CSVLogger(root / "events.csv")))

# Build the initial (unfitted) geometry scaffold and the differentiable structural parameters.
setup = from_experiment(structure, observations, cfg)

# The beam set re-derived at every trial orientation during the search below.
trial_coupling = TrialCoupling(
    policy=cfg.blochwave.to_policy(),
    scored=ScoredHklSelection(
        klar=cfg.blochwave.to_beam_selection(setup.integration),
        g_max=cfg.blochwave.g_max_refine,
    ),
)

# Fit orientation, then specimen thickness, per rotation.
prepare = pipeline(
    [
        build_orientation_plans(
            cfg.blochwave.to_rocking_curve(setup.integration),
            cfg.blochwave.mosaicity,
            coupling=cfg.blochwave.to_policy(),
            scoring_selection=cfg.blochwave.to_beam_selection(setup.integration),
        ),
        fit_orientation(
            setup.refinement,
            cfg.preprocess.orientation.to_search(),
            method=cfg.blochwave.solver.refine,
            coupling=trial_coupling,
            logger=logger,
        ),
        fit_thickness(
            setup.refinement,
            cfg.preprocess.thickness.to_grid(),
            method=cfg.blochwave.solver.refine,
            logger=logger,
        ),
    ],
    logger=logger,
)
plan = prepare(setup.plans.combined)

# Simulate and score the settled Plan without changing the structure.
inference = run_inference(
    plan,
    setup.refinement,
    method=cfg.blochwave.solver.inference,
    logger=logger,
)

# Refine: simulate, compare with experiment, and update atomic parameters by gradient descent.
engine = build_engine(
    plan,
    setup.refinement,
    loss=cfg.refinement.objective.to_loss(),
    method=cfg.blochwave.solver.refine,
)
model = build_refinement_model(initial=setup.refinement.params)
problem = build_refinement_problem()
result = run_refinement_model(
    engine,
    model,
    problem,
    trainable=cfg.refinement.trainable.to_spec(),
    steps=cfg.refinement.steps,
    optimizer=cfg.refinement.optimizer.name,
    lr=cfg.refinement.optimizer.lr,
    logger=logger,
)

print(inference.mean_r_obs)
print(result.best_step, result.best_loss)
```

For checkpoint/reuse behavior, the app layer wraps this same public composition with
`plan.npz`/`plan.lock` handling; see [Reproducibility](reproducibility.md).
