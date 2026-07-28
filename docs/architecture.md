# Architecture

diffBloch is organized around its core module: Bloch-wave simulation of multiple electron scattering.
The surrounding layers prepare inputs, build reusable geometry, run optimization, and report
progress without making those concerns part of the numerical core.

## End-to-end flow

The end-to-end flow is:

1. `.cif` + `.cif_pets` + config are parsed into typed records.
2. Preprocessing turns those records into a reusable `Plan`.
3. The `Plan` and differentiable structural parameters feed the Bloch-wave simulation.
4. The simulated diffraction intensities are compared with experiment and loss is calculated. 
5. Gradients from that loss update the differentiable structural parameters, then the loop repeats, improving the agreement between simulation and experiment.

## Layers

| Layer | Role | Guide |
|---|---|---|
| [IO](api/io.md) | Parse CIF/PETS files into validated typed records. | [Inputs](inputs.md) |
| [Config](api/config.md) | Validate experiment settings and lock input/checkpoint identity. | [Inputs](inputs.md), [Reproducibility](reproducibility.md) |
| [Preprocess](api/preprocess.md) | Build and improve the immutable `Plan` the simulator consumes. | [Preprocessing](preprocessing.md) |
| [Core](api/core.md) | Bloch-wave numerical kernels. | [Architecture](architecture.md), [Refinement](refinement.md) |
| [Params](api/params.md) | Differentiable structural parameters and physical constraints. | [Refinement](refinement.md) |
| [Engine](api/engine.md) | Combine `Plan` + parameters, simulate, score, and refine. | [Refinement](refinement.md) |
| [Observability](api/observability.md) | Emit typed events without coupling the core to logger backends. | [Observability](observability-guide.md) |
| [App](api/app.md) | CLI and default orchestration around the reusable API. | [Examples](examples.md), [Refinement](refinement.md) |

## API shape

This pseudocode shows the declarative public API shape underneath the CLI: load typed inputs,
compose a `Plan -> Plan` preprocessing recipe, build an engine from the settled `Plan`, then run the
refinement model while typed progress events flow to loggers. The example is close to runnable code,
but it is shown as API shape rather than a recipe to copy verbatim.

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
    integrate_rocking_curve,
    mosaicity,
    pipeline,
    run_inference,
    select_beams,
)
from diffBloch.specs import ScoredHklSelection, TrialCoupling

root = Path("examples/experiments/quartz-checkpoint")

# Boundary: parse experiment.yaml, verify experiment.lock, and read typed CIF/PETS records.
cfg, experiment_lock = load_experiment(root)
structure = read_structure(root / cfg.inputs.structure, load_hydrogens=cfg.inputs.load_hydrogens)
observations = read_observations(root / cfg.inputs.observations)

# Effects stay at the edge: loggers consume typed events from preprocessing/refinement.
logger = MultiLogger((ConsoleLogger(), CSVLogger(root / "events.csv")))

# Initial construction: raw input records become a Plan scaffold plus structure-side params/specs.
setup = from_experiment(structure, observations, cfg)

# Scientific choices are explicit typed values, not a string registry or hidden CLI behavior.
trial_coupling = TrialCoupling(
    policy=cfg.preprocess.coupling.to_policy(),
    scored=ScoredHklSelection(
        klar=cfg.numerics.to_beam_selection(setup.integration),
        g_max=cfg.numerics.g_max_refine,
    ),
)

# Preprocessing is declarative composition of Plan -> Plan steps.
prepare = pipeline(
    [
        select_beams(cfg.numerics.to_beam_selection(setup.integration)),
        build_orientation_plans(),
        integrate_rocking_curve(cfg.numerics.to_rocking_curve(setup.integration)),
        mosaicity(cfg.numerics.mosaicity),
        fit_orientation(
            setup.refinement,
            cfg.preprocess.orientation.to_search(),
            method=cfg.solver.refine,
            coupling=trial_coupling,
            logger=logger,
        ),
        fit_thickness(
            setup.refinement,
            cfg.preprocess.thickness.to_grid(),
            method=cfg.solver.refine,
            logger=logger,
        ),
    ],
    logger=logger,
)
plan = prepare(setup.plans.combined)

# Inference is the terminal score-only path: same settled Plan, no optimizer updates.
inference = run_inference(plan, setup.refinement, method=cfg.solver.inference, logger=logger)

# Refinement composes a pure engine/context with model/problem values, then runs the optimizer shell.
engine = build_engine(
    plan,
    setup.refinement,
    loss=cfg.refinement.objective.to_loss(),
    method=cfg.solver.refine,
    precision=cfg.refinement.precision,
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
