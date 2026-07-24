# Preprocessing and `Plan`

The first step after loading an experiment is to create a refinable `Plan`. A `Plan` is the
geometry and data scaffold around the differentiable structural parameters; it is not the structure
being refined.

A `Plan` carries things such as:

- shared structure-factor support;
- per-rotation orientations;
- solve beam sets;
- rocking-curve tilts;
- observed patterns and alignments;
- fitted nuisance values such as per-rotation thickness;
- coupled beam-union geometry when coupling is enabled.

Conceptually, a settled `Plan` looks like this:

```python
Plan(
    structure_factor_grid=StructureFactorGrid(
        structure_factor_hkl=...,  # shared Fgb support
        cell=...,
        reciprocal_basis=...,
        g_max=...,
    ),
    orientations=(
        OrientationPlan(
            orientation=...,      # one rotation/orientation
            beam_hkl=...,         # solve beam set
            beam_plans=...,       # precomputed Bloch geometry for those beams
            tilts=...,            # rocking-curve sub-orientations
            pattern=...,          # observed intensities for this rotation
            alignment=...,        # calculated/observed hkl alignment
            thickness=...,        # fitted nuisance value
        ),
        # ...more rotations...
    ),
    provenance=(...),             # ordered preprocessing recipe
)
```

## Beam sets and scoring inside a `Plan`

A `Plan` deliberately separates hkl sets that answer different questions:

| Set | Meaning |
|---|---|
| Structure-factor hkl | The shared support where `Fgb` is tabulated. |
| Solve beams | The beam basis used in a Bloch solve. |
| Observed hkl | Reflections present in the PETS observations. |
| Scored hkl | Reflections included in the objective/scoring comparison. |
| Matched hkl | The calculated/observed overlap recorded by the alignment. |

This split matters because the best solve basis is not always the same as the set of reflections you
want in the objective. Under coupled rocking-curve solves, the solve set may expand to a per-segment
beam union while the scored set stays pinned to the selected/scored hkl, so widening the solve basis
does not silently change what the objective compares.

## Rocking curve and mosaicity inside a `Plan`

A rotation electron-diffraction frame integrates intensity while the crystal rocks through a small
angular range. In a `Plan`, that is represented as virtual tilt sub-orientations on each rotation:
`integrate_rocking_curve` replaces a single static orientation with sampled tilts, and `mosaicity`
adds a tilt-axis moving-average reduction before intensities are summed. If beam coupling is enabled,
`couple_beams` can then replace one shared beam set with per-segment beam unions over the rocking
curve.

## API shape: from experiment records to an initial `Plan`

```python
from pathlib import Path

from diffBloch.config import load_experiment
from diffBloch.io import read_observations, read_structure
from diffBloch.preprocess import from_experiment

root = Path("examples/experiments/quartz-checkpoint")
cfg, _lock = load_experiment(root)
structure = read_structure(root / cfg.inputs.structure)
observations = read_observations(root / cfg.inputs.observations)

setup = from_experiment(structure, observations, cfg)
initial_plan = setup.plans.combined
refinement_setup = setup.refinement

print(len(initial_plan.orientations))
print(refinement_setup.params.asu_positions.shape)
```

## `Plan -> Plan` steps

Preprocessing is a composable `Plan -> Plan` pipeline. A `select_beams` step, for example, is a
function that takes a `Plan` and adjusts its geometry/data scaffold by replacing each rotation's
candidate beam set with the reflections selected by the beam-selection policy. More generally, any
function with that shape can be a step: it receives a `Plan`, does one focused piece of work, and
returns an updated `Plan`.

Real steps include:

- {func}`diffBloch.preprocess.steps.beams.select_beams` — choose each rotation's candidate solve beams.
- {func}`diffBloch.preprocess.steps.beams.build_orientation_plans` — build solvable per-orientation beam geometry.
- {func}`diffBloch.preprocess.steps.rocking_curve.integrate_rocking_curve` — expand orientations into virtual rocking-curve tilts.
- {func}`diffBloch.preprocess.steps.mosaicity.mosaicity` — apply tilt-axis mosaic broadening.
- {func}`diffBloch.preprocess.steps.fit_orientation.fit_orientation` — search nearby orientations and keep the best-scoring one.
- {func}`diffBloch.preprocess.steps.fit_thickness.fit_thickness` — search specimen thickness and keep the best-scoring value.
- {func}`diffBloch.preprocess.steps.coupling.couple_beams` — settle coupled per-segment beam unions.
- {func}`diffBloch.preprocess.driver.converge_numerics` — run coverage/convergence sweeps over numerical knobs.

The pipeline also provides composition helpers such as {func}`diffBloch.preprocess.pipeline.fork`,
{func}`diffBloch.preprocess.pipeline.iterate_until`, and
{func}`diffBloch.preprocess.pipeline.stateful_plan_step` for conditional, repeated, or stateful
preprocess logic. More advanced steps use the same contract: convergence sweeps repeatedly improve
numerical plan values, while `fit_orientation` and `fit_thickness` score candidate plans and return
the best updated `Plan`.

## API example: composing simple steps

This example shows the composition shape. It is intentionally small; the full default recipe also
captures refinement setup, coupling policy, device/precision choices, and logging.

```python
from diffBloch.preprocess import build_orientation_plans, pipeline, select_beams
from diffBloch.specs import BeamSelection

prepare = pipeline([
    select_beams(BeamSelection()),
    build_orientation_plans(),
])

# prepared_plan = prepare(candidate_plan)
```

## API example: loading a checkpointed `Plan`

The end-to-end preprocessing pipeline can take a long time, and it is deliberately decoupled from
refinement. You can checkpoint a settled `Plan` — for example, a plan whose orientations and
thicknesses have already been fitted — then run inference or refinement over that reusable scaffold
without recomputing the preprocessing recipe.

```python
from diffBloch.preprocess import read_plan, require_built_plans

plan = read_plan("examples/experiments/quartz-checkpoint/plan.npz")
orientations = require_built_plans(plan)

print(len(orientations))
print(plan.structure_factor_grid.structure_factor_hkl.shape)
```

## API shape: branching, looping, and stateful drivers

Advanced branching and looping pipelines can be composed with
{func}`diffBloch.preprocess.pipeline.fork`,
{func}`diffBloch.preprocess.pipeline.iterate_until`, and
{func}`diffBloch.preprocess.pipeline.stateful_plan_step`.

`fork` constructs a {class}`diffBloch.preprocess.pipeline.Fork` value: the lowercase function is the
user-facing combinator, while uppercase `Fork` is the returned dataclass/type used for recipe
resolution. It chooses one of two step lists from the immutable structure-factor grid, so the chosen
recipe can still be resolved before checkpointing; the default app recipe uses this shape to route
large cells through a coarser fp32 orientation/thickness-fit branch. `iterate_until` wraps a repeated
`Plan -> Plan` improvement behind the same step shape.

The convergence path is the stateful version of this idea. Its public pipeline surface is still a
single `Plan -> Plan` step, but internally {func}`diffBloch.preprocess.driver.converge_numerics`
runs a coordinate-search driver: it threads a `ConvergenceState` containing `g_max_refine`,
`integration_semiangle`, and `rocking_curve_sampling` through coverage and self-stability sweeps,
rebuilding candidate `Plan` values from each scalar setting. The generic shape is formalized as
{data}`diffBloch.preprocess.pipeline.StatefulPlanStep`,
{func}`diffBloch.preprocess.pipeline.stateful_pipeline`, and
{func}`diffBloch.preprocess.pipeline.stateful_plan_step`: an explicit immutable-state carry, similar
to Haskell's `State`, Elm's model-threading `update`, or JAX's `carry` in `scan`/`while_loop`.

```python
from dataclasses import dataclass

from diffBloch.preprocess import (
    fork,
    identity,
    iterate_until,
    stateful_pipeline,
    stateful_plan_step,
)

large_cell_branch = fork(
    lambda grid: grid.cell_volume > 1000.0,
    when_true=[identity()],   # e.g. coarse/faster steps for large cells
    when_false=[identity()],  # e.g. exact/default steps for small cells
)

repeat_until_stable = iterate_until(
    identity(),
    until=lambda previous, current: previous is current,
    max_iterations=1,
)

@dataclass(frozen=True)
class SearchState:
    tried: int = 0


def bump_trial_count(plan, state: SearchState):
    # A real driver would rebuild a candidate Plan from state, score it, and return the new carry.
    return plan, SearchState(tried=state.tried + 1)


stateful_driver = stateful_plan_step(
    init_state=lambda plan: SearchState(),
    step=stateful_pipeline([bump_trial_count, bump_trial_count]),
)
```

For real convergence, prefer the provided convergence steps and driver rather than the trivial
`identity` placeholders above.
