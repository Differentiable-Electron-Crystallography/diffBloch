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

The pipeline also provides composition helpers such as `iterate_until` and `Fork` for repeated,
conditional, or branching preprocess logic. More advanced steps use the same contract: convergence
sweeps repeatedly improve numerical plan values, while `fit_orientation` and `fit_thickness` score
candidate plans and return the best updated `Plan`.

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

## API shape: `Fork` and `iterate_until`

Advanced branching and looping pipelines can be composed with the `Fork` and `iterate_until`
utilities. `Fork` chooses one of two step lists from the immutable structure-factor grid, so the
chosen recipe can still be resolved before checkpointing; the default app recipe uses this shape to
route large cells through a coarser fp32 orientation/thickness-fit branch. `iterate_until` wraps a
repeated `Plan -> Plan` improvement behind the same step shape.

The convergence path is the stateful version of this idea. Its public pipeline surface is still a
single `Plan -> Plan` step, but internally `converge_numerics` runs a small coordinate-search driver:
it threads a `ConvergenceState` containing `g_max_refine`, `integration_semiangle`, and
`rocking_curve_sampling` through coverage and self-stability sweeps, rebuilding candidate `Plan`
values from each scalar setting. There is not currently a general-purpose stateful pipeline utility
for this shape; `converge_scalar`, `run_coverage_phase`, and `run_stability_phase` are the dedicated
helpers that formalize it for convergence.

```python
from diffBloch.preprocess import fork, identity, iterate_until

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
```

For real convergence, prefer the provided convergence steps and driver rather than the trivial
`identity` placeholders above.
