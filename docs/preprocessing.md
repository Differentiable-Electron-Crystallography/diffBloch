# Preprocessing and `Plan`

Preprocessing builds the immutable `Plan` consumed by inference and refinement. A `Plan` is not the
structure being refined; it is the settled geometry and data scaffold around the differentiable
structural parameters.

A `Plan` carries things such as:

- shared structure-factor support;
- per-rotation orientations;
- solve beam sets;
- rocking-curve tilts;
- observed patterns and alignments;
- fitted nuisance values such as per-rotation thickness;
- coupled beam-union geometry when coupling is enabled.

## `Plan -> Plan` steps

Preprocessing is a composable `Plan -> Plan` pipeline. Any function with that shape can be a step:
it receives a `Plan`, does one focused piece of work, and returns an updated `Plan`.

Real steps include:

- `select_beams`
- `build_orientation_plans`
- `integrate_rocking_curve`
- `mosaicity`
- `fit_orientation`
- `fit_thickness`
- `couple_beams`
- `converge_numerics`

The pipeline also provides composition helpers such as `iterate_until` and `Fork` for repeated,
conditional, or branching preprocess logic. More advanced steps use the same contract: convergence
sweeps repeatedly improve numerical plan values, while `fit_orientation` and `fit_thickness` score
candidate plans and return the best updated `Plan`.

## API example: loading a checkpointed `Plan`

```python
from diffBloch.preprocess import read_plan, require_built_plans

plan = read_plan("examples/experiments/quartz-checkpoint/plan.npz")
orientations = require_built_plans(plan)

print(len(orientations))
print(plan.grid.grid_hkl.shape)
```

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

## API shape: `Fork` and `iterate_until`

`Fork` chooses one of two step lists from the immutable grid, so the chosen recipe can still be
resolved before checkpointing. `iterate_until` wraps a repeated `Plan -> Plan` improvement behind the
same step shape.

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

## API shape: from records to an initial `Plan`

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
