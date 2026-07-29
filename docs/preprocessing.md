# Preprocessing and `Plan`

Dynamical diffraction is extremely sensitive to crystal orientation and thickness. diffBloch
therefore fits this experimental metadata before refining the structure and stores it in a `Plan`.

PETS2 reduces continuous-rotation data into overlapping **virtual frames**, allowing complete
rocking curves to be integrated and partial reflections to be rejected
([Klar *et al.*, 2023](https://doi.org/10.1038/s41557-023-01186-1)). diffBloch represents each
virtual frame by sampled tilt sub-orientations and sums their simulated intensities.

## Orientation

PETS2 supplies a best-fit **UB matrix**: {math}`B` maps the reciprocal lattice and {math}`U`
orients that lattice in the laboratory frame. For virtual frame {math}`i`, diffBloch constructs

```{math}
M_i = R_z(\omega_i)R_x(\alpha_i)R_y(\beta_i)(UB)B^{-1},
```

where {math}`\alpha` is the main varying goniometer angle.

Using a fixed trial thickness and the starting structure, orientation optimization searches nearby
orientations for better agreement with experiment. Three approaches are available:

| Method | Difference |
|---|---|
| Bayesian optimization | Explores a broad angular range; robust but expensive. |
| Nelder–Mead | Locally optimizes all three rotation angles; efficient but sensitive to the starting orientation. |
| Modified simplex | Searches progressively smaller tilts around the starting orientation; robust when the PETS2 estimate is close. |

## Thickness

Real crystals have irregular shapes. diffBloch offers two
approaches, both using the starting structure to improve agreement with experiment:

| Method | Difference |
|---|---|
| Grid search | Selects the best mean thickness independently for each rotation. |
| Neural network | Learns how apparent thickness varies smoothly with rotation angle. |

The default workflow optimizes orientation first and thickness second when both stages are enabled.
Either stage can be disabled independently:

```yaml
preprocess:
  optimize_orientation: true
  optimize_thickness: false
```

## The `Plan`

A `Plan` is the fitted geometry and observation scaffold around the structural parameters; it is
not the structure being refined. It carries optimized orientations, rocking-curve tilts,
per-rotation thicknesses, reflection sets, and preprocessing provenance.

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
            tilts=...,            # rocking-curve sub-orientations
            thickness=...,        # fitted value
        ),
        # ...more rotations...
    ),
    provenance=(...),             # ordered preprocessing recipe
)
```

## Beam sets and scoring inside a `Plan`

A `Plan` deliberately separates sets of hkls by purpose or origin:

| Set | Meaning |
|---|---|
| Structure-factor hkl | Stored structure factors calculated for all reciprocal space up to a given resolution. |
| Solve beams | The beam basis used in a given Bloch wave calculation. |
| Observed hkl | Experimentally observed reflections. |
| Scored hkl | Reflections included in the loss calculation. |
| Matched hkl | The calculated/observed overlap. |


## Rocking curves and mosaicity inside a `Plan`

The virtual frame's angular range is represented by tilt sub-orientations around its central
orientation. In the default app recipe, `build_orientation_plans` constructs those sub-tilts,
selects each tilt-dependent SOLVE basis from `g_max` and `sg_max`, builds the Bloch geometry, and
attaches the configured mosaic reduction. Rocking integration and mosaicity are parts of the built
orientation plan, not separately displayed default stages.

## API shape: from experiment records to an initial `Plan`

```python
from pathlib import Path

from diffBloch.config import load_experiment
from diffBloch.io import read_observations, read_structure
from diffBloch.preprocess import from_experiment

root = Path("examples/experiments/quartz-checkpoint")
cfg, _lock = load_experiment(root)
structure = read_structure(root / cfg.inputs.structure)
observations = read_observations(root / cfg.inputs.exp_data)

setup = from_experiment(structure, observations, cfg)
initial_plan = setup.plans.combined
refinement_setup = setup.refinement

print(len(initial_plan.orientations))
print(refinement_setup.params.asu_positions.shape)
```

## `Plan -> Plan` steps

Preprocessing is a composable `Plan -> Plan` pipeline. Any function with that shape receives a
`Plan`, does one focused piece of work, and returns an updated `Plan`.

The default app recipe begins with one displayed `build_orientation_plans` stage. It calculates the
shared structure-factor support, constructs every central orientation and rocking sub-tilt,
calculates excitation errors for reciprocal-lattice points inside the solve cutoff, selects the
tilt-dependent SOLVE beams, builds the Bloch geometry, and matches the selected scoring reflections
to PETS observations. It does not use experimental presence to choose the SOLVE basis.

Real steps include:

- {func}`diffBloch.preprocess.steps.beams.build_orientation_plans` — build the default coupled solve geometry, rocking sub-tilts, reduction, and scoring alignment.
- {func}`diffBloch.preprocess.steps.beams.select_beams` — lower-level tilt-independent candidate selection for custom API pipelines and convergence work; it is not a separate default app stage.
- {func}`diffBloch.preprocess.steps.rocking_curve.integrate_rocking_curve` — expand orientations into virtual rocking-curve tilts.
- {func}`diffBloch.preprocess.steps.mosaicity.mosaicity` — apply tilt-axis mosaic broadening.
- {func}`diffBloch.preprocess.steps.fit_orientation.fit_orientation` — search nearby orientations and keep the best-scoring one.
- {func}`diffBloch.preprocess.steps.fit_thickness.fit_thickness` — search mean specimen thickness and keep the best-scoring value.
- {func}`diffBloch.preprocess.driver.converge_numerics` — test convergence over `g_max`, `sg_max`, and `tilt_steps`.

The pipeline also provides composition helpers such as {func}`diffBloch.preprocess.pipeline.fork`,
{func}`diffBloch.preprocess.pipeline.iterate_until`, and
{func}`diffBloch.preprocess.pipeline.stateful_plan_step` for conditional, repeated, or stateful
preprocess logic. More advanced steps use the same contract: convergence sweeps repeatedly improve
numerical plan values, while `fit_orientation` and `fit_thickness` score candidate plans and return
the best updated `Plan`.

## API example: composing simple steps

This example mirrors the default geometry-build shape. The app additionally handles checkpointing,
device/precision choices, optional fitting stages, and logging.

```python
from pathlib import Path

from diffBloch.config import load_experiment
from diffBloch.io import read_observations, read_structure
from diffBloch.preprocess import build_orientation_plans, from_experiment, pipeline

root = Path("examples/experiments/quartz-checkpoint")
cfg, _lock = load_experiment(root)
structure = read_structure(root / cfg.inputs.structure, load_hydrogens=cfg.inputs.load_hydrogens)
observations = read_observations(root / cfg.inputs.exp_data)

setup = from_experiment(structure, observations, cfg)
base_plan = setup.plans.combined

prepare = pipeline([
    build_orientation_plans(
        cfg.blochwave.to_rocking_curve(setup.integration),
        cfg.blochwave.mosaicity,
        coupling=cfg.blochwave.to_policy(),
        scoring_selection=cfg.blochwave.to_beam_selection(setup.integration),
    ),
])

prepared_plan = prepare(base_plan)
```

From the CLI, preprocessing prints each completed stage as
`Preprocess stage N │ Name │ measurements`, then an aligned completion box, the resolved pipeline,
and the absolute `plan.npz` / `plan.lock` locations:

```bash
uv run diffbloch run preprocess examples/experiments/quartz --refresh
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
varies `g_max`, `sg_max`, and `tilt_steps`, rebuilding the simulation at each setting. The generic
shape is formalized as
{data}`diffBloch.preprocess.pipeline.StatefulPlanStep`,
{func}`diffBloch.preprocess.pipeline.stateful_pipeline`, and
{func}`diffBloch.preprocess.pipeline.stateful_plan_step`: an explicit immutable state threaded
between phases and dropped again at the `Plan -> Plan` boundary.

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
