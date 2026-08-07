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
orientations for better agreement with experiment. Two approaches are implemented, selected by
`preprocess.orientation.method`:

| Method | Difference |
|---|---|
| `palatinus_modified_simplex` (default) | Searches progressively smaller tilts around the starting orientation; robust when the PETS2 estimate is close. |
| `nelder_mead` | Local simplex search over all three goniometer-correction angles directly (`scipy.optimize.minimize`); efficient but sensitive to the starting orientation and the fixed `step_size` neighbourhood it explores. |

Bayesian optimization is not implemented.

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

A `Plan` is the fitted geometry and experimental scaffold around the structural parameters; it is
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
attaches the configured mosaic reduction. The configured reduction currently defaults to
`Mosaicity(window=5)`: a five-sampled-tilt moving average, not a value estimated from PETS
mosaic-spread metadata. Passing `mosaicity=None` at the lower-level API keeps the bare `PlainSum`.
Rocking integration and mosaicity are parts of the built orientation plan, not separately displayed
default stages.

## API shape: from experiment records to an initial `Plan`

```python
from pathlib import Path

from diffBloch.config import load_experiment
from diffBloch.io import read_experimental_data, read_structure
from diffBloch.preprocess import from_experiment

root = Path("examples/experiments/quartz-checkpoint")
cfg, _lock = load_experiment(root)
structure = read_structure(root / cfg.inputs.structure)
experimental_data = read_experimental_data(root / cfg.inputs.exp_data)

setup = from_experiment(structure, experimental_data, cfg)
initial_plan = setup.plans.combined
refinement_setup = setup.refinement

print(len(initial_plan.orientations))
print(refinement_setup.params.asu_positions.shape)
```

## `Plan -> Plan` steps

Preprocessing is a sequence of small, focused steps, each taking a `Plan` and returning an updated
one — one step builds the tilt geometry, another fits orientation, another fits thickness.

The default app recipe begins with one displayed `build_orientation_plans` stage. It calculates the
shared structure-factor support, constructs every central orientation and rocking sub-tilt,
calculates excitation errors for reciprocal-lattice points inside the solve cutoff, selects the
tilt-dependent SOLVE beams, builds the Bloch geometry, and matches the selected scoring reflections
to PETS experimental data. It does not use experimental presence to choose the SOLVE basis.

Real steps include:

- {func}`diffBloch.preprocess.steps.beams.build_orientation_plans` — build the default coupled solve geometry, rocking sub-tilts, reduction, and scoring alignment.
- {func}`diffBloch.preprocess.steps.beams.select_beams` — lower-level tilt-independent candidate selection for custom API pipelines and convergence work; it is not a separate default app stage.
- {func}`diffBloch.preprocess.steps.rocking_curve.integrate_rocking_curve` — expand orientations into virtual rocking-curve tilts.
- {func}`diffBloch.preprocess.steps.mosaicity.mosaicity` — apply tilt-axis mosaic broadening.
- {func}`diffBloch.preprocess.steps.optimize_orientation.optimize_orientation` — search nearby orientations and keep the best-scoring one.
- {func}`diffBloch.preprocess.steps.optimize_thickness.optimize_thickness` — search mean specimen thickness and keep the best-scoring value.
- {func}`diffBloch.preprocess.driver.converge_numerics` — test convergence over `g_max`, `sg_max`, and `tilt_steps`.

## API example: composing simple steps

This example mirrors the default geometry-build shape. The app additionally handles checkpointing,
device choices, optional fitting stages, and logging.

```python
from pathlib import Path

from diffBloch.config import load_experiment
from diffBloch.io import read_experimental_data, read_structure
from diffBloch.preprocess import build_orientation_plans, from_experiment, pipeline

root = Path("examples/experiments/quartz-checkpoint")
cfg, _lock = load_experiment(root)
structure = read_structure(root / cfg.inputs.structure, load_hydrogens=cfg.inputs.load_hydrogens)
experimental_data = read_experimental_data(root / cfg.inputs.exp_data)

setup = from_experiment(structure, experimental_data, cfg)
base_plan = setup.plans.combined

prepare = pipeline([
    build_orientation_plans(
        cfg.blochwave.to_rocking_curve(setup.integration),
        cfg.blochwave.mosaicity,  # default app config is Mosaicity(window=5); use None for PlainSum
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

## Routing on cell size

The coupled orientation search diagonalizes (or exponentiates) the structure matrix {math}`A` once
per trial, an {math}`O(N^3)` operation in the beam count {math}`N`, and {math}`N` grows with
unit-cell volume. Above a fixed volume threshold, diffBloch skips a per-trial integrity check in the
search that costs proportionally more on a large coupled beam set, without changing the search
itself — the fitted orientation is always re-scored under the full check once the search settles.
Below the threshold (quartz, at {math}`\sim 113\ \text{Å}^3`, included) the check runs on every
trial. This routing is built with {func}`diffBloch.preprocess.pipeline.fork`, which picks one of two
step lists from the structure-factor grid before the recipe is checkpointed, so a committed `Plan` is
unaffected by which branch produced it.

Numerical convergence testing — sweeping `g_max`, `sg_max`, and rocking-curve sampling until the
simulated intensities stop changing — uses the same `Plan -> Plan` step shape, built with
{func}`diffBloch.preprocess.driver.converge_numerics`. See
[Hyperparameter selection](hyperparameter-selection.md) for the physics behind that sweep and a
runnable example.
