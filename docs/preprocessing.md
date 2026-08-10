# Preprocessing and `Plan`

Dynamical diffraction is extremely sensitive to crystal orientation and thickness. diffBloch
therefore determines this experimental metadata before refining the structure and stores it in a `Plan`.

PETS2 reduces continuous-rotation data into overlapping **virtual frames**, allowing complete
rocking curves to be integrated and partial reflections to be rejected
([Klar *et al.*, 2023](https://doi.org/10.1038/s41557-023-01186-1)). diffBloch represents each
virtual frame by sampled tilt sub-orientations and sums their simulated intensities.

## Orientation

PETS2 supplies a **UB matrix**. The reciprocal-basis matrix {math}`B` is calculated from the unit
cell and maps integer reflection indices {math}`(h,k,l)` to reciprocal-space vectors. The matrix
{math}`U` places that reciprocal basis in the laboratory coordinate system. Their product {math}`UB`
therefore maps a reflection index directly into the measured laboratory geometry.

diffBloch separates the two matrices using

```{math}
U = (UB)B^{-1}.
```

For virtual frame {math}`i`, the PETS goniometer angles are then applied in their recorded order:

```{math}
M_i = R_z(\omega_i)R_x(\alpha_i)R_y(\beta_i)(UB)B^{-1},
```

where {math}`\alpha_i`, {math}`\beta_i`, and {math}`\omega_i` are given in degrees and
{math}`\alpha_i` is the main scan angle. {math}`M_i` is the starting crystal orientation for that
virtual frame.

Using the current thickness and starting unrefined structure, diffBloch searches for a small
correction to each {math}`M_i`. A trial correction is described by three angles
{math}`(\Delta\alpha,\Delta\beta,\Delta\omega)` and applied as

```{math}
M_i' = M_i R_z(\Delta\omega)R_x(\Delta\alpha)R_y(\Delta\beta).
```

The Bloch-wave intensities are recalculated for each trial and compared with the observed
intensities using `loss_metrics.residual`, which defaults to `wr2`.

### Orientation-search method

diffBloch applies the general Nelder--Mead optimization algorithm
([Nelder and Mead, 1965](https://doi.org/10.1093/comjnl/7.4.308)) to orientation refinement.

Nelder--Mead minimizes the orientation residual without requiring its derivatives. For the three
correction angles, the search holds four trial points: zero correction and one point displaced by
`step_size` along each angle. These four points form a simplex.

Each point is evaluated by running the Bloch-wave calculation and comparing its intensities with
experiment. Nelder--Mead ranks the four residuals and replaces the worst point by reflecting it
through the opposite face of the simplex. Depending on the new residual, it may expand farther in
that direction, contract toward the better points, or shrink the whole simplex around the best
point. The simplex therefore moves and changes size as it approaches a local minimum. The final
three coordinates are applied to the PETS orientation as
{math}`(\Delta\alpha,\Delta\beta,\Delta\omega)`.

This is a local, unconstrained search. `step_size` defines only the initial simplex; it neither
limits the correction angles nor guarantees that the search finds the global minimum. The PETS UB
matrix must already provide a close starting orientation.

| Method | Search | Status |
|---|---|---|
| `nelder_mead` | Varies all three correction angles simultaneously using the four-point simplex described above. | Implemented and used by the default preprocessing path. |


| Parameter | Default | Meaning |
|---|---:|---|
| `step_size` | `0.05` degrees | Edge length of the initial simplex. The four starting points are zero correction and one positive step along each correction angle. This is not a hard search bound. |
| `max_iterations` | `60` | Maximum number of simplex iterations for one rotation. |
| `x_tolerance` | `1e-3` degrees | Convergence tolerance for changes in the correction angles. |
| `f_tolerance` | `1e-3` | Convergence tolerance for changes in the comparison residual. |
| `penalize_fewer_reflections` | `true` | Prevents a trial from appearing better only because its orientation produces a smaller matched-reflection set. |

```yaml
preprocess:
  orientation:
    nelder_mead:
      step_size: 0.05
      max_iterations: 60
      x_tolerance: 0.001
      f_tolerance: 0.001
      penalize_fewer_reflections: true
```

### Routing on cell size

The coupled orientation search diagonalizes (or exponentiates) the structure matrix {math}`A` once
per trial, an {math}`O(N^3)` operation in the beam count {math}`N`, and {math}`N` grows with
unit-cell volume. Above a fixed volume threshold, diffBloch skips a per-trial integrity check in the
search that costs proportionally more on a large coupled beam set, without changing the search
itself — the fitted orientation is always re-scored under the full check once the search settles.
Below the threshold (quartz, at {math}`\sim 113\ \text{Å}^3`, included) the check runs on every
trial. This routing is built with {func}`diffBloch.preprocess.pipeline.fork`, which picks one of two
step lists from the structure-factor grid before the recipe is checkpointed, so a committed `Plan` is
unaffected by which branch produced it. The thickness fit has no such split and always runs the same
path regardless of cell size.

## Thickness

Real crystals have irregular shapes. The illuminated area therefore contains regions of different
thickness, and the thickness distribution changes as the crystal orientation changes. Under the
column approximation, each region may be simulated at its local thickness and the resulting intensities summed incoherently.

Multiple scattering transfers amplitude between coupled reflections as the electron wave propagates through the crystal.
Pendellösung oscillations therefore cause the intensity of an observed reflection to rise and fall
with thickness. A small change in thickness can produce a large, reflection-specific change in
intensity.

diffBloch represents the thickness distribution by an effective mean thickness for each orientation. It can use one shared effective thickness for the complete experiment, determine a separate mean value for each orientation, or learn a smooth orientation-dependent mean thickness distribution:

| Method | Difference |
|---|---|
| Shared thickness | Uses the value in `sample.thicknesses` for every orientation. |
| Grid search | Selects the lowest-residual thickness independently for each orientation. |
| Neural network | Learns how apparent thickness varies smoothly with orientation angle. |

We normally use one shared value for orientation refinement if the approximate specimen thickness is known:

```yaml
sample:
  thicknesses: [850.0]  # Angstroms

preprocess:
  optimize_thickness: false
```

If the thickness is not known, it is recommended to run thickness optimization before orientation optimization or structural refinement. The result establishes thickness before other parameters are
changed. A representative value can then be used as the shared thickness if separate values are not
required.

The grid search evaluates `n_steps` evenly spaced thicknesses from `min_thickness` to
`max_thickness`, inclusive, for every orientation. Each candidate is passed through the Bloch-wave
calculation and scored against experiment. The lowest-residual thickness is stored for that
orientation. The search uses the starting unrefined structure, so its purpose is to establish the
experimental geometry before atomic parameters are changed.

```yaml
preprocess:
  optimize_thickness: true
  thickness:
    min_thickness: 100.0
    max_thickness: 2000.0
    n_steps: 100
    plot: true
```

With `plot: true`, diffBloch writes one residual-versus-thickness PNG for each orientation to
`thickness_optim/` beside the structure input. Every tested thickness is shown and a dashed line
marks the selected minimum. These plots show whether the minimum is well defined, whether different
orientations favor the same thickness range, and whether the chosen search interval is too narrow.

## Numerical convergence

Determine `g_max`, `sg_max`, and `rocking_curve_sampling` before optimizing thickness or
orientation and before refining the structure. Numerical convergence depends mainly on the
simulation basis and integration grid, not on whether the starting orientation or atomic structure
has already been optimized. Thickness is the important exception because it changes the width of
the rocking curve. See [Convergence testing](convergence-testing.md) for the convergence procedure
and the thickness-dependent sampling check.

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

Mosaicity describes the spread of crystal orientations within the illuminated volume. Different
mosaic domains are slightly misoriented, so a reflection is excited over a wider angular range than
it would be in a single perfectly oriented crystal. Mosaicity therefore broadens the calculated
rocking curve and changes the integrated intensity, especially for reflections whose excitation
condition varies rapidly with angle.

## API shape: from experiment records to an initial `Plan`

```python
from pathlib import Path

from diffBloch.config import load_experiment
from diffBloch.io import read_experimental_data, read_structure
from diffBloch.preprocess import from_experiment

root = Path("examples/Colmey_et_al_2026_Acta_Cryst_A/data/quartz-no-abs")
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

root = Path("examples/Colmey_et_al_2026_Acta_Cryst_A/data/quartz-no-abs")
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
uv run diffbloch preprocess examples/Colmey_et_al_2026_Acta_Cryst_A/data/quartz-no-abs --refresh
```

## API example: loading a checkpointed `Plan`

The end-to-end preprocessing pipeline can take a long time, and it is deliberately decoupled from
refinement. You can checkpoint a settled `Plan` — for example, a plan whose orientations and
thicknesses have already been fitted — then run inference or refinement over that reusable scaffold
without recomputing the preprocessing recipe.

The path below is where a run deposits its checkpoint; it exists once you have run that experiment
at least once, since no example ships one pre-built.

```python
from diffBloch.preprocess import read_plan, require_built_plans

plan = read_plan("examples/Colmey_et_al_2026_Acta_Cryst_A/data/quartz-no-abs/reproducibility/plan.npz")
orientations = require_built_plans(plan)

print(len(orientations))
print(plan.structure_factor_grid.structure_factor_hkl.shape)
```
