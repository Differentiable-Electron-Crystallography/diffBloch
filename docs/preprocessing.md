# Preprocessing and `Plan`

Dynamical diffraction is extremely sensitive to crystal orientation and thickness. diffBloch
therefore determines this experimental metadata before refining the structure and stores it in a `Plan`.


## Convergence Testing

Determine suitable `g_max`, `sg_max`, and `rocking_curve_sampling` simulation parameters to be used before optimizing thickness or
orientation and before refining the structure. 

For more information, see [Convergence testing](convergence-testing.md).

## Orientation

The `.cif_pets` file supplies a **UB matrix**. The reciprocal-basis matrix {math}`B` is calculated from the unit
cell and reciprocal lattice. The matrix
{math}`U` places that reciprocal lattice in the laboratory coordinate system. Their product {math}`UB`
therefore maps a reflection index directly into the measured laboratory geometry.

diffBloch separates the two matrices using

```{math}
U = (UB)B^{-1}.
```

For virtual frame {math}`i`, the goniometer angles recorded in `.cif_pets` are applied in their recorded order:

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

The Bloch wave intensities are recalculated for each trial and compared with the observed
intensities using `loss_metrics.residual`, which defaults to `wr2`.

### Unit-cell authority

The reciprocal-basis matrix {math}`B` above is built from the cell recorded in `.cif_pets`, not the
structure CIF's. The `.cif_pets` cell is authoritative for every piece of simulation geometry that needs a unit
cell — the structure-factor grid, the reciprocal basis, the cell volume, the ADP {math}`U^*`-frame
conversion, and the beam geometry derived from that grid. The structure CIF still supplies every
piece of atomic content: positions, atom types, occupancies, ADPs, and symmetry operators.
Fractional coordinates are read from the CIF unchanged; they are interpreted using the `.cif_pets`
cell rather than the CIF's own.

diffBloch checks the CIF cell against the `.cif_pets` cell on load:

- **> 1% relative difference** on any of `a, b, c, alpha, beta, gamma` logs a warning stating that
  the `.cif_pets` value overrides the CIF value. 
- **> 5% relative difference** raises `ValueError` and stops. A gap this large usually means the two files describe different crystals or settings entirely.

For a multi-dataset experiment, the first `.cif_pets` file's cell is the shared authoritative cell. The
structure CIF and every further `.cif_pets` file are checked against it under the same thresholds.

### Orientation-search method

diffBloch applies the general Nelder--Mead optimization algorithm
([Nelder and Mead, 1965](https://doi.org/10.1093/comjnl/7.4.308)) to orientation refinement.

Nelder--Mead minimizes the orientation residual without requiring its derivatives. For the three
correction angles, the search holds four trial points: zero correction and one point displaced by
`step_size` along each angle. These four points form a simplex.

Each point is evaluated by running the Bloch wave calculation and comparing its intensities with
experiment. Nelder--Mead ranks the four residuals and replaces the worst point by reflecting it
through the opposite face of the simplex. Depending on the new residual, it may expand farther in
that direction, contract toward the better points, or shrink the whole simplex around the best
point. The simplex therefore moves and changes size as it approaches a local minimum. The final
three coordinates are applied to the `.cif_pets` orientation as
{math}`(\Delta\alpha,\Delta\beta,\Delta\omega)`.

This is a local, unconstrained search. `step_size` defines only the initial simplex; it neither
limits the correction angles nor guarantees that the search finds the global minimum. The `.cif_pets` UB
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
| Shared thickness | Uses the value in `sample.thickness` for every orientation. |
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
`max_thickness`, inclusive, for every orientation. Each candidate is passed through the Bloch wave
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

The neural-network option (`refinement.thickness_nn`) has its own `sample_thickness` switch: when
enabled, thickness itself is sampled as part of the network's forward pass during refinement,
rather than the network predicting one deterministic thickness per orientation. See
[Hyperparameter selection](hyperparameter-selection.md) for the full field
list.


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
            orientation=...,  # one rotation/orientation
            tilts=...,  # rocking-curve sub-orientations
            thickness=...,  # fitted value
        ),
        # ...more rotations...
    ),
    provenance=(...),  # ordered preprocessing recipe
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
orientation. `build_orientation_plans` constructs those sub-tilts, selects each tilt-dependent SOLVE
basis from `g_max` and `sg_max`, and builds the Bloch geometry.

Mosaicity describes the spread of crystal orientations within the illuminated volume. Different
mosaic domains are slightly misoriented, so a reflection is excited over a wider angular range than
it would be in a single perfectly oriented crystal. Mosaicity therefore broadens the calculated
rocking curve and changes the integrated intensity, especially for reflections whose excitation
condition varies rapidly with angle.

With `blochwave.mosaicity: true`, diffBloch reads the apparent mosaicity in degrees from the source
`.cif_pets` and converts it to an internal sampled-tilt span using

```{math}
\Delta\theta = \frac{2\theta_{\mathrm{semi}}}{N}, \qquad
s = \operatorname{round}\left(\frac{m}{\Delta\theta}\right).
```

{math}`\theta_{\mathrm{semi}}` is the rocking-curve integration semiangle, the tilt half-width
around each orientation's nominal angle. {math}`N` is `rocking_curve_sampling`, the number of tilt
samples spanning that full range. {math}`\Delta\theta` is the resulting angular spacing between
adjacent tilt samples. {math}`m` is the apparent mosaicity in degrees read from `.cif_pets`.
{math}`s` is the internal sample span, rounded to the nearest integer.

The calculated rocking curve is averaged over that span before integration. A span of zero or one
leaves the curve unchanged. This uses the existing {math}`N` Bloch wave solves rather than
adding orientations, so enabling mosaicity adds no preprocessing or refinement cost.

Set `blochwave.mosaicity: false` (the default) to evaluate only the nominal tilt orientations and
sum their intensities without mosaic broadening. In that mode any PETS mosaicity value is ignored.
