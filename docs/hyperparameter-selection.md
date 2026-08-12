# Hyperparameter selection

Running an experiment requires the user to select numerical, preprocessing, and refinement settings. Hyperparameters control how diffBloch performs the calculation but are not themselves determined by structural refinement.

These choices are recorded in `experiment.yaml`. diffBloch keeps this file short by supplying
defaults for common settings and automatically deriving quantities already defined by the input
data. Importantly, values specified in the `experiment.yaml` will override defaults.

This page lists every config, its default, and what it controls. For guidance in selecting appropriate `g_max`, `sg_max`, and `rocking_curve_sampling` specifically, see [Convergence testing](convergence-testing.md).

## Values read from `.cif` and `.cif_pets`

Some values that other refinement packages expose as settings are deliberately **not** config
fields in diffBloch — they are read from the structure `.cif` or `.cif_pets` file at load time, so
they cannot silently drift from the data they describe:

| Value | Source | Notes |
|---|---|---|
| Electron energy / wavelength | `.cif_pets` wavelength | Converted to energy and snapped onto the nearest standard TEM voltage when close (`snap_to_standard_energy`). The wavelength is recorded to limited precision, so this recovers the operator-selected voltage exactly. |
| Unit cell (`a, b, c, α, β, γ`) | `.cif_pets`, checked against `.cif` | The `.cif_pets` cell is authoritative for all simulation geometry; the structure CIF's own cell is only a consistency check (logs a warning past 1%, raises past 5%). See [Inputs and outputs](inputs.md). |
| UB orientation matrix | `.cif_pets`, per rotation | Each rotation's UB supplies its starting orientation, later refined by `preprocess.orientation` (below) if enabled. |
| Integration semiangle | `.cif_pets` precession angle | The tilt half-width, read per file; under `inputs.multi_dataset` each dataset's recipe uses its own (precession angles may differ between files). |
| Apparent mosaicity (degrees) | `.cif_pets` `_diffrn_measurement_details` | Only read when `blochwave.mosaicity: true`; missing from the file then raises rather than defaulting silently. |
| Atomic positions, ADPs, occupancies, symmetry ops | structure `.cif` | Unaffected by the `.cif_pets` cell override — fractional coordinates are interpreted using the `.cif_pets` cell. |
| Structure-factor support radius | derived, `2 * blochwave.g_max` | Not independently configurable: a beam set bounded by `\|g\| <= g_max` produces `F(g - h)` terms reaching `2 * g_max`, so a separate support setting could silently contradict the solve cutoff. |

## `sample`

Fixed sample properties (`diffBloch.config.schema.SampleConfig`).

| Field | Default | What it does |
|---|---|---|
| `thicknesses` | `(820.0,)` | Seed specimen thickness in Å, one shared value for every rotation (and every dataset under `inputs.multi_dataset`); `preprocess.optimize_thickness` then fits each rotation individually, so datasets with genuinely different thickness converge per rotation from the shared seed. |

## `blochwave`

Bloch wave simulation hyperparameters (`BlochwaveConfig`).

| Field | Default | What it does |
|---|---|---|
| `solver` | `"matrix_exp"` | Solver used for preprocessing, inference, and refinement. Use `matrix_exp` when absorption is enabled -- the alternative, `bloch_eigen`, isn't safe for the non-Hermitian absorptive structure matrix. |
| `absorption` | `false` | Include anomalous absorption as an imaginary structure-factor contribution. |
| `rsg` | `0.66` | Relative excitation-error cutoff. See [`rsg` and `dsg`](#rsg-and-dsg). |
| `dsg` | `0.0015` | Absolute excitation-error margin. See [`rsg` and `dsg`](#rsg-and-dsg). |
| `rocking_curve_sampling` | `50` | Tilt samples integrated per rocking curve. See [Convergence testing](convergence-testing.md). |
| `mosaicity` | `false` | `true` reads the apparent mosaicity in degrees from `.cif_pets` and converts it into an internal moving-average sample span. No additional orientations are simulated. |
| `coupling_mode` | `"union"` | See [Union coupling](#union-coupling). |
| `union_adaptive` | `true` | Choose union sections adaptively. See [Union coupling](#union-coupling). |
| `fixed_n_segments` | `12` | Number of union sections when adaptive splitting is disabled. See [Union coupling](#union-coupling). |
| `union_max_new_beams_pct` | `0.01` | Threshold for adaptive splitting. See [Union coupling](#union-coupling). |
| `g_max` | `2.25` | Largest reflection {math}`g` vector simulated (Å⁻¹). Off-diagonal structure factors extend to {math}`2g_\mathrm{max}`. |
| `sg_max` | `0.01` | Maximum excitation-error magnitude (Å⁻¹) for a beam to enter the simulation at a sampled tilt. |
| `ignore_orientations` | `()` | Zero-based `.cif_pets` rotation indices to exclude from the experiment. |

### `rsg` and `dsg`

Continuous-rotation data is recorded in the `.cif_pets` file as overlapping **virtual frames**
([Klar *et al.*, 2023](https://doi.org/10.1038/s41557-023-01186-1)). Because the frames overlap, a
given reflection can appear in several neighbouring frames, but should only be fully integrated in
one of them. `rsg` and `dsg` identify the frames in which a reflection passes sufficiently far
through the Bragg condition to be treated as fully integrated.

For each reflection, diffBloch calculates its excitation error {math}`|S_g|` at the centre of the
frame and the excitation-error half-range {math}`\Delta S_g` swept from the centre to the edge of
that frame. The reflection is retained when both conditions are satisfied:

```{math}
\frac{|S_g|}{\Delta S_g} < rsg, \qquad \Delta S_g - |S_g| > dsg.
```

`rsg` is dimensionless. It limits the reflection's central excitation error relative to its swept
range. Increasing `rsg` retains reflections farther from the centre of that range. `dsg` is an
absolute margin in Å⁻¹. Increasing `dsg` rejects reflections that only just enter the integration
range. These parameters select the reflections compared with experiment; they do not select the
beams included in the Bloch wave calculation.

### Union coupling

Each virtual frame covers a small angular range of the crystal's rotation. To simulate its
integrated intensity, diffBloch samples a rocking curve across that range. `rocking_curve_sampling`
sets the number of samples. Each sample is a tilt: one crystal orientation within the virtual
frame. The intensities calculated at all tilts are summed to give the simulated integrated
intensity for that frame.

Changing the tilt changes every reflection's excitation error {math}`S_g`. At each tilt, beams are
selected when {math}`|S_g| < sg_\mathrm{max}`. In `per_tilt` mode, diffBloch applies this test and
builds a separate structure matrix at every tilt. This keeps each matrix as small as possible, but
repeatedly calculates {math}`S_g` for all candidate reflections and prevents the tilts from being
solved together efficiently.

`union` mode groups neighbouring tilts. Every tilt in a group uses one combined beam set containing
the beams selected at the group's boundaries. The off-diagonal elements describe coupling between
these beams through the structure factors and are reused across the group. The diagonal contains
the excitation error of each beam and is recalculated for every tilt. The matrices are then solved
as a batch, making better use of GPU parallelization. This is usually faster when the individual
structure matrices are small or moderate in size.

The combined beam set is larger than the set needed by any one tilt. Union mode therefore solves
larger matrices and stores several of them on the GPU at once. When the matrices are already large,
or GPU memory is nearly full, this cost can outweigh the benefit of batching. In that case,
`coupling_mode: "per_tilt"` is likely to be faster and use less memory. See
[Devices and scaling](devices-and-scaling.md) for measured comparisons.

With `union_adaptive: true`, diffBloch makes shorter groups where the beam set changes rapidly and
longer groups where it remains stable: starting from one chunk spanning the whole rocking curve,
each chunk's midpoint tilt is checked against the beam union already covered by its two endpoints,
and the chunk is split in two and recursed into if the midpoint would add more than
`union_max_new_beams_pct` of new beams beyond that union. Adaptive chunking costs more beam-set
unions up front (to evaluate the split predicate) but generally solves fewer total beams than a
fixed split sized conservatively enough to avoid under-coupling.

With `union_adaptive: false`, `fixed_n_segments` sets the number of equal groups regardless of how
much the beam set actually changes across them -- more predictable when a fixed, known chunk count
is wanted instead.

## `preprocess`

Which preprocessing steps run and their search bounds (`PreprocessConfig`).

| Field | Default | What it does |
|---|---|---|
| `optimize_orientation` | `true` | Run the per-rotation Nelder-Mead orientation search. |
| `optimize_thickness` | `true` | Run the per-rotation thickness grid search. |
| `stage_order` | `"thickness_first"` | Which optimization stage runs first when both are enabled. `"thickness_first"`: optimize thickness against the seed orientation, then orientation against the optimized thickness. `"orientation_first"`: reversed. |

### `preprocess.orientation.nelder_mead`

Bounds for the local orientation search (`NelderMeadOptimizationConfig`).

| Field | Default | What it does |
|---|---|---|
| `step_size` | `0.05` | Initial simplex step size, degrees. |
| `max_iterations` | `60` | Maximum Nelder-Mead iterations (`maxiter`). |
| `x_tolerance` | `0.001` | Simplex convergence tolerance on the orientation parameters (`xatol`). |
| `f_tolerance` | `0.001` | Simplex convergence tolerance on the objective (`fatol`). |
| `penalize_fewer_reflections` | `true` | Penalize trial orientations that match fewer reflections than the seed, discouraging the search from drifting to a trivially-easier beam set. |

### `preprocess.thickness`

Bounds for the per-rotation thickness grid search (`ThicknessOptimizationConfig`).

| Field | Default | What it does |
|---|---|---|
| `min_thickness` | `5.0` | Lower bound, Å (one shared grid; the search itself is per rotation, so pooled datasets fit their own thicknesses within it). |
| `max_thickness` | `2000.0` | Upper bound, Å. |
| `n_steps` | `100` | Number of evenly-spaced grid candidates. |
| `plot` | `false` | Write one wR2-vs-thickness PNG per rotation (`<inputs.structure's directory>/thickness_optim`). Reporting-only — never affects the fitted `Plan` and is excluded from the reproducibility digest. |

## `loss_metrics`

The one residual driving both preprocessing search and gradient refinement (`LossMetricsConfig`).
Top-level, not nested under `refinement`, because it governs preprocessing too.

| Field | Default | What it does |
|---|---|---|
| `residual` | `"wr2"` | `"wr2"` or `"robs"`. Both re-fit an optimal multiplicative scale between calculated and observed intensities before scoring (`core.losses.optimal_scale`) because calculated and `.cif_pets` intensities use different scales. |

## `refinement`



| Field | Default | What it does |
|---|---|---|
| `steps` | `40` | Number of gradient-refinement epochs. |
| `optimizer.name` | `"lbfgs"` | `"lbfgs"`, `"adam"`, or `"adamw"`. |
| `optimizer.lr` | `1e-3` | Learning rate (Adam/AdamW; L-BFGS uses its own internal line search). |

### `refinement.trainable`


| Field | Default | What it does |
|---|---|---|
| `positions` | `"all"` | `"all"` or `"none"`: whether atomic positions are refined. |
| `adp` | `"all"` | `"all"` or `"none"`: whether atomic displacement parameters are refined. |
| `occupancy` | `"none"` | `"all"` or `"none"`: whether site occupancies are refined. |

### `refinement.split`

Train/validation split (`DataSplitConfig`).

| Field | Default | What it does |
|---|---|---|
| `train_test` | `false` | `false`: train on every rotation, no held-out set. `true`: hold out an evenly-spaced `val_frac` of rotations from the refinement objective (preprocessing still fits their orientation/thickness) and report their wR2/R_obs separately. |
| `val_frac` | `0.2` | Fraction of rotations held out when `train_test: true`. Must be strictly between 0 and 1. |

### `refinement.thickness_nn`

Apparent-thickness neural network used by the default refinement path (`ThicknessNNConfig`).

| Field | Default | What it does |
|---|---|---|
| `enabled` | `true` | Whether a learned thickness model replaces the fixed per-rotation seed during refinement. |
| `num_samples` | `40` | Number of thickness samples the network's input encoding uses. |
| `sample_thickness` | `false` | Whether thickness itself is sampled as part of the network's forward pass. |
| `form` | `"min_thickness"` | Functional form of the network's output (currently only one implemented). |
| `min_thickness` | `100.0` | Lower bound, Å, for the network's thickness output. |
| `max_thickness` | `2000.0` | Upper bound, Å, for the network's thickness output. |
| `init_seed` | `0` | Random seed for the network's initial weights. |

## `inputs`

Input file references, relative to the experiment directory only (`Inputs`).

| Field | Default | What it does |
|---|---|---|
| `structure` | required | Relative path to the structure `.cif`. |
| `exp_data` | required | Relative path to a `.cif_pets`, or (with `multi_dataset: true`) a list of 2+ paths. |
| `multi_dataset` | `false` | See [Combining multiple datasets](#combining-multiple-datasets) below. |
| `load_hydrogens` | `false` | Include hydrogen atom sites from the structure CIF (molecular crystals). |

### Combining multiple datasets

`inputs.multi_dataset: true` pools rotations from several `.cif_pets` files (`inputs.exp_data` as a
list of 2+ paths) into one experiment, each dataset keeping its own energy, orientation, and
thickness. Two situations call for it:

- **Beam damage series** — repeat measurements of the same crystal taken at increasing dose. Each
  dataset is its own `.cif_pets` file (its own UB, its own apparent thickness), but they refine one shared
  structure; combining them uses every rotation instead of picking one dataset and discarding the
  rest.
- **Low-symmetry structures** — a single tilt series from one crystal orientation range may not
  cover enough of reciprocal space to constrain the structure well when the space group has few
  symmetry operations to fill in the gaps. Multiple datasets from different crystal
  orientations/mounts fill in coverage that one series alone would leave thin.

Each dataset is preprocessed and checkpointed on its own (`plan.<stem>.npz` per file, with its own
integration geometry -- precession angles may differ between files), and the settled per-dataset
plans are pooled in memory just before refinement. See
[Inputs and outputs](inputs.md) for the mechanics (per-dataset checkpoints, the
first-file authoritative cell, one-energy rule, rotation-index offsets) and
[Reproducibility](reproducibility.md) for what each per-dataset lock verifies.
