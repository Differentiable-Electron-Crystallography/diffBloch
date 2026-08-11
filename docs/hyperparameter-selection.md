# Hyperparameter selection

An experiment requires simulation, preprocessing, and refinement hyperparameters. These control how diffBloch performs the calculation but are not themselves determined by structural refinement.

These choices are recorded in `experiment.yaml`. diffBloch keeps this file short by supplying
defaults for common settings and automatically deriving quantities already defined by the input
data. Importantly,

This page lists every config, its default, and what it controls. For guidance in selecting appropriate `g_max`, `sg_max`, and `rocking_curve_sampling` specifically, see [Convergence testing](convergence-testing.md).

## What is *not* config: auto-filled from CIF/PETS

Some values that other refinement packages expose as settings are deliberately **not** config
fields in diffBloch — they are read from the structure `.cif` or `.cif_pets` file at load time, so
they cannot silently drift from the data they describe:

| Value | Source | Notes |
|---|---|---|
| Electron energy / wavelength | `.cif_pets` wavelength | Converted to energy and snapped onto the nearest standard TEM voltage when close (`snap_to_standard_energy`). PETS records wavelength to limited precision, so this recovers the operator-selected voltage exactly. |
| Integration semiangle | `.cif_pets` precession angle | The tilt half-width; must be shared across every file when `inputs.multi_dataset` combines several `.cif_pets`. |
| Apparent mosaicity (degrees) | `.cif_pets` `_diffrn_measurement_details` | Used when `blochwave.mosaicity: true`. |

## `sample`

Sample properties.

| Field | Default | What it does |
|---|---|---|
| `thicknesses` | `(820.0,)` | Starting thickness in Å. For multiple datasets, provide one value per `.cif_pets` file. |

## `blochwave`

Bloch-wave simulation hyperparameters.

| Field | Default | What it does |
|---|---|---|
| `solver` | `"matrix_exp"` | Solver used for preprocessing, inference, and refinement. Use `matrix_exp` when absorption is enabled. |
| `absorption` | `false` | Include anomalous absorption as an imaginary structure-factor contribution. |
| `rsg` | `0.66` | Relative excitation-error cutoff. See [`rsg` and `dsg`](#rsg-and-dsg). |
| `dsg` | `0.0015` | Absolute excitation-error margin. |
| `rocking_curve_sampling` | `50` | Tilt samples integrated per rocking curve. See [Convergence testing](convergence-testing.md). |
| `mosaicity` | `false` | Convert the PETS apparent mosaicity into a moving-average width using the spacing between rocking-curve samples. No additional orientations are simulated. |
| `coupling_mode` | `"union"` | Beam-coupling method. See [Union coupling](#union-coupling). |
| `union_adaptive` | `true` | Choose union sections adaptively. See [Union coupling](#union-coupling). |
| `fixed_n_segments` | `12` | Number of union sections when adaptive splitting is disabled. See [Union coupling](#union-coupling). |
| `union_max_new_beams_pct` | `0.01` | Threshold for adaptive splitting. See [Union coupling](#union-coupling). |
| `g_max` | `2.25` | Largest reflection {math}`g` vector simulated (Å⁻¹). Off-diagonal structure factors extend to {math}`2g_\mathrm{max}`. |
| `sg_max` | `0.01` | Maximum excitation-error magnitude (Å⁻¹) for a beam to enter the simulation at a sampled tilt. |
| `ignore_orientations` | `()` | Zero-based PETS rotation indices to exclude from the whole experiment (damaged/empty/diagnostic frames). |

### `rsg` and `dsg`

PETS virtual frames overlap, so the same reflection may be recorded in neighbouring frames. `rsg`
and `dsg` identify the frames in which a reflection passes sufficiently far through the Bragg
condition to be treated as fully integrated.

For each reflection, diffBloch calculates its excitation error {math}`|S_g|` at the centre of the
frame and the excitation-error half-range {math}`\Delta S_g` swept from the centre to the edge of
that frame. The reflection is retained when both conditions are satisfied:

```{math}
\frac{|S_g|}{\Delta S_g} < rsg
```

```{math}
\Delta S_g-|S_g| > dsg.
```

`rsg` is dimensionless. It limits the reflection's central excitation error relative to its own
swept range. Increasing `rsg` retains reflections farther from the centre of that range. `dsg` is an
absolute margin in Å⁻¹. Increasing `dsg` rejects reflections that only just enter the integration
range. These parameters select the reflections compared with experiment for residual calculation; they do not set the beams
included in the Bloch-wave calculation.

### Union coupling

Each PETS virtual frame covers a small angular range of the crystal's rotation. To simulate its
integrated intensity, diffBloch samples a rocking curve across that range. `rocking_curve_sampling`
sets the number of samples. Each sample is a **tilt**: one crystal orientation within the virtual
frame. The intensities calculated at all tilts are summed to give the simulated integrated intensity for that
frame.

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
`coupling_mode: "per_tilt"` is likely to be faster and use less memory.

With `union_adaptive: true`, diffBloch makes shorter groups where the beam set changes rapidly and
longer groups where it remains stable. `union_max_new_beams_pct` controls when a group is split. With
adaptive grouping disabled, `fixed_n_segments` sets the number of equal groups.

## `preprocess`



| Field | Default | What it does |
|---|---|---|
| `optimize_orientation` | `true` | Run the per-rotation Nelder-Mead orientation search. |
| `optimize_thickness` | `true` | Run the per-rotation thickness grid search. |
| `stage_order` | `"thickness_first"` | Which fitting stage runs first when both are enabled. `"thickness_first"`: fit thickness against the starting orientation, then orientation against the fitted thickness. `"orientation_first"`: reversed. |


### `preprocess.orientation.nelder_mead`

Settings for the local orientation search.

| Field | Default | What it does |
|---|---|---|
| `step_size` | `0.05` | Initial simplex step size, degrees. |
| `max_iterations` | `60` | Maximum number of Nelder–Mead iterations. |
| `x_tolerance` | `0.001` | Convergence tolerance on the orientation angles, degrees. |
| `f_tolerance` | `0.001` | Convergence tolerance on the residual. |
| `penalize_fewer_reflections` | `true` | Penalize trial orientations that match fewer reflections. |

### `preprocess.thickness`

Bounds for the per-rotation thickness grid search (`ThicknessOptimizationConfig`).

| Field | Default | What it does |
|---|---|---|
| `min_thickness` | 10| Lower bound, Å. Under `inputs.multi_dataset`, may be a per-dataset list (same length/order as `inputs.exp_data`) instead of one shared scalar — both bounds must move to the per-dataset shape together. |
| `max_thickness` | 2000 | Upper bound, Å. Same per-dataset rule as `min_thickness`. |
| `n_steps` | 100 | Number of evenly-spaced grid candidates. Always shared across datasets (only the range, not the resolution, differs per dataset). |
| `plot` | `false` | Write one wR2-vs-thickness PNG per rotation (`<inputs.structure's directory>/thickness_optim`). Reporting-only — never affects the fitted `Plan` and is excluded from the reproducibility digest. |

## `loss_metrics`

The residual used for preprocessing and refinement.

| Field | Default | What it does |
|---|---|---|
| `residual` | `"wr2"` | Use weighted {math}`R2` or {math}`R_\mathrm{obs}`. |

For calculated intensities {math}`I_{\mathrm{calc},h}`, observed intensities
{math}`I_{\mathrm{obs},h}`, and fitted scale {math}`k`, `wr2` is

```{math}
wR2 = \sqrt{
\frac{\sum_h \left[w_h\left(kI_{\mathrm{calc},h}-I_{\mathrm{obs},h}\right)\right]^2}
     {\sum_h \left(w_hI_{\mathrm{obs},h}\right)^2}
}.
```

The weights are

```{math}
w_h = \frac{1}{\sqrt{\sigma_{\sqrt I,h}^2+
\left(0.01\sqrt{I_{\mathrm{obs},h}}\right)^2}},
```

with

```{math}
\sigma_{\sqrt I,h} =
\begin{cases}
5\sqrt{\sigma_h}, & I_{\mathrm{obs},h}<0.01\sigma_h,\\
\dfrac{\sigma_h}{2\sqrt{I_{\mathrm{obs},h}}}, & \text{otherwise}.
\end{cases}
```

`robs` is calculated only over reflections satisfying {math}`I_{\mathrm{obs},h}>3\sigma_h`:

```{math}
R_\mathrm{obs} =
\frac{\sum_{I_{\mathrm{obs},h}>3\sigma_h}
\left|\sqrt{I_{\mathrm{obs},h}}-\sqrt{kI_{\mathrm{calc},h}}\right|}
{\sum_{I_{\mathrm{obs},h}>3\sigma_h}\sqrt{I_{\mathrm{obs},h}}}.
```

For each comparison, diffBloch chooses {math}`k` to minimize the selected residual.

## `refinement`


| Field | Default | What it does |
|---|---|---|
| `steps` | `40` | Number of gradient-refinement epochs. |
| `optimizer.name` | `"lbfgs"` | `"lbfgs"`, `"adam"`, or `"adamw"`. |
| `optimizer.lr` | `1e-3` | Learning rate (Adam/AdamW; L-BFGS uses its own internal line search). |

### `refinement.trainable`

Whole-group trainable selections (`TrainableConfig`). Element-filtered selections (e.g. freeze
hydrogens) are Python/API composition, not config — see `engine.with_hydrogen_riding`.

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

Apparent-thickness neural network used during refinement.

| Field | Default | What it does |
|---|---|---|
| `enabled` | `true` | Whether a learned thickness model replaces the fixed per-rotation starting thickness during refinement. |
| `num_samples` | `40` | Number of thickness samples used by the network. |
| `sample_thickness` | `false` | Whether thickness is sampled during the network calculation. |
| `form` | `"min_thickness"` | Functional form of the network's output (currently only one implemented). |
| `min_thickness` | `100.0` | Lower bound, Å, for the network's thickness output. |
| `max_thickness` | `2000.0` | Upper bound, Å, for the network's thickness output. |

## `inputs`

Input file references, relative to the experiment directory only (`Inputs`).

| Field | Default | What it does |
|---|---|---|
| `structure` | required | Relative path to the structure `.cif`. |
| `exp_data` | required | Relative path to a `.cif_pets`, or (with `multi_dataset: true`) a list of 2+ paths. |
| `multi_dataset` | `false` | Combine rotations from every file in `exp_data` into one experiment. Requires `exp_data` to be a list of 2+ paths; every file must share one integration semiangle. |
| `load_hydrogens` | `false` | Include hydrogen atom sites from the structure CIF (molecular crystals). |
