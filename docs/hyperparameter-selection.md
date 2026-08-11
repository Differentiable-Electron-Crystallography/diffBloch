# Hyperparameter selection

Running an experiment requires the user to select numerical, preprocessing, and refinement settings. Hyperparameters control how diffBloch performs the calculation but are not themselves determined by structural refinement.

These choices are recorded in `experiment.yaml`. diffBloch keeps this file short by supplying
defaults for common settings and automatically deriving quantities already defined by the input
data.

This page lists every config, its default, and what it controls. For guidance in selecting appropriate `g_max`, `sg_max`, and `rocking_curve_sampling` specifically, see [Convergence testing](convergence-testing.md).

## What is *not* config: auto-filled from CIF/PETS

Some values that other refinement packages expose as settings are deliberately **not** config
fields in diffBloch — they are read from the structure `.cif` or `.cif_pets` file at load time, so
they cannot silently drift from the data they describe:

| Value | Source | Notes |
|---|---|---|
| Electron energy / wavelength | `.cif_pets` wavelength | Converted to electron energy for the simulation. |
| UB orientation matrix | `.cif_pets`, per rotation | Each rotation's own PETS UB seeds its starting orientation, later refined by `preprocess.orientation` (below) if enabled. |
| Integration semiangle | `.cif_pets` precession angle | The tilt half-width of the rocking curve. |
| Atomic positions, ADPs, occupancies, symmetry ops | structure `.cif` | All atomic content comes from the structure CIF. |
| Structure-factor support radius | derived, `2 * blochwave.g_max` | Not independently configurable: a beam set bounded by `\|g\| <= g_max` produces `F(g - h)` terms reaching `2 * g_max`, so a separate support setting could silently contradict the solve cutoff. |

## `sample`

Fixed sample properties (`diffBloch.config.schema.SampleConfig`).

| Field | Default | What it does |
|---|---|---|
| `thicknesses` | `(820.0,)` | Seed specimen thickness in Å, one shared value for every rotation — required whenever the seed actually reaches a solve; harmless if `preprocess.optimize_thickness` runs first and overwrites it anyway. |

## `blochwave`

Numerical-accuracy controls frozen into the simulation spec (`BlochwaveConfig`).

| Field | Default | What it does |
|---|---|---|
| `solver.refine` | `"matrix_exp"` | Dynamical solver used during gradient refinement. Must stay `matrix_exp` if `absorption: true` (the alternative, `bloch_eigen`, isn't safe for the non-Hermitian absorptive structure matrix). |
| `solver.inference` | `"matrix_exp"` | Dynamical solver used for `infer` / preprocessing forward solves. |
| `absorption` | `false` | Include anomalous absorption as an imaginary structure-factor contribution. |
| `rsg` | `0.9` | Klar beam-selection cutoff: relative excitation-error radius. |
| `dsg` | `0.0015` | Klar beam-selection cutoff: excitation-error offset. |
| `rocking_curve_sampling` | `42` | Tilt samples integrated per rocking curve. See [Convergence testing](convergence-testing.md). |
| `mosaicity` | `false` | `true` converts the apparent mosaicity from `.cif_pets` into a moving-average width from the rocking-curve angular spacing; `false` applies no smoothing. The legacy `{window: N}` form sets the width directly. |
| `fixed_n_segments` | `12` | Number of tilt-coupling segments when `coupling_mode: "union"` and `union_adaptive: false`. |
| `coupling_mode` | `"union"` | `"union"`: couple the union of excited beams across each tilt-chunk's boundary tilts. `"per_tilt"`: couple only each tilt's own excited beams (more accurate, more expensive). |
| `g_max` | `2.25` | Solve cutoff (Å⁻¹): maximum reciprocal-vector length of beams entering the Bloch-wave matrix. See [Convergence testing](convergence-testing.md). |
| `sg_max` | `0.01` | Maximum excitation-error magnitude (Å⁻¹) for a beam to enter the simulation at a sampled tilt. |
| `union_adaptive` | `true` | Use recursive-bisection adaptive tilt-chunk boundaries instead of `fixed_n_segments` evenly-sized chunks. |
| `union_max_new_beams_pct` | `0.01` | Adaptive-chunking split threshold: a chunk splits further while its midpoint tilt would add more than this fraction of new beams beyond its boundary union. |
| `ignore_orientations` | `()` | Zero-based PETS rotation indices to exclude from the whole experiment (damaged/empty/diagnostic frames). |

## `preprocess`

Which preprocessing steps run and their search bounds (`PreprocessConfig`).

| Field | Default | What it does |
|---|---|---|
| `optimize_orientation` | `true` | Run the per-rotation Nelder-Mead orientation search. |
| `optimize_thickness` | `true` | Run the per-rotation thickness grid search. |
| `stage_order` | `"orientation_first"` | Which fitting stage runs first when both are enabled. `"orientation_first"`: fit orientation against the seed thickness, then thickness against the fitted orientation. `"thickness_first"`: reversed. |
| `orientations_csv` | `None` | Path (relative to the experiment directory) to a `Rotation Index`/`Orientation Matrix` CSV that overwrites every candidate's orientation before the recipe's fitting steps. `optimize_orientation` still controls whether the search then refines from that imported seed. Overridden by the CLI's `--orientations-csv` flag. |

### `preprocess.orientation.nelder_mead`

Bounds for the local orientation search (`NelderMeadOptimizationConfig`).

| Field | Default | What it does |
|---|---|---|
| `step_size` | `0.05` | Initial simplex step size, degrees. |
| `max_iterations` | `60` | Maximum Nelder-Mead iterations (`maxiter`). |
| `x_tolerance` | `1e-3` | Simplex convergence tolerance on the orientation parameters (`xatol`). |
| `f_tolerance` | `1e-3` | Simplex convergence tolerance on the objective (`fatol`). |
| `penalize_fewer_reflections` | `true` | Penalize trial orientations that match fewer reflections than the seed, discouraging the search from drifting to a trivially-easier beam set. |

### `preprocess.thickness`

Bounds for the per-rotation thickness grid search (`ThicknessOptimizationConfig`).

| Field | Default | What it does |
|---|---|---|
| `min_thickness` | `5.0` | Lower bound, Å. |
| `max_thickness` | `2000.0` | Upper bound, Å. |
| `n_steps` | `100` | Number of evenly-spaced grid candidates (inclusive endpoints). |
| `plot` | `false` | Write one wR2-vs-thickness PNG per rotation (`<inputs.structure's directory>/thickness_optim`). Reporting-only — never affects the fitted `Plan` and is excluded from the reproducibility digest. |

## `loss_metrics`

The one residual driving both preprocessing search and gradient refinement (`LossMetricsConfig`).
Top-level, not nested under `refinement`, because it governs preprocessing too.

| Field | Default | What it does |
|---|---|---|
| `residual` | `"wr2"` | `"wr2"`, `"least_squares"`, or `"robs"`. Parses into the matching loss (gradient refinement) and per-thickness scores (preprocessing search) function pair — the two stages always agree on one metric. |

## `refinement`

Execution knobs for the default single-stage `refine` (`RefinementConfig`).

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
| `exp_data` | required | Relative path to a `.cif_pets`. |
| `load_hydrogens` | `false` | Include hydrogen atom sites from the structure CIF (molecular crystals). |
