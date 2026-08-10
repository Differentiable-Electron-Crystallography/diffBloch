# Hyperparameter selection

Running an experiment requires the user to select numerical, preprocessing, and refinement settings. Hyperparameters control how diffBloch performs the calculation but are not themselves determined by structural refinement. 

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
| Unit cell (`a, b, c, α, β, γ`) | `.cif_pets`, checked against `.cif` | PETS's cell is authoritative for all simulation geometry; the structure CIF's own cell is only a consistency check (warns past 1%, raises past 5%). See [Inputs](inputs.md#unit-cell-authority-pets-overrides-the-structure-cif). |
| UB orientation matrix | `.cif_pets`, per rotation | Each rotation's own PETS UB seeds its starting orientation, later refined by `preprocess.orientation` (below) if enabled. |
| Integration semiangle | `.cif_pets` precession angle | The tilt half-width; must be shared across every file when `inputs.multi_dataset` combines several `.cif_pets`. |
| Apparent mosaicity (Gaussian σ) | `.cif_pets` `_diffrn_measurement_details` | Only read when `blochwave.mosaicity: true`; missing from the file then raises rather than defaulting silently. |
| Atomic positions, ADPs, occupancies, symmetry ops | structure `.cif` | Unaffected by PETS's cell override — fractional coordinates are simply reinterpreted against PETS's metric. |
| Structure-factor support radius | derived, `2 * blochwave.g_max` | Not independently configurable: a beam set bounded by `\|g\| <= g_max` produces `F(g - h)` terms reaching `2 * g_max`, so a separate support setting could silently contradict the solve cutoff. |

## `sample`

Fixed sample properties (`diffBloch.config.schema.SampleConfig`).

| Field | Default | What it does |
|---|---|---|
| `thicknesses` | `(820.0,)` | Seed specimen thickness in Å, one shared value for every rotation. Under `inputs.multi_dataset: true`, must instead be a list of tuples, one per file in `inputs.exp_data` order (different specimens/regions can have genuinely different thickness) — required whenever the seed actually reaches a solve; harmless as a single value if `preprocess.optimize_thickness` runs first and overwrites it anyway. |

## `blochwave`

Numerical-accuracy controls frozen into the simulation spec (`BlochwaveConfig`).

| Field | Default | What it does |
|---|---|---|
| `solver.refine` | `"matrix_exp"` | Dynamical solver used during gradient refinement. Must stay `matrix_exp` if `absorption: true` (the alternative, `bloch_eigen`, isn't safe for the non-Hermitian absorptive structure matrix). |
| `solver.inference` | `"matrix_exp"` | Dynamical solver used for `run infer` / preprocessing forward solves. |
| `absorption` | `false` | Include anomalous absorption as an imaginary structure-factor contribution. |
| `rsg` | `0.9` | Klar beam-selection cutoff: relative excitation-error radius. |
| `dsg` | `0.0015` | Klar beam-selection cutoff: excitation-error offset. |
| `rocking_curve_sampling` | `50` | Tilt samples integrated per rocking curve. See [Convergence testing](convergence-testing.md). |
| `mosaicity` | `true` | Use PETS's apparent-mosaicity σ for Gaussian orientation averaging (three-point quadrature) instead of a single static solve per tilt. |
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
| `stage_order` | `"thickness_first"` | Which fitting stage runs first when both are enabled. `"thickness_first"`: fit thickness against the seed orientation, then orientation against the fitted thickness. `"orientation_first"`: reversed. |
| `orientations_csv` | `None` | Path (relative to the experiment directory) to a `Rotation Index`/`Orientation Matrix` CSV that overwrites every candidate's orientation before the recipe's fitting steps. `optimize_orientation` still controls whether the search then refines from that imported seed. Overridden by the CLI's `--orientations-csv` flag. |

### `preprocess.orientation.nelder_mead`

Bounds for the local orientation search (`NelderMeadOptimizationConfig`).

| Field | Default | What it does |
|---|---|---|
| `step_size` | scipy default | Initial simplex step size, degrees. |
| `max_iterations` | scipy default | Maximum Nelder-Mead iterations (`maxiter`). |
| `x_tolerance` | scipy default | Simplex convergence tolerance on the orientation parameters (`xatol`). |
| `f_tolerance` | scipy default | Simplex convergence tolerance on the objective (`fatol`). |
| `penalize_fewer_reflections` | scipy default | Penalize trial orientations that match fewer reflections than the seed, discouraging the search from drifting to a trivially-easier beam set. |

### `preprocess.thickness`

Bounds for the per-rotation thickness grid search (`ThicknessOptimizationConfig`).

| Field | Default | What it does |
|---|---|---|
| `min_thickness` | grid default | Lower bound, Å. Under `inputs.multi_dataset`, may be a per-dataset list (same length/order as `inputs.exp_data`) instead of one shared scalar — both bounds must move to the per-dataset shape together. |
| `max_thickness` | grid default | Upper bound, Å. Same per-dataset rule as `min_thickness`. |
| `n_steps` | grid default | Number of evenly-spaced grid candidates. Always shared across datasets (only the range, not the resolution, differs per dataset). |
| `plot` | `false` | Write one wR2-vs-thickness PNG per rotation (`<inputs.structure's directory>/thickness_optim`). Reporting-only — never affects the fitted `Plan` and is excluded from the reproducibility digest. |

## `loss_metrics`

The one residual driving both preprocessing search and gradient refinement (`LossMetricsConfig`).
Top-level, not nested under `refinement`, because it governs preprocessing too.

| Field | Default | What it does |
|---|---|---|
| `residual` | `"wr2"` | `"wr2"`, `"least_squares"`, or `"robs"`. Parses into the matching loss (gradient refinement) and per-thickness scores (preprocessing search) function pair — the two stages always agree on one metric. |

## `refinement`

Execution knobs for the default single-stage `run refine` (`RefinementConfig`).

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
| `enabled` | spec default | Whether a learned thickness model replaces the fixed per-rotation seed during refinement. |
| `num_samples` | spec default | Number of thickness samples the network's input encoding uses. |
| `sample_thickness` | spec default | Whether thickness itself is sampled as part of the network's forward pass. |
| `form` | `"min_thickness"` | Functional form of the network's output (currently only one implemented). |
| `min_thickness` | spec default | Lower bound, Å, for the network's thickness output. |
| `max_thickness` | spec default | Upper bound, Å, for the network's thickness output. |
| `init_seed` | spec default | Random seed for the network's initial weights. |

## `inputs`

Input file references, relative to the experiment directory only (`Inputs`).

| Field | Default | What it does |
|---|---|---|
| `structure` | required | Relative path to the structure `.cif`. |
| `exp_data` | required | Relative path to a `.cif_pets`, or (with `multi_dataset: true`) a list of 2+ paths. |
| `multi_dataset` | `false` | Combine rotations from every file in `exp_data` into one experiment. Requires `exp_data` to be a list of 2+ paths; every file must share one integration semiangle. |
| `load_hydrogens` | `false` | Include hydrogen atom sites from the structure CIF (molecular crystals). |
