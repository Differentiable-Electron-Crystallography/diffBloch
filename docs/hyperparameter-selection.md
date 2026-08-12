# Hyperparameter selection

Running an experiment requires the user to select numerical, preprocessing, and refinement settings. Hyperparameters control how diffBloch performs the calculation but are not themselves determined by structural refinement.

These choices are recorded in `experiment.yaml`. diffBloch keeps this file short by supplying
defaults for common settings and automatically deriving quantities already defined by the input
data. Importantly, values specified in the `experiment.yaml` will override defaults.

This page lists every config, its default, and what it controls. For guidance in selecting appropriate `g_max`, `sg_max`, and `rocking_curve_sampling` specifically, see [Convergence testing](convergence-testing.md).

## What is *not* config: auto-filled from CIF/PETS

Some values that other refinement packages expose as settings are deliberately **not** config
fields in diffBloch — they are read from the structure `.cif` or `.cif_pets` file at load time, so
they cannot silently drift from the data they describe:

| Value | Source | Notes |
|---|---|---|
| Electron energy / wavelength | `.cif_pets` wavelength | Converted to energy and snapped onto the nearest standard TEM voltage when close (`snap_to_standard_energy`). PETS records wavelength to limited precision, so this recovers the operator-selected voltage exactly. |
| Unit cell (`a, b, c, α, β, γ`) | `.cif_pets`, checked against `.cif` | PETS's cell is authoritative for all simulation geometry; the structure CIF's own cell is only a consistency check (logs a warning past 1%, raises past 5%). See [Inputs and outputs](inputs.md#unit-cell). |
| UB orientation matrix | `.cif_pets`, per rotation | Each rotation's own PETS UB seeds its starting orientation, later refined by `preprocess.orientation` (below) if enabled. |
| Integration semiangle | `.cif_pets` precession angle | The tilt half-width, read per file; under `inputs.multi_dataset` each dataset's recipe uses its own (precession angles may differ between files). |
| Apparent mosaicity (degrees) | `.cif_pets` `_diffrn_measurement_details` | Only read when `blochwave.mosaicity: true`; missing from the file then raises rather than defaulting silently. |
| Atomic positions, ADPs, occupancies, symmetry ops | structure `.cif` | Unaffected by PETS's cell override — fractional coordinates are simply reinterpreted against PETS's metric. |
| Structure-factor support radius | derived, `2 * blochwave.g_max` | Not independently configurable: a beam set bounded by `\|g\| <= g_max` produces `F(g - h)` terms reaching `2 * g_max`, so a separate support setting could silently contradict the solve cutoff. |

## `sample`

Fixed sample properties (`diffBloch.config.schema.SampleConfig`).

| Field | Default | What it does |
|---|---|---|
| `thicknesses` | `(820.0,)` | Seed specimen thickness in Å, one shared value for every rotation (and every dataset under `inputs.multi_dataset`); `preprocess.optimize_thickness` then fits each rotation individually, so datasets with genuinely different thickness converge per rotation from the shared seed. |

## `blochwave`

The beam-selection, coupling, and dynamical-solver settings (`BlochwaveConfig`), ordered here by
how often you'd actually reach for them.

| Field | Default | What it does |
|---|---|---|
| `g_max` | `2.25` | Solve cutoff (Å⁻¹): maximum reciprocal-vector length of beams entering the Bloch wave matrix. See [Convergence testing](convergence-testing.md). |
| `sg_max` | `0.01` | Maximum excitation-error magnitude (Å⁻¹) for a beam to enter the simulation at a sampled tilt. |
| `rocking_curve_sampling` | `50` | Tilt samples integrated per rocking curve. See [Convergence testing](convergence-testing.md). |
| `absorption` | `false` | Include anomalous absorption as an imaginary structure-factor contribution. |
| `mosaicity` | `false` | `true` converts the apparent mosaicity from `.cif_pets` into a moving-average width from the rocking-curve angular spacing; `false` applies no smoothing. The legacy `{window: N}` form sets the width directly. |
| `rsg` | `0.66` | Klar beam-selection cutoff: relative excitation-error radius. |
| `dsg` | `0.0015` | Klar beam-selection cutoff: excitation-error offset. |
| `coupling_mode` | `"union"` | `"union"`: couple the union of excited beams across each tilt-chunk's boundary tilts. `"per_tilt"`: couple only each tilt's own excited beams (more accurate, more expensive). |
| `union_adaptive` | `true` | See [Adaptive tilt-chunk coupling](#adaptive-tilt-chunk-coupling) below. |
| `union_max_new_beams_pct` | `0.01` | Adaptive-chunking split threshold — see below. |
| `fixed_n_segments` | `12` | Number of tilt-coupling segments when `coupling_mode: "union"` and `union_adaptive: false`. |
| `solver` | `"matrix_exp"` | Dynamical solver, used for preprocessing search, refinement, and `run infer` alike. Must stay `matrix_exp` if `absorption: true` (the alternative, `bloch_eigen`, isn't safe for the non-Hermitian absorptive structure matrix). |
| `ignore_orientations` | `()` | Zero-based PETS rotation indices to exclude from the whole experiment (damaged/empty/diagnostic frames). |

### Adaptive tilt-chunk coupling

`coupling_mode: "union"` couples, within each tilt chunk, the beam set that is the *union* of the
excited beams at the chunk's two boundary tilts (see [Preprocessing](preprocessing.md) for what
"coupling" means here). `union_adaptive` picks how those chunk boundaries are chosen:

- `true` (default): recursive bisection. Start with one chunk spanning the whole rocking curve;
  at each step, check the chunk's midpoint tilt against the union already covered by its two
  endpoints. If the midpoint would add more than `union_max_new_beams_pct` of *new* beams beyond
  that union, split the chunk in two at the midpoint and recurse into each half; otherwise the
  chunk is left as one piece. This puts more, smaller chunks where the excited beam set is
  changing quickly across the rocking curve (e.g. near a strong systematic row) and fewer, larger
  chunks where it's stable — without needing to know in advance where that is.
- `false`: `fixed_n_segments` evenly-sized chunks regardless of how much the beam set actually
  changes across them.

Adaptive chunking costs more beam-set unions up front (to evaluate the split predicate) but
generally solves fewer total beams than a fixed split sized conservatively enough to avoid
under-coupling; a fixed split is more predictable when you want a fixed, known chunk count instead.

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
| `step_size` | scipy default | Initial simplex step size, degrees. |
| `max_iterations` | scipy default | Maximum Nelder-Mead iterations (`maxiter`). |
| `x_tolerance` | scipy default | Simplex convergence tolerance on the orientation parameters (`xatol`). |
| `f_tolerance` | scipy default | Simplex convergence tolerance on the objective (`fatol`). |
| `penalize_fewer_reflections` | scipy default | Penalize trial orientations that match fewer reflections than the seed, discouraging the search from drifting to a trivially-easier beam set. |

### `preprocess.thickness`

Bounds for the per-rotation thickness grid search (`ThicknessOptimizationConfig`).

| Field | Default | What it does |
|---|---|---|
| `min_thickness` | grid default | Lower bound, Å (one shared grid; the search itself is per rotation, so pooled datasets fit their own thicknesses within it). |
| `max_thickness` | grid default | Upper bound, Å. |
| `n_steps` | grid default | Number of evenly-spaced grid candidates. |
| `plot` | `false` | Write one wR2-vs-thickness PNG per rotation (`<inputs.structure's directory>/thickness_optim`). Reporting-only — never affects the fitted `Plan` and is excluded from the reproducibility digest. |

## `loss_metrics`

The one residual driving both preprocessing search and gradient refinement (`LossMetricsConfig`).
Top-level, not nested under `refinement`, because it governs preprocessing too.

| Field | Default | What it does |
|---|---|---|
| `residual` | `"wr2"` | `"wr2"` or `"robs"`. Both re-fit an optimal multiplicative scale between calculated and observed intensities before scoring (`core.losses.optimal_scale`) — necessary since the Bloch wave solve comes off on an arbitrary structure-factor scale, not PETS's own intensity scale. Parses into the matching loss (gradient refinement) and per-thickness scores (preprocessing search) function pair — the two stages always agree on one metric. |

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
| `multi_dataset` | `false` | See [Combining multiple datasets](#combining-multiple-datasets) below. |
| `load_hydrogens` | `false` | Include hydrogen atom sites from the structure CIF (molecular crystals). |

### Combining multiple datasets

`inputs.multi_dataset: true` pools rotations from several `.cif_pets` files (`inputs.exp_data` as a
list of 2+ paths) into one experiment, each dataset keeping its own energy, orientation, and
thickness. Two situations call for it:

- **Beam damage series** — repeat measurements of the same crystal taken at increasing dose. Each
  dataset is its own PETS file (its own UB, its own apparent thickness), but they refine one shared
  structure; combining them uses every rotation instead of picking one dataset and discarding the
  rest.
- **Low-symmetry structures** — a single tilt series from one crystal orientation range may not
  cover enough of reciprocal space to constrain the structure well when the space group has few
  symmetry operations to fill in the gaps. Multiple datasets from different crystal
  orientations/mounts fill in coverage that one series alone would leave thin.

Each dataset is preprocessed and checkpointed on its own (`plan.<stem>.npz` per file, with its own
integration geometry -- precession angles may differ between files), and the settled per-dataset
plans are pooled in memory just before refinement. See
[Inputs and outputs](inputs.md#multiple-datasets) for the mechanics (per-dataset checkpoints, the
first-file authoritative cell, one-energy rule, rotation-index offsets) and
[Reproducibility](reproducibility.md) for what each per-dataset lock verifies.
