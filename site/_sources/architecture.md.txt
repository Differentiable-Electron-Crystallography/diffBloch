# Architecture

This page maps `src/diffBloch/` — the packages, the modules inside each, and how they depend on
each other. For the scientific calculation these modules implement, see [Workflow](workflow.md).

## Package layout

```text
src/diffBloch/
  io/            CIF / .cif_pets readers -> validated records
  core/          pure crystallography + Bloch-wave physics (no config, no IO, no app)
  engine/        the refinement engine: forward composition + the imperative optimizer loop
  preprocess/    the Plan -> Plan calibration pipeline (orientation, thickness, numerics)
  config/        pydantic experiment.yaml schema + reproducibility locks/digests
  app/           CLI, default runners, and logger backends
  observability.py   typed domain-observation events + the pluggable Logger sink
  params.py      raw <-> physical parameter transform (RefinableParams <-> PhysicalState)
  specs.py       validated value-types the preprocess calibration steps consume
```

## Dependency layering

Dependencies point one way; nothing here imports back up the chain:

```text
io -> core -> engine -> preprocess -> app
              config (schema/manifest) sits beside engine/preprocess, imported by app
```

- **`io` -> `core`**: readers produce validated records (`StructureRecord`, `ExperimentalRecord`,
  `AdpRecord`); nothing downstream re-parses a file.
- **`core` -> `engine`**: `engine.refine -> engine.forward -> core` — `core/` stays free of
  `torch.optim` entirely; the one deliberately stateful corner of the codebase (`engine.refine`) is
  quarantined at the top of that chain.
- **`engine` -> `preprocess`**: preprocess composes `engine`'s forward path to score trial
  orientations/thicknesses, but `engine` never imports `preprocess` — the settled `Plan` is the
  one-way handoff (`preprocess -> Plan -> refine`, and `refine` never re-enters preprocessing).
- **`config`** parses `experiment.yaml` into the typed values `engine`/`preprocess`/`core` consume
  (e.g. `LossMetricsConfig.to_loss()` resolves a config string to an `engine.losses` function); it
  is the boundary between YAML and everything else.
- **`app`** depends on all of the above (it is the orchestration layer — CLI, default runners,
  logger backends); nothing else imports `app`.

Within `core/`, there is a second, orthogonal split: a **NumPy planning path** (fixed setup
geometry — cells, symmetry, electron-optics constants — computed once, never refined) and a
**differentiable torch path** (structure factors, ADPs, constraints — everything gradients flow
through). The same split recurs in `core.dynamical` (`primitives.py` NumPy, `assembly.py` torch)
and mirrors `core.reciprocal` (NumPy) vs `core.scattering` (torch).

## `core/` — crystallography and Bloch-wave physics

**Planning (NumPy, setup-only, no gradients):**

| Module | What it does |
|---|---|
| `reciprocal.py` | Miller-index grid helpers (reciprocal-space geometry). |
| `crystal.py` | Unit-cell and lattice-centering helpers. |
| `symmetry.py` | Symmetry expansion with precomputed ASU membership. |
| `dynamical/primitives.py` | Relativistic electron-optics constants: wavelength, interaction parameter σ, excitation error, obliquity factors. |

**Differentiable (torch, gradients flow through):**

| Module | What it does |
|---|---|
| `scattering.py` | Electron structure factors (Lobato parametrization), vectorised. |
| `absorption.py` | Differentiable absorptive electron-scattering factors. |
| `absorptive_parameters.py` | Fitted absorptive-scattering parameters used by the legacy paper code. |
| `adp.py` | ADP transforms at the raw-parameter constraint boundary. |
| `constraints.py` | Pure tensor constraints/bijectors for raw refinement parameters. |
| `solver.py` | Bloch-wave propagators (`matrix_exp` / `bloch_eigen`) that integrate the assembled structure matrix — see [Bloch-wave simulation](bloch-wave-simulation.md#solving-two-equivalent-routes-one-gradient-safe). |
| `dynamical/assembly.py` | Differentiable structure-matrix assembly — the Bloch `A` path. |
| `products.py` | `AlignedIntensities` and the other typed products bridging propagation to losses. |
| `losses.py` | Intensity-space loss/metric functions (`rbragg`, `w_rbragg`, `optimal_scale`, ...). |

## `engine/` — the refinement engine

| Module | What it does |
|---|---|
| `plan.py` | Refinement-invariant geometry plans: the shared scattering grid + per-orientation bundles. |
| `forward.py` | The pure, differentiable forward spine (`RefinementEngine.objective_value`/`simulate`). |
| `refine.py` | The imperative `torch.optim` loop — the one deliberately stateful corner of the core. |
| `losses.py` | Named `LossFn`/`ScoresFn` builders (`wr2_loss`, `rbragg_loss`, ...) — the objective terms `loss_metrics.residual` picks. |
| `constraints.py` | Hard molecular constraints (e.g. hydrogen riding) as reparameterizations of the physical state. |
| `penalties.py` | Soft refinement penalties (restraints) on the bounded physical ASU state. |
| `components.py` | Refinement model components that supply forward-model values (e.g. `ApparentThicknessNN`). |
| `chemistry.py` | Shared covalent-radii constants for connectivity perception (bond penalties, hydrogen riding). |

## `preprocess/` — the `Plan -> Plan` calibration pipeline

| Module | What it does |
|---|---|
| `experiment.py` | `from_experiment` — the boundary constructor from parsed records + config to refinement inputs. |
| `plan.py` | The `Plan` value object: the invariant geometry differentiable refinement is conditioned on. |
| `pipeline.py` | Composition combinators: `pipeline` (sequencing) and the fixpoint driver. |
| `orientation.py` | Native crystal-orientation derivation from goniometer geometry. |
| `coupling.py` | Tilt-segment-union beam coupling across a rocking curve. |
| `scoring.py` | Assembles a `RefinementEngine` and scores orientations against data. |
| `inference.py` | The eval-only terminal: forward-model every rotation and score it, no refinement. |
| `driver.py` | Simulation-convergence testing over `g_max`/`sg_max`/tilt steps. |
| `serialize.py` | Serializes a settled `Plan` to a plan-checkpoint `.npz` and reads it back. |

**`preprocess/steps/`** — the composable steps the pipeline and driver assemble:

| Module | Step |
|---|---|
| `beams.py` | `select_beams` — prune each orientation's beams to its active set. |
| `frames.py` | `select_frames` — drop whole rotations whose observed pattern is too sparse. |
| `coupling.py` | `couple_beams` — choose the rocking curve's beam-coupling policy. |
| `rocking_curve.py` | `integrate_rocking_curve` — bake each rotation's tilt set into the geometry. |
| `optimize_orientation.py` | Per-rotation crystal-orientation refinement (Nelder–Mead). |
| `optimize_thickness.py` | Per-rotation specimen-thickness calibration by grid search. |
| `convergence.py` | Grow a simulation-accuracy knob until the pattern stops moving. |
| `coverage.py` | Grow a beam knob to the minimum that recovers the most matched reflections. |
| `report_coupling.py` | Pure identity step whose only effect is emitting observability events. |

## `io/` — readers and validated records

| Module | What it does |
|---|---|
| `cif.py` | CIF structure reader backed by `gemmi`. |
| `pets.py` | PETS CIF-like experimental-data reader. |
| `_cifio.py` | Shared CIF parsing primitives used by both readers. |
| `record.py` | The validated records themselves (`StructureRecord`, `ExperimentalRecord`, `AdpRecord`, ...). |
| `symmetry_setup.py` | Extracts special-position ADP/coordinate constraints at the IO boundary. |

## `config/` — experiment schema and reproducibility

| Module | What it does |
|---|---|
| `schema.py` | The pydantic `ExperimentConfig` tree parsed from `experiment.yaml` — see [Config reference](hyperparameter-selection.md). |
| `manifest.py` | `experiment.lock` / per-dataset `plan.<stem>.lock` / `refinement.lock` and the digest functions that key them — see [Reproducibility](reproducibility.md). |

## `app/` — CLI, runners, and logger backends

| Module | What it does |
|---|---|
| `cli.py` | Thin command-line entry point (the console-script wrapper). |
| `program.py` | The default recipe: `run_experiment` / `refine_experiment` / `preprocess_experiment`. |

**`app/loggers/`** — the only place vendor SDKs and terminal/plotting dependencies live:

| Module | What it does |
|---|---|
| `summary.py` | The human-readable `refinement_report.txt` backend. |
| `tui.py` | Live terminal dashboard (`diffBloch[tui]` extra, `--tui`). |
| `plotting.py` | Matplotlib thickness-vs-residual plots (`diffBloch[plot]` extra). |
| `wandb.py` | Weights & Biases backend — the only module touching the `wandb` SDK. |
| `comet.py` | Comet ML backend — the only module touching the `comet_ml` SDK. |

`ConsoleLogger` and `CSVLogger` live directly in `app/loggers/__init__.py` alongside the shared
`Event`-formatting helpers every backend uses.

## Top-level modules

| Module | What it does |
|---|---|
| `observability.py` | Typed domain-observation events (`Event`/`Logger` protocols) — distinct from stdlib `logging`, which carries solver diagnostics, not scientific ones. See [Observability](observability-guide.md). |
| `params.py` | The raw (unbounded, optimizer-facing) <-> physical (bounded, constrained) parameter transform: `RefinableParams` <-> `PhysicalState`. |
| `specs.py` | Validated value-types the preprocess calibration steps consume (`NelderMeadSearch`, `ThicknessGrid`, `RockingCurve`, ...) — the parsed counterparts of `config.schema`'s YAML blocks. |
