# Colmey et al. (2026), Acta Crystallographica A

This directory contains imported inputs and legacy Hydra configuration fragments for the quartz,
CsPbBr3, and borane calculations. The three `data/*/experiment.yaml` files are translations into
the current strict diffBloch schema. They establish runnable configuration boundaries; they do not
yet claim reproduction of the paper's final R factors.

`legacy_checkpoint_preprocess/` is a preserved snapshot of the exact relevant configs, CIF/PETS
inputs, and saved preprocessing outputs copied from the Desktop `checkpoint_preprocess` repository.
Its README records the source revision and scope. Use it as the legacy reproduction record; use
the sibling `data/*/experiment.yaml` directories for the current diffBloch CLI.

All three paper configs use `blochwave.coupling_mode: per_tilt`. Every rocking-curve sub-tilt
recomputes excitation errors over the radial candidate pool, selects its own `g_max`/`sg_max` beam
basis, and builds its own Bloch structure-matrix geometry. No beam union is shared between tilts.

## Material configs

| Material | Current config | Legacy source |
|---|---|---|
| Quartz | `data/quartz/experiment.yaml` | `configs/experiment/quartz.yaml` plus the base fragments |
| CsPbBr3 | `data/cspbbr3/experiment.yaml` | `configs/experiment/cspbbr3_zone.yaml` plus the base fragments |
| Borane | `data/borane/experiment.yaml` | No borane-specific override was imported; values are explicitly marked as inferred |

## Legacy-to-current mapping

| Legacy setting | Current setting | Translation |
|---|---|---|
| `atoms.data.cif_file_path` | `inputs.structure` | Reduced to a path relative to the material experiment directory |
| `refinement.data.pets_path` | `inputs.exp_data` | Reduced to a relative path |
| `atoms.data.load_hydrogens` | `inputs.load_hydrogens` | Preserved as `true` |
| `bloch.thicknesses` | `sample.thicknesses` | Preserved at the imported base value of 1000 A |
| `refinement.dataloader.ignore_orientations` | `blochwave.ignore_orientations` | Preserved in original zero-based PETS order |
| `bloch.g_max_refine` | `blochwave.g_max` | Merged: the seed/scored radius now reuses the SOLVE cutoff directly (no separate knob) |
| `bloch.sg_max` | `blochwave.sg_max` | Preserved |
| `structure_factor.absorption` | `blochwave.absorption` | Preserved; `false` selects the elastic control and `true` enables absorption |
| `refinement.data.rocking_curve_sampling` | `blochwave.rocking_curve_sampling` | Preserved |
| `refinement.data.dsg` / `rsg` | `blochwave.dsg` / `rsg` | Preserved |
| `preprocess.orientation.optim` | `preprocess.optimize_orientation` | Preserved |
| `preprocess.thickness.optim` | `preprocess.optimize_thickness` | Preserved |
| orientation search bounds and steps | `preprocess.orientation` | Preserved |
| thickness grid bounds and steps | `preprocess.thickness` | Preserved |
| `refinement.epochs` | `refinement.steps` | Set to 20 for the current reproduction runs |
| Adam and `lr=0.001` | `refinement.optimizer` | Preserved |
| position/ADP optimization | `refinement.trainable` | Preserved at whole-group granularity |
| `thicknessNN` | `refinement.thickness_nn` | Disabled for the fixed-840 A structure-refinement runs |

## Known gaps before reproduction

- The initially imported quartz `.cif_pets` was truncated after zone axis 2. It has been restored
  from the matching complete in-repository export: 99 declared orientations and 6,666 reflections.
- The imported borane PETS file parses successfully (52 rotations and 32,124 reflection rows).
  Its H1b3 and H1b8 anisotropic ADP matrices are not positive semidefinite, so those two sites use
  their supplied positive isotropic equivalents (0.049 and 0.035 A2); all coordinates and boron
  ADPs remain unchanged.
- The imported CsPbBr3 structure and PETS data parse successfully (4 ASU sites, 59 rotations, and
  32,669 reflection rows). The paper config retains all 59 source rotations.
- The legacy `rbragg_abs` objective is not a current loss literal. The translated configs use
  `wr2`; parity must be established or an explicit paper loss added.
- The legacy global `structure_factor.g_max` was independently configurable. Current diffBloch
  derives structure-factor support from the SOLVE cutoff, so the old oversized support values
  (quartz 4.5 and CsPbBr3 5.0) are recorded here but not independently represented.
- The old `isotropic_displacements_only` flag has no current whole-config equivalent. Current ADP
  behavior follows each CIF site's ADP kind and crystallographic constraints.
- The old random/sequential dataloader, batch size, and `num_rotations` controls do not map to the
  current full-batch deterministic default refinement.
- The current train/validation split language cannot express every legacy sampling mode. The
  translated files retain the standard current split until the paper's exact scoring population is
  established.
- Thickness-network settings, restraints, schedulers, visualization, output paths, devices, and
  logger settings are not copied into experiment config because they are either unsupported
  scientific composition or execution-only concerns in the current architecture.
- Borane needs its original material-specific refinement configuration or a documented
  reconstruction from the paper/method records.
- Each material still needs an `experiment.lock` before `run preprocess`, `run infer`, or
  `run refine` can be treated as a locked reproduction run.

## Absorption controls

Each material config records the elastic control by default:

```yaml
blochwave:
  absorption: false
```

For the absorption member of a paper comparison, change only
`blochwave.absorption` to `true` in a separate experiment directory. The model reproduces
the imported paper implementation's fitted absorptive atomic factors for elements Z=1--103,
including its B-factor interpolation and absorptive `U0'` diagonal term. Absorption makes the
Bloch operator non-Hermitian, so these configs must use `matrix_exp`; validation rejects
`bloch_eigen`.

## Quartz refinement with optimized orientations and thickness NN

After preprocessing has written `data/quartz/plan.npz`, this command reuses those optimized
orientations and jointly refines the structure plus the configured apparent-thickness network:

```bash
uv run diffbloch run refine \
  examples/experiments/papers/colmey_et_al_2026_Acta_Cryst_A/data/quartz
```

The trained network tensors are written to `refined_components.npz` alongside the refined CIF,
structural parameter snapshot, and refinement summary.

## Validate the translated schemas

```bash
uv run diffbloch validate \
  examples/experiments/papers/colmey_et_al_2026_Acta_Cryst_A/data/quartz/experiment.yaml

uv run diffbloch validate \
  examples/experiments/papers/colmey_et_al_2026_Acta_Cryst_A/data/cspbbr3/experiment.yaml

uv run diffbloch validate \
  examples/experiments/papers/colmey_et_al_2026_Acta_Cryst_A/data/borane/experiment.yaml
```
