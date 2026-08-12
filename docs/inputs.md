# Inputs and outputs

Each experiment has its own directory containing `experiment.yaml`, a starting structure CIF, and
experimental `.cif_pets` data.

## Required inputs

| File | Contents |
|---|---|
| `experiment.yaml` | Input filenames and settings for the experiment. |
| `.cif` | Starting crystal structure. |
| `.cif_pets` | Experimental diffraction data reduced by PETS2. |

Paths written in `experiment.yaml` are measured from the experiment directory:

```yaml
inputs:
  structure: structure.cif
  exp_data: experiment.cif_pets
```

The structure CIF supplies the atoms, fractional coordinates, occupancies, atomic displacement
parameters (ADPs), and space-group symmetry operations.

The `.cif_pets` file supplies the observed reflection intensities and uncertainties, crystal
orientations, goniometer angles, electron wavelength, unit cell, and apparent mosaicity.

## Unit cell

Both input files contain a unit cell. diffBloch uses the `.cif_pets` cell for the simulation and uses the
structure-CIF cell as a consistency check.

- A difference greater than 1% in any cell parameter produces a warning.
- A difference greater than 5% stops the calculation.



## Excluding rotations

Individual rotations can be excluded when their diffraction data should not be used, for example
after beam damage has degraded the crystal or when no usable reflections were recorded. Rotations
are selected by their zero-based indices:

```yaml
blochwave:
  ignore_orientations: [0, 1, 18, 56]
```

Excluded rotations are not used during preprocessing, inference, or refinement.

## Multiple datasets

Several `.cif_pets` files from the same material can be combined in one refinement:

```yaml
inputs:
  structure: structure.cif
  multi_dataset: true
  exp_data:
    - crystal_1.cif_pets
    - crystal_2.cif_pets

refinement:
  thickness_nn:
    enabled: false
```

**Each dataset is preprocessed and checkpointed on its own.** Every file runs the recipe with its
own rocking-curve integration geometry -- datasets recorded with *different precession angles can
be mixed* -- and settles into its own checkpoint pair,
`reproducibility/plan.<stem>.npz` + `plan.<stem>.lock`, named after the file
(`undamaged/frame_1.cif_pets` becomes `plan.undamaged__frame_1.npz`). Checkpoint identity follows
the file, not its list position: adding, removing, or reordering `exp_data` entries never
recomputes another dataset's preprocessing -- appending frame 5 of a damage series reuses frames
1-4's checkpoints untouched. See [Reproducibility](reproducibility.md) for what each per-dataset
lock verifies.

Each file also keeps its own UB matrix/cell for orientation derivation and its own
mean-inner-potential correction. The shared structure-factor grid and ADP metric come from the
first file's authoritative PETS cell; every per-dataset plan lock records that cell, so changing the
anchor file's cell restales every pooled checkpoint. The files' wavelength-derived beam energies
must snap to the same accelerating voltage: the Bloch engine solves the whole pooled experiment at
one energy, so mixing e.g. a 120 kV and a 200 kV dataset is rejected with an error rather than
silently mis-scored.

Just before refinement the settled per-dataset plans are pooled in memory (never re-checkpointed):
rotations concatenate file-by-file with a running offset, so `rotation_index` stays globally unique
across the combined set (the first file's rotations are `0..N-1`, the second file's are
`N..N+M-1`, and so on) -- `ignore_orientations` and the train/validation split both key off this
combined index, and ignoring a rotation leaves a gap rather than renumbering later frames.

Pooled files must also describe the same crystal: the structure CIF and each further file's
recorded cell are checked against the first file's PETS cell (log a warning past 1% relative
difference on any cell parameter -- ordinary damage-series drift stays under this -- raise past
5%, listing every offending parameter).

One current limitation, rejected with a clear error rather than silently mishandled:
`refinement.thickness_nn` (the network keys only on each rotation's tilt angle, so it cannot
represent per-dataset thickness differences -- it defaults **on**, so a pooled config must set
`refinement.thickness_nn.enabled: false` explicitly or fail at config load, before any compute).
Listing the same
file twice -- or two paths that would collide on one checkpoint name -- is rejected at config
validation, since a duplicated dataset would double-weight its reflections in the refinement.
`multi_dataset` defaults to `false`, and a single `exp_data` path is simply the one-dataset case
of the same flow.

## Outputs

Running preprocessing or refinement adds results to the experiment directory:

| Output | Contents |
|---|---|
| `refined_structure.cif` | Refined crystal structure. |
| `refinement_report.txt` | Final residuals, reflection counts, and refinement summary. |
| `thickness_optim/` | Thickness-search plots when plotting is enabled. |
| `reproducibility/experiment.lock` | Record of the input files. |
| `reproducibility/plan.<stem>.npz` | Saved preprocessing for each `.cif_pets` dataset. |
| `reproducibility/plan.<stem>.lock` | Inputs and settings used for the saved preprocessing. |
| `reproducibility/refined_parameters.npz` | Refined parameter values. |
| `reproducibility/refinement.lock` | Inputs and settings used for the refinement. |

See [Refinement](refinement.md#refinement-outputs) for the refinement report and
[Reproducibility](reproducibility.md) for the lock files.
