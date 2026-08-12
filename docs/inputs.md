# Inputs and experiment directories

A diffBloch run starts from an experiment directory. The directory contains a small YAML config, a
starting crystal structure, reduced experimental diffraction data, and lock/checkpoint artifacts when
available.

## Required inputs

| File | Role |
|---|---|
| `experiment.yaml` | Experiment config and relative references to the input files. |
| `structure.cif` | Starting crystal structure. |
| `exp_data.cif_pets` | Reduced experimental diffraction data. |
| `reproducibility/experiment.lock` | Content identity for the input CIF/PETS files. |

Checkpointed examples also include:

| File | Role |
|---|---|
| `reproducibility/plan.<stem>.npz` | Serialized preprocessed `Plan`, one per `inputs.exp_data` file (`<stem>` is the file's path with separators as `__` and no `.cif_pets` suffix). |
| `reproducibility/plan.<stem>.lock` | Provenance lock tying that dataset's checkpoint to its input bytes, authoritative PETS cell, dataset-scoped config, recipe, code version, and artifact hash. |

Bundled binary artifacts such as `.cif_pets` experimental data and committed plan checkpoints are
tracked with Git LFS. After cloning, run `git lfs pull` before validating or running checkpointed
examples so these files are present as real data rather than LFS pointer files.

## Reading the structure and the experimental data

Reading a CIF gives fractional atomic coordinates, ADPs (isotropic or anisotropic, per atom), site
occupancies, and the space-group symmetry operations. Reading a `.cif_pets` gives the reduced
experimental data: the UB matrix, cell parameters, per-rotation goniometer angles, and the observed
hkl/intensity/sigma triples. Both are parsed into validated records, so a malformed CIF, an unsupported ADP type, or a unit mismatch fails immediately at read time rather than surfacing later as a silently wrong simulation.

PETS2's apparent mosaicity is read from `_diffrn_measurement_details` in degrees. Set
`blochwave.mosaicity: true` to use it in the simulated rocking-curve geometry, or `false` to disable
mosaic averaging.

The public records are:

- `StructureRecord`
- `ObservationRecord`
- `AdpRecord`

## Unit-cell authority: PETS overrides the structure CIF

The structure CIF and the `.cif_pets` file each carry their own unit cell, and they are not always
identical. diffBloch uses **PETS's cell, not the CIF's**, for every piece of simulation geometry: the
structure-factor grid, the reciprocal basis, the cell volume, the ADP `U*`-frame conversion, and the
beam geometry derived from that grid. The structure CIF still supplies all atomic content —
fractional positions, atom types, occupancies, ADPs, and symmetry operators — unchanged; those
fractions are simply interpreted against PETS's cell instead of the CIF's own.

The CIF's cell is checked against PETS's on every load:

- **> 1% relative difference** on any of `a, b, c, alpha, beta, gamma` logs a warning, stating
  explicitly that the PETS value overrides the CIF value. Ordinary refinement/measurement drift
  stays well under this.
- **> 5% relative difference** raises `ValueError` and stops — listing every offending parameter,
  both values, and the percentage difference — rather than silently deriving the whole simulation's
  geometry from a mismatch that large.

Under `inputs.multi_dataset`, the first `.cif_pets` file's cell is the shared authoritative cell.
The CIF is checked against it, and the same two thresholds apply between combined `.cif_pets` files
too (each further file's cell checked against the first file's); see
[Combining multiple datasets](#combining-multiple-datasets) below.

## API example

This example is runnable against the bundled quartz checkpoint.

```python
from pathlib import Path

from diffBloch.config import load_experiment
from diffBloch.io import read_experimental_data, read_structure

root = Path("examples/Colmey_et_al_2026/data/quartz-no-abs")
cfg, experiment_lock = load_experiment(root)

structure = read_structure(root / cfg.inputs.structure)
experimental_data = read_experimental_data(root / cfg.inputs.exp_data)

print(cfg.name)
print(structure.frac_positions.shape)
print(experimental_data.hkl.shape)
```

Validate a config from the CLI before launching a longer job:

```bash
uv run diffbloch validate examples/Colmey_et_al_2026/data/quartz-no-abs/experiment.yaml
```

## Excluding rotations

Exclude damaged, empty, or diagnostic-only frames with zero-based indices in the original PETS
rotation order:

```yaml
blochwave:
  ignore_orientations: [0, 1, 18, 56]
```

Excluded rotations never enter beam construction, orientation optimization, thickness optimization,
inference, or structure refinement. Filtering occurs before the train/validation split, and does
not renumber later source rotations when deciding split membership. The selection is part of the recorded experiment config and invalidates an incompatible
preprocess checkpoint.

The two optimization stages in the standard preprocess command can be selected independently:

```yaml
preprocess:
  optimize_orientation: true
  optimize_thickness: false
```

## Combining multiple datasets

`inputs.exp_data` is a single path by default. Set `inputs.multi_dataset: true` and give
`exp_data` a list of two or more distinct paths to combine rotations from several `.cif_pets` files
into one experiment -- e.g. a damage series, or repeat measurements of the same crystal:

```yaml
inputs:
  structure: enantiomer_1.cif
  multi_dataset: true
  exp_data:
    - undamaged/frame_1.cif_pets
    - undamaged/frame_2.cif_pets
refinement:
  thickness_nn:
    enabled: false   # defaults on; unsupported for pooled experiments (see limitations below)
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

An experiment directory accumulates generated files as each command runs, alongside the inputs
above:

```text
experiment.yaml
structure.cif
exp_data.cif_pets
refined_structure.cif      # written by `run refine`
refinement_report.txt      # written by `run refine`
thickness_optim/           # written by preprocess.thickness.plot / --plot-thickness
reproducibility/
  experiment.lock
  plan.<stem>.npz          # written by `run preprocess` / `run refine`, one per exp_data file
  plan.<stem>.lock         # written by `run preprocess` / `run refine`, one per exp_data file
  refined_parameters.npz   # written by `run refine`
  refined_components.npz   # written by `run refine`, only when components are composed
  refinement.lock          # written by `run refine`
```

`refined_structure.cif` contains the best epoch's constrained structural values; the human-facing
summary lives in `refinement_report.txt` (best epoch, which objective selected it, HKL counts). The
raw `.npz` snapshots and every lock live under `reproducibility/`, since nobody reads them directly
-- see [Reproducibility](reproducibility.md) for what each lock verifies. `run refine` prints the
absolute path of every file it writes. See [Refinement](refinement.md#refinement-outputs) for the
full breakdown.
