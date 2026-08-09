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
| `experiment.lock` | Content identity for the input CIF/PETS files. |

Checkpointed examples also include:

| File | Role |
|---|---|
| `plan.npz` | Serialized preprocessed `Plan`. |
| `plan.lock` | Provenance lock tying the checkpoint to inputs, config, recipe, code version, and artifact hash. |

Bundled binary artifacts -- `.cif_pets` experimental data, and any `plan.npz` checkpoint committed
alongside an experiment -- are tracked with Git LFS. After cloning, run `git lfs pull` before
validating or running an example so these are real data rather than LFS pointer files.

## Reading the structure and the experimental data

Reading a CIF gives fractional atomic coordinates, ADPs (isotropic or anisotropic, per atom), site
occupancies, and the space-group symmetry operations. Reading a `.cif_pets` gives the reduced
experimental data: the UB matrix, cell parameters, per-rotation goniometer angles, and the observed
hkl/intensity/sigma triples. Both are parsed into validated records, so a malformed CIF, an unsupported ADP type, or a unit mismatch fails immediately at read time rather than surfacing later as a silently wrong simulation.

The public records are:

- `StructureRecord`
- `ObservationRecord`
- `AdpRecord`

## API example

This example is runnable against any of the bundled experiment directories.

```python
from pathlib import Path

from diffBloch.config import load_experiment
from diffBloch.io import read_experimental_data, read_structure

root = Path("examples/Colmey_et_al_2026_Acta_Cryst_A/data/quartz-no-abs")
cfg, experiment_lock = load_experiment(root)

structure = read_structure(root / cfg.inputs.structure)
experimental_data = read_experimental_data(root / cfg.inputs.exp_data)

print(cfg.name)
print(structure.frac_positions.shape)
print(experimental_data.hkl.shape)
```

Validate a config from the CLI before launching a longer job:

```bash
uv run diffbloch validate examples/Colmey_et_al_2026_Acta_Cryst_A/data/quartz-no-abs/experiment.yaml
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

The two fitting stages in the standard preprocess command can be selected independently:

```yaml
preprocess:
  optimize_orientation: true
  optimize_thickness: false
```

## Pooling multiple datasets

`inputs.exp_data` is a single path by default. Set `inputs.multi_dataset: true` and give
`exp_data` a list of two or more paths to pool rotations from several `.cif_pets` files into one
experiment -- e.g. a damage series, or repeat measurements of the same crystal:

```yaml
inputs:
  structure: enantiomer_1.cif
  multi_dataset: true
  exp_data:
    - undamaged/frame_1.cif_pets
    - undamaged/frame_2.cif_pets
```

Each file keeps its own wavelength-derived energy, UB matrix/cell, and mean-inner-potential
correction. Rotations are concatenated file-by-file with a running offset, so `rotation_index`
stays globally unique across the pool (the first file's rotations are `0..N-1`, the second file's
are `N..N+M-1`, and so on) rather than every file restarting at `0` -- `ignore_orientations`, the
train/validation split, and an imported orientation CSV all key off this pooled index. Pooled files
must share one rocking-curve integration semiangle: diffBloch builds a single shared
rocking-curve/beam-selection geometry for the whole experiment, so files recorded with different
precession angles cannot currently be mixed. `multi_dataset` defaults to `false`, and a single
`exp_data` path behaves exactly as before -- nothing about the single-dataset path changes.


## Generated refinement artifacts

`refine` writes `refined_structure.cif`, `refined_parameters.npz`, and
`refinement_summary.json` into the experiment directory and prints their absolute paths. The CIF
contains the best epoch's constrained structural values; the NPZ preserves its exact raw optimizer
parameters.
