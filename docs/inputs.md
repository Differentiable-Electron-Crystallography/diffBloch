# Inputs and experiment directories

A diffBloch run starts from an experiment directory. The directory contains a small YAML config, a
starting crystal structure, reduced experimental diffraction data, and lock/checkpoint artifacts when
available.

## Required inputs

| File | Role |
|---|---|
| `experiment.yaml` | Experiment config and relative references to the input files. |
| `structure.cif` | Starting crystal structure. |
| `observations.cif_pets` | Reduced experimental diffraction data. |
| `experiment.lock` | Content identity for the input CIF/PETS files. |

Checkpointed examples also include:

| File | Role |
|---|---|
| `plan.npz` | Serialized preprocessed `Plan`. |
| `plan.lock` | Provenance lock tying the checkpoint to inputs, config, recipe, code version, and artifact hash. |

Bundled binary artifacts such as `.cif_pets` observations and committed `plan.npz` checkpoints are
tracked with Git LFS. After cloning, run `git lfs pull` before validating or running checkpointed
examples so these files are present as real data rather than LFS pointer files.

## Typed IO boundary

diffBloch does not pass parser output directly into numerical kernels. It parses CIF/PETS inputs
into typed Pydantic records first, so unsupported shapes, units, missing fields, or invalid values
fail at the boundary.

The public records are:

- `StructureRecord`
- `ObservationRecord`
- `AdpRecord`

## API example

This example is runnable against the bundled quartz checkpoint.

```python
from pathlib import Path

from diffBloch.config import load_experiment
from diffBloch.io import read_observations, read_structure

root = Path("examples/experiments/quartz-checkpoint")
cfg, experiment_lock = load_experiment(root)

structure = read_structure(root / cfg.inputs.structure)
observations = read_observations(root / cfg.inputs.exp_data)

print(cfg.name)
print(structure.frac_positions.shape)
print(observations.hkl.shape)
```

Validate a config from the CLI before launching a longer job:

```bash
uv run diffbloch validate examples/experiments/quartz-checkpoint/experiment.yaml
```

## Excluding rotations

Exclude damaged, empty, or diagnostic-only frames with zero-based indices in the original PETS
rotation order:

```yaml
blochwave:
  ignore_orientations: [0, 1, 18, 56]
```

Excluded rotations never enter beam construction, orientation fitting, thickness fitting,
inference, or structure refinement. Filtering occurs before the train/validation split, and does
not renumber later source rotations when deciding split membership. The selection changes the
scientific result, so it is part of the recorded experiment config and invalidates an incompatible
preprocess checkpoint.

The two fitting stages in the standard preprocess command can be selected independently:

```yaml
preprocess:
  optimize_orientation: true
  optimize_thickness: false
```

The fixed stage order is orientation fitting followed by thickness fitting when both are enabled.

## Generated refinement artifacts

`run refine` writes `refined_structure.cif`, `refined_parameters.npz`, and
`refinement_summary.json` into the experiment directory and prints their absolute paths. The CIF
contains the best epoch's constrained structural values; the NPZ preserves its exact raw optimizer
parameters.
