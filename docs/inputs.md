# Inputs and experiment directories

A diffBloch run starts from an experiment directory. The directory contains a small YAML config, a
starting crystal structure, processed diffraction observations, and lock/checkpoint artifacts when
available.

## Required inputs

| File | Role |
|---|---|
| `experiment.yaml` | Experiment config and relative references to the input files. |
| `structure.cif` | Starting crystal structure. |
| `observations.cif_pets` | Experimental diffraction observations processed by [PETS2](https://pets.fzu.cz/). |
| `experiment.lock` | Content identity for the input CIF/PETS files. |

Checkpointed examples also include:

| File | Role |
|---|---|
| `plan.npz` | Serialized preprocessed `Plan`. |
| `plan.lock` | Provenance lock tying the checkpoint to inputs, config, recipe, code version, and artifact hash. |

## Typed IO boundary

DiffBloch does not pass parser output directly into numerical kernels. It parses CIF/PETS inputs
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
observations = read_observations(root / cfg.inputs.observations)

print(cfg.name)
print(structure.frac_positions.shape)
print(observations.hkl.shape)
```

Validate a config from the CLI before launching a longer job:

```bash
diffbloch validate examples/experiments/quartz-checkpoint/experiment.yaml
```
