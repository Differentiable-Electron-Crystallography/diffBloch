# Reproducibility

diffBloch records the files and settings used during preprocessing and refinement. These records
are stored in the experiment directory under `reproducibility/`.

| File | Purpose |
|---|---|
| `experiment.lock` | Identifies the structure CIF and `.cif_pets` input files. |
| `plan.<stem>.npz` | Stores the orientations, thicknesses, and simulation setup produced by preprocessing. |
| `plan.<stem>.lock` | Records the inputs and settings used to produce that preprocessing file. |
| `refinement.lock` | Records the preprocessing, refinement settings, code version, and final outputs used for a refinement. |

## Input files

`experiment.lock` records a checksum for each input file. A checksum changes when the contents of a
file change. This prevents a calculation from using input files that differ from those recorded for
the experiment.

`experiment.lock` is created automatically, from the current input files, the first time any command
(`converge`, `preprocess`, `infer`, `refine`) runs against an experiment directory that doesn't have
one yet. After that, if an input file's contents change, the lock is *not* rewritten automatically --
the mismatch is exactly the drift this file exists to catch, so the command fails instead. Rewrite it
explicitly once the change is intentional:

```bash
uv run diffbloch lock <experiment_dir>
```

## Preprocessing

Preprocessing results are saved under `reproducibility/` so they can be reused. Combined experiments
save the preprocessing for each `.cif_pets` dataset separately.

The corresponding `plan.<stem>.lock` records:

- The structure CIF and `.cif_pets` data.
- The unit cell used for the simulation.
- The preprocessing settings and steps.
- The diffBloch version.

The saved preprocessing is reused when these records still match the current experiment. If they
do not match, preprocessing runs again.

Use `--refresh` to rebuild preprocessing even when the existing files still match:

```bash
uv run diffbloch preprocess <experiment_dir> --refresh
```

Use `--no-checkpoint` to run preprocessing without reading or writing saved preprocessing:

```bash
uv run diffbloch preprocess <experiment_dir> --no-checkpoint
```

## Refinement

`refinement.lock` is written after refinement. It records:

- The preprocessing files used by the refinement.
- The refinement settings.
- The diffBloch version.
- Checksums for the refined CIF and refined parameters.

This file identifies how the refined structure was produced. It is not used to resume or reuse a
completed refinement.

## Limits

The lock files confirm the identity of the inputs, settings, preprocessing, and outputs. They do
not guarantee identical floating-point results on different hardware.
