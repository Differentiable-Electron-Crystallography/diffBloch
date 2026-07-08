# Example: quartz inference

A complete, runnable experiment directory — the published quartz (SiO₂) precession
electron-diffraction dataset. It is the canonical end-to-end example: given a structure and observed
intensities, `diffbloch` preprocesses all 99 rotations and scores the full dynamical Bloch-wave
forward model against the observations.

## Files

| File                       | Role                                                          |
| -------------------------- | ------------------------------------------------------------- |
| `experiment.yaml`          | the experiment definition (inputs + numerical settings)       |
| `experiment.lock`          | input-byte identity; `run infer` verifies the inputs against it |
| `quartz.cif`               | structure — SiO₂, space group P3₁21                           |
| `quartz_exp_data.cif_pets` | observed reflection intensities (PETS `.cif_pets`)            |

## Run

From the repository root:

```bash
diffbloch run infer examples/experiments/quartz            # ~6–16 min the first time (the fit)
diffbloch run infer examples/experiments/quartz --console  # stream per-rotation progress
```

Expected result: **mean R_obs = 0.0506** over the 99 rotations.

The recipe is the faithful default — beam selection, rocking-curve integration, mosaicity, then the
per-rotation orientation fit **under the private's per-trial beam coupling**, and the thickness fit.
The orientation fit is the expensive phase, so the first run writes a preprocess checkpoint
(`plan.npz` + `plan.lock`) into this directory; a second identical run reuses it in seconds. Both
checkpoint files are gitignored here. Recompute from scratch with `--refresh`, or skip the
checkpoint entirely with `--no-checkpoint`.

For a variant that **ships** a pre-computed checkpoint (so the first run is already instant), see the
sibling `quartz-checkpoint` example. For the checkpoint machinery itself, see the checkpoint/resume
tutorial.
