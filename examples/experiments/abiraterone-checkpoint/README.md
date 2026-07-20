# Example: abiraterone acetate with a frozen preprocess checkpoint

Abiraterone acetate (C₂₆H₃₃NO₂, space group P2₁2₁2₁) — a well-conditioned molecular crystal, and the
first organic-crystal example the **public config path expresses end to end**. It ships a
**pre-computed preprocess checkpoint** (`plan.npz` + `plan.lock`): the settled coupled `Plan` (fitted
orientations, tilt-segment couplings, pinned scored sets). The first run reuses the checkpoint and
scores in seconds instead of repeating the fit.

What makes this experiment expressible where it previously was not: the faithful abiraterone
preprocess needs hydrogens in the forward model **and** an experiment-specific per-trial coupling.
Both are now explicit config — `inputs.load_hydrogens: true` and a `preprocess.coupling` block — so no
bespoke script is required.

## Files

| File                            | Role                                                              |
| ------------------------------- | ----------------------------------------------------------------- |
| `experiment.yaml`               | the experiment definition (incl. `load_hydrogens` + `coupling`)   |
| `experiment.lock`               | input-byte identity                                               |
| `abiraterone.cif`               | structure — abiraterone acetate, space group P2₁2₁2₁              |
| `abiraterone_exp_data.cif_pets` | observed reflection intensities (PETS `.cif_pets`)                |
| `plan.npz`                      | the frozen coupled preprocess checkpoint (committed)              |
| `plan.lock`                     | binds the checkpoint to inputs + config + recipe + release version |

## Run

From the repository root:

```bash
diffbloch run preprocess examples/experiments/abiraterone-checkpoint            # settle the Plan (reuses the checkpoint)
diffbloch run preprocess examples/experiments/abiraterone-checkpoint --refresh  # recompute the coupled fit from scratch
diffbloch run infer examples/experiments/abiraterone-checkpoint                 # reuse the checkpoint, then score every rotation
```

The unit cell (~2186 Å³) is above the large-cell threshold, so the orientation search runs on the
coarse fp32 fast path; the fp64 terminal re-scores the fitted orientation, so the reported score
keeps full fidelity. Generating the checkpoint uses a CUDA device (`--device cuda`); the committed
checkpoint is then reused on CPU or GPU alike.

## Why it reuses across a fresh clone

`plan.lock` records what determined the checkpoint (input bytes, resolved config, the recipe, and the
software version). The reuse gate compares only the **release** `__version__`, not the git SHA — so
the committed checkpoint stays valid across commits within a diffBloch release. On a `__version__`
bump it is regenerated; `--refresh` forces a recompute at any time (and rewrites the checkpoint).

> **Data provenance.** `abiraterone.cif` / `abiraterone_exp_data.cif_pets` are real private-lineage
> 3D-ED data. Clear redistribution before any public release.
