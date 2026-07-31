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
diffbloch run refine examples/experiments/abiraterone-checkpoint                # reuse the checkpoint, then refine (default path)
```

The unit cell (~2186 Å³) is above the large-cell threshold, so the orientation search runs on the
fast path (per-trial gather integrity checks skipped); the terminal re-scores the fitted
orientation, so the reported score keeps full fidelity. Generating the checkpoint uses a CUDA
device (`--device cuda`); the committed
checkpoint is then reused on CPU or GPU alike.

## Refinement

`run refine` is the **default single-stage path**: it refines positions + ADPs with the optimizer /
step budget in `experiment.yaml` (`refinement.steps`, `refinement.optimizer`, `refinement.trainable`,
top-level `loss_metrics`). Those are stable execution knobs; the config does **not** author a
scientific program.

Scientific composition is typed Python, not config. Hydrogen **riding** — which the faithful
abiraterone match uses (each H derived from its parent heavy atom each step) — is composed with
`diffBloch.engine.with_hydrogen_riding`, not a YAML mode:

```python
from diffBloch.engine import build_refinement_problem, run_refinement_problem, with_hydrogen_riding

trainable, constraints = with_hydrogen_riding(structure, cfg.refinement.trainable.to_spec())
problem = build_refinement_problem(initial=params, trainable=trainable, constraints=constraints)
run_refinement_problem(engine, problem, steps=cfg.refinement.steps,
                       optimizer=cfg.refinement.optimizer.name, lr=cfg.refinement.optimizer.lr)
```

## Why it reuses across a fresh clone

`plan.lock` records what determined the checkpoint (input bytes, resolved config, the recipe, and the
software version). The reuse gate compares only the **release** `__version__`, not the git SHA — so
the committed checkpoint stays valid across commits within a diffBloch release. On a `__version__`
bump it is regenerated; `--refresh` forces a recompute at any time (and rewrites the checkpoint).

> **Data provenance.** `abiraterone.cif` / `abiraterone_exp_data.cif_pets` are real private-lineage
> 3D-ED data. Clear redistribution before any public release.
