# Example: quartz with a frozen preprocess checkpoint

The same quartz (SiO₂) experiment as the sibling `quartz` example, but this directory **ships a
pre-computed preprocess checkpoint** (`plan.npz` + `plan.lock`) — the settled coupled `Plan` (fitted
orientations, tilt-segment couplings, pinned scored sets). So you skip the expensive fit entirely:
the first run reuses the checkpoint and scores in seconds.

## Files

| File                       | Role                                                            |
| -------------------------- | --------------------------------------------------------------- |
| `experiment.yaml`          | the experiment definition                                       |
| `experiment.lock`          | input-byte identity                                             |
| `quartz.cif`               | structure — SiO₂, space group P3₁21                             |
| `quartz_exp_data.cif_pets` | observed reflection intensities (PETS `.cif_pets`)              |
| `plan.npz`                 | the frozen coupled preprocess checkpoint (committed)            |
| `plan.lock`                | binds the checkpoint to inputs + config + recipe + release version |

## Run

From the repository root:

```bash
diffbloch run infer examples/experiments/quartz-checkpoint            # instant: reuses the checkpoint
diffbloch run infer examples/experiments/quartz-checkpoint --refresh  # ~6–16 min: recompute from scratch
```

Expected result: **mean R_obs = 0.0506** over the 99 rotations — the same as recomputing, in
seconds.

## Why it reuses across a fresh clone

`plan.lock` records what determined the checkpoint (input bytes, resolved config, the recipe, and
the software version). The reuse gate compares only the **release** `__version__`, not the git SHA —
so the committed checkpoint stays valid across commits within a diffBloch release rather than going
stale on the next commit. On a `__version__` bump the checkpoint is regenerated. `--refresh` forces
a recompute at any time (and rewrites the checkpoint).
