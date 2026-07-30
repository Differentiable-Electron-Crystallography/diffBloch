# Reproducibility and checkpoints

Fitting orientation and thickness (see [Preprocessing](preprocessing.md)) is a greedy, local search
over a bumpy objective — the coupled wR2 landscape has many nearby local minima, so a small
floating-point difference between two runs can occasionally send the search into a different basin
and change the fitted orientation, and with it the reported R-factor. Reproducing a specific result
therefore means more than reproducing the code: it means starting from the exact same fitted `Plan`,
not merely re-running the same search and hoping it lands in the same place. diffBloch checkpoints
that fitted geometry and locks it to everything that determined it, so a later run can either verify
it is reusing the identical starting point or refuse to.

- `experiment.lock` pins the input structure and observation files by content hash.
- `plan.lock` ties a `plan.npz` checkpoint to the input lock, resolved preprocess-determining config,
  ordered preprocess recipe, producing code version, and checkpoint artifact hash.

In committed examples, `plan.npz` and `.cif_pets` files are Git LFS artifacts; `plan.lock` verifies
the realized file bytes after LFS checkout, not the small pointer file.

## What this guarantees

A valid preprocess checkpoint verifies:

- the input CIF/PETS files match their lock;
- the current preprocess-determining config matches the checkpoint lock;
- the requested recipe is compatible with the checkpoint recipe;
- the checkpoint artifact has not been silently changed.

## What this does not guarantee

The lock guarantees the identity of the fitted starting point, not hardware-independent optimizer
determinism downstream of it. Device, thread scheduling, and optimizer implementation details can
still shift the refinement's floating-point trajectory — most consequentially for a fresh orientation
search, since that is exactly the bumpy-landscape case above; a *reused* checkpoint sidesteps it
entirely, since the search was already run and its result is frozen.

## API example

```python
from pathlib import Path

from diffBloch.config import (
    code_version,
    config_digest,
    load_experiment,
    preprocess_lock_status,
    read_preprocess_lock,
    sha256_file,
)
from diffBloch.preprocess import read_plan

root = Path("examples/experiments/quartz-checkpoint")
cfg, _experiment_lock = load_experiment(root)
plan = read_plan(root / "plan.npz")
lock = read_preprocess_lock(root / "plan.lock")

status = preprocess_lock_status(
    lock,
    experiment_lock_sha256=sha256_file(root / "experiment.lock"),
    config_digest=config_digest(cfg),
    recipe=lock.recipe,
    code_version=code_version(),
    plan_path=root / "plan.npz",
    root=root,
)

print(status)  # "reuse", "resume", or "stale"
```

## CLI example

Run preprocessing and write/reuse the checkpoint:

```bash
uv run diffbloch run preprocess examples/experiments/quartz-checkpoint
```

Run refinement from a checkpointed plan:

```bash
uv run diffbloch run refine examples/experiments/quartz-checkpoint
```
