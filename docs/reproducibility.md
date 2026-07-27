# Reproducibility and checkpoints

diffBloch separates input identity from generated artifacts.

- `experiment.lock` pins the input structure and observation files by content hash.
- `plan.lock` ties a `plan.npz` checkpoint to the input lock, resolved preprocess-determining config,
  ordered preprocess recipe, producing code version, and checkpoint artifact hash.

This gives inference and refinement a verifiable starting point: a reused `plan.npz` is known to
match the inputs and preprocessing recipe that produced it. In committed examples, `plan.npz` and
`.cif_pets` files are Git LFS artifacts; `plan.lock` verifies the realized file bytes after LFS
checkout, not the small pointer file.

## What this guarantees

A valid preprocess checkpoint verifies:

- the input CIF/PETS files match their lock;
- the current preprocess-determining config matches the checkpoint lock;
- the requested recipe is compatible with the checkpoint recipe;
- the checkpoint artifact has not been silently changed.

## What this does not guarantee

The locks do not claim hardware-independent optimizer determinism. Device, thread scheduling,
precision, and optimizer implementation details can still affect floating-point trajectories,
especially for refinement. The lock guarantees the identity of the preprocessed starting point.

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
