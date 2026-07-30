# Examples

The repository ships runnable experiment directories under `examples/`. The top-level
`examples/experiments` entries are mostly baked-in demonstration runs: small or checkpointed
experiments that exercise preprocessing, inference, and refinement paths.

## Quick examples

### Quartz, end to end

Runs preprocessing and refinement from the ordinary quartz example directory.

```bash
uv run diffbloch run refine examples/experiments/quartz
```

The live refinement line reports `wR2`, `R_obs`, and diffraction loss for each epoch. The final
summary identifies the best epoch, reports matched observed / total matched HKLs, and lists the
refined CIF, raw parameter snapshot, JSON summary, and plan artifacts.

### Quartz, checkpointed

Starts from a committed `plan.npz` + `plan.lock`, so it skips preprocessing and reaches refinement
quickly.

```bash
uv run diffbloch run refine examples/experiments/quartz-checkpoint
```

### Abiraterone, checkpointed on CUDA

A larger compound; use an accelerator when available.

```bash
uv run diffbloch run refine examples/experiments/abiraterone-checkpoint --device cuda
```

## Catalog

| Example | Purpose |
|---|---|
| `examples/experiments/quartz` | Small full run from input files through preprocessing and refinement. |
| `examples/experiments/quartz-checkpoint` | Fast-start quartz run from a committed preprocess checkpoint. |
| `examples/experiments/abiraterone-checkpoint` | Larger molecular example; good CUDA/refine demonstration. |
| `examples/experiments/lta` | Larger zeolite example. |

## Python API example

```python
from diffBloch.app import run_experiment

result = run_experiment("examples/experiments/quartz-checkpoint")
print(result.n_evaluated, result.mean_r_obs)
```
