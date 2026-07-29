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

## Example precision

The example YAMLs choose:

```yaml
refinement:
  precision: fp32
```

That makes example refines faster and lighter. Switch to `fp64` for the conservative
highest-precision refinement path.

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
