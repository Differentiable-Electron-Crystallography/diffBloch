# Examples

The repository ships runnable experiment directories under `examples/`. The top-level
`examples/experiments` entries are mostly baked-in demonstration runs: small or checkpointed
experiments that exercise preprocessing, inference, and refinement paths. For layouts intended to be
studied, copied, and modified as research starting points, use `examples/papers`.

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

### Quartz chirality check

Compares two committed preprocessed quartz `Plan`s against the same observed PETS2 `.cif_pets`
data. One diffBloch simulation scores the P3_2 21 matching hand at mean R_obs = 0.0507 and the
P3_1 21 enantiomorphic candidate at mean R_obs = 0.2192.

```bash
uv run python -c 'from pathlib import Path; from diffBloch.app import run_experiment; print("running quartz chirality comparison...", flush=True); root = Path("examples/experiments/quartz-chirality"); matching = run_experiment(root / "matching-hand", checkpoint=True); opposite = run_experiment(root / "opposite-hand", checkpoint=True); print(f"matching-hand mean R_obs: {matching.mean_r_obs:.4f}"); print(f"opposite-hand mean R_obs: {opposite.mean_r_obs:.4f}"); print(f"opposite/matching ratio: {opposite.mean_r_obs / matching.mean_r_obs:.2f}x")'
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
| `examples/experiments/quartz-chirality` | Handedness-discrimination proof of concept using two committed quartz plans. |
| `examples/papers/...` | Advanced research/composition examples. |

## Python API example

```python
from diffBloch.app import run_experiment

result = run_experiment("examples/experiments/quartz-checkpoint")
print(result.n_evaluated, result.mean_r_obs)
```

For advanced Python workflows, see
[`examples/papers`](https://github.com/Differentiable-Electron-Crystallography/diffBloch/tree/main/examples/papers).
