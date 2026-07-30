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

The four bundled examples span the two axes that matter most for cost and behavior: unit-cell size
(which sets the beam count, and with it the O(N³) eigensolve cost and whether the large-cell fork
routes past it — see [Preprocessing](preprocessing.md#routing-on-cell-size)) and structural
complexity (whether hydrogens and their riding-model treatment are in play).

| Example | What it demonstrates |
|---|---|
| `examples/experiments/quartz` | Small unit cell (~113 Å³), no hydrogens; a full run from raw inputs, and the reproducibility anchor other changes are checked against. |
| `examples/experiments/quartz-checkpoint` | Same structure, starting from a committed fitted `Plan` — the fast path once orientation/thickness are already settled. |
| `examples/experiments/abiraterone-checkpoint` | A larger organic molecule with hydrogens, so the hydrogen riding-model constraint (see [Refinement](refinement.md#advanced-composition-constraints-restraints-and-learned-thickness)) is exercised; a CUDA/refine demonstration. |
| `examples/experiments/lta` | A zeolite well above the large-cell threshold — routes through the faster orientation-search branch and benefits most from a GPU. |

## Python API example

```python
from diffBloch.app import run_experiment

result = run_experiment("examples/experiments/quartz-checkpoint")
print(result.n_evaluated, result.mean_r_obs)
```
