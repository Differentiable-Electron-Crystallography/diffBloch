# Architecture

diffBloch is organized around one stable scientific centre: a deterministic Bloch-wave simulation.
The surrounding layers prepare inputs, build reusable geometry, run optimization, and report
progress without making those concerns part of the numerical core.

## End-to-end flow

The end-to-end flow is:

1. `.cif` + `.cif_pets` + config are parsed into typed records.
2. Preprocessing turns those records into a reusable `Plan`.
3. The `Plan` and differentiable structural parameters feed the deterministic Bloch-wave simulation.
4. The simulated pattern is compared with the observed pattern as an R-loss.
5. Gradients from that loss update the differentiable structural parameters, then the loop repeats.

## Layers

| Layer | Role |
|---|---|
| [IO](api/io.md) | Parse CIF/PETS files into validated typed records. |
| [Config](api/config.md) | Validate experiment settings and lock input/checkpoint identity. |
| [Preprocess](api/preprocess.md) | Build and improve the immutable `Plan` the simulator consumes. |
| [Core](api/core.md) | Deterministic crystallographic and Bloch-wave numerical kernels. |
| [Params](api/params.md) | Differentiable structural parameters and physical constraints. |
| [Engine](api/engine.md) | Combine `Plan` + parameters, simulate, score, and refine. |
| [Observability](api/observability.md) | Emit typed events without coupling the core to logger backends. |
| [App](api/app.md) | CLI and default orchestration around the reusable API. |

## API shape

This is the high-level shape used by the CLI internally. It is intentionally shown as API shape
rather than a full script because real runs need an experiment directory, locks, and preprocessing
configuration.

```python
from diffBloch.app import refine_experiment, run_experiment

# Simulate/score a checkpointed experiment.
inference = run_experiment("examples/experiments/quartz-checkpoint")
print(inference.mean_r_obs)

# Run the refinement loop against the same settled Plan.
refined = refine_experiment("examples/experiments/quartz-checkpoint")
print(refined.best_loss)
```

For lower-level composition, use the public `preprocess` and `engine` APIs directly; see
[Preprocessing](preprocessing.md) and [Refinement](refinement.md).
