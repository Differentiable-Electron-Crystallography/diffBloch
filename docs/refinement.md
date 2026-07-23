# Refinement

Refinement is the optimization loop around the deterministic Bloch-wave simulation. It holds a
settled `Plan` fixed, exposes selected structural quantities as differentiable parameters, simulates
diffraction, compares calculated and observed intensities, and updates the parameters with a PyTorch
optimizer.

## `infer` vs `refine`

- `infer` simulates and scores a settled plan. It does not update parameters.
- `refine` runs the optimization loop and updates differentiable structural parameters.

## Refinable groups

The default config exposes whole-group selections for:

- ASU positions;
- ADPs;
- occupancies;
- structure factors (`Fgb`).

The schema default keeps positions and ADPs trainable, and leaves occupancies and `Fgb` frozen.

## Precision

`refinement.precision` controls the Bloch solve precision used by the app refinement path:

```yaml
refinement:
  precision: fp32  # faster/lower-memory solve path
```

The schema default is `fp64` for the conservative complex128 path. The bundled examples opt into
`fp32` for faster iteration. This knob downcasts the solve path; it does not make every trainable
parameter or every structure-factor calculation float32.

## CLI examples

```bash
# Full quartz run, including preprocessing.
diffbloch run refine examples/experiments/quartz

# Faster start from a committed preprocessing checkpoint.
diffbloch run refine examples/experiments/quartz-checkpoint

# Larger checkpointed example on CUDA.
diffbloch run refine examples/experiments/abiraterone-checkpoint --device cuda
```

## API example: default app refinement

```python
from diffBloch.app.program import refine_experiment

result = refine_experiment("examples/experiments/quartz-checkpoint")

print(result.losses.shape)
print(result.best_step, result.best_loss)
```

## API shape: lower-level composition

This is the lower-level shape used when power users need custom constraints, penalties, or model
components.

```python
from pathlib import Path

from diffBloch.config import load_config
from diffBloch.engine import build_refinement_model, build_refinement_problem, run_refinement_model
from diffBloch.preprocess import build_engine, read_plan
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.io import read_structure

root = Path("examples/experiments/quartz-checkpoint")
cfg = load_config(root / "experiment.yaml")
plan = read_plan(root / "plan.npz")
structure = read_structure(root / cfg.inputs.structure)
refinement = RefinementSetup.from_structure(structure)

engine = build_engine(plan, refinement, precision="fp32")
model = build_refinement_model(initial=refinement.params)
problem = build_refinement_problem()

result = run_refinement_model(
    engine,
    model,
    problem,
    trainable=cfg.refinement.trainable.to_spec(),
    steps=2,
)
print(result.best_loss)
```
