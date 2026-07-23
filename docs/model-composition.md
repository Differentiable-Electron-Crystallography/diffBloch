# Model composition

The default CLI path runs a simple single-stage refinement. Power users can compose the refinement
problem directly in Python when they need molecular constraints, penalties, or model components.

## Taxonomy

| Concept | Meaning |
|---|---|
| Crystallographic constraints | Always-on transforms in `params.constrain`, such as site-symmetry position projection and ADP constraints. |
| Constraint transforms | Hard molecular constraints applied to the physical state before diffraction, such as hydrogen riding. |
| Penalties | Soft additive objective terms, such as bond-length penalties. |
| Components | Trainable model pieces that can feed forward-context values, such as apparent thickness. |

## API shape

This is the lower-level composition shape. The exact constraint/penalty construction depends on the
structure and the scientific question, so treat the middle block as an example pattern.

```python
from pathlib import Path

from diffBloch.config import load_config
from diffBloch.engine import build_refinement_model, build_refinement_problem, run_refinement_model
from diffBloch.engine.constraints import with_hydrogen_riding
from diffBloch.io import read_structure
from diffBloch.preprocess import build_engine, read_plan
from diffBloch.preprocess.experiment import RefinementSetup

root = Path("examples/experiments/abiraterone-checkpoint")
cfg = load_config(root / "experiment.yaml")
structure = read_structure(root / cfg.inputs.structure, load_hydrogens=True)
refinement = RefinementSetup.from_structure(structure)
plan = read_plan(root / "plan.npz")

engine = build_engine(plan, refinement, precision=cfg.refinement.precision)

# API shape: add hard constraints/penalties/components before running.
trainable, constraints = with_hydrogen_riding(
    structure,
    cfg.refinement.trainable.to_spec(),
)
model = build_refinement_model(initial=refinement.params, constraints=constraints)
problem = build_refinement_problem()

result = run_refinement_model(
    engine,
    model,
    problem,
    trainable=trainable,
    steps=2,
    optimizer=cfg.refinement.optimizer.name,
    lr=cfg.refinement.optimizer.lr,
)
```

See
[`examples/papers`](https://github.com/Differentiable-Electron-Crystallography/diffBloch/tree/main/examples/papers)
for fuller research examples.
