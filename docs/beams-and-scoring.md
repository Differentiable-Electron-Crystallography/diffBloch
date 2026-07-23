# Beams and scoring

DiffBloch keeps several hkl sets separate because they answer different questions.

| Set | Meaning |
|---|---|
| Solve beams | The beam basis used in the Bloch solve. |
| Scored hkl | Reflections included in the objective/scoring comparison. |
| Observed hkl | Reflections present in the PETS observations. |
| Matched hkl | Reflections aligned between calculated and observed patterns. |
| Structure-factor hkl | The support grid where `Fgb` is tabulated. |

Keeping these separate matters because a good solve basis is not always the same as the set of
reflections used to score the model. Coupled rocking-curve solves add another distinction: each tilt
segment may use a per-segment beam union, while the final pattern is reassembled onto a combined
union beam axis.

## API example: scoring a checkpointed plan

```python
from pathlib import Path

from diffBloch.config import load_config
from diffBloch.io import read_structure
from diffBloch.preprocess import read_plan, score_orientations
from diffBloch.preprocess.experiment import RefinementSetup

root = Path("examples/experiments/quartz-checkpoint")
cfg = load_config(root / "experiment.yaml")
plan = read_plan(root / "plan.npz")
structure = read_structure(root / cfg.inputs.structure)
refinement = RefinementSetup.from_structure(structure)

scores = score_orientations(plan, refinement, method=cfg.solver.refine)
print(len(scores), float(scores[0]))
```

## API shape: beam selection

```python
from diffBloch.preprocess import build_orientation_plans, pipeline, select_beams
from diffBloch.specs import BeamSelection

prepare = pipeline([
    select_beams(BeamSelection()),
    build_orientation_plans(),
])
```

See also the `solve-set-vs-scored-set` design decision in the context docs for the detailed
rationale behind the split.
