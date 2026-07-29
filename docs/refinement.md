# Refinement

Refinement is the optimization loop around the repeatable Bloch-wave simulation. It holds a
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

The schema default keeps positions and ADPs trainable, and leaves occupancies frozen.

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
from diffBloch.app import refine_experiment

result = refine_experiment("examples/experiments/quartz-checkpoint")

print(result.losses.shape)
print(result.best_step, result.best_loss)
```

## Advanced composition: constraints, penalties, and learned thickness

The default CLI refinement is intentionally conservative: it refines selected structural parameter
groups against diffraction. The lower-level API also supports richer composition:

- **crystallographic constraints** are built into `RefinementSetup.from_structure(...)`; special-
  position coordinates and ADP equality constraints are always enforced by the parameter
  `constrain(...)` path;
- **hard molecular constraints** are per-atom/structure transforms applied after crystallographic
  constraints, such as hydrogen riding, where H atoms stay in the forward scattering but ride on
  their parent heavy atoms;
- **soft penalties** are additive objective terms across atoms, such as bond-length penalties;
- **components** are trainable non-structural model pieces that provide dependent forward-model
  values, such as apparent thickness, alongside the structural parameters.

Compose a structure component, optional hard constraints, optional soft penalties, and optional
forward-model components, then optimize all selected leaves through one objective.

```python
from pathlib import Path

import torch

from diffBloch.config import load_config
from diffBloch.engine import (
    ApparentThicknessNN,
    ThicknessBounds,
    build_refinement_model,
    build_refinement_problem,
    mean_plan_thickness,
    perceive_bond_length_penalty,
    run_refinement_model,
    with_hydrogen_riding,
)
from diffBloch.io import read_structure
from diffBloch.preprocess import RefinementSetup, build_engine, read_plan

root = Path("examples/experiments/abiraterone-checkpoint")
device = torch.device("cpu")

cfg = load_config(root / "experiment.yaml")
structure = read_structure(root / cfg.inputs.structure, load_hydrogens=True)
refinement = RefinementSetup.from_structure(structure)
plan = read_plan(root / "plan.npz")
engine = build_engine(plan, refinement, precision=cfg.refinement.precision)

# Start from the config's structural trainable groups, then freeze H optimizer leaves and add a
# hard hydrogen-riding transform when hydrogens are present.
trainable, constraints = with_hydrogen_riding(
    structure,
    cfg.refinement.trainable.to_spec(),
)

# Soft objective term: tether perceived ASU-contiguous bonds to their starting distances. The
# optimizer may trade this against diffraction; unlike a hard constraint, it is not exact.
penalties = (
    perceive_bond_length_penalty(
        structure,
        include_hydrogen=False,
        sigma_angstrom=0.02,
        weight=0.1,
        criterion="flat_bottom_l1",
    ),
)

# Train a bounded apparent-thickness component alongside the structural parameters. This overrides
# each Plan orientation's baked fixed thickness through ForwardContext.thickness.
thickness_nn = ApparentThicknessNN(bounds=ThicknessBounds(100.0, 2000.0))
initial = refinement.params.to(device)
component_params = {
    thickness_nn.key: thickness_nn.initial_params(
        dtype=initial.asu_positions.dtype,
        device=device,
        initial_thickness=mean_plan_thickness(engine.orientations),
    )
}

model = build_refinement_model(
    initial=initial,
    constraints=constraints,
    components=(thickness_nn,),
    component_params=component_params,
)
problem = build_refinement_problem(penalties=penalties)

result = run_refinement_model(
    engine,
    model,
    problem,
    trainable=trainable,
    steps=2,
    optimizer="adam",
    lr=1e-3,
)
print(result.best_loss)
```

For thickness, `ApparentThicknessNN` is one component choice. Simpler alternatives include
`PerOrientationThickness` for one free positive thickness per rotation, or `QuadraticThicknessProfile`
for a low-dimensional bounded profile over orientation angle.
