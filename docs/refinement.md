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

## CLI examples

```bash
# Full quartz run, including preprocessing.
uv run diffbloch run refine examples/experiments/quartz

# Faster start from a committed preprocessing checkpoint.
uv run diffbloch run refine examples/experiments/quartz-checkpoint

# Larger checkpointed example on CUDA.
uv run diffbloch run refine examples/experiments/abiraterone-checkpoint --device cuda
```

The default refinement budget is 40 epochs. Set a different recorded budget in the experiment
config:

```yaml
refinement:
  steps: 40
```

Each live epoch reports `wR2`, `R_obs`, and the diffraction loss. Epoch numbering in the CLI starts
at 1. At completion, the CLI shows the best epoch and its metrics in an aligned summary.
`HKLs (Observed/total): X / Y` means matched observed reflections / all matched reflections, where
the observed classification uses the conventional `I > 3 sigma` test internally.

## Refinement outputs

The default app writes the best recorded epoch, not merely the final optimizer state:

| File | Contents |
|---|---|
| `refined_structure.cif` | Best constrained coordinates, occupancies, and ADPs in CIF form. |
| `refined_parameters.npz` | Exact raw optimizer parameters for the best epoch. |
| `refinement_summary.json` | Best epoch metrics, the compact HKL count, and artifact paths. |
| `plan.npz` / `plan.lock` | Settled preprocessing plan and its provenance lock. |

The completion summary prints the absolute location of every output.

## API example: default app refinement

```python
from diffBloch.app import refine_experiment

result = refine_experiment("examples/experiments/quartz-checkpoint")

print(result.losses.shape)
print(result.best_step, result.best_loss)
print(result.history[result.best_step].wr2)
print(result.history[result.best_step].r_obs)
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
engine = build_engine(plan, refinement)

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

# Train the legacy apparent-thickness component alongside the structural parameters. The input is
# the PETS alpha coordinate normalized over the full experiment to [-1, 1].
thickness_nn = ApparentThicknessNN(
    bounds=ThicknessBounds(100.0, 2000.0),
    normalized_alphas=(-1.0, 0.0, 1.0),
    sample_thickness=False,
)
initial = refinement.params.to(device)
component_params = {
    thickness_nn.key: thickness_nn.initial_params(
        dtype=initial.asu_positions.dtype,
        device=device,
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
