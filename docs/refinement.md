# Refinement

Refinement holds a settled `Plan` (fitted orientation, tilts, thickness — see
[Preprocessing](preprocessing.md)) fixed and minimizes the scaling-optimized weighted {math}`wR_2`
of [Klar *et al.* (2023)](https://doi.org/10.1038/s41557-023-01186-1) over the differentiable
structural parameters — ASU positions, ADPs, occupancies — by gradient descent through the full
dynamical calculation described in [Workflow](workflow.md#structural-refinement). `R_obs`, the Bragg
{math}`R`-factor restricted to {math}`I_{\mathrm{obs}} > 3\sigma` reflections, is reported alongside
{math}`wR_2` at every step as the conventional crystallographic residual, but is not itself the
optimized quantity.

## `infer` vs `refine`

- `infer` runs the forward simulation and reports {math}`wR_2`/{math}`R_{\mathrm{obs}}` for a settled
  `Plan`. It does not update parameters.
- `refine` runs the optimization loop: simulate, differentiate {math}`wR_2` back to the structural
  parameters, step the optimizer, repeat.

## Refinable groups

The default config exposes whole-group selections for:

- ASU positions;
- ADPs;
- occupancies;

The schema default keeps positions and ADPs trainable, and leaves occupancies frozen.

## CLI example

```bash
uv run diffbloch run refine <experiment_dir> --device cuda
```

Preprocesses (or reuses settled per-dataset plan checkpoints, see [Reproducibility](reproducibility.md)) and then
gradient-refines. Add `--refresh` to force a real preprocess recompute.

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

The default app writes the best recorded epoch, not merely the final optimizer state. With the
default `refinement.split.train_test = false`, "best" means the lowest recorded training objective
in the optimizer loop. When `train_test` is enabled, validation rotations are genuinely held out
from the training engine and the best epoch is selected by the held-out validation objective.
Validation still does not stop the run early; the loop always runs the configured
`refinement.steps`.

Which objective did the selecting is reported, not implied: `refinement_report.txt` carries a
`Best epoch selection` row, and the `RefinementCompleted` event reports its number under
`best_training_loss` or `best_validation_loss` accordingly. The per-epoch stream is always the
training objective.

| File | Contents |
|---|---|
| `refined_structure.cif` | Best constrained coordinates, occupancies, and ADPs in CIF form. |
| `refinement_report.txt` | Best epoch metrics, which objective selected it, and the compact HKL count. |
| `refined_parameters.npz` | Exact raw optimizer parameters for the best epoch. |
| `refined_components.npz` | Trainable component tensors for the best epoch, when components are composed. |
| `plan.<stem>.npz` / `plan.<stem>.lock` | Settled preprocessing plan and its provenance lock, one pair per `inputs.exp_data` file. |
| `refinement.lock` | Binds the refined outputs to every dataset's plan lock (`plan_lock_sha256s`), config digest, and code version; written only when every plan lock exists. |

`refined_structure.cif` and `refinement_report.txt` land in the experiment directory; the `.npz`
snapshots and locks go under `reproducibility/`.

The completion summary prints the absolute location of every output.

## Advanced composition: constraints, restraints, and learned thickness

The default CLI refinement is intentionally conservative: it refines selected structural parameter
groups against diffraction alone. Real structures often need more of the machinery familiar from
X-ray least-squares refinement, and the lower-level Python API exposes it directly:

- **special-position and ADP symmetry constraints** are enforced automatically wherever
  `RefinementSetup.from_structure(...)` builds the parameters — an atom on a special position never
  acquires an unphysical degree of freedom, and symmetry-equivalent ADPs never drift apart;
- **hard constraints** fix the geometric relationship between atoms exactly, e.g. a riding model
  where hydrogens keep contributing to the calculated scattering but move rigidly with their parent
  heavy atom rather than refining independent coordinates;
- **restraints** (soft penalties) pull a quantity toward a target without fixing it, e.g. keeping a
  bond length close to a chemically reasonable value; the optimizer can still trade the restraint off
  against the diffraction fit, unlike a hard constraint;
- **thickness models** treat the apparent specimen thickness at each tilt as a trainable quantity
  alongside the atomic parameters, rather than a fixed value fitted once during preprocessing.

Compose a structure, optional hard constraints, optional restraints, and an optional thickness model,
then refine every selected parameter through one objective. The composition entry points --
`build_refinement_model`, `build_refinement_problem`, `with_hydrogen_riding`,
`perceive_bond_length_penalty`, `run_refinement_model` -- and the thickness component choices
(`ApparentThicknessNN` for a learned network, `PerOrientationThickness` for one free positive
thickness per rotation, `QuadraticThicknessProfile` for a low-dimensional bounded profile over
orientation angle) are documented in [`diffBloch.engine`](api/engine.md).
