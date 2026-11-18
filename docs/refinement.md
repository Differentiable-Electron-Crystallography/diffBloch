# Refinement

Refinement holds a settled `Plan` (fitted orientation, tilts, thickness — see
[Preprocessing](preprocessing.md)) fixed and minimizes a scaling-optimized residual, chosen with
`loss_metrics.residual`, over the differentiable structural parameters — ASU positions, ADPs,
occupancies — by gradient descent through the full dynamical calculation described in
[Workflow](workflow.md#structural-refinement). Three residuals are available, and all three are
reported at every step whichever one is minimized:

| `residual` | Quantity | Definition |
|---|---|---|
| `wr` (default) | {math}`wR` on amplitudes, as printed by Jana2020 and Klar *et al.* (2023) | {math}`\sqrt{\sum w(F_o - F_c)^2 / \sum w F_o^2}`, {math}`w = 1/(\sigma(F)^2 + (0.01 F_o)^2)` |
| `wr2` | {math}`wR_2 = wR(F^2)` on intensities | {math}`\sqrt{\sum w(I_o - I_c)^2 / \sum w I_o^2}`, {math}`w = 1/(\sigma(I)^2 + (0.02 I_o)^2)` |
| `robs` | {math}`R_{\mathrm{obs}}` on amplitudes | {math}`\sum \lvert F_o - F_c \rvert / \sum F_o` over {math}`I_o > 3\sigma(I_o)` |

Here {math}`F = \sqrt{I}`, {math}`\sigma(F) = \sigma(I)/(2F)` floored at {math}`5\sqrt{\sigma(I)}`
for {math}`I < 0.01\,\sigma(I)`, and a negative measured intensity enters {math}`wR` at
{math}`F_o = 0`. The intensity weight carries the same 0.01 instability factor as the amplitude
weight, propagated to intensities. Klar *et al.*'s SI prints the {math}`wR` equation without the
square roots; their published values reproduce only with them.

## `infer` vs `refine`

- `infer` runs the forward simulation and reports {math}`wR`/{math}`wR_2`/{math}`R_{\mathrm{obs}}` for a settled
  `Plan`. It does not update parameters.
- `refine` runs the optimization loop: simulate, differentiate the residual back to the structural
  parameters, step the optimizer, repeat.

## Refinable groups

The default config exposes whole-group selections for:

- ASU positions;
- ADPs;
- occupancies;

The schema default keeps positions and ADPs trainable, and leaves occupancies frozen.

## CLI example

```bash
uv run diffbloch refine <experiment_dir> --device cuda
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
`Matched HKLs (I>3σ/total): X / Y` means matched reflections with `I > 3 sigma` / all matched reflections.

## Fit statistics comparable with Jana2020

The per-epoch `wR`, `wR2` and `R_obs` above are means of per-frame values, each frame scored at its
own optimal scale for that metric. Jana2020 instead pools every reflection into one sum, with one
scale per frame, so the per-frame means differ slightly from Jana's numbers on the same model. After
refinement, diffBloch therefore also reports Jana-convention statistics of the
best model, pooled over the fitted frames (event `FitStatisticsReported`, and the
"Fit statistics" section of `refinement_report.txt`):

- refinement on {math}`|F|`, with {math}`F_o = \sqrt{\max(I_{\mathrm{obs}}, 0)}` and one
  weighted least-squares amplitude scale per frame;
- {math}`\sigma(F) = \sigma(I)/(2F_o)`, floored at {math}`5\sqrt{\sigma(I)}` for
  {math}`I < 0.01\,\sigma(I)`, and weights {math}`w = 1/(\sigma(F)^2 + (0.01\,F_o)^2)`;
- {math}`R_{\mathrm{obs}} = \sum|F_o - F_c| / \sum F_o` over {math}`I > 3\sigma(I)`, and
  {math}`wR = \sqrt{\sum w(F_o - F_c)^2 / \sum w F_o^2}` over the obs and over all matched reflections;
- {math}`\mathrm{GoF} = \sqrt{\sum w(F_o - F_c)^2 / (N - p)}`, with {math}`N` counting per-frame
  measurements.

The parameter count {math}`p` is itemised: structural parameters after symmetry and constraints, every
trainable component element (thickness-network weights included, an upper bound on their freedom), a
per-frame thickness when the model uses the preprocess-fitted one, and the per-frame orientation
corrections. GoF is given both without and with the orientation corrections, and under both Jana
weights and {math}`1/\sigma(F)^2`. Per-frame scales are not charged in {math}`p`: Jana2020 lists them
among its parameters but leaves them out of GoF, and only that convention reproduces its published
values. The definition is pinned against Jana2020's own per-reflection output for quartz and borane
(`tests/unit/test_fit_statistics.py`).

## Refinement outputs

The default app writes the best recorded epoch, not merely the final optimizer state. With the
default `refinement.split.train_test = false`, "best" means the lowest recorded training objective
in the optimizer loop. When `train_test` is enabled, validation rotations are genuinely held out
from the training engine and the best epoch is selected by the held-out validation objective.
Validation still does not stop the run early; the loop always runs the configured
`refinement.steps`.

Which objective did the selecting is reported, not implied: the canonical JSONL report's
`RefinementCompleted` event reports its number under `best_training_loss` or
`best_validation_loss` accordingly. The per-epoch stream always carries the training objective and,
when a train/test split is enabled, the validation wR2/R_obs values computed during best-epoch
selection.

| File | Contents |
|---|---|
| `refined_structure.cif` | Best constrained coordinates, occupancies, and ADPs in CIF form. |
| `reproducibility/reports/report-*.jsonl` | Canonical versioned event report for completed commands, consumed by post-run notebooks and visualizers. |
| `refined_parameters.npz` | Exact raw optimizer parameters for the best epoch. |
| `refined_components.npz` | Trainable component tensors for the best epoch, when components are composed. Thickness-network tensors are stored per dataset under `apparent_thickness[<ref>]__<parameter>` names. |
| `plan.<stem>.npz` / `plan.<stem>.lock` | Settled preprocessing plan and its provenance lock, one pair per `inputs.exp_data` file. |
| `refinement.lock` | Binds the refined outputs to every dataset's plan lock (`plan_lock_sha256s`), config digest, and code version; written only when every plan lock exists. |

`refined_structure.cif` lands in the experiment directory; the completed-run JSONL report, `.npz`
snapshots, and locks go under `reproducibility/`. Any summary view is derived from the JSONL report
by whatever consumes it, not an extra refinement side effect.

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
