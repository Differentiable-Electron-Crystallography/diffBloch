# Architecture

Electron scattering cross sections are orders of magnitude larger than X-ray cross sections, so a
diffracted beam in a crystal only tens to a few hundred nanometres thick re-scatters before it exits
— dynamical, multiple-beam diffraction rather than the kinematical (first Born, single-scattering)
regime X-ray refinement relies on. The measured intensity of reflection {math}`hkl` is not
{math}`\propto |F_{hkl}|^2`; it depends on the full coupled system of every beam excited at that
orientation and thickness. diffBloch solves that coupled system directly (a Bloch-wave calculation)
and backpropagates through it, so a structure can be refined against the dynamical intensities rather
than a kinematical approximation of them.

## The forward model

For one crystal orientation, the excited beam set {math}`\{g\}` obeys the coupled dynamical
diffraction equation

```{math}
\frac{d\boldsymbol{\psi}}{dz} = \frac{i\pi}{k_n} A\,\boldsymbol{\psi}, \qquad \boldsymbol{\psi}(0) = \boldsymbol{\psi}_0,
```

where {math}`\boldsymbol{\psi}` is the vector of beam amplitudes, {math}`k_n` is the wavevector
magnitude corrected for the mean inner potential, {math}`\boldsymbol{\psi}_0` is unity on the
transmitted (000) beam and zero elsewhere, and the structure matrix {math}`A` is built from the
structure factors {math}`F_{g-h}` off-diagonal and the excitation error {math}`S_g` on the diagonal:
{math}`A_{gh} = \pi F_{g-h} / (k_n \Omega)` for {math}`g \neq h`, {math}`A_{gg} = 2 k_n S_g`. With
absorption, {math}`F` is complex (`Sec. Absorption` below); without it, {math}`A` is Hermitian and
the propagator is unitary — flux is conserved between beams, not lost. Solving this at a fixed
thickness {math}`t` is a matrix exponential, {math}`\boldsymbol{\psi}(t) = \exp(i\pi t A / k_n)\,\boldsymbol{\psi}_0`,
evaluated either by diagonalizing {math}`A` once per orientation (cheap for many thicknesses, but its
gradient is ill-conditioned near degenerate eigenvalues) or directly as a matrix exponential (the
refinement-safe default).

Each experimental rotation is not one orientation but a narrow angular range — a **virtual frame**
([Klar *et al.*, 2023](https://doi.org/10.1038/s41557-023-01186-1)) — so the forward model sums
{math}`|\boldsymbol{\psi}(t)|^2` over sampled tilts spanning that range before comparing to the
integrated experimental intensity. See [Preprocessing](preprocessing.md) for how the tilt sampling,
orientation, and {math}`t` are fitted, and [Hyperparameter selection](hyperparameter-selection.md)
for how many beams and tilts are enough.

## Absorption

Thermal diffuse scattering removes flux from the coherent beams without literally absorbing it —
electrons are redistributed into a diffuse background rather than destroyed, unlike X-ray photon
absorption. It is modelled as an imaginary addition to the structure factor,
{math}`F^{\text{tot}}_g = F_g + iF'_g`, with {math}`F'_g` the absorptive form factor (Lobato/Thomas
parametrization) evaluated per atom from its equivalent isotropic Debye–Waller factor. This makes
{math}`A` non-Hermitian, so `matrix_exp` (not the eigendecomposition path) is required whenever
absorption is enabled.

## Refinement

Comparing calculated to observed intensities uses the same residuals reported in the crystallographic
literature: the Bragg {math}`R_{\mathrm{obs}}`,

```{math}
R_{\mathrm{obs}} = \frac{\sum_{I_{\mathrm{obs}} > 3\sigma} \left| \sqrt{I_{\mathrm{obs}}} - \sqrt{I_{\mathrm{calc}}} \right|}{\sum_{I_{\mathrm{obs}} > 3\sigma} \sqrt{I_{\mathrm{obs}}}},
```

and the scaling-optimized weighted {math}`wR_2` of Klar *et al.* (2023), the default refinement
objective. Gradients of {math}`wR_2` with respect to atomic positions, ADPs, and occupancies flow
back through the structure-factor sum and the matrix exponential to a PyTorch optimizer (Adam or
L-BFGS). See [Refinement](refinement.md) for the trainable-parameter groups and
[Observability](observability-guide.md) for what gets reported per step.

## Where each piece lives

| Physical role | Code | Guide |
|---|---|---|
| Parse `.cif` / `.cif_pets` into validated records | [`io`](api/io.md) | [Inputs](inputs.md) |
| Validate `experiment.yaml`, pin input identity | [`config`](api/config.md) | [Inputs](inputs.md), [Reproducibility](reproducibility.md) |
| Fit orientation and thickness, build the `Plan` | [`preprocess`](api/preprocess.md) | [Preprocessing](preprocessing.md) |
| Structure factors and the Bloch-wave propagator | [`core`](api/core.md) | this page |
| Atomic positions, ADPs, occupancies, and their crystallographic constraints | [`params`](api/params.md) | [Refinement](refinement.md) |
| Simulate, score, and run the optimization loop | [`engine`](api/engine.md) | [Refinement](refinement.md) |
| Report per-rotation R-factors and progress | [`observability`](api/observability.md) | [Observability](observability-guide.md) |
| CLI and default recipes | [`app`](api/app.md) | [Examples](examples.md) |

## Running the whole calculation from Python

The CLI (`diffbloch run refine ...`) is a thin wrapper over the same public functions shown below:
read the inputs, fit orientation/thickness into a `Plan`, then simulate and refine against it.

```python
from pathlib import Path

from diffBloch.app import CSVLogger, ConsoleLogger
from diffBloch.config import load_experiment
from diffBloch.engine import build_refinement_model, build_refinement_problem, run_refinement_model
from diffBloch.io import read_observations, read_structure
from diffBloch.observability import MultiLogger
from diffBloch.preprocess import (
    build_engine,
    build_orientation_plans,
    fit_orientation,
    fit_thickness,
    from_experiment,
    pipeline,
    run_inference,
)
from diffBloch.specs import ScoredHklSelection, TrialCoupling

root = Path("examples/experiments/quartz-checkpoint")

# Read the starting structure and the rocking-curve observations.
cfg, experiment_lock = load_experiment(root)
structure = read_structure(root / cfg.inputs.structure, load_hydrogens=cfg.inputs.load_hydrogens)
observations = read_observations(root / cfg.inputs.exp_data)

# Progress (per-rotation wR2/R_obs, orientation-search steps, ...) streams to these loggers.
logger = MultiLogger((ConsoleLogger(), CSVLogger(root / "events.csv")))

# Build the initial (unfitted) geometry scaffold and the differentiable structural parameters.
setup = from_experiment(structure, observations, cfg)

# The beam set re-derived at every trial orientation during the search below.
trial_coupling = TrialCoupling(
    policy=cfg.blochwave.to_policy(),
    scored=ScoredHklSelection(
        klar=cfg.blochwave.to_beam_selection(setup.integration),
        g_max=cfg.blochwave.g_max_refine,
    ),
)

# Fit orientation, then specimen thickness, per rotation.
prepare = pipeline(
    [
        build_orientation_plans(
            cfg.blochwave.to_rocking_curve(setup.integration),
            cfg.blochwave.mosaicity,
            coupling=cfg.blochwave.to_policy(),
            scoring_selection=cfg.blochwave.to_beam_selection(setup.integration),
        ),
        fit_orientation(
            setup.refinement,
            cfg.preprocess.orientation.to_search(),
            method=cfg.blochwave.solver.refine,
            coupling=trial_coupling,
            logger=logger,
        ),
        fit_thickness(
            setup.refinement,
            cfg.preprocess.thickness.to_grid(),
            method=cfg.blochwave.solver.refine,
            logger=logger,
        ),
    ],
    logger=logger,
)
plan = prepare(setup.plans.combined)

# Simulate and score the settled Plan without changing the structure.
inference = run_inference(
    plan,
    setup.refinement,
    method=cfg.blochwave.solver.inference,
    logger=logger,
)

# Refine: simulate, compare with experiment, and update atomic parameters by gradient descent.
engine = build_engine(
    plan,
    setup.refinement,
    loss=cfg.refinement.objective.to_loss(),
    method=cfg.blochwave.solver.refine,
)
model = build_refinement_model(initial=setup.refinement.params)
problem = build_refinement_problem()
result = run_refinement_model(
    engine,
    model,
    problem,
    trainable=cfg.refinement.trainable.to_spec(),
    steps=cfg.refinement.steps,
    optimizer=cfg.refinement.optimizer.name,
    lr=cfg.refinement.optimizer.lr,
    logger=logger,
)

print(inference.mean_r_obs)
print(result.best_step, result.best_loss)
```

For checkpoint/reuse behavior, the app layer wraps this same public composition with
`plan.npz`/`plan.lock` handling; see [Reproducibility](reproducibility.md).
