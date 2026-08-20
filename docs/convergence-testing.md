# Convergence testing

Bloch wave calculations use a finite subset of an infinite reciprocal lattice and a finite number
of tilts across each rocking curve. Too few beams or tilts can change the calculated
integrated intensities; too many increase the computational cost without affecting the result. The required values depend on the material, orientation, and thickness and so cannot be generalized.

## Running the test

The convergence test is usually run before preprocessing or refinement:

```bash
uv run diffbloch convergence-test <experiment_dir>
```

## Simulation convergence

Simulation convergence is different from agreement with experiment: convergence compares two
simulations against each other, while refinement compares one simulation against measurement. A converged simulation can still disagree with experiment because the structure or experimental metadata is wrong.

The main controls governing simulation convergence are:

| Control | What it changes |
|---|---|
| `g_max` | Largest reflection {math}`g` vector simulated. Off-diagonal structure factors included in structure matrix extend to {math}`2g_\mathrm{max}`. |
| `sg_max` | A beam is simulated at a sampled orientation when {math}`|S_g| < sg_\mathrm{max}`. |
| `rocking_curve_sampling` | Number of crystal orientations sampled across each `.cif_pets` virtual frame and summed to give its integrated intensity. |

Increasing `g_max` or `sg_max` increases the number of beams {math}`N`, with simulation cost scaling
approximately as {math}`N^3`. Increasing `rocking_curve_sampling` adds more structure-matrix solves,
but its runtime depends on union grouping, GPU batching, matrix size, and available memory.

For `g_max`, a useful starting value is the magnitude of the largest reciprocal vector observed in the experimental data. This is only a lower bound. Dynamical scattering can reach an observed
reflection through coupling to unobserved beams outside the measured range, so the SOLVE basis
normally extends beyond the largest observed {math}`|\mathbf{g}|`.

## Effect of thickness

Thickness requires separate consideration. In the thin-crystal, two-beam limit, the rocking-curve
profile has the form

```{math}
I(s_g) \propto \operatorname{sinc}^2(\pi t s_g),
```

where {math}`t` is thickness and {math}`s_g` is excitation error. Its width is therefore inversely
proportional to thickness: a thinner crystal produces a broader rocking curve. If the actual experimental crystal
thickness is different than the value used for convergence testing, preserving the same angular sampling may requires a different number of tilt samples. The user should therefore consider performing convergence testing again if thickness optimization returns a substantially different value than was used in initial convergence test.

## Stopping the sweep

For each control, diffBloch increases the value by a chosen step, rebuilds the simulation, and
compares consecutive calculated intensity sets. The sweep stops when their R-factor falls below
`r_factor_threshold`; failure to reach the threshold within `max_iterations` raises an error.

Every comparison is recorded in the run's JSONL report as a `ConvergenceTrial`, so the sweep can be
inspected afterwards. `tools/event_report`'s `convergence_sweeps` figure plots one panel per
control: the between-steps R-factor against the candidate value, the threshold as a rule, and a
marker on the crossing. Because the y axis is the *change* between consecutive settings rather than
a measure of quality, a curve that flattens above the rule is a sweep that ran out of range, not a
converged one.

## Values used for refinement

The values returned by convergence testing do not have to be used for refinement. They are upper
limits for a fully converged simulation. Smaller values may give the same refined parameters at a
much lower computational cost. Palatinus *et al.* found that agreement with experimental data and
the refined structure could become insensitive to the beam cutoffs before the calculated
intensities were fully converged, and selected smaller practical values to keep the calculation
manageable ([Palatinus *et al.*, 2013](https://doi.org/10.1107/S010876731204946X)).
