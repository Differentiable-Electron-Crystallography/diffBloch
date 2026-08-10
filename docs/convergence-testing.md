# Convergence testing

Bloch-wave calculations use a finite subset of an infinite reciprocal lattice and a finite number
of tilts across each rocking curve. Too few beams or tilts can change the calculated
intensities; too many increase the cost without improving the result. The required values depend on
the material, orientation, and thickness.

## Convergence

A simulation is converged when increasing its size produces little to no change in the
simulated intensities. This is different from agreement with experiment: convergence compares two
simulations, while refinement compares simulation with measurement. A converged calculation can
still disagree with experiment because the structure or experimental metadata is wrong.

The main numerical controls are:

| Control | What it changes |
|---|---|
| `g_max` | Maximum reciprocal-vector length of beams included in the Bloch-wave simulation. |
| `sg_max` | Maximum excitation-error magnitude for a beam to enter the simulation at a sampled tilt. |
| `rocking_curve_sampling` | Number of tilt samples used to integrate each rocking curve. |

Increasing `g_max` or `sg_max` increases the number of beams {math}`N`, with simulation cost scaling
approximately as {math}`N^3`. Cost scales approximately linearly with
`rocking_curve_sampling`.

For `g_max`, a useful starting value is the magnitude of the largest reciprocal vector observed in
the experimental data. This is only a lower bound. Dynamical scattering can reach an observed
reflection through coupling to unobserved beams outside the measured range, so the SOLVE basis
normally extends beyond the largest observed {math}`|\mathbf{g}|`. Start the convergence sweep at
the experimental limit and increase `g_max` until adding the more distant beams no longer changes
the calculated intensities. The larger structure-factor support required for beam differences is
derived automatically from `g_max`.

## Convergence testing in diffBloch

Convergence testing should be performed before thickness optimization, orientation optimization,
or structural refinement. Its purpose is to determine whether the finite beam basis and tilt grid
are large enough, not whether the model agrees with experiment. These numerical requirements are
usually insensitive to small orientation corrections and to the structural changes produced by
refinement. There is little value in optimizing the model first and then discovering that the
simulation used to optimize it was not converged.

Thickness requires separate consideration. In the thin-crystal, two-beam limit, the rocking-curve
profile has the form

```{math}
I(s_g) \propto \operatorname{sinc}^2(\pi t s_g),
```

where {math}`t` is thickness and {math}`s_g` is excitation error. Its width is therefore inversely
proportional to thickness: a thinner crystal produces a broader rocking curve. If the actual experimental crystal
thickness is much thicker than the value used for convergence testing, preserving the same angular sampling requires
more tilt samples. The user should therefore consider performing convergence testing again if thickness optimization returns a substantially different value.

For each control, diffBloch increases the value by a chosen step, rebuilds the simulation, and
compares consecutive calculated intensity sets. The sweep stops when their R-factor falls below
`r_factor_threshold`; failure to reach the threshold within `max_iterations` raises an error.

Convergence testing sweeps `g_max`, `sg_max`, and `rocking_curve_sampling`. It should include
representative experimental orientations because reciprocal-space density varies through a tilt
series.
## API example

Convergence testing is an optional preprocessing step, so its settings are explicit Python values
rather than fields in `experiment.yaml`:

```python
from pathlib import Path

from diffBloch.config import load_experiment
from diffBloch.io import read_experimental_data, read_structure
from diffBloch.preprocess import (
    ConvergenceTest,
    ConvergenceTolerance,
    build_orientation_plans,
    converge_numerics,
    from_experiment,
    select_beams,
)

root = Path("examples/experiments/quartz")
cfg, _lock = load_experiment(root)
structure = read_structure(root / cfg.inputs.structure)
experimental_data = read_experimental_data(root / cfg.inputs.exp_data)
setup = from_experiment(structure, experimental_data, cfg)
plan = build_orientation_plans()(
    select_beams(cfg.blochwave.to_beam_selection(setup.integration))(setup.plans.combined)
)

test = ConvergenceTest(
    g_max_step=0.1,
    sg_max_step=0.005,
    tilt_steps_step=5,
    num_passes=2,
)
tolerance = ConvergenceTolerance(
    r_factor_threshold=0.01,
    max_iterations=100,
)

converged_plan = converge_numerics(
    test,
    cfg.blochwave.to_rocking_curve(setup.integration),
    cfg.blochwave.to_policy(),
    setup.refinement,
    tolerance,
)(plan)
```
