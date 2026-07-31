# Hyperparameter selection

Bloch-wave calculations use a finite subset of an infinite reciprocal lattice and a finite number
of samples across each rocking curve. Too few beams or samples can change the calculated
intensities; too many increase the cost without improving the result. The required values depend on
the material, orientation, and thickness.

## Convergence

A calculation is converged when increasing its numerical resolution produces little to no change in the
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
approximately as {math}`N^3` (the propagator diagonalizes or exponentiates the {math}`N \times N`
structure matrix once per orientation/thickness). Cost scales approximately linearly with
`rocking_curve_sampling`.

## Convergence testing in diffBloch

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
    tilt_steps_step=2,
    num_passes=2,
)
tolerance = ConvergenceTolerance(
    r_factor_threshold=0.005,
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

Smaller sweep steps resolve the convergence boundary more precisely but require more simulations.
The threshold states the numerical accuracy required; it should be chosen before inspecting the
result and reported with the selected hyperparameters.
