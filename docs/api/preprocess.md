# Preprocess

The composable `Plan → Plan` preprocess pipeline: build the initial plan from an experiment, sharpen
it with swappable steps (beam selection, orientation/thickness determination, rocking-curve integration,
mosaicity, convergence/coverage sweeps), then hand the final plan to a terminal (`run_inference` or
`engine.refine`).

## Spine

```{eval-rst}
.. automodule:: diffBloch.preprocess.plan
```

```{eval-rst}
.. automodule:: diffBloch.preprocess.experiment
```

```{eval-rst}
.. automodule:: diffBloch.preprocess.pipeline
```

```{eval-rst}
.. automodule:: diffBloch.preprocess.orientation
```

```{eval-rst}
.. automodule:: diffBloch.preprocess.scoring
```

```{eval-rst}
.. automodule:: diffBloch.preprocess.coupling
```

## Steps

```{eval-rst}
.. automodule:: diffBloch.preprocess.steps.beams
```

```{eval-rst}
.. automodule:: diffBloch.preprocess.steps.optimize_orientation
```

```{eval-rst}
.. automodule:: diffBloch.preprocess.steps.optimize_thickness
```

```{eval-rst}
.. automodule:: diffBloch.preprocess.steps.rocking_curve
```

```{eval-rst}
.. automodule:: diffBloch.preprocess.steps.convergence
```

```{eval-rst}
.. automodule:: diffBloch.preprocess.steps.coverage
```

```{eval-rst}
.. automodule:: diffBloch.preprocess.steps.coupling
```

## Orchestration and terminals

```{eval-rst}
.. automodule:: diffBloch.preprocess.driver
```

```{eval-rst}
.. automodule:: diffBloch.preprocess.inference
```

## Checkpoint / resume

Serialize a settled `Plan` to a portable `.npz` and read it back (source persisted, compiled
geometry rebuilt on load). The `infer` CLI checkpoints/resumes against this plus the
`plan.lock` provenance in `diffBloch.config.manifest`.

```{eval-rst}
.. automodule:: diffBloch.preprocess.serialize
```
