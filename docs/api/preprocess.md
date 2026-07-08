# Preprocess

The composable `Plan → Plan` preprocess pipeline: build the initial plan from an experiment, sharpen
it with swappable steps (beam selection, orientation/thickness fits, rocking-curve integration,
mosaicity, convergence/coverage sweeps), then hand the final plan to a terminal (`run_inference` or
`engine.refine`).

## Spine

::: diffBloch.preprocess.plan

::: diffBloch.preprocess.experiment

::: diffBloch.preprocess.pipeline

::: diffBloch.preprocess.orientation

::: diffBloch.preprocess.scoring

::: diffBloch.preprocess.coupling

## Steps

::: diffBloch.preprocess.steps.beams

::: diffBloch.preprocess.steps.fit_orientation

::: diffBloch.preprocess.steps.fit_thickness

::: diffBloch.preprocess.steps.rocking_curve

::: diffBloch.preprocess.steps.mosaicity

::: diffBloch.preprocess.steps.convergence

::: diffBloch.preprocess.steps.coverage

::: diffBloch.preprocess.steps.coupling

## Orchestration and terminals

::: diffBloch.preprocess.driver

::: diffBloch.preprocess.inference

## Checkpoint / resume

Serialize a settled `Plan` to a portable `.npz` and read it back (source persisted, compiled
geometry rebuilt on load). The `run infer` CLI checkpoints/resumes against this plus the
`plan.lock` provenance in `diffBloch.config.manifest`.

::: diffBloch.preprocess.serialize
