---
title: diffBloch
---

# diffBloch

Differentiable Bloch-wave electron-diffraction structure refinement.

The valuable core is a small **differentiable map from a handful of structural parameters to a
scalar R-loss**, minimised by gradient descent so that simulated and observed diffraction
intensities agree. Everything else — parsing, config, logging, orchestration — exists to make that
map inspectable, reproducible, and safe to evolve, and lives *around* the core, never inside it.

## Overview

### Core simulation and refinement

The [core](api/core.md) starts with a deterministic Bloch-wave simulation, then the
[refinement engine](api/engine.md) exposes selected simulation inputs as differentiable
[parameters](api/params.md). In PyTorch terms, those parameters sit inside an optimization loop:
each pass simulates diffraction from the current structure, compares the calculated beam intensities
with the experimental observations, and uses gradients to improve the inputs so the simulated
diffraction pattern tends towards congruence with the experiment. The Ewald-sphere geometry
determines which reciprocal-lattice beams are excited and scored; the refinement objective then
compares their calculated and observed intensities. The differentiable parameters are structural
quantities — primarily asymmetric-unit (ASU) atom coordinates, with ADPs, occupancies, and structure
factors available as refinable groups.

### Core flow

The core flow is:

1. `.cif` + `.cif_pets` + config are parsed into typed records.
2. Preprocessing turns those records into a reusable `Plan`.
3. The `Plan` and differentiable structural parameters feed the deterministic Bloch-wave simulation.
4. The simulated pattern is compared with the observed pattern as an R-loss.
5. Gradients from that loss update the differentiable structural parameters, then the loop repeats.

### Inputs and typed records

diffBloch is driven by two input files: experimental diffraction data processed by
[PETS2](https://pets.fzu.cz/) into `.cif_pets`, and a starting crystal structure as a standard
`.cif`. Because the simulator makes concrete assumptions about the shapes, units, indices, and
crystallographic meaning in those files, diffBloch does not pass parser output directly into the
numerical core. It reads CIF/PETS data into typed [Pydantic](https://docs.pydantic.dev/)
[IO models](api/io.md) first, so invalid or unsupported input fails at the boundary with explicit
validation errors.

| File | Role |
|---|---|
| `experiment.yaml` | Experiment config and references to the input files. |
| `structure.cif` | Starting crystal structure. |
| `observations.cif_pets` | Processed experimental diffraction observations. |
| `experiment.lock` | Content identity for the input CIF/PETS files. |
| `plan.npz` / `plan.lock` | Reusable preprocessing checkpoint and its provenance lock. |

### Reproducibility locks

To make refinements reproducible across the expensive preprocessing boundary, diffBloch records the
identity of the input CIFs and the preprocessing recipe. `experiment.lock` pins the structure and
observation files by content hash; `plan.lock` then ties a `plan.npz` checkpoint to that input lock,
the resolved preprocess-determining config, the ordered preprocess steps, the producing code
version, and the checkpoint artifact hash. This gives diffBloch-driven inference and refinement runs
a verifiable starting point: a reused checkpoint is known to match the inputs and preprocessing that
produced it, rather than being an unlabelled intermediate file.

### Preprocessing and `Plan`

The surrounding orchestration is deliberately a shell around that stable, physics-grounded
deterministic simulation. Its central layer is [preprocessing](api/preprocess.md), which produces a
`Plan`: the immutable geometry and data context the simulator will consume — shared
structure-factor support, per-rotation orientations, beam sets, rocking-curve tilts, observed
patterns, alignments, and fitted nuisance values such as thickness. The `Plan` is not the structure
being refined; it is the settled scaffold around the differentiable structural parameters that enter
the optimization loop.

Preprocessing is a composable `Plan → Plan` pipeline. Any function with that shape can become a
pipeline step: it receives a `Plan`, does one focused piece of work, and returns an updated `Plan`.
As a rule, each step should do one thing. For example, the rocking-curve integration step takes the
current per-rotation plans, expands each orientation into sampled virtual tilts across the
integration semiangle, builds the corresponding tilt geometry, and returns a new `Plan` whose
orientations carry those tilts for the forward model. Other steps select solve beams, build
per-orientation plans, apply mosaicity, fit orientation and thickness against
simulation/observation agreement, optionally sweep numerical coverage/convergence knobs, and settle
coupled beam unions. The pipeline also provides higher-level composition utilities, such as
`iterate_until` and `Fork`, for repeated steps, conditional branching, and optimization-style steps
that improve `Plan` values before the final refinement loop begins.

Because this settled `Plan` is expensive and reusable, diffBloch checkpoints it as `plan.npz` plus a
lock file, so later inference or refinement runs can resume from the same validated preprocessing
state instead of recomputing it. [Config](api/config.md), [IO](api/io.md), logging, checkpointing,
and the [CLI](api/app.md) wrap that pipeline so the workflow stays reproducible and inspectable
without becoming part of the scientific core.

### Observability

As the pipeline runs, diffBloch emits typed [observability](api/observability.md) events for
preprocessing, coupling, inference, and refinement progress. The default CLI prints those events
through a console logger, and the same event stream can be sent to CSV, Weights & Biases, or Comet;
writing another logger is just a matter of implementing the small logger protocol for the event
objects you care about.

### CLI and examples

For day-to-day use, diffBloch ships with a [CLI](api/app.md) that has sane defaults for both
preprocessing and the refinement loop, and can be run out of the box against bundled compounds such
as quartz, abiraterone, LTA, and CsPbBr3. To run the small quartz example end to end — including
preprocessing and refinement — point the CLI at the ordinary quartz experiment directory:

```bash
diffbloch run refine examples/experiments/quartz
```

For a quicker first run, use a checkpointed example such as `quartz-checkpoint`, which already
includes `plan.npz` and `plan.lock`, so the CLI can skip preprocessing and start from the reusable
`Plan`:

```bash
diffbloch run refine examples/experiments/quartz-checkpoint
```

For a larger compound, run on an accelerator when available:

```bash
diffbloch run refine examples/experiments/abiraterone-checkpoint --device cuda
```

The examples choose `refinement.precision: fp32` for faster iteration; switch that field to `fp64`
for the conservative highest-precision refinement path.

It also supports scientific research workflows for power users who need to compose their own
pipelines, constraints, penalties, or model components. To get started with those patterns, review
[`examples/papers`](https://github.com/Differentiable-Electron-Crystallography/diffBloch/tree/main/examples/papers),
which shows more advanced usage than the default CLI path.

## Quickstart

Run the small quartz example end to end:

```bash
diffbloch run refine examples/experiments/quartz
```

Start faster from the bundled quartz preprocessing checkpoint:

```bash
diffbloch run refine examples/experiments/quartz-checkpoint
```

Run a larger checkpointed compound on CUDA:

```bash
diffbloch run refine examples/experiments/abiraterone-checkpoint --device cuda
```

Use `infer` when you want to simulate and score a settled plan without changing parameters. Use
`refine` when you want to run the optimization loop and update the structural parameters. Run
`diffbloch validate <experiment.yaml>` to check an experiment config before launching a longer job.

Python users can compose preprocessing steps, constraints, penalties, and refinement problems
directly with the public API; the CLI is the friendly default runner, not the only path. See
`examples/papers` for more advanced composition patterns.

## API

API docs are generated from docstrings and type signatures via `mkdocstrings`. The navigation
covers every public layer: **Config**, **IO**, **Core**, **Params**, **Specs**, **Engine**,
**Preprocess**, **Observability**, and **App**.
