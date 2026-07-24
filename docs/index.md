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

### Simulation and refinement

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

### End-to-end flow

The end-to-end flow is:

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

For day-to-day use, diffBloch ships with a [CLI](api/app.md) that has sane defaults for
preprocessing, inference, and refinement. The bundled `examples/experiments` directories are baked-in
demonstration runs; [`examples/papers`](https://github.com/Differentiable-Electron-Crystallography/diffBloch/tree/main/examples/papers)
shows more advanced composition patterns for research workflows.

The examples choose `refinement.precision: fp32` for faster iteration; switch that field to `fp64`
for the conservative highest-precision refinement path.

## Quickstart

Prerequisite: install [uv](https://docs.astral.sh/uv/getting-started/installation/), then sync the
project environment from the repository root:

```bash
uv sync --dev
```

Run CLI commands through `uv run` unless you have separately installed the `diffbloch` console script
on your shell `PATH`.

Validate a config before launching a longer job:

```bash
uv run diffbloch validate examples/experiments/quartz-checkpoint/experiment.yaml
```

Simulate and score the settled quartz checkpoint without changing parameters:

```bash
uv run diffbloch run infer examples/experiments/quartz-checkpoint
```

Run the small quartz example end to end, including preprocessing and refinement:

```bash
uv run diffbloch run refine examples/experiments/quartz
```

Start refinement faster from the bundled quartz preprocessing checkpoint:

```bash
uv run diffbloch run refine examples/experiments/quartz-checkpoint
```

Use `infer` to run the forward simulation and scoring pass over a settled `Plan`; it does not update
structural parameters. Use `refine` to run the optimization loop, repeatedly simulating, scoring,
and updating the selected trainable structural parameters.

Python users can compose preprocessing steps, constraints, penalties, and refinement problems
directly with the public API; the CLI is the friendly default runner, not the only path.

## API

API docs are generated from the package's docstrings and type signatures. The navigation covers
every public layer: **Config**, **IO**, **Core**, **Params**, **Specs**, **Engine**,
**Preprocess**, **Observability**, and **App**.

```{toctree}
:hidden:
:caption: Guides

architecture.md
inputs.md
preprocessing.md
reproducibility.md
refinement.md
observability-guide.md
examples.md
```

```{toctree}
:hidden:
:caption: API

api/config.md
api/io.md
api/core.md
api/params.md
api/specs.md
api/engine.md
api/preprocess.md
api/observability.md
api/app.md
```
