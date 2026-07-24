---
title: diffBloch
---

# diffBloch

Differentiable Bloch-wave structure refinement for rotating-stage 3D electron diffraction.

The valuable core is a small **differentiable map from a handful of structural parameters to a
scalar R-loss**, minimised by gradient descent so that simulated and observed diffraction
intensities agree. Because rotating-stage 3DED is dynamical — a multiple-scattering diffraction
problem — that map uses a Bloch-wave simulation rather than a kinematical approximation. Everything
else — parsing, config, logging, orchestration — exists to make the map inspectable, reproducible,
and safe to evolve, and lives *around* the core, never inside it.

## Overview

diffBloch starts from continuous-rotation 3DED observations: diffraction frames collected while the
crystal is tilted/rocked through reciprocal space and reduced upstream by PETS2 into `.cif_pets`.
It models refinement as two complementary values: a raw crystal structure and a settled `Plan`. The
structure supplies the differentiable crystallographic parameters; the `Plan` supplies the geometry
and data scaffold around them — typed observations, per-frame orientations, beam/scoring sets,
rocking-curve geometry, fitted nuisance values, and checkpoint provenance. Together they feed the
refinement engine, which runs a deterministic Bloch-wave simulation, compares calculated and
observed intensities, and iteratively minimizes the objective by updating selected trainable
parameters. The guides below unpack that path from experiment inputs through preprocessing,
refinement, reproducibility, observability, and runnable examples.

| Guide | What it covers |
|---|---|
| [Architecture](architecture.md) | The whole shape of the system. |
| [Inputs](inputs.md) | The raw data needed to refine a structure. |
| [Preprocessing](preprocessing.md) | How raw input data becomes a reusable `Plan`. |
| [Refinement](refinement.md) | With a settled `Plan`, choose evaluation or optimization. |
| [Reproducibility](reproducibility.md) | How `experiment.lock`, `plan.npz`, and `plan.lock` let later runs verify and reuse expensive preprocessing results. |
| [Observability](observability-guide.md) | How to track progress and debug refinements. |
| [Examples](examples.md) | Runnable example experiments for small and large compounds, and implementations of papers that demonstrate doing science with diffBloch. |

For day-to-day use, start with the CLI quickstart below. For custom scientific workflows, the same
pieces are available through the public Python API.

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
