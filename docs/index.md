---
title: diffBloch
---

# diffBloch

![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fdifferentiable-electron-crystallography.github.io%2FdiffBloch%2Fcoverage-endpoint.json)
![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white)
![uv](https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-D7FF64?logo=ruff&logoColor=black)
![mypy](https://img.shields.io/badge/mypy-strict-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-97ca00)

Differentiable Bloch-wave structure refinement for 3D electron diffraction.

This codebase is entirely open-source, and we welcome contributions as well as questions.


## Overview

diffBloch is a crystallographic refinement software for 3D ED. To perform a refinement both an initial structural model and experimental data are required. The initial unrefined atomic structure in the form of a 'cif' (crystallographic information file) can be obtained from previous experiments if the structure is known or via structure solution. Experimental data in the form of diffraction frames collected are collected while the crystal is tilted/rocked through reciprocal space and reduced upstream by one of PETS2 or DIALS, into the`.cif_pets` data format.

diffBloch performs the refinement as two complementary values: a crystal structure, consisting of atomic coordinates and thermal displacement parameters, and a settled `Plan`, consisting of crystallographic metadata such as thickness or orientation. Together they feed the refinement engine, which runs a repeatable Bloch-wave simulation, compares calculated and observed intensities, and iteratively minimizes the objective by updating selected trainable parameters. The guides below unpack that path from experiment inputs through preprocessing, refinement, reproducibility, observability, and runnable examples.

| Guide | What it covers |
|---|---|
| [Workflow](workflow.md) | The calculation from experiment setup through convergence, preprocessing, inference, refinement, and outputs. |
| [Bloch-wave simulation](bloch-wave-simulation.md) | The physics: structure factor, relativistic interaction parameter, structure matrix, and its two solvers. |
| [Inputs](inputs.md) | The starting structure and rocking-curve data a refinement needs. |
| [Preprocessing](preprocessing.md) | Fitting crystal orientation and specimen thickness before the structure is touched. |
| [Convergence testing](convergence-testing.md) | Choosing beam and rocking-curve settings using convergence tests. |
| [Config reference](hyperparameter-selection.md) | Every `experiment.yaml` switch, its default, and what's auto-filled from CIF/PETS instead. |
| [Refinement](refinement.md) | The optimization loop: constraints, restraints, and thickness models alongside the structure. |
| [Reproducibility](reproducibility.md) | How `experiment.lock` and the per-dataset `plan.<stem>.npz`/`.lock` checkpoints pin a fitted `Plan` so a result can be reproduced exactly. |
| [Observability](observability-guide.md) | Tracking wR2, R_obs, and diffraction loss as a run progresses. |
| [Examples](examples.md) | Runnable example experiments for small and large compounds, and implementations of papers that demonstrate doing science with diffBloch. |

For day-to-day use, start with the CLI quickstart below. For custom scientific workflows, the same
pieces are available through the public Python API.

## Quickstart

Prerequisite: install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Git LFS](https://git-lfs.com/) for the bundled `.cif_pets` experimental data and plan-checkpoint `.npz`
checkpoints, then sync the project environment from the repository root:

```bash
git lfs install
git lfs pull
uv sync --dev
```

Run CLI commands through `uv run` unless you have separately installed the `diffbloch` console script
on your shell `PATH`. Every command below takes an experiment directory (see
[Inputs](inputs.md)) -- try one under `examples/`, e.g. `examples/Colmey_et_al_2026/data/quartz-no-abs`.

```bash
uv run diffbloch convergence-test <experiment_dir>   # test numerical convergence
uv run diffbloch preprocess <experiment_dir>         # settle orientation/thickness, write plan checkpoints
uv run diffbloch infer <experiment_dir>              # forward-simulate and score a settled Plan
uv run diffbloch refine <experiment_dir>             # preprocess (or reuse) + gradient-refine
```

Add `--refresh` to any command to force a real recompute instead of reusing a checkpoint, and
`--device cpu`/`--device cuda` to pick the execution device. `infer` scores without changing
parameters; `refine` runs the optimization loop, repeatedly simulating, scoring, and updating the
selected trainable structural parameters -- see [Refinement](refinement.md) for its outputs.



## Citation

If you use diffBloch in your research, please cite it:

```bibtex
@misc{diffBloch,
  author  = {Doherty, Tiarnan and Malik, Shreshth and Colmey, Benjamin and Maitland, Iain, and Midgley, Paul},
  title   = {diffBloch},
  version = {0.2.0},
  year = {2026},
  url     = {https://github.com/Differentiable-Electron-Crystallography/diffBloch}
}
```


```{toctree}
:hidden:
:caption: Guides

workflow.md
bloch-wave-simulation.md
inputs.md
preprocessing.md
convergence-testing.md
hyperparameter-selection.md
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
