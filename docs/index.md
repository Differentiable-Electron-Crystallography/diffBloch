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
| [Architecture](architecture.md) | Code Structure. |
| [Inputs](inputs.md) | Data needed to refine a structure. |
| [Preprocessing](preprocessing.md) | Initial steps before refinement. |
| [Hyperparameter selection](hyperparameter-selection.md) | Choosing beam and rocking-curve settings using convergence tests. |
| [Refinement](refinement.md) | Optimizing crystal structure |
| [Reproducibility](reproducibility.md) | How `experiment.lock`, `plan.npz`, and `plan.lock` let later runs verify and reuse expensive preprocessing results. |
| [Observability](observability-guide.md) | How to track progress and debug refinements. |
| [Examples](examples.md) | Runnable example experiments for small and large compounds, and implementations of papers that demonstrate doing science with diffBloch. |

For day-to-day use, start with the CLI quickstart below. For custom scientific workflows, the same
pieces are available through the public Python API.

## Quickstart

Prerequisite: install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Git LFS](https://git-lfs.com/) for the bundled `.cif_pets` observations and `plan.npz`
checkpoints, then sync the project environment from the repository root:

```bash
git lfs install
git lfs pull
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

The default refinement runs for 40 epochs, streams `wR2`, `R_obs`, and diffraction loss, then writes
the best refined CIF, exact raw parameter snapshot, and JSON summary into the experiment directory.
The CLI prints aligned preprocess/refinement completion summaries and absolute artifact paths.

Python users can compose preprocessing steps, constraints, penalties, and refinement problems
directly with the public API; the CLI is the friendly default runner, not the only path.

## Citation

If you use diffBloch in your research, please cite it (see
[CITATION.cff](https://github.com/Differentiable-Electron-Crystallography/diffBloch/blob/main/CITATION.cff)):

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

architecture.md
inputs.md
preprocessing.md
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
