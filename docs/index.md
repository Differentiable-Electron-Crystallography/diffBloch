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

Differentiable Bloch wave structure refinement for 3D electron diffraction.

This codebase is entirely open-source, and we welcome user contributions as well as questions.


## Overview

diffBloch is a crystallographic refinement software for 3D ED. To perform a refinement both an initial structural model and experimental data are required. The initial unrefined atomic structure in the form of a 'cif' (crystallographic information file) can be obtained from previous experiments if the structure is known or via structure solution. Experimental data in the form of diffraction frames are collected while the crystal is tilted/rocked through reciprocal space and reduced upstream by one of PETS2 or DIALS, into the`.cif_pets` data format.

diffBloch performs the refinement as two complementary values: a crystal structure, consisting of atomic coordinates, occupancies and thermal displacement parameters, and a settled `Plan`, consisting of crystallographic metadata such as thickness or orientation. Together they feed the refinement engine, which runs a repeatable Bloch wave simulation, compares calculated and observed intensities, and iteratively minimizes the objective by updating selected trainable parameters. The guides below unpack that path from experiment inputs through preprocessing, refinement, reproducibility, observability, and runnable examples.

| Guide | Contents |
|---|---|
| [Workflow](workflow.md) | Refinement pipeline from input files to refined structure. |
| [Inputs and outputs](inputs.md) | Files required for a refinement and files produced by diffBloch. |
| [Hyperparameter selection](hyperparameter-selection.md) | Simulation, preprocessing, and refinement hyperparameters and their defaults. |
| [Convergence testing](convergence-testing.md) | Convergence of calculated intensities with respect to `g_max`, `sg_max`, and rocking-curve sampling. |
| [Preprocessing](preprocessing.md) | Crystal-orientation and thickness determination before structural refinement. |
| [Bloch wave simulation](bloch_wave_simulation.md) | Theory and equations used to calculate dynamical diffraction intensities. |
| [Refinement](refinement.md) | Structural parameter optimization against experimental intensities. |
| [Devices and scaling](devices-and-scaling.md) | CPU and GPU execution, memory controls, and refinement profiling. |
| [Reproducibility](reproducibility.md) | Records identifying the inputs and preprocessing used for a refinement. |
| [Examples](examples.md) | Runnable experiments included with diffBloch. |

## Quickstart

### Command line

The command-line interface (CLI) runs diffBloch from a terminal. This is the standard way to run a
complete experiment. `uv` installs the required Python packages and runs diffBloch inside the
project environment. Git LFS downloads the larger experimental-data files stored in the repository.

From the repository directory:

```bash
git lfs install
git lfs pull
uv sync --dev
```

Each diffBloch command takes the path to an experiment directory containing `experiment.yaml`, the
starting CIF, and the `.cif_pets` data:

```bash
uv run diffbloch refine <experiment_dir>
```



## Citation

If diffBloch is used in research, please cite:

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
inputs.md
hyperparameter-selection.md
convergence-testing.md
preprocessing.md
bloch_wave_simulation.md
refinement.md
devices-and-scaling.md
reproducibility.md
examples.md
```

```{toctree}
:hidden:
:caption: Python API

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
