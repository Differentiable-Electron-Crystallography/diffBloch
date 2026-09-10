# Installation

diffBloch is currently tagged as a release candidate and is under active development. The `v0.2.0`
release will be the first stable public release.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Other Python package managers
also work (`pip install` and equivalents), but uv is what these guides use.

## From PyPI

```bash
uv tool install 'diffBloch>=0.2.0rc1'
```

The requirement names the candidate explicitly. Avoid `--pre`: it allows pre-releases for every
package in the resolution, not just diffBloch, and installs beta builds of pydantic and SQLAlchemy.

The `diffbloch` CLI is then on `PATH`:

```bash
diffbloch --version
diffbloch --help
```

`diffbloch --version` reports {{ version }} for the release these docs describe.

To import diffBloch from your own Python, install it into a virtual environment instead:

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install 'diffBloch>=0.2.0rc1'
```

Installation pulls `torch>=2.13`, which is a large download. `--torch-backend=auto` selects the
build matching the host accelerator.

The logging backends are optional extras, and the core never imports them:
`uv tool install 'diffBloch[wandb]>=0.2.0rc1'`, likewise `comet`.

## Running your own data

[Understand how experiments are structured](inputs.md) first. A run needs a directory containing
`experiment.yaml`, the starting structure CIF, and the experimental `.cif_pets` data. The
[examples directory](https://github.com/Differentiable-Electron-Crystallography/diffBloch/tree/main/examples/Colmey_et_al_2026)
holds reference experiments against a variety of crystals.

```bash
diffbloch validate   my-experiment/experiment.yaml
diffbloch preprocess my-experiment --device cpu
diffbloch refine     my-experiment --device cpu
```

`--device` defaults to `cuda`, so a machine without CUDA requires `--device cpu`.

## From a git clone

A checkout is required to develop diffBloch and to run the bundled examples. Git LFS supplies the
experimental data and plan checkpoints; without it those paths are pointer stubs rather than files.

```bash
git lfs install
git clone https://github.com/Differentiable-Electron-Crystallography/diffBloch
cd diffBloch && git lfs pull && uv sync --dev

EXP=examples/Colmey_et_al_2026/data/quartz-no-abs
uv run diffbloch validate $EXP/experiment.yaml
uv run diffbloch refine   $EXP --device cpu
```

The wheel and sdist contain the package only. Every Git LFS path lives under `tests/` and
`examples/`, which the sdist excludes, so a clone is the only route to the bundled experiments.

[Workflow](workflow.md) covers the pipeline from input files to refined structure.
