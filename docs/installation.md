# Installation

diffBloch requires Python 3.12 or newer. Two installations are available: the package published on
PyPI, and a development checkout of the repository. The checkout is required to develop diffBloch or
to run the bundled examples.

## From PyPI

diffBloch is published to PyPI as a release candidate. `pip` installs a pre-release only when asked
for one, so `--pre` is required.

Check the interpreter with `python3 --version`, then install into a virtual environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --pre diffBloch
```

uv fetches a suitable interpreter when the host has none:

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install --pre diffBloch
```

Two further forms install the command-line entry point without a project environment:

```bash
uv tool install --pre diffBloch                               # console script on PATH
uvx --prerelease=allow --from diffBloch diffbloch --version    # run without installing
```

Installation pulls `torch>=2.13`, which is a large download. On Linux with CUDA, install the
required torch build from the PyTorch index first, then install diffBloch.

### Optional extras

The logging backends are optional extras. The core never imports them, and each confines its SDK to
one module.

| Extra | Command | Enables |
|---|---|---|
| `wandb` | `pip install --pre 'diffBloch[wandb]'` | Weights & Biases logging backend. |
| `comet` | `pip install --pre 'diffBloch[comet]'` | Comet logging backend. |

## From a checkout

Git LFS supplies the experimental data and plan checkpoints stored in the repository. Without it
those paths are pointer stubs rather than files.

```bash
git lfs install
git clone https://github.com/Differentiable-Electron-Crystallography/diffBloch
cd diffBloch
git lfs pull
uv sync --dev
```

Commands then run inside the project environment as `uv run diffbloch ...`, which is the form used
throughout these guides. An installed package provides the same commands as `diffbloch ...`.

## Verifying the installation

```bash
diffbloch --version
diffbloch --help
```

`diffbloch --version` reports the installed version, {{ version }} for the release these docs
describe. `diffbloch --help` lists the available subcommands.

## Experiment data

The wheel and sdist contain the package only. Example data is excluded because every Git LFS path
lives under `tests/` and `examples/`; an sdist built from a pointer-only checkout would carry LFS
stubs that install cleanly and then fail when read.

An installed package therefore requires an experiment directory of your own, described in
[Inputs and outputs](inputs.md). The bundled experiments listed in [Examples](examples.md) require
the checkout above.

Once an experiment directory is available, [Workflow](workflow.md) covers the pipeline from input
files to refined structure.
