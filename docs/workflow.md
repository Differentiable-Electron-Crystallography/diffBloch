# Workflow

diffBloch converts a starting crystal structure and 3D electron-diffraction data into a refined structure.

This page walks through that calculation end to end.

Before running diffBloch, an `experiment.yaml` file must be created in the directory containing the
starting structure `.cif` and experimental `.cif_pets` data. The YAML identifies those files and
specifies the simulation, preprocessing, and refinement settings that are not selected from the
input data. See [Hyperparameter selection](hyperparameter-selection.md) for the available settings
and their defaults.

Commands are run from the repository root with `uv run`. In the examples below,
`<experiment_dir>` denotes this directory.

## Experiment directory

A calculation begins with an experiment directory containing:

```text
<experiment_dir>/
  experiment.yaml
  structure.cif
  exp_data.cif_pets
```

The structure CIF supplies the starting atomic model. The `.cif_pets` file supplies the observed
intensities, uncertainties, orientations, wavelength, goniometer angles, and unit cell parameters. Both
files are specified in `experiment.yaml`:

```yaml
name: example-experiment

inputs:
  structure: structure.cif
  exp_data: exp_data.cif_pets

sample:
  thicknesses: [800.0]  # Angstroms
```

The simulation hyperparameters include the reciprocal-space cutoffs and the number of rocking-curve
samples. Suitable values depend on the experiment and should be established by convergence testing
before preprocessing or refinement. 

With `mosaicity: true`, diffBloch reads the apparent mosaicity from `.cif_pets` and converts it to a
moving-average sample span using the angular spacing between sampled orientations. The calculated
rocking curve is smoothed over that span before it is summed. This does not add Bloch-wave solves.
`mosaicity: false` (the default) applies no smoothing and ignores any PETS mosaicity value. The
output is one calculated diffraction pattern for each experimental rotation.

For more information, see [Inputs and outputs](inputs.md).

Every command below verifies the structure CIF and `.cif_pets` file(s) against a checksum recorded in
`reproducibility/experiment.lock`, creating that lock automatically the first time it runs. This file
is a raw-input blessing gate: if those input bytes change later, the command fails instead of
rewriting the lock. For a new experiment, the lock can also be created explicitly:

```bash
uv run diffbloch lock-experiment <experiment_dir>
```

That command refuses to overwrite an existing lock. After an intentional CIF or `.cif_pets` change,
delete `experiment.lock` and rerun, or force a rewrite:

```bash
uv run diffbloch lock-experiment --force <experiment_dir>
```

Warning: accepting changed input bytes invalidates existing plan and refinement locks, so
preprocessing and refinement outputs must be regenerated.

See [Reproducibility](reproducibility.md#input-files).

## Convergence testing

The convergence test determines suitable values for the main simulation hyperparameters:

```bash
uv run diffbloch convergence-test <experiment_dir>
```

The command reports settled values for `gmax`, `sgmax`, and `tilt_steps`. These correspond to
`blochwave.g_max`, `blochwave.sg_max`, and `blochwave.rocking_curve_sampling` in
`experiment.yaml`. 

For more infromation, see [Convergence testing](convergence-testing.md).

## Preprocessing

Preprocessing establishes specimen thickness and optimizes the experimental orientations before structural refinement.  

For more information, see [Preprocessing](preprocessing.md).

Preprocessing is run with:

```bash
uv run diffbloch preprocess <experiment_dir>
```

When the approximate mean thickness is known, one shared starting value may be used for
orientation optimization:

```yaml
sample:
  thicknesses: [800.0]

preprocess:
  optimize_thickness: false
```

When the thickness is uncertain, a thickness grid search can be run before orientation optimization:

```yaml
preprocess:
  optimize_thickness: true
  optimize_orientation: true
  thickness:
    min_thickness: 100.0
    max_thickness: 2000.0
    n_steps: 100
    plot: true
```

With `plot: true`, residual-versus-thickness plots are written to `thickness_optim/`.  

## Structural refinement

The structural parameters and optimizer settings are specified in `experiment.yaml`. Refinement is
run with:

```bash
uv run diffbloch refine <experiment_dir>
```

The objective and validation metrics are reported throughout the run. 

For more information, see [Refinement](refinement.md). 

## Outputs

A completed refinement writes the main results beside the inputs and under `reproducibility/`:

```text
<experiment_dir>/
  refined_structure.cif
  refinement_report.txt
  thickness_optim/                 # when thickness plots are enabled
  reproducibility/
```

`refined_structure.cif` contains the refined structural model, while `refinement_report.txt`
summarizes the run. The `experiment.yaml`, `.cif`, `.cif_pets`, and complete `reproducibility/` directory form the record associated with a reported result. The locks verify the inputs and preprocessed starting point; they do not guarantee identical floating-point optimizer trajectories on different hardware. 

For more information, see [Reproducibility](reproducibility.md).

## Installing from PyPI (pre-release)

diffBloch is published to PyPI as a release candidate. `pip` installs a pre-release only when asked
for one, so `--pre` is required.

### Install

diffBloch requires Python 3.12 or newer. Check the interpreter with `python3 --version` before
creating the virtual environment.

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
uv tool install --pre diffBloch                              # console script on PATH
uvx --prerelease=allow --from diffBloch diffbloch --version   # run without installing
```

The logger backends are optional extras: `pip install --pre 'diffBloch[wandb]'` or
`pip install --pre 'diffBloch[comet]'`.

Installation pulls `torch>=2.13`, which is a large download. On Linux with CUDA, install the
required torch build from the PyTorch index first, then install diffBloch.

`diffbloch --version` reports the installed version, {{ version }} for the release these docs
describe. `diffbloch --help` lists the available subcommands.

### Obtaining an experiment directory

The wheel and sdist contain the package only. Example data is excluded because every Git LFS path
lives under `tests/` and `examples/`; an sdist built from a pointer-only checkout would carry LFS
stubs that install cleanly and then fail when read.

Either point the CLI at an experiment directory of your own, described in
[Inputs and outputs](inputs.md), or clone the repository to obtain the quartz example:

```bash
git lfs install
git clone https://github.com/Differentiable-Electron-Crystallography/diffBloch
cd diffBloch && git lfs pull
```

### Running the installed CLI

The console script runs the same pipeline as the `uv run` commands used elsewhere on this page:

```bash
EXP=examples/Colmey_et_al_2026/data/quartz-no-abs

diffbloch validate $EXP/experiment.yaml     # configuration check, no calculation
diffbloch converge $EXP --device cpu        # convergence testing
diffbloch preprocess $EXP --device cpu      # settle the Plan and write the checkpoint
diffbloch refine $EXP --device cpu          # gradient refinement
```

`--device` defaults to `cuda`, so a machine without CUDA requires `--device cpu`. `--max-batch`
raises the propagator block size on a larger GPU, and `--refresh` discards existing preprocess
checkpoints. `reproducibility/experiment.lock` is written on the first run; after an intentional
change to the inputs, refresh it with `diffbloch lock-experiment --force $EXP`.
