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

For more information, see [Inputs and outputs](inputs.md).

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
