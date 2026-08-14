# Inputs and outputs

Each experiment has its own directory containing `experiment.yaml`, a starting structure CIF, and
experimental `.cif_pets` data.

## Required inputs

| File | Contents |
|---|---|
| `experiment.yaml` | Input filenames and settings for the experiment. |
| `.cif` | Starting crystal structure. |
| `.cif_pets` | Experimental diffraction data reduced by PETS2. |

Paths written in `experiment.yaml` are measured from the experiment directory:

```yaml
inputs:
  structure: structure.cif
  exp_data: experiment.cif_pets
```

The structure CIF supplies the atoms, fractional coordinates, occupancies, atomic displacement
parameters (ADPs), and space-group symmetry operations.

The `.cif_pets` file supplies the observed reflection intensities and uncertainties, crystal
orientations, goniometer angles, electron wavelength, unit cell, and apparent mosaicity.

## Unit cell

Both input files contain a unit cell. diffBloch uses the `.cif_pets` cell for the simulation and uses the
structure-CIF cell as a consistency check.

- A difference greater than 1% in any cell parameter produces a warning.
- A difference greater than 5% stops the calculation.



## Excluding rotations

Individual rotations can be excluded when their diffraction data should not be used, for example
after beam damage has degraded the crystal or when no usable reflections were recorded. Rotations
are selected by their zero-based indices:

```yaml
blochwave:
  ignore_orientations: [0, 1, 18, 56]
```

Excluded rotations are not used during preprocessing, inference, or refinement.

## Multiple datasets

Several `.cif_pets` files from the same material can be combined in one refinement:

```yaml
inputs:
  structure: structure.cif
  multi_dataset: true
  exp_data:
    - crystal_1.cif_pets
    - crystal_2.cif_pets
```

The rotations from all files are combined against one structure. Each dataset keeps its own
orientations and thicknesses. The files must describe the same crystal and use the same electron
energy. The first `.cif_pets` file supplies the unit cell used for the combined refinement.

When the learned thickness model is enabled (`refinement.thickness_nn`, the default), each dataset
trains its own network over its own tilt range — pooled datasets with overlapping tilt ranges are
never forced onto one shared thickness-vs-alpha curve. The refinement report carries one
`Thickness NN -- <ref>` section and one `thickness_nn_shape_<stem>.png` plot per dataset.

## Outputs

Running preprocessing or refinement adds results to the experiment directory:

| Output | Contents |
|---|---|
| `refined_structure.cif` | Refined crystal structure. |
| `refinement_report.txt` | Final residuals, reflection counts, and refinement summary. |
| `thickness_optim/` | Thickness-search plots when plotting is enabled. |
| `thickness_nn_shape_<stem>.png` | Learned thickness curve per `.cif_pets` dataset, when the thickness network and plotting are enabled. |
| `reproducibility/plan.<stem>.npz` | Saved preprocessing for each `.cif_pets` dataset. |
| `reproducibility/plan.<stem>.lock` | Inputs and settings used for the saved preprocessing. |
| `reproducibility/refined_parameters.npz` | Refined parameter values. |
| `reproducibility/refinement.lock` | Inputs and settings used for the refinement. |

`reproducibility/experiment.lock` is not one of these -- it identifies the input files, not a
calculation result. It is created automatically the first time any command runs against the
experiment directory; see [Reproducibility](reproducibility.md#input-files).

See [Refinement](refinement.md#refinement-outputs) for the refinement report and
[Reproducibility](reproducibility.md) for the lock files.
