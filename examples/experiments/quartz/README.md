# Example: quartz inference

A runnable diagnostic configuration for the published quartz (SiO₂) electron-diffraction dataset.
It currently retains raw PETS rotation 62 only, runs its orientation fit, and skips thickness
fitting.

## Files

| File                       | Role                                                          |
| -------------------------- | ------------------------------------------------------------- |
| `experiment.yaml`          | the experiment definition (inputs + numerical settings)       |
| `experiment.lock`          | input-byte identity; `run infer` verifies the inputs against it |
| `quartz.cif`               | structure — SiO₂, space group P3₁21                           |
| `quartz_exp_data.cif_pets` | observed reflection intensities (PETS `.cif_pets`)            |

## Run

From the repository root:

```bash
uv run diffbloch run preprocess examples/experiments/quartz --refresh
```

The recipe selects coupled SOLVE beams from `g_max`/`sg_max`, builds the final rocking-curve plans
(including the configured mosaic reduction), then matches simulator HKLs to PETS observations and
runs orientation optimization. Thickness optimization is disabled in `experiment.yaml`.
The orientation fit writes a preprocess checkpoint
(`plan.npz` + `plan.lock`) into this directory; a second identical run reuses it in seconds. Both
checkpoint files are gitignored here. Recompute from scratch with `--refresh`, or skip the
checkpoint entirely with `--no-checkpoint`.

The command prints polished per-stage progress followed by a completion box, the resolved pipeline,
and the absolute checkpoint paths.

To run the 40-epoch structure refinement and write its best result:

```bash
uv run diffbloch run refine examples/experiments/quartz
```

Each epoch reports `wR2`, `R_obs`, and diffraction loss. The completion box reports the best epoch
and `HKLs (Observed/total)` as matched observed / all matched reflections, then lists
`refined_structure.cif`, `refined_parameters.npz`, `refinement_summary.json`, `plan.npz`, and
`plan.lock`.

For a variant that **ships** a pre-computed checkpoint (so the first run is already instant), see the
sibling `quartz-checkpoint` example. For the checkpoint machinery itself, see the checkpoint/resume
tutorial.
