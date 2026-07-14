# Example: LTA (zeolite A) inference — the large-cell path

A complete, runnable experiment directory for **zeolite A (LTA)** — a large cubic aluminosilicate
framework (a ≈ 12.3 Å, cell volume ~1861 Å³), real 3D electron-diffraction data. It is the
large-cell counterpart to the `quartz` example: the same faithful coupled recipe, but the cell is
big enough to exercise `diffbloch`'s large-cell GPU path.

## Files

| File                    | Role                                                            |
| ----------------------- | --------------------------------------------------------------- |
| `experiment.yaml`       | the experiment definition (inputs + numerical settings)         |
| `experiment.lock`       | input-byte identity; `run infer` verifies the inputs against it |
| `lta.cif`               | structure — zeolite A, space group Pm-3m                        |
| `lta_exp_data.cif_pets` | observed reflection intensities (PETS `.cif_pets`)              |

> **Data lineage:** `lta_exp_data.cif_pets` is real, private-repo-lineage data. Clear its
> redistribution before publishing this directory outside the project.

## Run

LTA's cell is above the large-cell threshold, so the recipe routes the coupled orientation search to
the **coarse fp32 branch** (halved eigensolve, gather integrity checks skipped behind the coverage
guard); the fp64 terminal re-scores for the reported number. The O(N³) eigensolve wants a GPU, and
the per-trial cost is host-bound, so parallelise rotations and **cap host BLAS/OpenMP threads** (in a
pod the node-sized thread pools otherwise oversubscribe the cores). From the repository root, on a
CUDA box:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  diffbloch run infer examples/experiments/lta --device cuda --workers 4 --console
```

The orientation fit is the expensive phase, so the first run writes a preprocess checkpoint
(`plan.npz` + `plan.lock`) into this directory; a second identical run reuses it in seconds (both are
gitignored here). Recompute from scratch with `--refresh`, or skip the checkpoint with
`--no-checkpoint`. No reference `R_obs` is pinned for LTA (unlike quartz's 0.0506) — this experiment
produces the representative mean, it does not assert against a published value.

See also: the `gpu_fit_performance`, `large_cell_gpu_path`, and `coupling_coverage_guard` tutorials
for the thread-cap / fp32-fork / coverage-guard machinery this run exercises.
