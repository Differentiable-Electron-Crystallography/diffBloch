# Devices and scaling

`--device`, `--workers`, and `--max-batch` control how a run executes. They do not change the
experiment settings or invalidate preprocessing checkpoints.

## Device

```bash
uv run diffbloch refine <experiment_dir> --device cuda
uv run diffbloch refine <experiment_dir> --device cpu
```

CUDA is the default. If CUDA is unavailable, diffBloch falls back to the CPU. Both devices perform
the same calculation, but a GPU is much faster for large structure matrices.

## Workers

`--workers` runs independent orientation-plan builds and orientation searches in parallel. The
default is one worker:

```bash
uv run diffbloch preprocess <experiment_dir> --workers 4
```

When using multiple workers, limit the host thread pools to avoid assigning a full set of CPU
threads to every worker.

## Maximum batch size

`--max-batch` limits the number of structure matrices passed to one matrix-exponential operation.
It controls memory use rather than the scientific result. The default selects a batch size from
the matrix size and available memory.

## GPU and CPU timings

The following timings were measured on the same machine with identical experiment settings:

| Experiment | GPU | CPU | CPU/GPU time |
|---|---:|---:|---:|
| Quartz, absorptive refinement, 40 epochs | 57.1 s | 72.8 s | 1.3× |
| CsPbBr₃, absorptive preprocessing, 59 rotations | 16 min 57 s | 5 h 00 min | 18× |
| CsPbBr₃, absorptive refinement, 40 epochs | 20 min 52 s | approximately 15.3 h | 44× |

CPU execution is reasonable for small structure matrices. GPU acceleration becomes increasingly
important as the number of dynamically coupled beams increases.

## Profiling refinement

`--profile` reports the time spent calculating structure factors, solving each rotation, computing
gradients, and updating the structural parameters:

```bash
uv run diffbloch refine <experiment_dir> --device cuda --profile
```

Profiling synchronizes the GPU during each measured section and therefore slows the run. It is
intended for diagnosis, not routine refinement.

The following results used a GPU and three refinement epochs:

| Structure | Refined parameters | Solved sections | Mean structure-matrix size | Forward per epoch | Gradient calculation per epoch |
|---|---:|---:|---:|---:|---:|
| Methyl, multiple datasets | 962 | 176 | 297 | approximately 0.62 s | approximately 3.06 s |
| CsPbBr₃, absorptive | 48 | 236 | 690 | approximately 4.17 s | approximately 25.5 s |

The gradient calculation dominates both examples. CsPbBr₃ has far fewer refined parameters but is
slower because its structure matrices are larger. Runtime is therefore governed mainly by the
number of dynamically coupled beams, not the number of refined structural parameters.
