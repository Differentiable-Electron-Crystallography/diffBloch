# Devices and scaling

diffBloch is optimized for GPU use. It checks for CUDA automatically and uses it if available;
otherwise it falls back to CPU.

```bash
uv run diffbloch refine <experiment_dir> --device cuda
uv run diffbloch refine <experiment_dir> --device cpu
```

Both devices perform the same calculation to the same accuracy; only speed differs. The fastest
`coupling_mode` and the phase-specific `solver` choices are not the same on CPU as on GPU. The config defaults
(`coupling_mode: union`) are tuned for GPU and are not optimal for CPU.

## CPU

`union` batches tilts into fewer, larger matrices. That pays off on GPU, where it improves
parallelism and amortizes launch overhead, but not on CPU, where the larger matrices simply cost
more per solve with no batching benefit. `per_tilt` is faster than `union` on CPU once the
structure matrix is larger than quartz's (N≈40); `union` is faster only at that small a scale.

`bloch_eigen` is faster than `matrix_exp` in every case measured, sometimes by 4-5x. `matrix_exp`'s
backward pass carries a fixed penalty (see [Forward vs backward time](#forward-vs-backward-time-gpu)
below) that is worst on CPU.

Measured on one rotation, three refinement epochs, steady-state (excludes the first, cache-warm
epoch):

| Structure | `union` + `bloch_eigen` | `union` + `matrix_exp` | `per_tilt` + `bloch_eigen` | `per_tilt` + `matrix_exp` |
|---|---:|---:|---:|---:|
| Quartz (tiny, N≈40) | 20.9 ms | 24.1 ms | 75.8 ms | 96.2 ms |
| Borane (N≈450) | 4144 ms | 16483 ms | 2311 ms | 4853 ms |
| CsPbBr₃ (N≈680) | 4268 ms | 23483 ms | 2221 ms | 6784 ms |

The fastest combination per structure is `union` for quartz and `per_tilt` for borane and CsPbBr₃.

## GPU

`union` (the default) is faster than `per_tilt` almost everywhere on GPU, since fewer, larger
batched calls suit GPU parallelism. One exception was measured: CsPbBr₃ with `matrix_exp`, where
`per_tilt` is faster (390 ms vs 499 ms). `union`'s larger matrices make `matrix_exp`'s backward
penalty worst there, and it outweighs the batching benefit.

`bloch_eigen` is faster than `matrix_exp` under `union`. Under `per_tilt`, `matrix_exp` is faster
than `bloch_eigen` for borane and CsPbBr₃: the smaller matrices shrink `matrix_exp`'s backward
penalty enough that its cheaper forward pass, which does not need a full eigendecomposition, wins
overall.

Same methodology as the CPU table:

| Structure | `union` + `bloch_eigen` | `union` + `matrix_exp` | `per_tilt` + `bloch_eigen` | `per_tilt` + `matrix_exp` |
|---|---:|---:|---:|---:|
| Quartz (tiny, N≈40) | 14.3 ms | 18.2 ms | 106.9 ms | 115.7 ms |
| Borane (N≈450) | 218.7 ms | 419.1 ms | 901.5 ms | 521.9 ms |
| CsPbBr₃ (N≈680) | 289.9 ms | 499.5 ms | 624.8 ms | 389.7 ms |

`bloch_eigen` requires a Hermitian, non-absorptive structure matrix and is not the refine default
even where it is faster: its backward pass is ill-conditioned near degenerate eigenvalues, which
symmetric crystals routinely produce. `matrix_exp` remains necessary for absorptive structures, and
for non-absorptive structures where `bloch_eigen` gives unstable refinement.

## Forward vs backward time (GPU)

`matrix_exp`'s backward pass costs several times its own forward pass; `bloch_eigen`'s does not.
This tracks structure-matrix size, the number of dynamically coupled beams, not the number of
refined parameters.

| Structure | Trainable params | Structure matrix N | `matrix_exp` forward / backward | ratio | `bloch_eigen` forward / backward | ratio |
|---|---:|---:|---|---:|---|---:|
| Quartz | 24 | ≈40 | 9.6 / 8.6 ms | 0.9x | 6.7 / 7.7 ms | 1.15x |
| CsPbBr₃ | 48 | ≈680 | 72 / 428 ms | 5.97x | 136 / 154 ms | 1.13x |
| Borane | 240 | ≈450 | 60 / 359 ms | 5.98x | 102 / 117 ms | 1.15x |

CsPbBr₃ has fewer trainable parameters than borane (48 vs 240) but a larger structure matrix, and
its `matrix_exp` backward pass is correspondingly slower (428 ms vs 359 ms). Parameter count does
not predict backward cost; matrix size does. `bloch_eigen`'s ratio stays close to 1x regardless of
matrix size, since it does not carry `matrix_exp`'s backward penalty.

The rows above use `union` coupling; the `per_tilt` figures in the CPU/GPU tables show the same
forward/backward split scaled by that coupling choice.

## GPU and CPU timings, full runs

The single-rotation figures above isolate the coupling and solver effect. The following are full
experiment timings measured on the same machine with identical settings:

| Experiment | GPU | CPU | CPU/GPU time |
|---|---:|---:|---:|
| Quartz, absorptive refinement, 40 epochs | 57.1 s | 72.8 s | 1.3x |
| CsPbBr₃, absorptive preprocessing, 59 rotations | 16 min 57 s | 5 h 00 min | 18x |
| CsPbBr₃, absorptive refinement, 40 epochs | 20 min 52 s | approximately 15.3 h | 44x |

CPU execution is reasonable for small structure matrices. GPU acceleration becomes increasingly
important as the number of dynamically coupled beams increases, and so does the choice of
`coupling_mode` and `solver`.

## Profiling refinement

`--profile` reports the time spent calculating structure factors, solving each rotation, computing
gradients, and updating the structural parameters:

```bash
uv run diffbloch refine <experiment_dir> --device cuda --profile
```

Profiling synchronizes the GPU during each measured section and therefore slows the run. It is
intended for diagnosis, not routine refinement.
