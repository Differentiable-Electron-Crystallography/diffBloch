# Known issues

Latent bugs and deferred fixes discovered during development, recorded here so they stay
discoverable instead of being buried in commit messages. Each entry gives a precise location, the
impact, and the intended fix. Close an entry by deleting it in the commit that fixes it.

## Precision is hardcoded to float64, which blocks Apple-GPU runs and throttles consumer NVIDIA GPUs

The codebase hardcodes `torch.float64` in ~75 places. That keeps the dynamical-diffraction maths
(a dense `matrix_exp` / `eigh` of the Bloch operator, ill-conditioned near degenerate eigenvalues)
numerically safe, but it boxes in where the code can run:

- **Apple Silicon GPUs (MPS) cannot do float64 at all** -- Metal has no `double` type, so PyTorch
  raises *"Cannot convert a MPS Tensor to float64 dtype..."*. This is a hardware floor, not a
  PyTorch bug that will be fixed. On a Mac this code is therefore CPU-only.
- **NVIDIA segments float64 by market**, which is the surprising part: consumer GeForce cards
  (Ampere/Ada, e.g. RTX 3090/4090) run float64 at **1/64** of their float32 rate -- a deliberate
  product-tiering choice, not a silicon limit -- while datacenter cards run it fast (A100 ~1:2,
  9.7 TFLOPS FP64; H100 ~60 TFLOPS FP64). So "GPU-accelerated *and* float64" is only genuinely fast
  on datacenter hardware; on a consumer card float64 works but is heavily throttled.

So the realistic story today is: **Mac = CPU/float64; fast GPU = Linux + datacenter NVIDIA + CUDA.**
Intended fix is a *precision policy* rather than a hardcode: thread a configurable `dtype` (float64
on CPU/datacenter, float32 on consumer/MPS GPUs) -- gated on first **validating that float32 is
numerically adequate for the propagator** (the `matrix_exp`/`eigh` conditioning is the open
question). This couples with the device co-location issue below and is deferred future work, not a
stage-10 change. Sources: PyTorch MPS notes (docs.pytorch.org/docs/stable/notes/mps.html); NVIDIA
Ada FP64 1:64 ratio (en.wikipedia.org/wiki/GeForce_RTX_40_series); A100/H100 FP64 TFLOPS
(nvidia.com data-center pages).

## Each operation moves its static tensors onto the GPU by hand, and CPU-only CI can't catch a miss

To run refinement on a GPU you put the parameters on the GPU, so everything derived from them (the
structure factors, the Bloch operator, the exit wave) lands there too. But the static, build-once
inputs -- the hkl grid, reciprocal basis, atomic numbers, beam plans, thicknesses -- are created on
the CPU. PyTorch refuses to combine tensors on different devices, so just before each such input
meets a parameter-derived tensor it is copied onto the parameter's device with `.to(device)`.

Today that copy is written out by hand at every place it is needed -- in `engine/forward.py`
(`_structure_factors`, `_solve`), `core/products.py::align`, and the `core/solver.py` propagators.
Because it is repeated in many places, forgetting a single `.to(device)` is a real bug -- but one
that is invisible on CPU (there the copy is a harmless no-op) so the CPU-only CI stays green and the
failure only appears when someone actually runs on a GPU. That is exactly the gap commit `ae8eb78`
had to fix for `propagate`. Intended fix: do the move in one place instead of many -- e.g. a single
`.to(device)` on the plan/engine objects -- and/or add a GPU smoke test so a missed copy is caught.

## `ScatteringGrid` stores its grid as tensors, then immediately converts them back to NumPy

`ScatteringGrid.from_cell` computes the grid in NumPy, then stores `grid_hkl`/`reciprocal_basis` as
tensors (`engine/plan.py:52`). But `OrientationPlan.build` turns them straight back into NumPy with
`np.asarray(...)` to call `build_beam_plan` (`engine/plan.py:86`). So neither form is clearly the
owner, and the back-and-forth conversion is wasted work. As of the self-describing reshape this now
also covers `cell` (stored as a tensor on the grid, `np.asarray`-ed in `build`) and `orientation`
(stored as a tensor on `OrientationPlan`, normalized back to NumPy in `build`, including the
rebuild-from-prior-plan path). Intended fix: pick one representation -- either also keep the NumPy
arrays on the grid, or let `build_beam_plan` take tensors directly.

## `fit_orientation` must not re-orthonormalize the orientation matrices

The real `optim_orientation.csv` matrices are `orientation = R_goni . U` with `U = UB . B^-1`,
which folds a constant ~1% anisotropic *measured-vs-ideal cell correction* into the transform
(singular values 1.0118/1.0114/1.0095, identical across all 99 rotations -- not optimization drift).
They are therefore **non-orthonormal** (`det ~ 1.033`), and the per-orientation reciprocal basis the
geometry uses is `reciprocal_cell(cell @ orientation.T) = reciprocal_basis @ orientation^-1`, which
is *not* `reciprocal_basis @ orientation^T`. The distinction is observable on real data (~0.008
A^-1, ~1% on `|g|`, enough to move `Sg`/`Mii`).

The trap: a future `fit_orientation` (stage 11 slice 5) that parametrizes orientation as a pure
rotation, or that re-orthonormalizes these matrices (e.g. via polar/SVD), will silently discard the
cell correction and shift every `|g|` by ~1%. If orientation is re-fit, the measured-cell scale must
be preserved (fit the rotation *around* the existing `U`, or carry the cell correction separately).
Pinned by `tests/unit/test_orientation_oracle.py` (including a guard that the `M^T` convention is
observably wrong); see commit `f8b82bd` and `tests/unit/test_orientation_oracle.py` for the
pinned convention.

## `from_experiment` uses an all-free position mask (no special-position constraints)

`refinement_setup` builds the `ConstraintSpec.position_mask` as all-ones, so every atomic
coordinate is refined freely. Special-position atoms (e.g. quartz Si on `(x, 0, 1/3)`) have fewer
free degrees of freedom than 3; with an all-free mask they are over-parameterized and can drift off
their special positions under refinement, breaking the spacegroup symmetry. The faithful fix is the
diffpy-backed special-position / ADP constraint expansion behind the
`diffBloch.io.symmetry_setup` seam (currently a placeholder returning only counts), which is a
later constraints stage. Until then, treat refined special-position coordinates as unconstrained.
Seeded in `refinement_setup` (`src/diffBloch/preprocess/experiment.py`); see
`tests/unit/test_from_experiment.py`.
