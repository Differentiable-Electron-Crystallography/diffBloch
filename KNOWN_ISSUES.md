# Known issues

Latent bugs and deferred fixes discovered during development, recorded here so they stay
discoverable instead of being buried in commit messages. Each entry gives a precise location, the
impact, and the intended fix. Close an entry by deleting it in the commit that fixes it.

## `converge_pool` cannot grow `g_max_refine` past the grid: dependent grid-resize is unimplemented

- **Location:** `preprocess/convergence.py`, `converge_pool` (the `build` closure's guard).
- **What:** the pool lever re-seeds beams from the shared `ScatteringGrid` at a wider
  `g_max_refine`. That stays inside the `Fgb` difference support only while
  `2 * g_max_refine <= grid.g_max`; a candidate beyond it needs the grid *itself* to grow (a
  dependent-sizing step: rebuild `ScatteringGrid.from_cell` at a larger `g_max`, re-tabulate `Fgb`).
  That resize is **not implemented** -- `converge_pool` raises a `ValueError` instead.
- **Impact:** none on the quartz anchor (`g_max_refine = 1.6`, `g_max = 4.5`, so the pool can nearly
  double before the bound bites) and low in general (convergence settles well before the pool
  approaches half the grid extent). It only bites a dataset whose converged pool would exceed
  `grid.g_max / 2`, which then fails loudly rather than silently truncating.
- **Intended fix:** when the guard trips, grow the grid `g_max` as a dependent sizing step before
  re-seeding (the partial-order dependency already named in `design/decisions/stage11-convergence.md`,
  "growing `g_max_refine` implies a grid-`g_max` sizing step before re-selection"), then continue the
  sweep. Deferred until a dataset needs it.
- **Found:** diffBloch 2.0 stage 11 (convergence, pool lever).

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

`RefinementSetup.from_structure` builds the `ConstraintSpec.refinable_position_mask` as all-ones, so every
atomic
coordinate is refined freely. Special-position atoms (e.g. quartz Si on `(x, 0, 1/3)`) have fewer
free degrees of freedom than 3; with an all-free mask they are over-parameterized and can drift off
their special positions under refinement, breaking the spacegroup symmetry. The faithful fix is the
diffpy-backed special-position / ADP constraint expansion behind the
`diffBloch.io.symmetry_setup` seam (currently a placeholder returning only counts), which is a
later constraints stage. Until then, treat refined special-position coordinates as unconstrained.
Seeded in `RefinementSetup.from_structure` (`src/diffBloch/preprocess/experiment.py`); see
`tests/unit/test_from_experiment.py`.

## `fit_orientation`'s `max_iterations` default is calibrated on quartz only

`fit_orientation` caps the Palatinus search at `max_iterations` passes per orientation and raises
`RuntimeError` rather than spin (matching `iterate_until`'s posture; see DIVERGENCE.md). The cap is
sound and exposed at the boundary (`OrientationFitConfig.max_iterations`, unpacked into the pure
function). Its default (**2000**) is **calibrated on the quartz anchor under the integrated recipe**:
the static from-scratch fit's slowest legitimate search converged in 526 passes, but with
rocking-curve integration (the C3 coupled recipe, now pinned) the landscape is bumpier and the
slowest rotation needs **1288** passes -- every rotation still terminates by the radius floor (this
is not a runaway; monotone descent guarantees termination), it simply takes more passes. Four
rotations exceed 600 under integration (1288, 764, 715, 699), which is why the earlier static-
calibrated 600 spuriously aborted the full-99 integrated run. 2000 leaves cross-platform headroom
over 1288 while still catching a genuine runaway.

The calibration is single-dataset: `diffBloch_private`'s search has no cap at all, so there is no
ported precedent, and a dataset with shallower minima than quartz could still need a larger cap. It
is overridable via `preprocess.orientation.max_iterations`. Where: `config/schema.py`
(`OrientationFitConfig.max_iterations`), `specs.py` (`HexagonalSearch.max_iterations`),
`preprocess/fit_orientation.py`.

## The convergence driver re-simulates the current Plan every step

`converge_scalar` calls `measure(current, candidate)` each iteration, and `simulation_rfactor`
simulates *both* Plans inside `measure` -- so the `current` Plan is re-simulated on every step even
though its simulation is unchanged until `current` advances. That doubles the simulation cost of the
sweep (each step pays for two solves instead of one). The fix is to thread the `current` solution
through the loop and only re-solve the `candidate`, e.g. a measure that takes a precomputed
previous-solution -- deferred as a hot-path optimisation, not a correctness issue (the sweep is a
one-off preprocess step, not in the refinement inner loop). Where: `preprocess/convergence.py`
(`converge_scalar` / `simulation_rfactor`).

## The train/validation split is constructed but unconsumed (and over-documented)

`from_experiment` builds a `train` / `validation` `Plan` pair (`PlanSplit`), and `PlanSplit` is
documented as if the split were live ("`refine` fits on `train` while scoring the held-out
`validation`"). Nothing downstream actually distinguishes them: both `run_inference` and the engine
take a single `Plan`, and whole-experiment work uses `PlanSplit.combined` (all rotations). Worse, a
whole-*rotation* holdout is a weak cross-validation guard for over-determined physics refinement --
the principled analog holds out *reflections* (R_free, Brünger 1992), not orientations, and the
private ran this experiment with no holdout (`val_prop = 1.0`). The split machinery is kept dormant
because it earns its keep for the future learned modes (`ThicknessNN`), but until refinement wiring
lands it should either be wired to a real use (learned modes, and/or an R_free-style reflection
holdout for physics refinement) or stop advertising an unimplemented behaviour. The `PlanSplit`
docstring has been corrected to describe reality; the deeper wiring decision is deferred. Crosses
the config / preprocess / engine layers. Where: `preprocess/experiment.py` (`PlanSplit`,
`from_experiment`), `config/schema.py` (`DataSplitConfig`). See
`design/decisions/train-validation-split.md`.

## Rocking-curve integration tilt-batching — LANDED (2026-07); large-N memory seam noted

- **Location:** `engine/forward.py` (`RefinementEngine._solve`); batched assembly in
  `core/dynamical/assembly.py` (`stack_beam_plans` / `build_bloch_systems`) and the rank-polymorphic
  `core/solver.py::propagate`.
- **Resolved:** each rotation's N rocking-curve tilts (which share the beam set, differing only in
  the per-tilt geometry `mii`/`diagonal`) are now solved as **one batched `eigh`/`matrix_exp` over
  `(B, N, N)`** with a single shared F-gather, replacing the Python per-tilt loop. Numerically
  identical: `matrix_exp` bit-for-bit, `eigh` to machine precision (pinned by
  `tests/unit/test_core_solver.py::test_batched_propagate_matches_the_per_tilt_loop` and
  `test_core_products.py::test_integrate_batched_matches_the_per_tilt_integrate`); the untilted
  (`B=1`) path is byte-identical (anchor unchanged).
- **Measured win:** ~3× on a full 99-rotation integrated eval (917→302 ms `bloch_eigen`; 1247→400 ms
  `matrix_exp`, quartz anchor `sampling=42`). Modest rather than the naive ~40× because the selected
  beam set is small (N=13–31), so per-matrix compute was already cheap and the win is fewer
  dispatches + one gather. (This was **not** the fit bottleneck -- the redundant gather rebuild
  below was; batching contributes on pure-eval paths.)
- **Deferred (large-N memory seam):** the batched `matrix_exp` materialises a `(B, T, N, N)` complex
  transfer buffer; at anchor scale this is trivial (<0.07 GB even with the 100-step thickness grid),
  but a future large-N system (hundreds of beams) could make it painful. Fix then would be internal
  thickness-chunking inside `propagate` (an implementation detail, no API change) or a value-typed
  strategy (never a boolean toggle). YAGNI now; recorded so the seam is discoverable.

## Redundant structure-factor gather rebuild — RESOLVED (2026-07); the real preprocess hog

- **Location (was):** `core/dynamical/assembly.py::build_beam_plan`, which rebuilt the entire
  `StructureFactorGather` (index-map ravel + `np.unique` + `np.isin` coverage validation) on every
  call — once per rocking-curve tilt and once per orientation-search trial.
- **What it was:** the gather (F index topology) is a pure function of `(grid_hkl, beam_hkl, gpts)`,
  identical across all N tilts of a `build` *and* every trial of a `fit_orientation` search (the beam
  set is fixed once `select_beams` has run). Profiling one rotation's integrated fit showed
  `build_structure_factor_gather` called **3403 times = 94% of the fit** (the batched `matrix_exp`
  was ~2%) — i.e. the tilt-batching perf pass optimised the wrong thing; the bottleneck was this
  rebuild, not the `eigh`. Lesson recorded: **profile before optimising** (a 30 s profile would have
  pointed here first).
- **Fix:** thread a precomputed gather through the build chain (`build_beam_plan(gather=...)`,
  `OrientationPlan.build(gather=...)`); `OrientationPlan.build` builds it once and shares it across
  the tilts, and `fit_orientation` / `integrate_rocking_curve` pass the seed plan's
  `op.beam_plans[0].gather` on every rebuild. 3403 -> 1 gather build per rotation fit. Byte-identical
  (reuse == recompute; pinned by
  `tests/unit/test_core_dynamical.py::test_build_beam_plan_reuses_a_precomputed_gather_byte_identically`),
  a cheap shape guard rejects a mismatched gather. Ports the private's precomputed HKL->index map
  (`diffBloch_private` 6bb3031).
- **Measured:** the coupled integrated fit dropped **14 -> 0.78 s/rotation (~18×); full-99 ~1.3 min**
  (was ~23 min), R_obs unchanged (0.0519 on the first 6 rotations). The static anchor e2e also fell
  76 -> 28 s. This is what actually unblocked the integrated anchor pin (1c), now landed as
  `tests/e2e/test_anchor.py::test_quartz_integrated_anchor` (full-99 mean R_obs 0.0655; opt-in
  subset via `DIFFBLOCH_ANCHOR_ROTATIONS`).
