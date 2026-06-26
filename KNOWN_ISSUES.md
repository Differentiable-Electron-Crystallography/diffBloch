# Known issues

Latent bugs and deferred fixes discovered during development, recorded here so they stay
discoverable instead of being buried in commit messages. Each entry gives a precise location, the
impact, and the intended fix. Close an entry by deleting it in the commit that fixes it.

## `core.solver.propagate` is not device-safe (BlochSystem geometry fields)

- **Found:** 2026-06, stage 10 (review of engine slice `5db3ee1` / fix `53e3ed5`).
- **Location:** `src/diffBloch/core/solver.py` `_propagate_matrix_exp` / `_propagate_bloch_eigen`,
  fed by `src/diffBloch/core/dynamical/assembly.py::build_bloch_system`.
- **What:** `build_bloch_system` carries `plan.psi0`, `plan.k_n`, `plan.mii`, `plan.mask` straight
  from the geometry `BeamPlan` (built CPU-side). `propagate` only *dtype*-casts them
  (`system.psi0.to(a.dtype)`, `thicknesses / system.k_n`, `mii.to(a.dtype)`) and never moves them to
  `a.device`. `a` itself is on the active (Fgb/params) device.
- **Effect:** with params on CUDA/MPS, `thicknesses / system.k_n` and `transfer @ system.psi0` mix a
  device tensor with CPU tensors and fail. The engine-level fix (`53e3ed5`) co-locates the engine's
  own invariants but cannot reach these BlochSystem fields, so an accelerator run still breaks one
  layer down.
- **Fix:** in `propagate` (the hot-path consumer, per the repo's "consumers `.to(device)` at the use
  site" convention), co-locate `psi0` / `k_n` / `mii` / `mask` onto `a.device` before use. Add a
  `test_core_solver` case asserting the exit wavefunction lands on the operator's device (CPU
  co-location assertion documents the contract; meaningful on accelerators).
- **Status:** open; deliberately not folded into the engine review-fix (rule 3 — avoid a
  solver-layer change inside an engine commit). Wants its own focused commit + test.
