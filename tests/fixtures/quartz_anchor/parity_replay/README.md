# Coupling-policy parity replay inputs

Captured forward-model inputs for a small subset of quartz rotations, used by
`tests/e2e/test_coupling_parity.py` to prove our Bloch solver reproduces a reference model given
its exact dynamical coupling — a forward-solver parity check that complements the from-scratch
accuracy anchors in `test_anchor.py`.

## What each file is

- `rot_{13,27,60,61,64}.npz` — one rotation each. Keys used by the replay:
  - `orientation` (3×3), `thickness` (1,), `u0` () — the per-rotation solve inputs
    (`u0 = |Fgb(000)|·prefactor`, the mean-inner-potential correction).
  - `n_segments` (), `seg{k}_hkl` (n_k×3 int), `seg{k}_cover` (m_k int) for k in 0..n_segments-1 —
    the per-tilt-segment **union coupling sets** and the tilt indices each segment integrates
    over. This is the `union_splits=12`, non-adaptive tilt-union coupling policy (cap
    `|g| < g_max/2 - 0.2 = 2.05`, `sg_max = 0.01`).
  - `hkl_matched` (M×3 int), `exp_ints` (M,), `sigmas` (M,) — the matched experimental reflections
    (from the PETS data) the R-factor is computed over.
  - `sim_matched_unscaled` (M,) — the reference per-reflection simulated intensities at these
    inputs (a finer per-reflection cross-check; not required for the R-factor assertion).
- `tilts.npz` — `tilts` (42×3×3): the shared rocking-curve tilt-orientation matrices; each
  segment integrates over `tilts[seg{k}_cover]`.

The **parity target** is not vendored here: it is the already-present, hash-verified
`reference_results.json` (`R_obs` per `rotation_idx`), which matches these inputs' reference
R-factors to ~1e-6.

## Legacy mosaicity reference

This replay records the former five-sample moving-average calculation. Current plans instead use
the apparent mosaicity angle from `.cif_pets` and normalized Gaussian orientation sampling, so this
legacy result is not a current mosaicity oracle.
