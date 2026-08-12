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

## Rot 61: a mosaicity-reduction effect, not a corner beam

Rot 61 was once treated as a "corner-beam outlier" (its lone `(4,0,5)` reflection diverged ~1.6×).
That was a *tilt-reduction* artifact, not a solve difference: its per-tilt `|psi(4,0,5)|^2` is
byte-identical to the reference, and the reference's five-sample mosaicity moving-average
down-weights that sharp peak (it sits near the boundary of its coupled-tilt range, so the
valid smoothing counts 1..4 instead of 5). The replay reassembles each reflection's full
rocking curve across segments and applies the same five-sample smoothing, so rot 61 reproduces the
reference like the others — no cap or corner-beam work was needed (that hypothesis was falsified;
`(4,0,5)` has mid-range `|g|=1.318`).
