# Divergences from `diffBloch_private`

The diffBloch 2.0 rewrite is a clean-room port of `diffBloch_private`. Most of it reproduces the
original behaviour faithfully (pinned against extracted oracles and textbook values). This file
records the places where the port **intentionally differs** -- principally bugs in the original that
we corrected rather than replicated -- so the *where / how / why* of each difference stays
discoverable instead of living only in commit messages and the private repo's notes.

Scope, vs the neighbouring docs:

- **`DIVERGENCE.md`** (this file) -- intentional, behaviour-changing differences from the original.
- **`KNOWN_ISSUES.md`** -- open latent bugs / deferred fixes in *our* code.
- **`REFERENCES.md`** -- the literature each algorithm is pinned to.

Each entry is self-contained: the original location and defect, the corrected behaviour, where it
lives in this codebase, and the test that pins it. The defects are also recorded at source in
`diffBloch_private/KNOWN_ISSUES.md`, with a suggested fix for that codebase.

---

## Corrected bugs (we do **not** reproduce the original)

### `Atoms.A_matrix()` mislabels `|a*|` as `c*` (anisotropic-cell ADP frame)

- **Original:** `diffBloch/atoms.py`, `A_matrix()` (~line 976). The Trueblood et al. (1996) eq. 50
  orthogonalization matrix should set `A[2,2] = 1/c*` with `c* = |a x b| / V`, but the code computes
  `c_star = norm(cross(unit_cell[2], unit_cell[1]) / V) = cross(c, b) = |a*|` -- so the variable
  named `c_star` actually holds `|a*|`.
- **Effect:** for an orthorhombic `diag(a, b, c)` cell, `A = diag(a, b, a)` instead of
  `diag(a, b, c)` -- wrong whenever `a != c` (e.g. quartz). Blast radius is narrow: `A` is only used
  in the **Uiso** ADP branch (`U* = A^-1 (Uiso I) A^-T`); the **Uani** branch cancels `A`
  algebraically, so only isotropic-ADP atoms on anisotropic cells are perturbed.
- **2.0 behaviour:** the rewrite drops the reconstructed `A` and uses the convention-correct
  reciprocal metric directly -- `U* = Uiso G*` with `G* = B B^T`, `B = reciprocal_cell`, and
  `U*_ij = d*_i d*_j U_cif_ij` for Uani.
- **Where:** `core/adp.py` (`cartesian_adp_to_star`); see `REFERENCES.md` (ADP frame conventions).
- **Found:** diffBloch 2.0 stage 10 (params/engine) port.

### `filter_hkls()` transverse component mixes the along-beam axis into `sg_max`

- **Original:** `diffBloch/diffraction_dataset.py`, `filter_hkls()` (~line 345). The Klar et al.
  (2023) rsg/dsg beam filter uses `sg_max = |k_perp| * deg2rad(semiangle)`, where `k_perp` is the
  component *perpendicular to the beam*. The code computes `norm(k[:, 1:]) = norm(g_y, g_z)`, but
  `excitation_errors` fixes the beam along `-z` (`K = [0, 0, -Kmag]`), so the perpendicular plane is
  `(x, y)` -- the transverse component should be `norm(g_x, g_y) = k[:, :2]`. The `(y, z)` form folds
  the along-beam component `g_z` into the "transverse" distance.
- **Effect:** `sg_max` is inflated for reflections with large `g_z` (HOLZ), loosening the filter
  exactly where it should tighten; ZOLZ reflections (`g_z ~ 0`) are barely affected. The `-z` beam
  convention is corroborated by the dynamical simulation reaching `R_obs = 0.0438`.
- **2.0 behaviour:** `preprocess.select_beams` / `klar_beam_mask` use the consistent transverse
  component `(g_x, g_y)` and do not reproduce the `(y, z)` quantity.
- **Where:** `preprocess/beams.py`; pinned by `tests/unit/test_select_beams.py`
  (`test_klar_mask_keeps_near_ewald_in_plane_drops_on_axis` -- keeps an x-offset beam, drops a
  z-offset one, the exact swap of the original convention). See `REFERENCES.md` (Klar 2023 rsg/dsg).
- **Found:** diffBloch 2.0 stage 11 (preprocess) port.

---

## Deliberate simplifications (we narrow, with justification)

### `fit_orientation` holds the active beam set fixed across the search

- **Original:** `diffBloch/programs/preprocess.py`, `palatinus_modified_simplex()` /
  `orientation_optim()`. Every trial orientation re-runs `results.filter_hkls(...)`, so the active
  beam set -- and hence the set of reflections the wR2 is computed over -- is re-derived for each
  tilt inside the search loop.
- **2.0 behaviour:** `fit_orientation` selects beams once (at the seed orientation, via the separate
  `select_beams` step) and holds that set fixed across the whole hexagonal search. Each trial still
  recompiles the *dynamics* (`Sg` / structure matrix) for its tilted orientation via
  `OrientationPlan.build`; only beam *membership* is frozen.
- **Justification:** the search is sub-degree (`max_search_angle = 0.4`, shrinking to
  `min_search_angle = 0.001`), over which the Klar rsg/dsg membership is effectively unchanged, so
  re-filtering per trial is near-identical at much higher cost. Freezing it also keeps each
  preprocess step single-responsibility (`fit_orientation` tilts + scores, `select_beams` selects --
  recompose `select_beams` afterwards to re-prune) and keeps the wR2 objective over a *fixed*
  reflection domain across trials, so trial scores are strictly comparable.
- **Where:** `preprocess/fit_orientation.py` (`_refine_one`, the fixed `beam_hkl`); exercised by
  `tests/unit/test_fit_orientation.py`. Decision context in
  `design/decisions/stage11-fit-orientation.md`.
- **Found:** diffBloch 2.0 stage 11 (5b) -- `fit_orientation`.

### Convergence drops the grid-`g_max` knob and treats the beam knobs as coupled levers

- **Original:** `convergence_testing.py` sweeps three independent knobs (`g_max`, `sg_max`,
  `tilt_steps`) as separate `optimize_*` passes; `g_max` grows the structure-factor grid that *is*
  the beam source (`filter_hkls` draws beams from it).
- **2.0 behaviour:** 2.0 splits the `Fgb` support grid from the active beam set. The Bloch matrix is
  a gather `A[i, j] = F(g_j - g_i)` over active beam pairs, so the grid only has to cover the beam
  differences (~2x the beam `g_max`); growing grid `g_max` beyond that changes nothing. There is
  therefore **no `converge_g_max`** -- grid extent is *sized-to-cover*, not converged. The private's
  `g_max` and `sg_max` instead map onto two coupled levers of one quantity, beam-set inclusiveness:
  `g_max -> g_max_refine` (seed pool) and `sg_max -> integration_semiangle` (the Klar excitation
  window), whose intersection is the active set. `tilt_steps` (rocking-curve sampling) is a separate
  axis, deferred until the forward model integrates rocking curves.
- **Justification:** the gather makes grid extent a sizing constraint, not an accuracy knob;
  consolidating the two beam knobs under one concern reflects their actual coupling (each bounded by
  the other) and follows "decompose by coupled home, not false independence".
- **Where:** `preprocess/convergence.py`; `design/decisions/stage11-convergence.md`.
- **Found:** diffBloch 2.0 stage 11 -- implementing `converge_beams`.

## Deliberate generalizations (we extend, not correct, the original)

### `fit_orientation` enforces an iteration cap; the private search has none

- **Original:** `diffBloch/programs/preprocess.py`, `palatinus_modified_simplex()`. A bare
  `while search_angle > min_search_angle:` with no iteration counter -- termination rests entirely
  on monotone wR2 descent plus the halving radius reaching the floor.
- **2.0 behaviour:** `fit_orientation` caps the total passes per orientation at `max_iterations`
  and raises `RuntimeError` if it is reached, matching this package's `iterate_until` posture that
  silent non-convergence is never returned. The search still terminates by construction on a
  non-degenerate objective; the cap only guards pathological ridge-walking on (near-)degenerate
  landscapes (e.g. the trivial high-symmetry synthetic system, which walks 900+ passes toward a
  symmetry-equivalent minimum).
- **Caveat:** the default cap is uncalibrated -- there is no private precedent and no real-data
  statistics yet (KNOWN_ISSUES.md tracks moving it into `OrientationFitConfig` and tuning it).
- **Where:** `preprocess/fit_orientation.py` (`fit_orientation` / `_refine_one`); exercised by
  `tests/unit/test_fit_orientation.py`.
- **Found:** diffBloch 2.0 stage 11 (5b) -- `fit_orientation`.

### Orientation scoring reduces over thickness by the best-fitting value, not the first

- **Original:** `diffBloch/programs/preprocess.py`, `orientation_optim()`. When scoring an
  orientation it simulates over the candidate thicknesses and then uses
  `combined_filtered_intensities[0]` -- the **first** thickness -- to compute the wR2 the simplex
  minimizes.
- **2.0 behaviour:** `RefinementEngine.score_orientation` computes the scaling-optimized wR2 per
  thickness and returns the **minimum** (the best-fitting thickness's score). Thickness is a nuisance
  when scoring orientation, so "orientation's best achievable fit over the candidate thicknesses" is
  the more faithful objective than "the fit at whichever thickness happens to be first".
- **Equivalence:** identical to the original whenever a single thickness is in play -- which is the
  preprocess case the original ran (one current thickness per rotation). The two differ only when
  more than one thickness is scored at once, where the original's `[0]` is order-dependent.
- **Where:** `engine/forward.py` (`RefinementEngine.score_orientation`); exercised by
  `tests/unit/test_scoring.py`. Decision context in
  `design/decisions/stage11-fit-orientation.md`.
- **Found:** diffBloch 2.0 stage 11 (5a) -- the wR2-scoring seam.

### Convergence revisits knobs to a fixpoint, not the private's fixed two order-varying passes

- **Original:** `diffBloch/programs/convergence_testing.py`, `_run_hyperparams_optimization` /
  `run_pass`. The hyperparameter suite runs a hard-coded `num_passes = 2`, and *changes the sweep
  order between passes* (pass 1: `g_max`, `tilt_steps`, `sg_max`; pass 2: `tilt_steps`, `g_max`,
  `sg_max`). The revisit count and the per-pass order are empirical -- no stated principle.
- **2.0 behaviour:** the suite is one ordered `pipeline` (`converge_g_max`, `converge_sampling`,
  `converge_beams` -- respecting the hard grid-before-beams partial order) driven by `iterate_until`
  to a genuine cross-knob fixpoint: the pass repeats until a whole pass leaves every knob unchanged,
  or the `ConvergenceTolerance` cap raises (silent non-convergence is never returned, matching the
  `iterate_until` posture). This **generalises** the private's fixed 2 passes (a fixpoint subsumes
  any sufficient fixed count) and drops the unprincipled per-pass order-swap.
- **Equivalence:** when two ordered passes already reach the fixpoint, 2.0 stops after detecting it
  rather than running a hard-coded second pass; results coincide whenever the private's 2 passes
  were themselves converged.
- **Where:** `preprocess/` convergence steps + `iterate_until` (`preprocess/pipeline.py`). Decision
  context in `design/decisions/stage11-convergence.md`.
- **Found:** diffBloch 2.0 stage 11 -- convergence testing.
