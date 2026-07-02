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

### `select_beams` `sg_max` lever arm: retracted divergence (2.0 was wrong, the private is correct)

- **Status: RETRACTED.** An earlier 2.0 port diverged here in error and this entry recorded that
  divergence; the divergence has been removed and 2.0 now matches `diffBloch_private`. Kept as a
  record because a false upstream bug report was filed (and has since been retracted from
  `../diffBloch_private/KNOWN_ISSUES.md`).
- **What happened:** the Klar et al. (2023) rsg/dsg filter keeps a reflection when its excitation
  error `|Sg|` is small relative to `sg_max = |g_lever| * deg2rad(semiangle)`, the excitation-error
  span the reflection *sweeps during integration*. The private computes
  `sg_max = norm(k[:, 1:]) = norm(g_y, g_z)`. The 2.0 port "corrected" this to `norm(g_x, g_y)`,
  reasoning that with the beam along `-z` the transverse plane is `(x, y)`.
- **Why that was wrong:** the integration is a **single-axis continuous rotation about the
  goniometer `x` axis** (`rocking_curve_tilts` builds `R_x`; the private's own docstring: "in the
  pets2 coordinate frame, the goniometer axis is x"), *not* an isotropic precession cone about the
  beam. Under `R_x(phi)` the along-beam component becomes `g_z' = sin(phi) g_y + cos(phi) g_z`, whose
  excursion amplitude is `norm(g_y, g_z)` -- the distance from the **rock axis**. So `norm(g_y, g_z)`
  is the geometrically correct lever arm; a reflection on the rock axis (`g_y = g_z = 0`) never
  sweeps and is correctly dropped. `norm(g_x, g_y)` (distance from the beam) is the lever arm only
  for precession -- a different experiment. The "in-plane anisotropy" the old entry called a bug is
  the correct behaviour of a single-axis rock.
- **Evidence:** with the wrong `(g_x, g_y)` lever arm the anchor admitted ~1.7x too many reflections
  (1643 vs 958) -- the extras cluster near the `x` rock axis (median 20.6 deg from it), barely sweep,
  and inflate `R_obs` to 0.337. Restoring `(g_y, g_z)` reproduces the reference reflection counts
  (965 vs `N_int_obs` 958) and `R_obs = 0.0594` (reference 0.0438).
- **2.0 behaviour now:** `preprocess.klar_beam_mask` selects the lever arm by `BeamSelection.geometry`
  -- `(g_y, g_z)` for `continuous_rotation` (matching the private), `(g_x, g_y)` for `precession`.
- **Where:** `preprocess/beams.py`; pinned by `tests/unit/test_select_beams.py`
  (`..._drops_on_axis` keeps a y-offset beam and drops an on-rock-axis one;
  `..._precession_uses_beam_transverse` covers the precession lever arm). Full narrative in
  `SCIENCE_FORK.md` and `DEBUGGING.md`; lesson in `LESSONS.md`.
- **Found / corrected:** diffBloch 2.0 stage 11 (preprocess) port.

### `rbragg_abs()` applies the `I > 3*sigma` cut by multiplication, poisoning the sum with `NaN`

- **Original:** `diffBloch/metrics.py`, `rbragg_abs()` (~line 200). The Bragg R(obs) factor takes
  `sqrt(I_obs)` / `sqrt(I_calc)` over *all* reflections, then applies the `I_obs > 3*sigma`
  significance cut by multiplying a 0/1 mask: `sum(|sqrt(I_obs) - sqrt(I_calc)| * mask)`. But
  experimental intensities can be **negative** (background-subtracted), and a negative reflection is
  always below the `3*sigma` cut, so `sqrt(negative) = NaN` at a masked-*out* reflection and
  `NaN * 0 = NaN` poisons the reduced sum -- the R-factor is `NaN` for any rotation with a single
  negative observed intensity.
- **Effect:** latent in the private (its upstream data path never presented a negative to this
  function). In 2.0 the observed patterns include negatives (real PETS data: e.g. the quartz
  anchor's first rotation has 6), so the multiply-mask returns `NaN` and the rotation silently drops
  out of the aggregate `R_obs`.
- **2.0 behaviour:** `core.losses.rbragg` applies the cut by *selection* (`torch.where`), with
  `clamp(min=0)` guarding the square roots against numerical noise. Masked-in reflections have
  `I_obs > 3*sigma > 0` and `I_calc = |psi|^2 >= 0`, so the result is identical to the original on
  all-positive data and merely finite (excluding them) where the original was `NaN`.
- **Where:** `core/losses.py` (`rbragg`); pinned by `tests/unit/test_core_losses.py`
  (`test_rbragg_is_nan_safe_for_negative_masked_reflections` -- a negative masked reflection leaves
  the result finite and equal to the observed-subset R). `test_rbragg_matches_private` still holds
  on clean data.
- **Found:** diffBloch 2.0 stage 11 (preprocess) port, wiring the executable quartz anchor.

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
- **Caveat:** the default cap (600) is calibrated on quartz only -- the private has no precedent,
  and the quartz anchor's slowest legitimate search needs 526 passes, so 600 has headroom but a
  shallower-minima dataset could need more (KNOWN_ISSUES.md; overridable via config).
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
