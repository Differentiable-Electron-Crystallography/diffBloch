# Debugging log

Active, in-flight debugging investigations. When an investigation closes, distil the outcome
into the right home (`DIVERGENCE.md`, `KNOWN_ISSUES.md`, `LESSONS.md`, a decision doc, or a test)
and trim the entry here to a one-line pointer.

---

## R_obs forward-model gap on the quartz anchor (open)

**Symptom.** The quartz anchor reproduces the reference `R_obs = 0.043766` only to ~0.34, not
~0.044. The gap survives every faithful ingredient we can control.

**Reference recipe (from `../diffBloch_private/tests/e2e/inference/quartz/config.yml`).** The
private e2e that produced `reference_results.json` runs `evaluate_over_rotations` with:
pre-optimised orientations from `optim_orientation.csv`; fixed thickness **820 Å**
(`optim_thicknesses_path: null`, `thicknesses: [820]`, single value, no grid); integrated
intensities on (`integration_semiangle: 1`, `rocking_curve_sampling: 42`, continuous rotation);
mosaicity on; `g_max_refine: 1.6`, `rsg: 0.9`, `dsg: 0.0015`, `sg_max: 0.01`.

**Inputs verified byte-identical** (SHA-256) between the private e2e `cifs/` and
`tests/fixtures/quartz_anchor/`:
- `enantiomer_1.cif` `4a1ed50a…` (== private `ref_structure.cif`)
- `exp_data.cif_pets` `96c730df…` (config points here; `benchmark_sim.cif_pets` is unused)
- `optim_orientation.csv` `70c7290f…`

So the gap is **not** an input mismatch.

**What has been ruled out (measured, optim orientations through our forward model):**

| variant | mean R_obs |
|---|---|
| optim orientations, fixed 820, **static** (no rocking) | 0.720 |
| optim orientations, fixed 820, **integrated** (42 tilts) — *the exact reference recipe* | **0.337** |
| optim orientations + `fit_thickness` grid, static | 0.285 |
| optim orientations + `fit_thickness` grid, integrated | 0.258 |
| our *fitted* orientations (full 99), static | 0.298 |
| our fitted orientations, integrated | 0.294 |
| **reference** | **0.044** |

Established facts:
- Rocking-curve geometry matches the private exactly (`R_x` tilts, `linspace(-1, 1, 42)` deg,
  `R2 @ orientation`; `rc_width` is vestigial/unused in the private; incoherent sum of `|psi|^2`).
- Mosaicity is a **1-frame moving average** for quartz
  (`round(0.05 / (2/42)) = round(1.05) = 1`) → effectively identity. Not the gap.
- Thickness is fixed **820 Å** in both; `fit_thickness` is *not* the reference recipe and does not
  close the gap.
- The optim orientations are legitimately the converged ones (raw `max|seed - opt| ≈ 0.10`, a
  few-degrees refinement). They are **integrated-optimal**: great when integrated (0.337) but
  terrible under a static solve (0.720). (An earlier orientation-vs-seed angular-distance check
  read 0.0 for all 99 — a *false negative*: the seed "orientation matrices" carry the cell metric
  and are not orthonormal, so the `arccos((tr(R)-1)/2)` rotation formula is invalid on them. Use
  raw element differences, not the rotation-angle formula, on these matrices.)

**Coupling invariant discovered (necessary, not sufficient).** The rocking curve is a property of
the *forward model*, not a terminal add-on: whatever physics the terminal integrates over, every
earlier fit must score under the *same* physics. Fitting on the static single-solve while
evaluating integrated optimises for one model and grades on another (0.72 static vs 0.34 integrated
for the same orientations). Consequence: decision #1b's "integrate the rocking curve *last*, after
the fits" is **wrong** — the tilt geometry must be present *during* the fits. In our architecture
this needs no engine change: `fit_orientation` must propagate `op.tilts` into each trial rebuild
(today `_refine_one` drops them, reverting each trial to static); `fit_thickness`
(`replace(op, thickness=…)`) and `refine`/`inference` already integrate via `_solve` over
`beam_plans`. **Deferred** until the forward-model gap below is understood — coupling the fit to a
forward model that is itself 6x off would just converge to the wrong optimum (~42x slower).

**ROOT CAUSE FOUND AND FIXED — the Klar-filter `sg_max` lever arm (our bug, not the private's).**
With the exact reference recipe we integrated **~1.7x more reflections** than the reference (1643 vs
`N_int_obs` 958). The single knob responsible is the transverse lever arm in `sg_max`
(`preprocess/beams.py::klar_beam_mask`):

| `sg_max` lever arm | mean R_obs | Σ n_observed | Σ n_beams |
|---|---|---|---|
| `|(g_x, g_y)|` = `[:, :2]` — was in 2.0 (wrong here) | 0.337 | 1643 | 2914 |
| `|(g_y, g_z)|` = `[:, 1:]` — private, correct | **0.0594** | **965** | **1875** |
| reference | 0.044 | 958 | 1740 |

**The diagnosis was initially inverted; corrected here.** `sg_max` is the excitation-error span a
reflection sweeps *during the actual integration*, which is a **single-axis continuous rotation
about the goniometer `x` axis** (`rocking_curve_tilts` builds `R_x`; private docstring: "in the
pets2 coordinate frame, the goniometer axis is x"). Under `R_x(phi)` the along-beam component is
`g_z'(phi) = sin(phi) g_y + cos(phi) g_z`, excursion amplitude `|(g_y, g_z)|` = the distance from
the rock axis. So `|(g_y, g_z)|` (the private's `norm(k[:, 1:])`) is **correct**: a reflection on
the rock axis never sweeps and is rightly dropped. `|(g_x, g_y)|` (distance from the `-z` beam) is
the lever arm only for **precession** — a different experiment. The 2.0 port selected as if
precessing while integrating by rocking about x: internally inconsistent. Confirmed: the
over-admitted reflections cluster near the x rock axis (median 20.6 deg from it, vs 59.6 deg for the
kept set). `exp_data.cif_pets` is **real** data (negative intensities, noisy sigmas), so the
correct filter genuinely selects the better-measured reflections.

**Fix applied.** `klar_beam_mask` now selects the lever arm by `BeamSelection.geometry`:
`(g_y, g_z)` for `continuous_rotation` (matching the private), `(g_x, g_y)` for `precession`. The
false divergence (`DIVERGENCE.md`) and the false upstream bug report
(`../diffBloch_private/KNOWN_ISSUES.md`) are retracted. Full resolved reasoning in `SCIENCE_FORK.md`;
lesson in `LESSONS.md`. Two caveats checked: the private uses the amplitude form (coincides with the
first-order rate for near-Ewald reflections); the private's filter frame has beam ∥ z and rock ∥ x.

**Residual after the fix: 0.0594 vs reference 0.0438** (optim orientations + fixed 820 + integrated,
*no* fit-coupling yet). Candidates for the last bit: exact obs matching / `I > 3σ` bookkeeping (965 vs
958), mosaicity (1-frame, tiny), minor forward-model details, and the fit-coupling invariant above.

**Superseded lead.** Earlier hypothesis (private takes a per-tilt Klar *union* via
`collect_unique_hkls`, so we were *missing* reflections) was wrong on direction: `filter_hkls`
computes the mask **once at the untilted/avg goniometer position**, and we had *more* reflections,
not fewer.
