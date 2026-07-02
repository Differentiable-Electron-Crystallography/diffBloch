# SCIENCE_FORK.md — RESOLVED: reflection-selection geometry (no fork)

**Status:** resolved. **Audience:** a crystallographer / electron-diffraction scientist.
**Outcome:** there is **no fork**. The apparent "reproduce the reference *or* be geometrically
correct" dilemma was based on an inverted diagnosis. The private reference filter is the
geometrically correct one for continuous-rotation 3D ED; the diffBloch 2.0 port had introduced the
error. The honest fix (use the distance from the goniometer rock axis) is *both* correct *and*
reproduces the reference — the second as a byproduct of the first.

This memo records the resolved reasoning; the earlier "fork" framing (preserved in git history) was
wrong. Live diagnostic narrative in `DEBUGGING.md`; the retracted divergence in `DIVERGENCE.md`; the
retracted false upstream bug report in `../diffBloch_private/KNOWN_ISSUES.md`; the lesson in
`LESSONS.md`.

---

## 1. Setup

α-quartz, 3D electron diffraction, **continuous rotation** (precession-free), PETS-processed. Per
crystal orientation the pipeline builds the dynamical (Bloch) scattering, propagates to a fixed
820 Å thickness, **integrates over the rocking curve** (the crystal rocks through a small angle
during each frame; `integration_semiangle = 1°`, 42 tilt samples), and compares to observed
integrated intensities via the scaling-optimised Bragg R-factor `R_obs`, over reflections that pass
a **selection filter** and have `I_obs > 3σ`.

Reference (private, CI): `R_obs = 0.043766`, `N_int_obs = 958`, `N_int_all = 1740`. All inputs
(structure CIF, PETS file `exp_data.cif_pets`, pre-optimised orientation list, fixed thickness) are
**byte-identical** (SHA-256) to the 2.0 fixtures. `exp_data.cif_pets` is **real experimental data**
(it contains negative background-subtracted intensities and realistic noisy sigmas — a simulator
would not); the unused `benchmark_sim.cif_pets` is a simulated counterpart from an early iteration.

## 2. The two geometries — beam vs. rock axis

Two axes matter and they are **different**:

- **Beam** along **−z** (`excitation_errors`: `K = [0,0,−K_mag]`), giving
  `S_g ≈ g_z − |g|²/(2·K_mag)` (the along-beam component is `g_z`).
- **Rock (goniometer) axis** along **x** (`rocking_curve_tilts` / the private
  `generate_integration_rotation_matrices` both build `R_x`; the private docstring: *"in the pets2
  coordinate frame, the goniometer axis is x"*).

`sg_max` is the excitation-error span a reflection **sweeps as the crystal rocks**. Under the rock
`R_x(φ)`, the along-beam component becomes `g_z'(φ) = sin(φ)·g_y + cos(φ)·g_z`, an oscillation of
amplitude

```
|(g_y, g_z)|  =  distance of the reflection from the rock axis x.
```

So `sg_max = |(g_y, g_z)|·θ`. This is the private's `norm(k[:, 1:])`. **It is correct.** A reflection
lying *on* the rock axis (`g_y = g_z = 0`) does not move as the crystal rotates about x, never
sweeps through the Ewald sphere, and correctly gets `sg_max → 0` (rejected). The in-plane anisotropy
is not a bug — a **single-axis** rock is anisotropic in-plane by construction.

`|(g_x, g_y)|` (distance from the **beam**) is the correct lever arm only for an **isotropic
precession cone about z** — a different experiment. The 2.0 port used `|(g_x, g_y)|` while its own
integrator rocked about x: internally inconsistent (select as if precessing, integrate by rocking).

## 3. Evidence

Exact reference recipe (private orientations, 820 Å, 42-tilt integration), toggling only the lever
arm:

| `sg_max` lever arm | mean `R_obs` | Σ `n_observed` | Σ beams |
|---|---|---|---|
| `|(g_x, g_y)|` — beam-transverse (was in 2.0, **wrong** here) | 0.337 | 1643 | 2914 |
| `|(g_y, g_z)|` — rock-axis distance (private, **correct**) | **0.0594** | **965** | **1875** |
| reference | 0.0438 | 958 | 1740 |

The reflections the wrong lever arm over-admits cluster **near the x rock axis** (median 20.6° from
it, vs 59.6° for the kept set) — precisely the non-sweeping, poorly-integrated reflections that
inflate R. Fixing the lever arm reproduces the reference reflection counts (965 ≈ 958) and lands
`R_obs = 0.0594`.

## 4. Resolution and consequences

- The fix (`preprocess/beams.py::klar_beam_mask`): choose the lever arm by
  `BeamSelection.geometry` — `|(g_y, g_z)|` for `continuous_rotation`, `|(g_x, g_y)|` for
  `precession`. This makes selection consistent with the integrator's own rock geometry.
- The private is **self-consistent and correct** (beam ∥ z, rock ∥ x, `sg_max` = distance from x).
  The `(g_y, g_z)` choice is deliberate — the private author documented the goniometer-axis frame —
  not a `[:, 1:]`/`[:, :2]` slip.
- Two caveats checked before retracting the upstream report: (1) the private uses the **amplitude**
  `|(g_y, g_z)|·θ`, not a first-order rate `|g_y|·θ`; the two coincide for near-Ewald reflections
  (`g_z ≈ 0`) and differ only for HOLZ, which are rejected anyway. (2) the private's filter frame
  has beam ∥ z and rock ∥ x (the `R_z(−rotation_axis_position=0.484°)` correction only aligns the
  measured goniometer azimuth to x), so `[:, 1:]` genuinely drops the rock-axis component.

## 5. Open (a genuinely smaller residual)

After the fix, the exact reference recipe gives `R_obs = 0.0594` vs reference `0.0438` — a small
residual, not a fork. Candidate sources, roughly ordered: (a) exact observed-reflection matching /
`I > 3σ` bookkeeping (we keep 965 vs 958); (b) mosaicity broadening (here a 1-frame moving average
≈ identity, likely negligible); (c) minor forward-model details (structure-factor conventions,
absorption); (d) the fit/evaluation consistency invariant — any orientation/thickness fitting must
score under the *same* integrated forward model used at evaluation (tracked in `DEBUGGING.md`).

## 6. Reproduction pointers

- Fix: `src/diffBloch/preprocess/beams.py::klar_beam_mask` (lever arm switched on
  `BeamSelection.geometry`); pinned by `tests/unit/test_select_beams.py`.
- Private (correct) filter: `../diffBloch_private/diffBloch/diffraction_dataset.py::filter_hkls`
  (`norm(k_avg_goni_pos[:, 1:], axis=1) * deg2rad(semiangle)`).
- Table (§3): evaluate `tests/fixtures/quartz_anchor/optim_orientation.csv` at 820 Å with 42-tilt
  integration; a one-line `[:, :2]` ↔ `[:, 1:]` swap reproduces both rows.
- Reference: `tests/fixtures/quartz_anchor/reference_results.json`.
