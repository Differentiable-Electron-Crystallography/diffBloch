# Decision: rocking-curve integration — tilts as sub-orientations, mosaicity as a composable knob

**Status:** accepted (stage 11, pre-implementation).
**Context:** 2.0's forward model point-samples one crystal orientation per rotation. A
rotation-electron-diffraction frame instead records each reflection's intensity *integrated* as it
sweeps through the Ewald sphere during the exposure. A point sample of a rapidly-varying rocking
curve cannot match an integrated measurement — a spike of the quartz anchor gave `R_obs ≈ 0.6`/NaN
against the private reference `0.0438`, dominated by this missing integration (the other gaps being
unfit orientations and a `0/0` NaN; see `ROADMAP.md` "The executable e2e anchor"). This records the
faithful 2.0 shape of rocking-curve integration, the last physics gap before the anchor's
`R_obs` pin can go green.
**Reference:** `diffBloch_private` — `rotation_dataset.generate_integration_rotation_matrices` /
`generate_precession_rotations`, `dynamical.BlochNet.forward(tilts=…)`,
`diffraction_dataset.DiffractionDataset.get_integrated_intensities`,
`tests/e2e/inference/quartz/config.yml` (`rocking_curve_sampling: 42`,
`integration_semiangle: 1`, `mosaicity: true`, continuous-rotation geometry). The private is the
authoritative source for the algorithm; this records the faithful 2.0 port.

## What the rocking curve does (physics)

During a single virtual frame the crystal is not at one fixed orientation: the goniometer sweeps,
the
beam converges, and the crystal mosaic spreads. The measured intensity of a reflection is therefore
the *integral* of its dynamical intensity over that small angular range as the reflection passes
through the Ewald sphere — its rocking curve. Near the Bragg condition intensity changes very fast
with tilt, so a single static Bloch solve is a poor estimate of the integrated measurement.
Integrating over a spread of tilts recovers it.

Private mechanics, in order:

1. **Generate tilts.** `angles = linspace(−semiangle, +semiangle, sampling)` (quartz: `−1°…+1°`, 42
   samples); each angle → a rotation about **x** (the goniometer axis in the PETS frame) for
   continuous-rotation geometry, or a precession cone for precession geometry.
2. **Tilt the nominal orientation.** `orientation_matrices = [R_tilt @ orientation for R_tilt in
   tilts]` → N slightly-rotated orientations of the one rotation.
3. **Simulate each tilt.** Rotate the cell → tilted reciprocal basis → different excitation errors
   `Sg` → different Bloch `A` → propagate at thickness → `|ψ|²`. N Bloch solves per rotation; the
   beam *set* is shared, the *geometry* per tilt differs.
4. **Integrate.** For each hkl, **sum** its intensity over the N tilts (per thickness). Optional
   **mosaicity** applies a moving average over the tilt axis before the sum.

The integrated pattern (per hkl, per thickness) is what feeds alignment → `R_obs`/loss. The beam set
is selected **once at the nominal (untilted) orientation** and reused across tilts (the private
`union_adaptive` path that grows the beam set per tilt is off for quartz).

## Decisions

### 1. A tilt is just another orientation

2.0 already models orientation via a per-orientation `reciprocal_basis` on `OrientationPlan`. So the
rocking curve is **N tilted sub-orientations of one rotation plus a sum-over-tilts reduction**,
reusing all existing per-orientation machinery — no new physics primitive, and `simulate` stays a
pure function. The tilt matrices are pure geometry (depend only on semiangle, sampling, axis), so
they are precomputed into the `Plan` like `BeamPlan`, not regenerated at simulation time. Rejected:
bolting an "integrated intensities" mode onto the forward model (the private shape), which couples
the integrator to the solver and is harder to test in isolation.

### 2. Shared beam set across tilts

Select beams **once at the nominal orientation** (`select_beams` unchanged), and reuse that beam set
for every tilt; only the geometry (`Sg`/`A`) varies per tilt. This is faithful to the non-adaptive
private path (`union_adaptive: false`). Growing the beam set per tilt (the private adaptive-union
optimization) is deferred; it is a cost/accuracy refinement, not a correctness requirement for the
reference.

### 3. Geometry mode — continuous rotation now, precession later

Implement **continuous-rotation (x-axis tilts)** now; quartz and the anchor are continuous rotation.
Precession is a later **discriminated mode** (a different tilt generator — a cone), not an optional
field. This needs `data_collection_geometry` surfaced from the PETS reader.

### 4. Mosaicity is a composable knob, not a baked-in step

This is scientific software: a modeller must be able to claim *"enabling mosaicity improved (or
degraded) `R_obs`"* by toggling one thing and re-running. So mosaicity broadening (the private
moving-average over the tilt axis before the sum) is factored as an **optional, composable step** —
selected by a `mosaicity` config/value-type, **off by default**, added to the pipeline as its own
function (`mosaicity(...)`) rather than hard-wired into the integrator. Consequences:

- The rocking-curve slice lands **plain tilt-integration first**; mosaicity is a **second slice**.
- The private reference has `mosaicity: true`, so the full per-rotation `atol ≈ 1e-4` match against
  the reference needs the mosaicity knob **on**. The plain-integration slice tightens the tolerance
  partway; mosaicity closes the remainder.
- Keeping the two separable is exactly what lets an experiment *measure* the effect of mosaicity
  rather than assume it.

### 5. `integration_semiangle` has a double role; `rocking_curve_sampling` is the density knob

The same angle sets **both** the Klar beam-selection window (already `BeamSelection.
integration_semiangle`) *and* the rocking-curve tilt half-width — one physical quantity (the angular
integration range), two roles. `rocking_curve_sampling` (currently an unused `NumericsConfig` field)
is the tilt count, and it is the axis `converge_sampling` sweeps. Introduce a
`RockingCurve(semiangle, sampling, geometry)` value-type (frozen dataclass in `specs.py`, boundary
validation) that shares `integration_semiangle` with beam selection.

### 6. Cost — naive first, batched later

N× the Bloch solves per rotation (42× for quartz), the eigendecomposition being the expensive part.
Land a **naive N×-loop first** (correctness), then a batched `eigh` over tilts as an optimization
(the private amortizes via `union_splits`). The e2e will be slow until the batch optimization lands;
this matches the codebase's correctness-first posture.

## Consequences

- Unblocks `converge_sampling` (deferred in `stage11-convergence.md`): once the rocking-curve
  forward
  model exists, sweeping `rocking_curve_sampling` against it is a real convergence study.
- Slots into plan C as (C3): plain tilt-integration, then mosaicity, tightening the anchor's
  per-rotation `R_obs` tolerance toward the private `1e-4` (see `ROADMAP.md`).
- The tilt reduction (sum over tilts of `|ψ|²`) is differentiable and preserves device; it is a new
  reduction axis on the intensities, not a change to the Bloch kernel.
