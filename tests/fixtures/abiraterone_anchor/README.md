# abiraterone_anchor — private-lineage test fixture (NOT a public example)

Real 3D electron-diffraction data for **abiraterone acetate**, a molecular organic crystal
(C₂₆H₃₃NO₂, space group P2₁2₁2₁ / sg 19). Used as a **test fixture** for two purposes:

1. a **forward parity** anchor — the public forward model reproduces the private reference's
   rotation-0 `R_obs` on this exact input
   (`tests/e2e/test_anchor.py::test_abiraterone_forward_parity_private_rotation0`): public `0.0978`
   vs private `0.09697` (Δ < 0.001);
2. a **refinement positive control** — the well-conditioned counterpart to LTA, to check that the
   scale-normalised refinement objective (`scaled_w_rbragg_loss`) actually lowers `R_obs` where the
   physics is well-conditioned (run on-accelerator, not committed).

## Source & lineage

- `abiraterone.cif` ← `diffbloch_private/data/abiraterone_acetate/abiraterone_acetate_clean.cif`.
- `abiraterone_exp_data.cif_pets` ← **`diffbloch_private/tests/e2e/inference/abiraterone_acetate/cifs/exp_data.cif_pets`** — the observation file the **private reference (0.097) was computed on**. This is the file to compare against.
- Note: `diffbloch_private/data/abiraterone_acetate/p2_abiraterone_original.cif_pets` is a *different real reduction* of the same experiment (111 rotations vs 55, a different reflection set) and is **intentionally not used** — comparing against the private number requires the private's own reduction.

Both files are **real** measured data (not synthetic — ~19% of reflections carry negative,
background-subtracted intensities). **Private-lineage data**; treat as such — the repo is public, so
do not redistribute this dataset outside the project until its redistribution is cleared.

## Reproducing forward parity — two non-obvious requirements

1. **Hydrogens.** The structure has 33 H (a light-atom organic); `read_structure(..., load_hydrogens=True)` is required. The app/config path does not yet express `load_hydrogens`, so the parity test loads it directly (see below).
2. **Explicit solve-set coupling.** A forward-only pipeline (no fit) under-couples: the public wires `couple_beams` only *inside* the fit steps. The parity test composes it explicitly — `TiltSegmentUnion(n_splits=4, g_max=1.5, cap_margin=0.2, sg_max=0.02)` (cap 1.3), matching the private's dynamical-matrix beam mask (`|Sg| < 0.02 ∩ |g| < g_max_sf/2 − 0.2`). Without it, rot-0 `R_obs` sits at ~0.137 instead of 0.098.

## Why this is a fixture, not a canonical `examples/` experiment

Reproducing the private forward requires **loading hydrogens** and **explicit solve-set coupling**,
neither of which the public **app/config** path expresses yet (`load_hydrogens` is not a config
field; the coupling policy is hardcoded to the quartz/LTA `TiltSegmentUnion` defaults). Adding those
config fields would restale the committed quartz checkpoint (`config_digest`), so it is deferred.
Until they are config-driven (and redistribution is cleared), abiraterone stays a `tests/fixtures`
asset and is **not** promoted to `examples/experiments/`.

## Config notes (parity with the private)

`experiment.yaml` sets the matchable private-abiraterone knobs exactly (from
`diffbloch_private/.../abiraterone_acetate/config.yml`): `numerics.g_max = 3`, `g_max_refine = 1`,
`sample.thicknesses = [1460]`, and the rocking-curve integration geometry `dsg = 0.0025`,
`rsg = 0.6`, `rocking_curve_sampling = 30`, `integration.semiangle = 1.422`. Energy (200 keV) is read
from the `.cif_pets` wavelength (0.0251 Å). Lobato scattering and the no-absorption path are
hardcoded and already match. The coupling policy is supplied by the test (not the yaml).
