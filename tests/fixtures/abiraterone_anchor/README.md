# abiraterone_anchor — test fixture (NOT a public example)

Real 3D electron-diffraction data for **abiraterone acetate**, a molecular organic crystal
(C₂₆H₃₃NO₂, space group P2₁2₁2₁ / sg 19). Used as a **test fixture** for two purposes:

1. a **forward parity** anchor — the public forward model reproduces a reference rotation-0
   `R_obs` on this exact input: public `0.0978` vs reference `0.09697` (Δ < 0.001);
2. a **refinement positive control** — the well-conditioned counterpart to LTA, to check that the
   scale-normalised refinement objective (`wr2_loss`) actually lowers `R_obs` where the
   physics is well-conditioned (run on-accelerator, not committed).

Both `abiraterone.cif` and `abiraterone_exp_data.cif_pets` are **real** measured data (not
synthetic — ~19% of reflections carry negative, background-subtracted intensities).

## Reproducing forward parity — two non-obvious requirements

Both are now expressed in `experiment.yaml`, so the parity test reads them from config rather than
hardcoding them:

1. **Hydrogens.** The structure has 33 H (a light-atom organic); it needs
   `read_structure(..., load_hydrogens=True)`. `inputs.load_hydrogens: true` is set in the config and
   the test reads it from there.
2. **Explicit solve-set coupling.** A forward-only pipeline (no fit) under-couples: the public wires
   `couple_beams` only *inside* the fit steps. The config declares a `preprocess.coupling` block —
   `TiltSegmentUnion(n_splits=4, g_max=1.5, sg_max=0.02)` — and the test composes `couple_beams` from
   it. Post-#154 the dynamical-matrix beam mask is `|Sg| < 0.02 ∩ |g| < g_max` (the `-0.2` cap margin
   was dropped), so the coupling radius is the physical `g_max = 1.5`. Without the coupling, rot-0
   `R_obs` sits at ~0.137 instead of ~0.098.

## Why this is a fixture, not a canonical `examples/` experiment

`load_hydrogens` and `preprocess.coupling` are now both config fields, so the API gap that used to
justify fixture status is gone. It stays a `tests/fixtures` asset and is **not** promoted to
`examples/Colmey_et_al_2026/data/`.

## Config notes

`experiment.yaml` sets `sample.thicknesses = [1460]`, and the rocking-curve integration geometry
`dsg = 0.0025`, `rsg = 0.6`, `rocking_curve_sampling = 30`, `integration.semiangle = 1.422`. The
structure-factor support grid is **derived** from the coupling radius (`2 * 1.5 + 0.5 = 3.5`), not
declared. Energy (200 keV) is read from the `.cif_pets` wavelength (0.0251 Å). Lobato scattering
and the no-absorption path are hardcoded and already match. The coupling policy lives in
`preprocess.coupling`.
