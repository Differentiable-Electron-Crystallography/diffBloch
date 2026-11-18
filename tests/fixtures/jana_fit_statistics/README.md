# Jana2020 fit-statistics oracle

Per-reflection output of two published Jana2020 dynamical refinements, used to pin
`diffBloch.engine.fit_statistics` to Jana's own R/wR/GoF convention.

- `quartz.csv.gz`, `borane.csv.gz`: the `_refln` loop of the Jana2020 CIFs `Quartz_IAM.cif` and
  `borane_IAM_4.cif` (independent-atom-model refinements, shipped in the diffBloch 0.3.0
  `examples/Suresh_et_al/*/CIF_files/`), as printed by Jana, one row per reflection: `intensity_calc` (already on the observed scale), `intensity_meas`,
  `intensity_sigma`, and `scale_group` (the zone, i.e. the frame carrying its own `EDScale`).
- `expected.json`: the same CIFs' `_refine_ls_*` / `_reflns_number_gt` values (fractions).

The GoF parameter count is Jana's `_refine_ls_number_parameters` minus the number of zones: Jana
lists one `EDScale` per zone among its parameters but does not charge those scales in GoF.
