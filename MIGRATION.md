# Coming from diffBloch v1? — vocabulary map

diffBloch v2 is a clean-room reimplementation of the v1 code. The physics is the same; much of the
vocabulary was deliberately renamed for internal consistency and to keep v1 domain terms (`union`,
`structure factor`, `rocking_curve_sampling`, `orientation_matrix`, `g_max_refine`) where they carried
meaning. This table is the fastest way to find where a v1 name went.

Public symbols carry a one-line `v1 analog: …` in their docstring where the mapping is non-obvious.

| v1 term | v2 landing |
|---|---|
| `union_hkl` (per-chunk beam set) | `Segment.union_hkl` |
| `cell_chunks` (tilt groups) | `Segment` objects; count `n_coupling_segments` |
| `union_splits` | `SegmentedUnionCoupling.fixed_n_segments` |
| `union_adaptive` | `SegmentedUnionCoupling.union_adaptive` |
| coupling `A_batches` / reassembly | `CoupledOrientationPlan` (composite) + `SegmentPlan` (per segment) |
| `BlochNet` | `RefinementEngine` (invariant context) + `RefinableParams` / `RefinementModel` (the variable state) |
| `StructureFactorNet` | `core.scattering.structure_factors` + `StructureFactorGrid` + the `Fgb` parameter |
| `ApparentThicknessNN` | `engine.components.ApparentThicknessNN` |
| `calculate_structure_matrix` / `A` | `core.dynamical.structure_matrix` / `BlochSystem.a` |
| `calculate_dynamical_scattering` | `core.solver.propagate` (over a `BlochSystem`) |
| `calculate_M_matrix` / `Mii` | `core.dynamical.mii_factors` / `mii` |
| `dynamical_solver` (`bloch_eigen`/`matrix_exp`) | `core.solver.SolverMethod` |
| `DiffractionDataset` | `BlochSolution` + `PatternBatch` + `AlignmentPlan` + `AlignedIntensities` |
| `RotationDataset` | `preprocess.Plan` + `io.ObservationRecord` |
| `filter_hkls` / `filter_reciprocal_space_vectors` | `select_beams` (solve basis) + `ScoredHklSelection` (scored set) |
| `resolution_filter` | `ScoredHklSelection.g_max` |
| `g_max_refine` / `g_min_refine` | `NumericsConfig.g_max_refine` (kept; no lower bound modelled) |
| `sg_max` | `BeamSelection.sg_max` (Klar) + `SegmentedUnionCoupling.sg_max` (coupling) |
| `Uij_layer` / `thermal_displacements` | `params.constrain` + `core.adp`; `RefinableParams.uij_raw` / `u_iso_raw` |
| `BlochLoss` / `*_loss` bundle | `LossFn` (data) + `PenaltyTerm`s + `ConstraintTransform`s → `ObjectiveValue` |
| `wRbragg` / `rbragg_abs` | `core.losses.w_rbragg` / `rbragg` |
| Hydra `cfg` / `DictConfig` | `config.ExperimentConfig` + the `specs` value-types |
| preprocess CSV "plans" | the serialized `Plan` (`plan.npz` + `plan.lock`) |

**Note on radii.** `g_max` is one *name* but several *values*: the structure-factor grid radius is ~2× the
solve/coupling radius (it must span beam-difference support), the scored cap is smaller still, and the seed
pool is `g_max_refine`. Each lives as a bare `.g_max` on its own role-named object — the owner tells you
which radius it is.

*This file tracks an in-progress source-vocabulary alignment; entries land as each area is renamed.*
