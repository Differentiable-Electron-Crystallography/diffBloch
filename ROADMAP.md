# diffBloch 2.0 — staged refactor roadmap

A from-scratch rewrite, ported from the research codebase (`diffBloch_private`) into this package
**stepwise**: each stage is one discrete, tested, well-commented commit with a comprehensive message.
Architecture and rationale: `notebooks/iain/principled_refactor_synthesis.ipynb`.

**The invariant:** the single-rotation quartz characterization anchor (stage 0) stays green at every
commit — the executable form of *"the core physics model has not changed."*

## Stages

- [x] **0 — Scaffolding.** uv package (src-layout, Py ≥3.12), pre-commit (ruff + hooks), Pydantic,
  justfile, mkdocstrings docs, unit + e2e harness, CI, and the anchor placeholder. `just check` green.
- [x] **1 — Config + experiment packaging.** Flesh out `config/` with split/objective/optimizer
  submodels; add experiment-as-directory loader + `experiment.lock` (SHA256, fail-fast); add
  generated-artifact `RunManifest`; add `diffbloch run pack` export surface. Relative paths only.
  Fixed sample thickness belongs in `sample.thicknesses`, not `numerics`, so later nuisance/refinable
  thickness work has one home.
- [x] **2 — Minimal quartz anchor fixture.** Copy the self-contained quartz CIF/PETS/orientation
  assets and private `results.json` reference into public tests; wrap them in `experiment.yaml`,
  `experiment.lock`, and `anchor_manifest.json`. Anchor now verifies fixture discovery + hashes before
  skipping only the not-yet-ported physics execution. This fixture is the seed experiment-directory
  artifact; subsequent stages should evolve it in place toward the final experiment/run layout.
  Intermediate tensor goldens remain marked pending until stages 7-8.
- [x] **3 — `io/` parser boundary.** gemmi adapter → Pydantic `StructureRecord` / `ObservationRecord`
  **with explicit `@model_validator` contracts** (shape, occupancy ∈ [0,1], ADP PSD, symop shapes,
  σ ≥ 0). diffpy `symmetry_constraints(record)` behind its seam. Conformance/golden tests.
  *Gate: the validators must exist before any `core/` code trusts the boundary.*
- [x] **4 — `core/crystal` + `reciprocal`.** Relocate the already-tested NumPy helpers:
  cell-matrix construction, reciprocal basis, reciprocal-grid sizing, HKL grid generation,
  g-vector lengths/masks, raveled HKL indexing, and lattice-centering reflection conditions.
  The public helpers intentionally fix the private A/B/C centering row-slice bug while preserving
  the primitive/body/face-centered behavior used by downstream stages.
- [x] **5 — `core/constraints` + `adp`.** Add torch-backed lower-triangular Cholesky ADP,
  isotropic ADP, unit-interval, positive, and symmetry-mask transforms, plus the
  `RefinableParams -> constrain -> PhysicalState` seam. Public constraint application avoids
  mutating grad-carrying inputs; `ConstraintSpec` carries the current Uiso/Uani/missing ADP kind
  metadata without occupying the richer future `StructureSpec` name. diffpy extraction remains behind
  the `io.symmetry_setup` seam until full special-position handling lands.
- [x] **6 — `core/symmetry::expand_asu`.** Add precomputed ASU membership plans and torch-only
  expansion for positions, atomic numbers, ADPs, and occupancies. Duplicate detection happens once
  in `build_asu_expansion_plan` with returned duplicate diagnostics; `expand_asu` gathers by plan
  indices and preserves gradients. Expanded positions remain unwrapped in the differentiable path so
  scattering can use periodic phases without introducing a modulo discontinuity. Quartz pins the
  atom-major/symop-minor membership order.
- [x] **7 — `core/scattering`.** Elastic structure factors ported from `StructureFactorNet`: native
  Lobato form factors (coefficients vendored to `core/data/lobato.json`, cited to Lobato & Van Dyck
  2014 in `REFERENCES.md`; abTEM/diffsims are test-only oracles, not runtime deps), anisotropic
  Debye–Waller, resolution cutoff, and a vectorised (unique-Z + batched phase-sum) `structure_factors`.
  Differentiable in positions/ADP/occupancy; `f_e` is a setup constant. Absorption (`U0'`) deferred;
  `structure_factors` consumes U*-frame ADPs (Cartesian→U* conversion is owed to the ADP/spec layer).
- [x] **8 — `core/dynamical` + `solver`.** `build_bloch_system(...) -> BlochSystem` (carries
  `A`, `Mii`, `psi0`, `k_n`, `mask` — so `bloch_eigen` needs no geometry); `Propagator(system,
  thickness)` with `matrix_exp` (refine default) + `bloch_eigen` (eval) both first-class and
  swappable off the same system value. Precompute `BeamPlan` **keyed on geometry/numerics** (NOT ADP)
  replaced the private `A_offdiag`/`sparse_prebuilt` caches. The `T ∈ {1,8,42,500}` timing + parity
  experiment (`scripts/stage8_propagator_experiment.py`) confirmed the methods coincide only at
  `Mii==1` and set the default; decision recorded in `design/decisions/stage8-bloch-propagators.md`.
  *Deferred (oracle methodology):* real-CIF `Fgb` via the new parser, an R-factor/intensity oracle,
  and a composite-step (gather→A→propagate→intensity→loss) breakdown.
- [ ] **9 — `core/losses` + `products`.** Move loss bodies out; retire `DiffractionDataset` for
  `PatternBatch` / `BlochSolution`.
  - [x] (1/n) `products.intensities` (`|psi|^2`) + intensity-space losses (`mse`, `l1`,
    `weighted_mse`, `rbragg`, `w_rbragg`), each pinned against the verbatim private `metrics.py`
    bodies (`41d22bf`).
  - [x] (2/n) typed products `BlochSolution` / `PatternBatch` (built from `io.ObservationRecord`) +
    precomputed `AlignmentPlan`/`align` bridge; device-safe (`2e635ac`, `22982eb`).
  - [x] (3/n) end-to-end calculated pipeline on the synthetic-Friedel oracle: composite-step
    breakdown (A -> psi -> intensities vs golden) + products->align->loss chain + end-to-end
    differentiability to `Fgb`.
  - Deferred: symmetry-equivalent hkl merging in alignment; engine/eval position metrics
    (`rmsd`, `euclidean_distance`, flat-bottomed/convexity regularisers); the physically-real
    R-factor pin against a CIF dataset (the e2e quartz anchor).
- [ ] **10 — `params`/`specs` + `engine`.** `RefinableParams` (incl. `b_dose_raw`) → `constrain` →
  `PhysicalState` (incl. `b_dose`). `RefinementEngine`: `from_experiment`/`from_config`/
  `from_snapshot`, `forward`, `simulate(solver=…)`, chainable `refine(targets, …, checkpoint_every)`,
  `activate(component)` (explicit; targeting an inactive component raises), explicit `OptimizerState`
  + threaded `torch.Generator`, `snapshot`/`history`.
  - [x] (1/n) ADP frame: `constrain` emits `PhysicalState.uij_star` (reciprocal `U*`) via
    `core.adp.cif_adp_to_star` (Uani) / `cartesian_adp_to_star` (Uiso); device-safe
    (`6870ee3`, `0991b92`). Private `A_matrix` `c_star` anomaly recorded (REFERENCES.md +
    private `KNOWN_ISSUES.md`).
  - [x] (2/n) stateless forward spine `RefinementEngine.forward`/`simulate` + `ScatteringGrid` /
    `OrientationPlan` plans; shared-grid coupling enforced; device-safe (`5db3ee1`, `53e3ed5`).
  - [x] (3/n) the imperative refinement loop: `engine/` promoted to a package (`plan` / `engine` /
    `refine`), `refine(targets, optimizer, lr)` over `adam`/`adamw`/`lbfgs` via
    `engine.refine.run_refinement` (functional contract: caller params never mutated),
    `RefinementResult` (final + `losses` trajectory + `best_*`). See
    `design/decisions/stage10-refinement-loop.md`.
  - Deferred: `from_config`/`from_experiment`/`from_snapshot` (need stage-11 beam selection);
    `activate(component)` + `b_dose` target (nothing optional to activate yet); per-group learning
    rates; `least_squares` (Gauss–Newton/LM); `OptimizerState` / threaded `Generator` /
    `snapshot`/`history` / `checkpoint_every`; refinable-thickness wiring and multi-thickness
    reduction beyond summation.
- [ ] **11 — `preprocess/`** (a composable `Plan -> Plan` pipeline; name finalized — see naming
  note). A distinct stage that **emits the invariant `Plan`** the differentiable refinement is
  conditioned on. Structurally a scikit-learn-style **Pipeline**: a sequence of `Plan -> Plan`
  transformers (each *fits* something and returns a sharpened `Plan` via `dataclasses.replace`),
  with the terminal estimator being `refine` (`Plan -> Result`). Orientation and thickness are
  **locked in here**, never entering the structural autograd graph — preprocess-locked, not
  joint-refine, is the confirmed default (the `RefinableParams.thickness_raw` seam stays dormant for
  an optional future joint path).
  - **Data flow.** `Plan` is the spine (immutable, threaded through, sharpened step by step);
    `simulate` is the kernel `(Plan, params) -> calculated` — one pass in eval, but **looped inside
    `refine`** every iteration (it is the loop body, not a stage); `refine` is terminal and loops
    `simulate`. So `plan0 -> converge_numerics -> fit_orientation -> fit_thickness -> plan*` is
    linear, then `refine(plan*, params) -> loop(simulate -> loss -> step) -> result`.
  - **Numeric convergence** (`converge_numerics()`): numerical-fidelity hyperparameters (`sg_max`,
    `g_max_sf`, `g_max_refine`, `rocking_curve_sampling`, `dsg`, `rsg`) set by convergence testing
    (coarsen/refine until the observable stops moving) — objective is accuracy-vs-cost, **not**
    fit-to-data, and several are discrete (beam count, sampling) so it is a sweep, not backprop.
    These *parameterize Plan construction*: `g_max` sizes the `ScatteringGrid`, `sg_max` selects the
    `BeamPlan` beams, sampling drives the rocking-curve integration.
  - **Physical nuisance calibration** (`fit_orientation()` / `fit_thickness()`): per-rotation
    orientation and thickness, fit to the data. **Thickness is per-rotation because the specimen's
    3D shape is irregular** (each orientation presents a different beam path length), so it moves
    into `OrientationPlan`, retiring the engine-level shared `thicknesses`. The forward model reads
    thickness per orientation from `OrientationPlan.thickness`; when thickness is being refined the
    optimiser's value overrides it for every orientation. Thickness is
    gridsearch today; a learned **`ThicknessNN`** (`theta -> thickness`, swapped in via that same
    override point -- config `thickness.mode: frozen|learned` or programmatic
    `refine(..., thickness=...)`; its
    `.parameters()` join the optimizer and the convexity penalty attaches only in learned mode,
    optionally warm-started from the baked thicknesses) is **committed future work for the v1
    release of the refactor**, faithful to private `ApparentThicknessNN` / `cfg.thicknessNN` /
    `convexity_loss_fn`. So thickness has two *modes* (a fixed per-rotation value vs a learned
    model), not two homes for one default value. Orientation likewise — 3D-ED practice already refines per-frame
    orientation by least-squares on simulated patterns (PETS2/Jana2020).
  - **Composition is a partial order, not free reordering.** The steps couple (convergence needs a
    rough orientation; thickness needs converged numerics; orientation/thickness can be mutually
    dependent), so the composition operator is sequencing **plus a fixpoint combinator**
    (`iterate_until(converged, ...)`), not an unordered set. A fixpoint of `Plan -> Plan` is itself
    `Plan -> Plan`, so it still composes.
  - *Naming note (finalized).* `preprocess` kept as the umbrella, justified by the scikit-learn
    **Pipeline** reading (transformers + a terminal estimator). Known wart: in ML "preprocessing"
    usually means *data* transforms, which here is `io/`'s job (the parser boundary; the 3D-ED
    "data reduction" step, PETS2) — this stage instead fits the *forward-model configuration*.
    Rejected: `calibrate` (excludes numeric convergence), `setup` (vaguer, less signal). The precise
    sub-verbs `converge_numerics` / `fit_orientation` / `fit_thickness` carry the meaning the
    umbrella elides.
  - **Slices.** (1/n) the `preprocess/` spine + `Plan -> Plan` combinators (`Plan` value object;
    `pipeline` sequencing, `iterate_until` fixpoint with a `RuntimeError` on non-convergence,
    `identity`) — pure scaffolding, engine unaware of `Plan` (done). **Reordered** so
    orientation-in-the-physics comes first (the faithful Klar rsg/dsg beam filter is
    orientation-dependent, so it sits *on top of* the rotated `g`): (2/n) **orientation in the
    physics** (done) — per-orientation `reciprocal_basis` feeds `g -> Sg/Mii` via
    `OrientationPlan.build`, default = the shared (untilted) grid basis; (3/n) `from_experiment`
    building a
    grid-sharing **`(train_plan, val_plan)`** pair (rotation train/val split; closes the stage-10
    `from_*` deferral); (4/n) `select_beams` (`Plan -> Plan`, rsg/dsg filter); (5/n)
    `fit_orientation`; (6/n) per-rotation thickness in `OrientationPlan` + `fit_thickness`; (7/n)
    `converge_numerics`. Decisions: orientation enters via a **per-orientation `reciprocal_basis`**
    (not a rotation in core); the quartz pull-in showed the real orientation matrices are
    non-orthonormal (a ~1% measured-cell correction), so `reciprocal_cell(cell @ M.T)` is faithful
    while `@ M.T` is wrong -- pinned by `test_orientation_oracle.py` against a private-impl golden.
    Train/val are two `Plan`s sharing one grid (engine stays split-agnostic). Orientations are
    **derived natively from the PETS goniometer geometry** (`preprocess/orientation`:
    `R_z R_x R_y @ UB B^-1`), not read from a side-car CSV -- per the effects/observability decision
    (`design/decisions/effects-and-observability.md`), `fit_orientation`/`fit_thickness` return
    `Plan`s and never write per-facet CSVs (persistence = checkpoint the whole `Plan`; CSV/visualize
    are boundary reporters). Landed: (1/n) spine; (2/n) orientation in physics; (3/n) a -- native
    orientation derivation pinned to a private golden (8.6e-7) + `orientation_basis` convention home.
- [ ] **12 — `logging` + `app/`.** Pluggable `Logger` (NullLogger default; wandb/CSV/MLflow as
  swappable backends — no vendor SDK in core); thin `cli.py`; pluggable `sweep.py`.
- [ ] **13 — Cleanup.** Delete deprecated adapters; final e2e + full unit run. `RunRef` op-boundary
  (orchestration §18) when a production orchestrator actually arrives.

## Design corrections folded in (review round)

1. **Solver seam** widened to a `BlochSystem` value object (the bare `(A, thickness, k_n, mask)` seam
   was too narrow for `bloch_eigen`, which needs `Mii`/`psi0`).
2. **Beam damage** is wired end-to-end: `PhysicalState.b_dose` + explicit `engine.activate(...)`;
   target selection never silently activates.
3. **Record validation** is encoded in explicit validators, not `arbitrary_types_allowed` alone.
4. **`BeamPlan`** is geometry/numerics-keyed (immutable across refinement); ADP-driven coverage is a
   separate `ScatteringTablePlan` — no `did_overflow(adp)`.
5. **Resume/checkpoint** lives in `from_snapshot(...)` + `refine(..., checkpoint_every=N)`.

## Cross-cutting

- Every stage: discrete tested commit, comprehensive message, `just check` + e2e anchor green.
- Scientific deps (numpy, torch, gemmi, diffpy.structure, abtem) are added to `pyproject.toml` as
  their modules land — not all up front.
