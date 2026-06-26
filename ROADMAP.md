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
- [ ] **11 — `preprocess/`.** Clean reimplementation of orientation + thickness + numeric-convergence
  behind `engine.fit_*()` / `converge_numerics()`.
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
