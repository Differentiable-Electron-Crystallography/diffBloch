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
    - *2.0 correction (stage 11):* the private's three knobs do **not** map 1:1. The `Fgb` support
      grid is sized-to-cover (the Bloch gather only needs the beam differences), so grid `g_max` is
      **not** a convergence knob; the real levers are beam-set inclusiveness
      (`g_max_refine` pool + `integration_semiangle` window, coupled) plus rocking-curve sampling.
      **Deferred subsequent task:** 2.0's forward model has **no rocking-curve integration** yet
      (`rocking_curve_sampling` is an unused config field), so `converge_sampling` waits on that
      forward-model feature (its own decision + oracle). See
      `design/decisions/stage11-convergence.md` and `DIVERGENCE.md`.
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

## The executable e2e anchor — the physically-real R-factor pin (plan C)

The quartz anchor (`tests/e2e/test_anchor.py`) currently verifies fixture discovery + hashes then
**skips the physics**. The goal is to make it *run a real experiment end to end* and pin
`R_obs ≈ 0.0438` (the private reference `reference_results.json`, mean of per-rotation `R_obs`,
matched per rotation the way the private `evaluate_over_rotations` does). A spike showed 2.0's
current single-orientation forward gives `R_obs ≈ 0.6`/NaN — **three gaps** separate us from the
reference:

1. **No rocking-curve integration.** The reference integrates each rotation over **42 tilts**
   (`rocking_curve_sampling: 42`, `linspace(−1°,+1°,42)`, `mosaicity: true`); 2.0 point-samples one
   orientation. A point sample of a rapidly-varying rocking curve cannot match an integrated
   measurement. *(**Correction, 2026-07:** this was originally called “the dominant gap” — see the
   dated note under C3. The rocking curve is essential physics, but its benefit is only realised
   when orientations are scored/fit under the integrated model; the dominant static-baseline
   correction turned out to be a beam-selection geometry bug, not the rocking curve.)*
2. **Unfit orientations.** The reference loads post-`fit_orientation` values
   (`optim_orientation.csv`, `apply_u_matrix: true`); 2.0 uses native-derived (pre-fit)
   orientations, which is why R sits near 0.6 (unfit) rather than ~0.05.
3. **`0/0` NaN** on rotations where no reflection passes `I > 3σ` in the current (unfit) beam set —
   secondary, expected to resolve once orientations are fit.

**Idiomatic-API principle (confirmed steer).** The e2e must read like a real experiment — load from
the **filesystem** experiment directory through the **public API** and assert a metric — *not* reach
into internals (`build_engine` / `engine._solve` / `align` / `optimal_scale`). That public surface
does not yet exist (there is `load_experiment`/`load_config` at the boundary but no public
"build the experiment → run the forward model → per-rotation metrics" entry, the 2.0 analog of the
private `evaluate_over_rotations`). Building it is part of this work.

**Sequencing — plan C (slice, tightening the tolerance at each step):**

- [x] **(C1) Public inference harness.** Stand up the idiomatic surface: `load_experiment(root)` →
  a public `run_inference`/driver → per-rotation `R_obs`. **Decision (b): C1 is the harness plus a
  synthetic-system unit test only** — the quartz aggregate `R_obs` is *not* pinned here (a captured
  pre-fit baseline over 20/99 finite rotations at `R_obs ≈ 0.71` would be a meaningless regression
  guard), it moves to **C2** where the number is meaningful. First `src/` caller that makes the
  value-type contract bite (the preprocess driver). *(landed: `run_inference` / `InferenceResult` /
  `RotationInference` in `preprocess/inference.py`, built from public `engine.simulate` +
  `core.products.align` + `core.losses`, with `PlanSplit.combined`; synthetic `test_inference.py`.)*
- [x] **(C2) Fit orientations (closes gaps 2/3).** Wire `select_beams → fit_orientation` (and
  `fit_thickness`) through the harness so the anchor runs the fit pipeline, not raw derived
  orientations. **Rewrite the anchor to run `run_inference` over `PlanSplit.combined` and pin the
  quartz aggregate `R_obs`** (the C1-deferred pin). Tighten the tolerance toward the reference.
  *(landed: `test_anchor.py` runs the full `select_beams → fit_orientation → fit_thickness` pipeline
  over all 99 rotations via the public API and pins `n_evaluated == 99` + `mean_r_obs` at `abs=1e-2`.
  Two blockers were cleared first: the `rbragg` NaN-safety fix (`1c79693`, all 99 now finite) and
  calibrating `fit_orientation`'s `max_iterations` to 600 (`768786b`). The captured baseline was
  `≈ 0.2977`; after the C3-note beam-selection geometry fix (`e198ad1`) it dropped to `≈ 0.174`.)*
- [ ] **(C3) Rocking-curve integration (closes gap 1).** New forward-model feature (see below).
  Tighten to a per-rotation `atol` approaching the private `1e-4`.

> **Correction (2026-07): the 0.298 → 0.044 gap was *not* the rocking curve.** A diagnostic against
> the exact reference recipe (optim orientations + 820 Å + 42-tilt integration) found two things.
> **(i)** The rocking curve is *negligible* on the from-scratch static-fit orientations
> (full-99 static `0.2977` → integrated `0.2939`) — because `fit_orientation` scored statically and
> so converged to *static-optimal* orientations, which the integration barely improves. On seed or
> reference-optim orientations the rocking curve is instead a **7–10×** effect (seed `0.60 → 0.08`;
> optim `0.63 → 0.06`) — see the tutorial. This is the **fit/eval consistency invariant**: tilts must
> be present *during* the fit, not bolted on after. **(ii)** The dominant static-baseline error was a
> **beam-selection geometry bug** — the Klar `sg_max` lever arm used the distance from the *beam*
> (a precession assumption) instead of from the *goniometer rock axis*, admitting ~1.7× too many
> non-sweeping reflections and inflating R. Fixing it (`e198ad1`) reproduced the reference reflection
> counts and, on optim orientations + integration, reached `R_obs = 0.0594` (reference `0.0438`);
> it also lowered the from-scratch static baseline `0.2977 → 0.174`. Full narrative in
> `SCIENCE_FORK.md` / `DEBUGGING.md` / `LESSONS.md`. **Remaining residual `0.0594 → 0.0438`** and the
> fit-coupling wiring are the live C3 work.
>
> **Scheduling: the residual chase is deferred to *after* stage 11 completes.** Stage 11's remaining
> convergence slices (pool lever + cross-lever fixpoint, `converge_sampling`, the coverage sweep, and
> the preprocess driver) land first; the `0.0594 → 0.0438` residual and the C3 fit/eval-integration
> coupling are then taken up as a distinct post-stage-11 workstream (candidates: obs matching /
> `I>3σ` bookkeeping 965 vs 958, mosaicity, minor forward-model details, and threading `op.tilts`
> through `fit_orientation`). Tracked in `DEBUGGING.md`.

### Rocking-curve integration — design decisions (approved)

*What it is:* a rotation-electron-diffraction frame integrates each reflection's intensity as it
sweeps through the Ewald sphere during the exposure (goniometer sweep + convergence + mosaicity). A
static single-orientation Bloch solve is a point sample of that curve; integrating over a spread of
tilts approximates the measured (integrated) intensity. Private mechanics: tilt matrices
`linspace(−semiangle,+semiangle, sampling)` about the goniometer axis (**x** in the PETS frame) →
tilt the nominal orientation → one Bloch solve per tilt (shared beam set, per-tilt geometry) →
**sum** `|ψ|²` over tilts per hkl. Faithful to private
`generate_integration_rotation_matrices` / `BlochNet.forward(tilts=…)` /
`DiffractionDataset.get_integrated_intensities`.

1. **A tilt is just another orientation.** Model the rocking curve as **N tilted sub-orientations of
   one rotation + a sum-over-tilts reduction**, reusing the existing per-orientation
   `reciprocal_basis` machinery (no new physics primitive; `simulate` stays pure). Tilt matrices are
   pure geometry → precompute into the `Plan` like `BeamPlan`. The integration itself is a
   **composable, toggleable `Plan → Plan` step** (`integrate_rocking_curve(...)`), off/identity by
   default (no step, or `sampling = 1`, = single static solve, byte-identical) — so a run can claim
   *"enabling rocking-curve integration improved/degraded R_obs"* by composing one unit in or out,
   the same modularity as mosaicity (#4), of which the rocking curve is the enabling structure.
2. **Shared beam set across tilts.** Select beams **once at the nominal orientation**
   (`select_beams` as-is), reuse for every tilt; only the geometry (Sg/A) varies per tilt (the
   non-adaptive private path; `union_adaptive: false`).
3. **Geometry mode.** Continuous-rotation (x-axis tilts) now; precession as a later discriminated
   mode. Needs `data_collection_geometry` from the PETS reader.
4. **Mosaicity is a composable knob (scientific-modularity steer).** Mosaicity broadening (private:
   a moving average over the tilt axis before the sum) is factored as an **optional, toggleable step
   composed into the pipeline** — not baked into the integrator — so a run can claim *"enabling
   mosaicity improved/degraded R_obs"* by adding or removing one composed function
   (`mosaicity(...)`, off by default; a `Plan → Plan` or a tilt-reduction transform selected by a
   `mosaicity: bool`/window config). This means (C3) lands plain tilt-integration first, then
   mosaicity as a second composable slice; the reference has `mosaicity: true`, so the full
   `1e-4` match needs the knob on.
5. **`integration_semiangle` double role.** The same angle sets **both** the Klar beam window *and*
   the rocking-curve tilt half-width; `rocking_curve_sampling` (currently an unused `NumericsConfig`
   field) is the tilt count and the axis `converge_sampling` sweeps. Introduce a
   `RockingCurve(semiangle, sampling, geometry)` value-type sharing `integration_semiangle`.
6. **Cost.** Naive N×-loop of Bloch solves first (correctness), batched `eigh` over tilts as a later
   optimization; the e2e is slow until then.

Once (C3)+mosaicity land, `converge_sampling` (deferred from stage 11) unblocks — it sweeps
`rocking_curve_sampling` against the now-real rocking-curve forward model.

### Deferred: explicit rotation exclusion

Some datasets need rotations dropped (bad frames, misindexing, outliers). The private does this
implicitly via a hand-edited `dataloader.ignore_orientations` list; 2.0 will make it **explicit,
reproducible config** — an `exclude_rotations` list `from_experiment` honours, versioned in
`experiment.yaml` + `experiment.lock`, justified per index. An **automatic** mechanism is also
planned, framed as a *robust-outlier fixpoint* (not a `converge_scalar` scalar sweep but an
`iterate_until` fixpoint over the excluded set: flag statistical outliers → exclude → refit until
the set stabilizes). The non-negotiable constraint: the flag criterion is an **independent
robustness statistic** of the residual distribution (e.g. `> k*MAD` from the median per-rotation
`R`), *never* chosen to minimize the reported `R` (that is data-dredging), report-first, and
recorded in config + lock. Auto-dropping search non-convergers stays report-only (it hides signal).
The executable quartz anchor needs no exclusions (the `max_iterations` calibration fixed its only
failing searches). See `design/decisions/rotation-exclusion.md`.

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
- **De-jargonify comments (housekeeping sweep).** Comments and docstrings should stand on their own
  against the code in their immediate context — not lean on other docs, decision records, or past
  conversation to be understood. The goal is not merely "use fewer fancy terms" but "a reader of
  this function needs nothing but this function to understand the comment." Known residue: the word
  "seam" used in the milder boundary sense in `params.py`, `core/dynamical/assembly.py`,
  `io/symmetry_setup.py`, `design/decisions/stage10-refinement-loop.md`, and ROADMAP's "Solver
  seam"; plus any remaining cross-doc pointers in comments. Sweep these in a dedicated pass.
