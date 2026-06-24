# diffBloch 2.0 — staged refactor roadmap

A from-scratch rewrite, ported from the research codebase (`diffBloch_private`) into this package
**stepwise**: each stage is one discrete, tested, well-commented commit with a comprehensive message.
Architecture and rationale: `notebooks/iain/principled_refactor_synthesis.ipynb`.

**The invariant:** the single-rotation quartz characterization anchor (stage 0) stays green at every
commit — the executable form of *"the core physics model has not changed."*

## Stages

- [x] **0 — Scaffolding.** uv package (src-layout, Py ≥3.11), pre-commit (ruff + hooks), Pydantic,
  justfile, mkdocstrings docs, unit + e2e harness, CI, and the anchor placeholder. `just check` green.
- [x] **1 — Config + experiment packaging.** Flesh out `config/` with split/objective/optimizer
  submodels; add experiment-as-directory loader + `experiment.lock` (SHA256, fail-fast); add
  generated-artifact `RunManifest`; add `diffbloch run pack` export surface. Relative paths only.
- [x] **2 — Minimal quartz anchor fixture.** Copy the self-contained quartz CIF/PETS/orientation
  assets and private `results.json` reference into public tests; wrap them in `experiment.yaml`,
  `experiment.lock`, and `anchor_manifest.json`. Anchor now verifies fixture discovery + hashes before
  skipping only the not-yet-ported physics execution. This fixture is the seed experiment-directory
  artifact; subsequent stages should evolve it in place toward the final experiment/run layout.
  Intermediate tensor goldens remain marked pending until stages 7-8.
- [ ] **3 — `io/` parser boundary.** gemmi adapter → Pydantic `StructureRecord` / `ObservationRecord`
  **with explicit `@model_validator` contracts** (shape, occupancy ∈ [0,1], ADP PSD, symop shapes,
  σ ≥ 0). diffpy `symmetry_constraints(record)` behind its seam. Conformance/golden tests.
  *Gate: the validators must exist before any `core/` code trusts the boundary.*
- [ ] **4 — `core/crystal` + `reciprocal`.** Relocate the already-tested helpers (mechanical move).
- [ ] **5 — `core/constraints` + `adp`.** Cholesky + symmetry-mask bijectors; delete the in-place Uij
  loop.
- [ ] **6 — `core/symmetry::expand_asu`.** Precomputed membership; remove per-step numpy dedup
  (membership-order golden).
- [ ] **7 — `core/scattering`.** From `StructureFactorNet`; vectorise per-atom loops (grid golden).
- [ ] **8 — `core/dynamical` + `solver`.** `build_bloch_system(...) -> BlochSystem` (carries
  `A`, `Mii`, `psi0`, `k_n`, `mask` — so `bloch_eigen` needs no geometry); `Propagator(system,
  thickness)` with `matrix_exp` (refine default) + `bloch_eigen` (eval-only) both first-class.
  Precompute `BeamPlan` **keyed on geometry/numerics** (NOT ADP); ADP→coverage is a separate
  `ScatteringTablePlan`. Two sub-steps (assert identical `A` with caches, then remove caches). Run the
  `T ∈ {1,8,42}` timing + gradient-parity experiment before locking `matrix_exp`. *(Highest risk.)*
- [ ] **9 — `core/losses` + `products`.** Move loss bodies out; retire `DiffractionDataset` for
  `PatternBatch` / `BlochSolution`.
- [ ] **10 — `params`/`specs` + `engine`.** `RefinableParams` (incl. `b_dose_raw`) → `constrain` →
  `PhysicalState` (incl. `b_dose`). `RefinementEngine`: `from_experiment`/`from_config`/
  `from_snapshot`, `forward`, `simulate(solver=…)`, chainable `refine(targets, …, checkpoint_every)`,
  `activate(component)` (explicit; targeting an inactive component raises), explicit `OptimizerState`
  + threaded `torch.Generator`, `snapshot`/`history`.
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
