# AGENTS.md for diffBloch

Guidance for coding agents working on diffBloch. It records the architecture, invariants, and principles that constrain every change, plus the way to derive the right implementation shape for a new feature. Read it before touching `src/diffBloch/`. This document stands alone: do not depend on files outside the repository as implementation authority.

When something is unclear, ask before changing architecture. When running unattended, choose the smallest reasonable interpretation, proceed, and state the assumption in your summary.

## What diffBloch is

diffBloch refines crystal structures against rotating-stage 3D electron-diffraction data: continuous-rotation 3DED, a sequence of diffraction frames collected as the crystal is tilted/rocked through reciprocal space around a goniometer axis, reduced upstream by PETS2 into `.cif_pets` experimental data. Because the diffraction is dynamical (multiple-scattering), diffBloch uses a differentiable Bloch-wave simulation rather than a kinematical approximation.

The object of value is small: a differentiable map from a handful of structural parameters (atom positions, ADPs, occupancies, structure factors) to simulated intensities and a single scalar R-loss. Gradients of that loss update the selected trainable parameters. Everything else in the package prepares inputs for that map, runs the optimizer around it, checkpoints expensive setup, or reports progress — around the numerical core, never inside it.

`infer` means simulate and score a settled `Plan` without updating parameters. `refine` means run optimizer steps that update selected trainable parameters.

## Architecture: functional core, imperative shell

The design is a pure functional core wrapped in a thin imperative shell. The core is deterministic mathematics; the shell does I/O, optimization, device placement, checkpointing, and logging. Keep that boundary sharp: it is what makes runs inspectable, reproducible at the artifact boundary, and attributable.

Preserve this system shape:

```text
.cif + .cif_pets + experiment.yaml
  -> typed IO / config boundary
  -> Plan-building preprocess shell
  -> deterministic Bloch-wave core
  -> refinement engine / objective
  -> quarantined optimizer / app / logging shell
```

Three quick smell-tests for any change:

- A change that makes a lower layer know about a higher layer is suspect.
- A change that hides a result-changing input outside config/provenance is suspect.
- A change that mutates caller-owned scientific values in place is suspect.

### Layers and dependency direction

Dependencies point one way. A layer may import from layers below it, never from orchestration above.

| Layer | Role | Hard rule |
|---|---|---|
| `io` | Parse CIF/PETS into validated typed records. | Parser details stop at typed records; numerical code never consumes raw parser objects. |
| `params` | Raw refinable parameters and crystallographic constraints. | `RefinableParams -> constrain -> PhysicalState`; no optimizer state here. |
| `specs` | Frozen value-types for preprocess/scientific knobs. | Defaults and validation live here. |
| `core` | Deterministic crystallographic and Bloch-wave kernels. | Tensor/value in, tensor/value out; no parser, optimizer, vendor SDK, filesystem, or app imports. |
| `engine` | Combine `Plan` + parameters, simulate, score, refine. | Forward/objective code stays pure; optimizer mutation is quarantined in the refinement loop. |
| `preprocess` | Build and improve the immutable `Plan`. | Public shape is `Plan -> Plan`; steps return new values, never mutations. |
| `config` | Validate experiment settings and lock identity. | Pydantic boundary, `extra="forbid"`; no Hydra/`DictConfig` reaches the core. |
| `observability` | Typed event values and the logger protocol. | Emit event data; backend I/O lives at the app/logger boundary. |
| `app` | CLI and default orchestration around reusable APIs. | Compose public APIs; do not hide science-only behaviour here. |

`io`, `params`, `specs` sit below `core`; `core -> engine -> preprocess`; `config` and `app` wire the lower layers at the top. The engine never imports `preprocess`.

### "Core" means the functional core

`core` is the `src/diffBloch/core/` layer — pure tensor-in/tensor-out kernels from structural parameters to simulated intensities and the R-loss — not "whatever is important." Keep this wording precise in prose.

### What is pure and what has effects

- **Pure:** every kernel in `core/*`, `params.constrain`, all `specs` value-types, the objective/simulation in `engine/forward.py`, and every `Plan -> Plan` step in `preprocess/steps/`.
- **The one quarantined stateful corner:** the `torch.optim` loop in `engine/refine.py`. It clones selected trainable fields into fresh leaves and never mutates the caller's parameters. `core/` never imports `torch.optim`.
- **Effects/I/O:** `io` (file reads), `config` (YAML/lock/manifest, git/version stamps), `preprocess/serialize.py` (`Plan` checkpoint read/write), and `app` (CLI, orchestration, logger backends).

### Invariants of the Bloch-wave core

These make the numerical core trustworthy. Treat a change that violates one as a defect even if the forward value looks right.

- **Differentiability.** The forward path is autograd-differentiable end to end. A change that produces the correct value but breaks or NaNs the gradient — an in-place op on a leaf, a stray `.item()`/`detach`, non-differentiable indexing — is a defect. Preserve gradient flow to the source leaves and keep gradients finite.
- **Determinism.** Same inputs produce the same outputs; the simulation carries no hidden state.
- **Precision is a boundary, not a sprinkle.** The Bloch solve always runs at fp32/complex64 (`core.solver.propagate`), never threaded ad hoc through individual kernels.
- **Device/dtype preserving.** Kernels preserve the incoming device and dtype. `params.asu_positions.device` is the authoritative forward device, and invariants co-locate onto it at the use site. Only a boundary selects a device policy.
- **Strategy-as-value.** `SolverMethod` is a literal; `matrix_exp` and `bloch_eigen` are swappable over one `BlochSystem`. Do not duplicate geometry logic per solver.
- **Cache geometry, not gradients.** Geometry-only caches such as `StructureFactorGather` integer lookup geometry and `BeamPlan` constants are allowed; never cache differentiable values that can go stale. `Fgb` stays live.

## Observability is data, not a dependency

The pure core only names, returns, or emits typed `Event` values and runs correctly with zero sinks attached (`NULL_LOGGER`). Logger backends (console/CSV/W&B/Comet) live at `app/loggers` and import their vendor SDKs lazily, only when used.

- Never thread a logger or file handle into the mathematics.
- Never route scientific results through the diagnostic (`logging`) channel. Library code only calls `logging.getLogger(__name__)` with a package-level `NullHandler`; never call `basicConfig` in library code.
- Persistence means serializing a whole `Plan` value, never writing a per-step CSV that a later step reads back. A human-readable CSV/log is a write-only report, never re-imported as pipeline truth.
- Event measurements are public-ish data contracts: rename them deliberately and update dashboards/docs/tests together. Use `rotation_index` for experimental frame/rotation indices in new events; avoid a vague `index`.
- Emit already-materialized scalar values; do not introduce accidental device synchronization inside hot kernels beyond what the refinement loop already needs.

## Design principles that constrain every change

**`result = pure_fn(config)`.** The recorded config must completely describe what ran on the default app/CLI path. Config carries only what changes the result there. Execution-only knobs — `device`, `workers`, checkpoint reuse/refresh, output paths, loggers — are function arguments or CLI execution options, never config fields and never part of the lock. There is no third category: a knob that changes the result but is not recorded is a hidden input and is forbidden. Never add a CLI override of a config value such as `--steps` or `--lr`; it makes the recorded artifact lie. A smoke run is a separate experiment directory with its own data slice, never an unrecorded override on the real one.

**Config mirrors the value-types, and fails fast.** Each config block is a 1:1 edge onto a frozen `specs` value-type that owns its defaults and validation. Pydantic models use `extra="forbid"`, so unknown keys fail fast; a config field with no consumer is forbidden. Defaults live once on the value-type. Mutually exclusive options are a discriminated union, not many nullable fields. A physical quantity shared by two consumers is modelled once and composed into both, never copied into two homes. No config-string reflection, no `getattr` registry, no pipeline-as-config program; if config toggles a stable method, map validated flags into a canonical, code-owned recipe order.

**Scientific composition is typed-Python-first.** A new constraint, penalty, component, stage, schedule, or multi-stage workflow is a typed API parameter passed to builders. Promote it into config only once the default recipe commits to it as stable public behaviour. Config that switches on a string to compose a program is the anti-pattern. Do not build declared/serialized/hashed refinement-recipe infrastructure until refinement outputs actually need caching.

**The typed I/O boundary is one-directional.** `.cif` and `.cif_pets` parse into validated typed records at the `io` boundary before any numerical code sees them; no raw parser or vendor object crosses into `core`. Symmetry data is extracted at the boundary so the core stays free of crystallography libraries.

**Plans are complete, self-describing, immutable values.** No optional field encodes a phase; a phase is a type (`CandidatePlan` before geometry is built, `OrientationPlanLike` after), re-parsed at runtime boundaries via `require_candidate_plans` / `require_built_plans` / `require_orientation_plans`. A `Plan` is always complete for its phase. Ordering lives in the recipe (`pipeline([...])`) and in total construction — never in a `completed_steps` history. Provenance is write-only identity; never branch on it to decide what data exists. Ask whether the value you need is present by type/phase.

**Recipe identity keys on recordable inputs, never the control-flow trace.** A data-directed combinator may read only a value that is pipeline-invariant (the same at every step) and read it purely — that is why `fork`'s predicate reads the `StructureFactorGrid`, not the `Plan`. A step's result-changing spec goes into its record; execution-only kwargs stay out. A bare closure with no recordable spec stamps `OPAQUE` and forces a safe checkpoint miss.

**Methods are compositions of swappable typed units.** `SolverMethod`, `LossFn`, each `Plan -> Plan` step, the tilt reduction, constraints, penalties, and components should be named typed units that can be composed in or out and tested independently. No config-string reflection, no registry, no god-object with flags.

**SOLVE and SCORED reflection sets stay independent.** The SOLVE set (`beam_hkl`) is which beams couple dynamically in the Bloch solve; the SCORED set (`alignment`/`pattern`) is which reflections enter the R-factor. You can only score what you solved (scored ⊆ solved), but they are separate concerns: a coupling change must not drift the scoring set, and widening the solve basis must not silently change the objective domain. `g_max` is one bare name whose value depends on its owning object, and those values are not interchangeable: scored ⊆ solve ⊆ structure-factor support, with support covering beam differences (`hkl_j - hkl_i`), roughly twice the solve cutoff. In serialized keys and events (no owning object) always scope the name — never emit a bare `g_max`.

**Let real data referee the physics.** A change that looks geometrically plausible but makes agreement with measured R worse is almost certainly wrong — reconstruct the physical experiment the code models before calling the old behaviour a bug. Treat the tilt reduction as a first-class typed choice, not an afterthought: matching structure factors and per-tilt solves does not guarantee matching integrated intensities. Profile before adding performance infrastructure.

## The `Plan -> Plan` pipeline

A `Plan` is the geometry/data scaffold around the differentiable structural parameters — not the crystal structure being refined. A settled `Plan` owns or references the shared `StructureFactorGrid` / `Fgb` support, per-rotation `OrientationPlan`s (or coupled plans), solve beam sets, scored/observed/matched hkl alignment, rocking-curve tilt samples around the goniometer sweep, fitted nuisance values such as per-rotation thickness, and ordered provenance records.

- **Step types** (`preprocess/pipeline.py`): `type PlanStep = Callable[[Plan], Plan]`, `type ConvergenceCheck = Callable[[Plan, Plan], bool]`, `type StatefulPlanStep[State] = Callable[[Plan, State], tuple[Plan, State]]`, `type StateInitializer[State] = Callable[[Plan], State]`.
- **One focused thing per step.** A step selects beams, builds orientation plans, integrates a rocking curve, applies mosaicity, fits orientation, fits thickness, couples beams, converges numerics, or another similarly narrow task. It returns a replaced immutable value.
- **Self-provenance.** A checkpointable step is a factory returning `as_step(name, spec, run)`. Only the result-changing `spec` enters the `StepRecord`; execution-only kwargs stay out. `pipeline` owns provenance stamping — steps preserve existing provenance and do not append records themselves.
- **Combinators.** `pipeline(steps)` folds left-to-right; `iterate_until(step, until=...)` is the fixpoint; `fork(predicate, when_true=, when_false=)` with `resolve_recipe(steps, grid)` is the static branch; `stateful_pipeline`/`stateful_plan_step` thread an explicit immutable state between phases and drop it at the `Plan -> Plan` boundary.
- **Phase narrowing.** A step that only makes sense at one phase asserts it at entry with a `require_*` guard — a tilt-independent-only step must precede `couple_beams` and enforce ordering with a clear boundary error.

### Candidate vs built plan

`CandidatePlan` is source-only and intentionally unsolvable: candidate beams and experimental data without built Bloch geometry. `build_orientation_plans` is the declared build boundary that constructs the gathers/beam plans.

- A `Plan` is simulatable only after the build phase; never encode the phase with nullable fields, optional gathers, or an `is_built` flag.
- Never build `StructureFactorGather` over the full candidate pool. Build gather geometry only over the active solve set or the coupled SOLVE union. This is the large-cell out-of-memory lesson: the candidate pool can be far larger than what is solved.

## Refinement composition

Refinement holds a settled `Plan` fixed and optimizes selected model leaves against it. The objective assembles in this order:

```text
raw RefinableParams
  -> crystallographic constraints (ConstraintSpec / constrain)
  -> PhysicalState
  -> molecular hard constraints (ConstraintTransform)
  -> diffraction term + soft penalties
  -> scalar ObjectiveValue
```

Decision tree for what a change is:

```text
Provides a value to the forward model?  -> component        (engine/components.py)
Enforces an invariant exactly?          -> constraint       (params.constrain OR engine/constraints.py)
Adds a soft objective cost?             -> penalty          (engine/penalties.py)
Selects which optimizer leaves vary?    -> trainable spec   (TrainableSpec / AtomSelection)
Changes the discrete problem setup?     -> a Plan -> Plan step
```

- **Refinable parameters** (`params.py`): `RefinableParams` are the raw, unbounded optimizer variables; `constrain(params, spec)` is the crystallographic hard-constraint layer (site-symmetry projection, ADP equalities, positivity/[0,1] bounds) producing the bounded `PhysicalState`.
- **Two hard-constraint homes, deliberately separate:** crystallographic constraints live inside `constrain`; molecular constraints are `ConstraintTransform`s (`engine/constraints.py`) applied to `PhysicalState` after `constrain`, in tuple order, duplicate names rejected, only on the refinement objective path — do not silently alter the inference/scoring path. `HydrogenRiding` is the worked example: constant parent-to-H offset, general-position H only, H stays in scattering, H rows excluded from position/ADP leaves, H ADP rides from the parent with a scale. Hydrogen riding is a constraint, not a component: it rewrites the physical state, it does not supply an independent forward value.
- **Soft penalties** are `PenaltyTerm`s (`engine/penalties.py`): additive objective terms with a `name`, `weight`, and `value(state) -> scalar`. The raw value is the scientific diagnostic; weighting happens through `ObjectiveComponent`. A penalty owns its own invariant context (pairs, targets, sigmas, metric) rather than bloating `PhysicalState`.
- **Components** implement the `ModelComponent` protocol (`engine/forward.py`), keyed and trainable, supplying values through `ForwardContext` (currently `thickness`), e.g. `ApparentThicknessNN`, `PerOrientationThickness`, `QuadraticThicknessProfile`. Component behaviour and component parameter tensors are separate: the component owns behaviour, `component_params[component.key]` owns the trainable tensors.
- **Assembly:** `build_refinement_model(initial=, constraints=, components=, component_params=)`, `build_refinement_problem(penalties=)`, and `run_refinement_model(engine, model, problem, trainable=, steps=, optimizer=, lr=)`. `TrainableSpec` + `AtomSelection` choose which leaves vary. The optimizer shell knows nothing of bonds, hydrogens, special positions, or thickness networks — it calls a scalar objective over selected leaves. `run_refinement`/`run_refinement_model` never mutate the caller's params.

Bounds and activation choices for advanced components are typed API values on the component, never global optional config soup.

## Adding a feature: derive it from the principles

There is no per-feature cookbook, and none is wanted: the sections above give the vocabulary and the homes; the principles decide the rest. For any change — including a kind not named here — work it through these questions in order. The answers determine what you build and where it goes.

1. **Does it change the result?** If no, it is an execution knob — a function argument or CLI execution option, out of config and out of the lock. If yes, it must be recorded where the artifact captures it.
2. **Is it a stable knob of the default path, or scientific composition?** A stable default-path knob becomes a config block that parses into a frozen value-type. Scientific composition is a typed Python API parameter first, promoted to config only when the default recipe commits to it.
3. **What is it, structurally?** Supplies a forward-model value -> component; enforces an invariant exactly -> constraint; adds a soft cost -> penalty; selects optimizer leaves -> trainable spec; changes discrete geometry/problem setup -> `Plan -> Plan` step. Match the nearest existing mechanism before inventing one.
4. **Where does coordination state live?** On the `Plan`, one shot -> a step; on the `Plan`, iterated to a fixpoint -> `iterate_until`; an alternative fixed by a pipeline-invariant input -> `fork`; transient driver state that should not pollute the `Plan` -> `stateful_pipeline` adapted back with `stateful_plan_step`.
5. **Is its identity recordable?** A checkpointable step needs a stable result-changing spec; execution-only kwargs stay out. If identity cannot be reduced to a recordable spec, stamp `OPAQUE` and accept a safe checkpoint miss.

Uniform mechanics, whatever the answers:

- A tunable knob gets a frozen value-type in `specs.py`, validating in `__post_init__` and exported from `__all__`; this is the single home for its defaults and validation.
- A `Plan -> Plan` step is a factory returning `as_step(name, spec, run)`, with `run` returning `dataclasses.replace(plan, ...)` and narrowing its phase with a `require_*` guard.
- A refinable parameter earns a raw field on `RefinableParams` only if it is truly a structural leaf. Otherwise it is a component. Move the tensor in `RefinableParams.to`; produce its physical form in `constrain`; extend `_TRAINABLE_FIELDS`, `TrainableSpec`, and seeding in `RefinementSetup.from_structure`; check device/dtype co-location through the params device.
- A component implements `ModelComponent` with a unique `key`; a constraint implements `ConstraintTransform`; a penalty implements `PenaltyTerm`. Each usually ships a `perceive_*` / `with_*` builder and consumes explicit value data, such as a plan-derived thickness, never a live engine or CLI runner. A component that supplies a new forward value widens `ForwardContext` deliberately and is consumed in the forward. Pair a dependent-coordinate constraint such as H-riding with a trainable-selection change so derived atoms are not also optimizer leaves.
- Every new public name joins its layer's `__all__`. A step joins the default path only by being wired into `app/program.py`; if serialized `Plan` state changes, bump `_FORMAT_VERSION` in `preprocess/serialize.py` so old checkpoints are rejected rather than misread.
- Keep the addition pure; give it a self-sufficient docstring; add gradient-routing, no-in-place, and objective/physics-invariance tests for anything on the forward path; then verify as below.

Worked example: a bond-length penalty changes the result, is scientific composition rather than a stable default-path knob, and adds a soft cost. Therefore it implements `PenaltyTerm` with a `perceive_*` builder and is passed through `build_refinement_problem(penalties=...)`: no config field, no new combinator, no `Plan` change.

## Performance and caching

- The expensive object is the dense Bloch structure matrix for `N` beams: matrix assembly is roughly `N²` and dense solves are worse, so beam count dominates cost. Avoid repeated gather construction and unnecessarily large union beam sets.
- Reuse one shared `StructureFactorGrid` per `Plan`.
- Precompute `StructureFactorGather` lookup geometry and `BeamPlan` constants; do not cache differentiable structure-matrix values that can go stale. `Fgb` remains live.
- For coupled rocking-curve orientation trials, reuse segment beam sets, tilt coverage, union alignment, and per-segment gathers when only orientation-dependent quantities change.
- Adaptive union coupling should split only where the active beam set changes enough to justify the cost.
- Profile before adding performance infrastructure: the anchor tells you when you have changed the science, a profile tells you where the time actually goes.

## Reproducibility and checkpoints

- `experiment.lock` pins the input CIF/PETS bytes.
- `plan.<stem>.npz` stores one dataset's settled preprocess checkpoint (stem = its exp_data ref, path separators as `__`, no `.cif_pets` suffix); `plan.<stem>.lock` ties it to that dataset's input bytes, the dataset-scoped config digest, its file-local ignored rotations, the ordered recipe, the code version/release gate, and the artifact hash.
- Locks verify the identity of the preprocessed starting point. Do not claim hardware-independent optimizer determinism: floating-point trajectories are not guaranteed across hardware.
- If a serialized `.npz` key or format changes, bump the plan format version; never silently reinterpret old checkpoints. Reading old formats is out of scope unless a feature explicitly adds a migration reader.
- A recipe containing `OPAQUE` cannot be reused safely. Force recompute rather than false reuse. Reuse/resume/stale logic stays append-only and conservative: exact recipe -> reuse, proper prefix -> resume, otherwise stale.

## Public API and code conventions

- The canonical usage shape is load inputs/config -> preprocess/build or read a `Plan` -> build an engine -> compose model/problem -> run inference/refinement -> inspect typed results/events.
- Prefer flat, layer-level public imports in examples and docs: `from diffBloch.engine import build_refinement_model`, not deep module paths, unless documenting that module. Each layer's `__init__.py` owns its public `__all__`; new public names go there.
- Use Python 3.12+ syntax where the project already does, including `type` aliases and PEP 695 generics. Keep ruff/mypy strictness intact; do not silence a check without a narrow explanation. Environment and commands run under uv.
- Docstrings are self-sufficient: what the object does, what it is for, important failure modes, and key tradeoffs. Cite papers inline where the implementation depends on literature; the fuller list is `REFERENCES.md`.
- Keep generated files, checkpoints, examples, and docs out of source refactor commits unless the family explicitly requires them.

### Naming new symbols

This document is the naming guide; do not depend on files outside the repository to choose names.

- Name the result, not the mechanism. Public dataclasses, config fields, event keys, and serialized keys get explicit names; locals inside focused math kernels may use conventional short forms (`sf_hkl`, `fgb`, `mii`, `sg`).
- Use `StructureFactorGrid`, not `ScatteringGrid`, for the shared support grid; use `SolverMethod`, not `DynamicalSolver`, for the solver-method literal; keep `Fgb` as the structure-factor convention; prefer `rotation_index` for experimental frame/rotation indices.
- `beam_hkl` means the solve basis. Use hkl/reflection vocabulary for scored/observed/matched sets; do not call scored reflections “beams” unless they are actually solve beams.
- `g_max` may be bare only when the owning object scopes it. Public metric/event/serialized keys must qualify radius/cutoff meanings when the owner is absent.
- Keep these names unless a deliberate family refactor changes them: `Plan`, `CandidatePlan`, `OrientationPlan`, `Step`, `StepRecord`, `Fork`, `OPAQUE`, `RefinableParams`, `PhysicalState`, `ConstraintSpec`, `constrain`, `RefinementEngine`, `LossFn`, `Fgb`.
- Rename semantic families together, not one isolated symbol. Make `src/` coherent by family before updating examples/docs/fixtures.

## Product and docs conventions

- Spell the project name `diffBloch`, never `DiffBloch`.
- Prefer plain project language; do not introduce external-language analogies in code comments or docs unless explicitly requested.
- User-facing quickstarts use fresh-checkout `uv` commands, not `just` recipes:

  ```bash
  uv sync --dev
  uv run diffbloch validate examples/Colmey_et_al_2026_Acta_Cryst_A/data/quartz-no-abs/experiment.yaml
  uv run diffbloch refine examples/Colmey_et_al_2026_Acta_Cryst_A/data/quartz-no-abs
  ```

  The six experiment directories under `examples/Colmey_et_al_2026_Acta_Cryst_A/data/` are the only
  runnable examples; quartz is the smallest and cheapest. None ships a preprocess checkpoint, so a
  first run settles the `Plan` from raw inputs before refinement starts.

- Documentation snippets are runnable where practical; if an example is only API shape, say so.
- Documentation must not overclaim reproducibility: locks verify identity, not floating-point trajectories.
- When explaining input data, say rotating-stage 3D electron diffraction / continuous-rotation 3DED where useful. PETS2 is upstream data reduction; diffBloch consumes its `.cif_pets` experimental data plus a starting structure `.cif`.

## Verify a change

Choose the smallest validation set that covers the change, but do not skip type/lint checks for source changes.

Useful direct commands:

```bash
uv run ruff check src/diffBloch tests
uv run mypy src/diffBloch
uv run pytest -q
uv run sphinx-build -W -b html docs site
```

Targeted examples:

```bash
uv run pytest tests/unit/test_pipeline.py tests/unit/test_driver.py -q
uv run pytest tests/e2e -q -m e2e
uv run diffbloch validate examples/Colmey_et_al_2026_Acta_Cryst_A/data/quartz-no-abs/experiment.yaml
uv run diffbloch refine examples/Colmey_et_al_2026_Acta_Cryst_A/data/quartz-no-abs
```

Change-type guidance:

| Change | Minimum checks to consider |
|---|---|
| Core numerical/objective change | Unit oracle or invariance test, gradient finite/routing test, dtype/device checks, relevant e2e anchor. |
| Preprocess step | Phase error tests, provenance tests, behavior tests, checkpoint/digest tests if the settled `Plan` changes. |
| Config change | Validation tests, default/value-type parity, digest-scope test if preprocessing identity changes. |
| Refinement constraint/penalty/component | Scalar/objective tests, gradient routing, non-mutation, device/dtype behavior. |
| Serialization change | Round-trip test, format-version bump, stale/reject behavior for old incompatible data. |
| Docs-only change | Strict Sphinx build. |

The quartz anchor is the scientific-safety contract and stays green at every commit. Any change on the objective or forward path should add a focused physics-invariance test; an end-to-end aggregate can pass while the numerical objective drifts, so the anchor alone is necessary, not sufficient. Judge parity per reflection under a global scale, not only by aggregate R.

Moving the anchor is deliberate, not incidental. A legitimate physics change must be isolated, justified against measured experimental R, and re-blessed through the project’s promote/regeneration path — never by hand-editing a fixture to accept a new number.

For docs-only changes, run strict Sphinx. Remove generated `site/` before committing unless explicitly publishing it. Do not commit `__pycache__`, temporary profiling scripts, generated notebook output, or large artifacts unless the change explicitly requires them.

## In-repository references

This guide stands alone; do not cite files outside the repository as implementation authority. For more detail, consult in-repo docs and current source:

- `docs/architecture.md` — user-facing architecture.
- `docs/inputs.md` — experiment directory, CIF/PETS inputs, typed records.
- `docs/preprocessing.md` — `Plan`, beam sets, rocking curve, composition.
- `docs/refinement.md` — infer/refine, constraints, penalties, components.
- `docs/reproducibility.md` — lock/checkpoint guarantees and limits.
- `docs/observability-guide.md` — event/logger behaviour.
- `docs/examples.md` — runnable experiments and paper-style composition examples.
- `src/diffBloch/**` — final authority for current behaviour when docs and source disagree.
