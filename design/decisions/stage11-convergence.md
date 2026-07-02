# Decision: convergence testing — `converge_*` steps, composition, and the value-type convention

**Status:** accepted (slice 11, pre-implementation).
**Context:** before refinement, the preprocess pipeline must pick simulation-accuracy
hyperparameters (reciprocal-grid extent, per-orientation beam-set inclusiveness, rocking-curve
integration density) that are *converged* — large enough that the simulated pattern no longer moves
when they grow, but no larger (accuracy vs cost). This is distinct from the `fit_*` steps, which
match the simulation to *observed* data.
**Reference:** `diffBloch_private` branch `pattern-vis-convergence-testing` (`14383cf`),
`programs/convergence_testing.py` + `configs/convergence_test/base.yaml`. The private is the
authoritative source for the algorithm; this records the faithful 2.0 shape.

## The metric is self-stability, not fit-to-data — so the verb is `converge_*`, not `fit_*`

`fit_orientation` / `fit_thickness` minimise a weighted R-factor of **simulation vs observation**.
Convergence testing instead measures **simulation vs the previous simulation**: increase a knob,
re-simulate, and compare the new pattern to the one before it. When the change between two
*consecutive* simulations falls below a threshold, that knob has converged. The private computes
this as `rbragg_abs` averaged per-orientation between the two simulation tables
(`_compute_step_rfactor`, faithful to `core.losses.optimal_scale` with
`r_value_method="rbragg_abs"`), with the threshold `r_factor_threshold = 0.005`.

This is a different *kind* of comparison than the existing scoring path (which is
simulation-vs-observed wR2), so it needs a **simulation-vs-simulation R-factor** — a new, small
function, not a reuse of `score_orientation`.

### Why comparing a simulation to a *previous simulation* is valid

Convergence testing does not ask *"is the pattern more accurate?"* — accuracy needs observed data
and is the refinement's job. It asks *"has the calculation stopped depending on this numerical
knob?"* — a standard resolution / truncation study. Beam count and rocking-curve sampling are
numerical truncation parameters: a Bloch calculation with too few beams is *wrong*, but wrong from
under-resolution, not from a bad model. Growing the knob and watching the pattern stop moving
locates the regime where truncation error is below tolerance, independent of experiment. The two
questions are orthogonal and sequential: converge the numerics first (sim-vs-sim), so that the
residual the refinement later minimises (sim-vs-observed) is model error, not truncation noise. A
converged simulation can still be *wrong* (converged ≠ correct); resolving that is refinement, not
convergence.

## The knobs consolidate: the private's `g_max`/`sg_max` are two levers on *one* quantity

The private sweeps three knobs (`g_max`, `sg_max`, `tilt_steps`) as three independent `optimize_*`
passes. Porting them showed that framing is a *false independence*: in 2.0 they do not correspond to
three independent cost axes, because 2.0 separates concerns the private conflates. Applying
"decompose by coupled home, not false independence", the corrected model has **one** beam concern
(with two coupled levers) plus a separate, deferred sampling axis.

- **The reciprocal-grid `g_max` is not a 2.0 convergence knob.** The Bloch structure matrix is
  a gather `A[i, j] = F(g_j - g_i)` over the *active beam pairs*; the grid only has to *cover the
  beam differences* (~2x the beam `g_max`). Growing the grid `g_max` past that bound adds `Fgb`
  entries that are never gathered — a strict no-op that would "converge" trivially at R = 0. The
  private's `g_max` sweep only bites because *its* grid **is** the beam source (`filter_hkls` draws
  beams from the structure-factor grid); 2.0 splits the `Fgb` support grid from the active beam set,
  so grid extent is *sized-to-cover*, never converged. (Recorded in `DIVERGENCE.md`.)

- **Both private beam knobs are levers on one quantity — beam-set inclusiveness — and they are
  coupled.** The active set is `seed(g_max_refine) ∩ Klar-window(integration_semiangle)`:
  - `g_max` (private candidate pool) -> **`g_max_refine`**, seed radius `from_experiment` uses to
    lay down each orientation's candidate reflections.
  - `sg_max` (private excitation-error cutoff in `filter_hkls`) -> **`integration_semiangle`**, the
    `BeamSelection` field that sets the Klar window `sg_max = |g_transverse|*deg2rad(semiangle)` in
    `klar_beam_mask`; growing it monotonically widens the excitation-error window and admits more
    near-Ewald beams.
  Because the set is the *intersection*, each lever is bounded by the other — widening the window
  admits nothing the pool has already clipped, and vice versa. That mutual bounding is the whole
  reason the private needs its multi-pass revisiting, and the reason 2.0 files both levers under one
  beam-inclusiveness concern reached by a cross-lever fixpoint, rather than as two "independent"
  axes. It is the same principle as `select_beams` owning the whole active-set selection.

- **Rocking-curve sampling (`tilt_steps`) is a genuinely separate axis — and has nothing to converge
  yet.** It drives a rocking-curve integration (the pattern averaged over many beam tilts) that
  **2.0's forward model does not implement**; `rocking_curve_sampling` is an unused config field
  today. `converge_sampling` therefore waits on that forward-model feature, recorded as a subsequent
  task in `ROADMAP.md`.

### First step — `converge_beams` sweeps the Klar window (`integration_semiangle`)

`converge_beams` grows **`integration_semiangle`** from its starting value by a fixed step,
re-selecting each orientation's beams (reusing `select_beams`) and re-simulating, comparing each new
simulation to the previous with `simulation_converged`, until the stopping rule below is met or the
`max_iterations` cap is hit. It is the direct analogue of the private's `sg_max` sweep and the
physically-primary "how many near-Ewald beams" lever — reusing `select_beams` +
`simulation_converged`, no new machinery.

The **pool lever** (`g_max_refine`) is the coupled second beam knob: growing the seed radius
re-seeds the candidate reflections and — to keep the `Fgb` difference-support valid — may grow the
grid `g_max` as a *dependent sizing* constraint. It is deferred until the window sweep is
calibrated; when it lands, the two levers are driven to a joint fixpoint together (block coordinate
descent, below).

### Stopping rule — established convergence utilities, not the private's fixed-step stop

The private stops the first time the consecutive-simulation R-factor drops below threshold. That has
a real premature-termination failure: `integration_semiangle` is continuous but the beam set is
*discrete*, so two increments can yield the **same** beam set, an identical simulation, R = 0, and a
false "converged" — even though a larger increment would still admit beams. 2.0 does not replicate
the flaw; it corrects it with standard convergence utilities (recorded in `DIVERGENCE.md`):

- **Skip null steps.** "Improvement" is only defined when the active beam set actually changes; an
  increment that leaves every orientation's set unchanged is not an evaluation — keep growing the
  angle until the set changes, *then* compare. This removes the R = 0 plateau at its source (the
  discrete-knob analogue of only measuring where the model can move).
- **Patience.** Across *real* changes the R-factor need not be monotone, so — as in early-stopping
  *patience* — require R below threshold for **`patience` consecutive** changed steps before
  declaring convergence, not one dip. This targets the *asymptotic range* (where more beams stop
  mattering), the same idea grid-convergence studies (Richardson / the Grid Convergence Index)
  formalise for mesh refinement.
- **Hard cap.** `max_iterations` bounds increments; exceeding it raises rather than returning a
  silently non-converged plan (the `iterate_until` posture).

This makes composition **block coordinate descent**: each lever is swept to its own fixpoint and
the sweep repeats until it leaves every lever unchanged — the textbook cyclic-coordinate stopping
rule. **Note (superseded mechanism):** the paragraphs below originally expressed this descent as
`iterate_until(pipeline([...]))`. Landing the pool lever showed that naive composition does not
work (seed/pruned mismatch + shared scalar state the `Plan` does not carry), so the cross-lever
fixpoint is instead assembled by the **preprocess driver** — see
`design/decisions/stage11-cross-lever-fixpoint.md`. The descent *model* here is unchanged;
only *where it is assembled* moved. The stopping-rule parameters (`r_factor_threshold`, `patience`,
`max_iterations`) are the invariant bundle carried by **`ConvergenceTolerance`**; `patience`'s
default is a calibration target (`KNOWN_ISSUES.md`), like `max_iterations`. See `REFERENCES.md` for
coordinate descent, early-stopping patience, and the Grid Convergence Index.

### Build vs adopt a sweep framework

The sweep is expressed with the project's own `pipeline` + `iterate_until` combinators plus a small
reusable `converge_scalar` higher-order component (skip-null + patience + cap, written once and
shared by the window lever, the pool lever, and later `converge_sampling`) — **not** a
hyperparameter-optimization framework (Optuna, Ray Tune, Ax, scikit-optimize, Weights & Biases
Sweeps, the Hydra sweeper). Those tools solve a *different* problem: black-box optimization that
*samples* a search space (random / TPE / Bayesian) to minimise an objective, with pruners to abandon
poor trials, at the cost of 4–11 transitive runtime dependencies (and, variously, a study database,
a cloud service, or a distributed runtime). Convergence testing is not optimization: the knob grows
*monotonically* until the output stops changing (a fixpoint at diminishing returns), there is no
objective to search, and the structure is fully known — sampling would be pointless. Adopting one
would also contradict the project's deliberate minimal-dependency, anti-Hydra posture. We borrow the
*concepts* (pruning/patience, coordinate descent, the GCI asymptotic range) and cite them; we do not
take the dependency.

## The convergence engine is a higher-order component (parameter-agnostic)

The loop knows nothing about beams. Its essence is *propose a clicked knob, rebuild the Plan,
re-simulate, and compare against the previous simulation* — so it is written once as a higher-order
component and reused for any parameter:

    converge_scalar(build, *, start, step, stabilized, tolerance) -> PlanStep

- **`build: value -> Plan`** rebuilds the Plan at a knob value (for beams,
  `select_beams(replace(selection, integration_semiangle=value))`). `start` / `step` seed and
  *click* the value; a negative `step` clicks *down*.
- **`stabilized: (previous, candidate) -> bool`** is the in-loop check against the previous
  simulation (`simulation_converged`). It is what makes the loop wait for the *diffraction pattern*
  to settle, independent of which knob moved.
- **`tolerance`** carries `patience` + `max_iterations`; skip-null + patience + cap live in-loop.

`converge_beams` is a small adapter that binds `build` to the beam re-selection; the pool lever,
`converge_sampling`, and any other experiment-config scalar are further adapters supplying their own
`build`. **Independently vs together** and **up vs down** are not special cases — they are just how
the click is supplied:

- **one param, up:** `converge_scalar(build, start, +step, ...)`.
- **one param, down (minimal-sufficient):** `step` negative.
- **several params together:** a `build` over a small parameter *tuple* clicked by a step *vector*.
- **several params independently:** separate `converge_scalar` steps driven as coordinate descent
  (the cross-lever fixpoint). This was originally sketched as `iterate_until(pipeline([...]))`; that
  naive composition is superseded — the fixpoint is assembled by the preprocess driver (see
  `design/decisions/stage11-cross-lever-fixpoint.md`).

**Guardrail (typed closures, not config reflection).** The genericity comes from *higher-order
functions over typed closures*, exactly the `Plan -> Plan` / combinator idiom in the codebase
— **not** a stringly-typed engine that introspects a config by key path and mutates it. Each `build`
is a small, explicit, type-checked function the caller writes; the convergence engine stays a pure
combinator with no knowledge of the config schema, keeping the value-object vocabulary and the
"no `DictConfig` in the core" posture intact.

## Composition — the driver assembles the cross-lever fixpoint (block coordinate descent)

Today only `converge_beams` (the window lever) and `converge_pool` (the pool lever) exist, each
self-contained. The composition below — block coordinate descent — activates once they are driven
together by the preprocess driver:

> **Implementation note (slice 3, pool lever landed).** The joint fixpoint is **not** a naive
> `iterate_until(pipeline([converge_beams, converge_pool]))` — `converge_beams` re-selects from an
> *unpruned* seed while `converge_pool` emits a *window-pruned* Plan, and the two levers share
> scalar state (`integration_semiangle`, `g_max_refine`) the `Plan` does not carry. The cross-lever
> fixpoint therefore lives in the **preprocess driver**, which holds that state explicitly. See
> `design/decisions/stage11-cross-lever-fixpoint.md` for the full reasoning.

- **Partial order (sizing dependency):** the grid must contain any beam a wider pool keeps, so
  growing `g_max_refine` implies a grid-`g_max` *sizing* step **before** re-selection.
  `converge_sampling` is independent of the beam levers.
- **One pass** orders the levers (window, then pool).
- **Cross-lever fixpoint:** because the pool and window levers are coupled (each bounded by the
  other), widening one can leave the other room to grow, so the suite must revisit. The private
  hard-codes `num_passes = 2` and *varies order* between passes (pass 2 leads with `tilt_steps`).
  2.0 expresses the revisit as a fixpoint over the whole pass — driven to stability rather than a
  fixed count. This is a **two-level fixpoint**: each lever converges internally, the driver
  converges across levers. (Originally sketched as `iterate_until(pipeline(...))`; that naive
  composition is superseded — see `design/decisions/stage11-cross-lever-fixpoint.md`.)

### Decision — fixpoint-until-stable (chosen), assembled by the driver

The private's "two passes, with the order changed on the second" is an empirical detail with no
stated principle; the order-variation in particular looks like a hand-tuned heuristic. 2.0 drives
one ordered pass to a genuine cross-knob fixpoint — the suite repeats the ordered pass until a whole
pass leaves every knob unchanged (or the `ConvergenceTolerance` cap raises). This **generalises**
the private's fixed count and is recorded as a deliberate generalization in `DIVERGENCE.md` (like the
`fit_orientation` iteration cap), and removes the unprincipled per-pass order-swap.

The *mechanism* is the **preprocess driver**, not a `pipeline` composition: the pool lever proved
that `iterate_until(pipeline([window, pool]))` cannot compose (seed/pruned mismatch + shared scalar
state the `Plan` does not carry), so the driver holds the unpruned candidate pool and the two live
scalars as explicit coordinate-descent state and threads each lever's settled value into the next.
See `design/decisions/stage11-cross-lever-fixpoint.md`.

## Two operations are two *kinds* of objective — a discriminated union, not a mode flag

The private also has an `initial_minimum_param_sweep`: it grows the same three knobs but accepts a
candidate only when it **increases the count of matched experimental reflections** (coverage), not
when consecutive simulations stabilise. That is a different objective (match-count vs
self-stability) and a different stopping rule, so it is a **separate step**, not a flag on the
convergence steps.

The private dispatches on `convergence_test.operation ∈ {initial_minimum_param_sweep,
hyperparams_optimization, both}`. In 2.0 this is a **discriminated union** on the operation
(a `Literal` + per-operation config block, never coexisting optional fields), faithful to the
private dispatcher. `both` is simply a `pipeline`: the coverage sweep first, its converged
parameters handed to the self-stability suite as its starting point (the private's
`_run_initial_minimum_param_sweep` → `_run_hyperparams_optimization(initial_*=...)` handoff).

## The self-convergence check is a value, and the tolerance is a value-type

The stopping rule compares a step's (previous, just-produced) Plans, so it is a
`ConvergenceCheck = (Plan, Plan) -> bool` — the type `iterate_until` already takes. Its parameters
(`r_factor_threshold` and the `patience` + `max_iterations` / `MAX_SWEEP_ITERATIONS` caps) are a
small invariant bundle that crosses into the algorithm, so by the convention below they become **one
frozen value-type, `ConvergenceTolerance`** (`specs.py`), validated once at construction
(`r_factor_threshold > 0`, `patience >= 1`, `max_iterations >= 1`). This also folds in
`iterate_until`'s lone
`max_iterations` guard — the
borderline scalar flagged in the value-type audit — giving it a proper home.

## The value-type convention this records (systemic)

The convergence steps are *born into* the convention settled across `f5b0676` / `5fc7250`: **every
config-derived parameter bundle that crosses into an algorithm arrives as one validated frozen
dataclass value-type** (`HexagonalSearch`, `ThicknessGrid`, `BeamSelection`, and now
`ConvergenceTolerance`). The rules:

- **Parse, don't validate.** Each value-type's `__post_init__` is the *single* home of its
  invariants, so an invalid spec is unrepresentable and the pure step never re-validates — the same
  posture `io` already earns its keep on.
- **Plain frozen dataclass, not pydantic.** The codebase has one value-object vocabulary (every
  algorithm-facing value — `ScatteringGrid`, `OrientationPlan`, `RefinableParams` — is a plain
  frozen dataclass), and the algorithm contract / public API stays decoupled from pydantic's version
  cycle. "No pydantic in core" is its real target: the differentiable kernel and the plain-values
  algorithm contract, not the config edge.
- **Config parses into the value-type.** The pydantic block at the YAML edge holds the same fields
  and a `to_*()` method that constructs the value-type, plus a `model_validator` that calls it so an
  invalid config fails fast *at load*. One rule home, no drift between config and function. The
  value-types live in `specs.py` — a leaf both `config` and the algorithms import (mirroring
  `params.py`), which avoids a `config → preprocess` import cycle.
- **Boundary-vs-math line.** The convention reaches only *config-derived parameter bundles with
  invariants*. Pure math/physics functions whose scalar **is** the argument (`excitation_errors`,
  `resolution_cutoff`, `hexagonal_tilt`, `klar_beam_mask`) keep their float signatures — wrapping
  them would be noise.

### Rollout audit (state of adoption)

| function | parameters | status |
| --- | --- | --- |
| `fit_orientation` | `HexagonalSearch` | done (`f5b0676`) |
| `fit_thickness` | `ThicknessGrid` | done (`f5b0676`) |
| `select_beams` | `BeamSelection` | done (`5fc7250`) |
| `converge_beams` | `BeamSelection` + `ConvergenceTolerance` | this slice |
| `klar_beam_mask`, `excitation_errors`, `resolution_cutoff`, `hexagonal_tilt`, … | scalars | not candidates (pure math) |

The convergence steps complete the convention: after them, every config-derived parameter bundle
crossing into an algorithm is a validated value-type, and the remaining loose scalars are all
legitimately-pure-math arguments.

## Validation failures: raise now, `Result` at the boundary later

The value-types raise (`ValueError` / `pydantic.ValidationError`) on bad input. That is correct
*today*: the only callers are config-load and direct construction, both of which fail fast. When the
`app` layer (stage 12) needs to surface validation errors **as values** — to a TUI or batch runner
that reports rather than crashes — a thin boundary adapter `parse(raw) -> Result[Spec,
ValidationError]` will wrap these raising constructors. `Result` stays confined to that boundary
adapter and never enters a step; the pure core only ever sees the unwrapped `Ok` value. This is the
same errors-as-values-to-the-shell posture as `design/decisions/effects-and-observability.md`.

## Sequencing

1. `ConvergenceTolerance` value-type + the sim-vs-sim R-factor check (`simulation_converged`).
   **Done** (`6f3d694`, `1b5681e`) — `patience` field lands with `converge_beams`.
2. `converge_scalar` HOF + `converge_beams` — the `integration_semiangle` window sweep
   (skip-null-steps + patience + cap).
3. The coupled pool lever (`g_max_refine` + dependent grid sizing). **Done** (`10fd0f9`,
   `converge_pool`). The cross-lever fixpoint is re-homed to the driver — see
   `design/decisions/stage11-cross-lever-fixpoint.md`.
4. `converge_sampling` — waits on rocking-curve integration in the forward model (`ROADMAP.md`).
5. The coverage `initial_minimum_param_sweep` step (match-count objective).
6. The operation discriminated union + the `both` pipeline + the cross-lever fixpoint, wired by the
   preprocess driver (lands with `refine`).
