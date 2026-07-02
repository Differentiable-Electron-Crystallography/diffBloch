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

**Private structure (accurate names).** This work lives **only on the branch**
`pattern-vis-convergence-testing` (`14383cf`), not on the private's `main` —
`programs/convergence_testing.py` is absent from the checked-out private submodule HEAD. On that
branch the module exposes three
operations dispatched by `convergence_testing(cfg)`:

- `_run_initial_minimum_param_sweep` — coverage / match-count objective;
- `_run_hyperparams_optimization` — self-stability via `_compute_step_rfactor`, sweeping `g_max` ->
  `tilt_steps` -> `sg_max` **inline** (there are no per-knob `optimize_*` functions);
- `both` — the sweep then the optimization.

The 2.0 analog of `_run_hyperparams_optimization`'s inline sweeps is the `converge_*` step family
(`converge_beams` / `converge_pool` / `converge_sampling`); `_run_initial_minimum_param_sweep` is
the coverage step; `both` is the operation discriminated union. *(The inline sweeps are the nested
closures `optimize_gmax` / `optimize_sgmax` / `optimize_tilt_steps` inside
`_run_hyperparams_optimization`. An earlier note in this doc claimed those names were fabricated —
that was wrong: they exist as nested functions, which a top-level `def` grep missed. Commits
`10fd0f9` / `20ca73b` citing `optimize_gmax` / `optimize_tilt_steps` were therefore correct.)*

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

The private sweeps three knobs (`g_max`, `sg_max`, `tilt_steps`) as three nested closures
(`optimize_gmax` / `optimize_sgmax` / `optimize_tilt_steps`) inside `_run_hyperparams_optimization`.
Porting them showed that framing is a *false independence*: in 2.0
they do not correspond to
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
  beam-inclusiveness concern reached by the driver's coupled multi-pass sweep, rather than as two
  "independent" axes. It is the same principle as `select_beams` owning the whole active-set
  selection.

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
calibrated; when it lands, the two levers are driven together by the driver's fixed multi-pass
coordinate sweep (below).

### Stopping rule — a faithful port of the private's first-below-threshold stop

The private stops the first time the consecutive-simulation R-factor drops below
`r_factor_threshold`
(`_run_hyperparams_optimization`'s nested `optimize_gmax` / `optimize_sgmax` / `optimize_tilt_steps`
each do `if r_value < r_threshold: break`), bounded by a `MAX_SWEEP_ITERATIONS` cap. 2.0 ports that
rule **exactly**: `converge_scalar` grows the knob, compares consecutive builds, and returns the
first candidate whose R-factor is below threshold; `max_iterations` is the hard cap that raises
rather than returning a silently non-converged plan (the `iterate_until` posture).

> **History (superseded over-correction).** An earlier 2.0 revision added *skip-null* + *patience*
> to
> `converge_scalar`, on the theory that the private's first-dip stop was a "plateau bug": because
> `integration_semiangle` is continuous but the beam set is discrete, two increments can yield the
> same beam set, R = 0, and a false "converged". That was reverted as an unwarranted divergence (the
> same over-correction class as the Klar-geometry episode): it invented a stopping rule the
> reference never had and complicated the driver. The discrete-plateau *sensitivity* is real but is
> handled by choosing a `step` coarse enough to move the beam set (as the adapters' tuned steps do),
> not by second-guessing the reference stop. `ConvergenceTolerance` therefore carries only
> `r_factor_threshold` + `max_iterations`; there is no `patience` field.

Composition is the private's **fixed multi-pass coordinate sweep**: each lever is swept to its own
first-below-threshold stop and the ordered pass is run a fixed `num_passes` (default 2), with the
private's per-pass order-swap. **Note (superseded mechanism):**
the paragraphs below originally expressed this as `iterate_until(pipeline([...]))`. Landing
the pool lever showed that naive composition does not work (seed/pruned mismatch + shared scalar
state the `Plan` does not carry), so the coupled sweep is instead assembled by the
**preprocess driver** — see `design/decisions/stage11-cross-lever-fixpoint.md`. The sweep *model*
here is unchanged; only *where it is assembled* moved. The stopping-rule parameters
(`r_factor_threshold`, `max_iterations`) are the invariant bundle carried by
**`ConvergenceTolerance`**; `max_iterations`'s default is the private's `MAX_SWEEP_ITERATIONS`.

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
- **several params independently:** separate `converge_scalar` steps run as the driver's coupled
  multi-pass sweep. This was originally sketched as `iterate_until(pipeline([...]))`; that naive
  composition is superseded — the sweep is assembled by the preprocess driver (see
  `design/decisions/stage11-cross-lever-fixpoint.md`).

**Guardrail (typed closures, not config reflection).** The genericity comes from *higher-order
functions over typed closures*, exactly the `Plan -> Plan` / combinator idiom in the codebase
— **not** a stringly-typed engine that introspects a config by key path and mutates it. Each `build`
is a small, explicit, type-checked function the caller writes; the convergence engine stays a pure
combinator with no knowledge of the config schema, keeping the value-object vocabulary and the
"no `DictConfig` in the core" posture intact.

## Composition — the driver coordinates the coupled levers (faithful fixed passes)

Today only `converge_beams` (the window lever) and `converge_pool` (the pool lever) exist, each
self-contained. The composition below — the private's fixed multi-pass coordinate sweep — activates
once they are driven together by the preprocess driver:

> **Implementation note (slice 3, pool lever landed).** The coupled sweep is **not** a naive
> `iterate_until(pipeline([converge_beams, converge_pool]))` — `converge_beams` re-selects from an
> *unpruned* seed while `converge_pool` emits a *window-pruned* Plan, and the two levers share
> scalar state (`integration_semiangle`, `g_max_refine`) the `Plan` does not carry. The coupled
> sweep therefore lives in the **preprocess driver**, which holds that state explicitly. See
> `design/decisions/stage11-cross-lever-fixpoint.md` for the full reasoning.

- **Partial order (sizing dependency):** the grid must contain any beam a wider pool keeps, so
  growing `g_max_refine` implies a grid-`g_max` *sizing* step **before** re-selection.
  `converge_sampling` is independent of the beam levers.
- **One pass** orders the levers (the private: `g_max` → `tilt_steps` → `sg_max`).
- **Coupled revisit:** because the pool and window levers are coupled (each bounded by the other),
  widening one can leave the other room to grow, so the suite revisits. The private hard-codes
  `num_passes = 2` and *varies order* between passes (pass 1 leads with `g_max`, pass 2 with
  `tilt_steps`). 2.0 **ports that faithfully**: a fixed `num_passes` (default 2) coordinated sweep,
  same per-pass order, each lever converging internally to its first-stop and the driver threading
  each settled scalar into the next lever.

### Decision — faithful fixed `num_passes`, assembled by the driver

2.0 runs the private's scheme unchanged: `num_passes` fixed passes (default 2) over the ordered
levers, with the per-pass order-swap, each lever driven to its own first-stop. An earlier revision
proposed *generalising* the fixed count to a repeat-until-stable cross-knob fixpoint (and dropping
the order-swap as unprincipled); that was reverted under the faithful-port directive — it invented a
stopping rule the reference never had, the same over-correction class as the patience/skip-null
episode. There is therefore **no divergence** here; 2.0 matches the private pass-for-pass.

The *mechanism* is still the **preprocess driver**, not a `pipeline` composition: the pool lever
proved that `iterate_until(pipeline([window, pool]))` cannot compose (seed/pruned mismatch + shared
scalar state the `Plan` does not carry), so the driver holds the unpruned candidate pool and the two
live scalars as explicit state and threads each lever's settled value into the next. The driver is
needed for the fixed passes just as much as it would be for a fixpoint — the coupling, not the
stopping rule, is what forces it. See `design/decisions/stage11-cross-lever-fixpoint.md`.

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

## Why convergence matters: fewer beams = a faster experiment (visualization, for later)

The point of converging is **not** accuracy for its own sake — it is to find the *minimal* beam set
that still reproduces the pattern. What fundamentally sets the cost of the whole experiment is
**which beams are included and how many**: the Bloch structure matrix is dense `N x N` in the active
beam count `N`, solved (`matrix_exp` / `eigh`) *per orientation, per rocking-curve tilt, per
thickness candidate*, so `N` dominates both simulation and inference time. Converging the beam
levers down to the minimal sufficient set is therefore a **speed** lever: fewer beams -> a faster
sim and a faster fit, with no loss of agreement. (This is the flip side of the coverage sweep, which
finds the *smallest* parameters that still recover the matched reflections.)

**Visualization (future work).** A colormap makes the tradeoff legible over the two beam levers:

- **x-axis:** `sg_max` (the excitation-error window; in 2.0, `integration_semiangle`).
- **y-axis:** `g_max` (the candidate pool radius; in 2.0, `g_max_refine`).
- **colour:** the number of beams included at that `(x, y)` — the size of the active set
  `seed(g_max_refine) ∩ Klar-window(integration_semiangle)`.

Overlaid (or a second panel) on the same axes: **which of those beams are useful** — e.g. how many
of the included beams actually match an observed reflection (the `plan_coverage` count) and/or carry
non-negligible dynamical intensity, versus dead weight that only slows the solve. That turns the
convergence path into a picture: you can see the region where adding beams stops buying either
coverage or pattern-change, and pick the `(sg_max, g_max)` corner that is cheapest for the accuracy
it delivers. Natural home: a convergence tutorial notebook (`../notebooks/iain/tutorials/`), driven
by the `converge_*` / `plan_coverage` public API once the driver lands.

**The raw beam count is boring — the useful count is the point.** Total beams included is *always*
maximal in the **top-right quadrant** (both knobs wide open admit the most beams): it is a trivially
monotone ramp toward `(max sg_max, max g_max)`, so a colormap of it alone says nothing. The signal
is the *useful*-beam overlay: the count of beams that match an observed reflection / carry real
intensity **saturates** somewhere well short of the top-right, and past that contour every extra
beam is pure cost. So the target is not the top-right corner but the **cheapest point on the
useful-saturation contour** — the lower-left-most `(sg_max, g_max)` that still captures every useful
beam. The viz earns its keep precisely by showing that gap between the (monotone) total-count ramp
and the (saturating) useful-count contour.

## Sequencing

1. `ConvergenceTolerance` value-type + the sim-vs-sim R-factor check (`simulation_converged`).
   **Done** (`6f3d694`, `1b5681e`; faithful strip of `patience`/skip-null landed later, `244c010`).
2. `converge_scalar` HOF + `converge_beams` — the `integration_semiangle` window sweep
   (first-below-threshold stop + cap; faithful to the private).
3. The coupled pool lever (`g_max_refine` + dependent grid sizing). **Done** (`10fd0f9`,
   `converge_pool`). The coupled multi-pass sweep is re-homed to the driver — see
   `design/decisions/stage11-cross-lever-fixpoint.md`.
4. `converge_sampling` — waits on rocking-curve integration in the forward model (`ROADMAP.md`).
   **Done** (`20ca73b`).
5. The coverage `initial_minimum_param_sweep` step (match-count objective). **Done** (`16410c5`,
   `plan_coverage` / `maximize_scalar` / `cover_beams` / `cover_pool`).
6. The operation discriminated union + the `both` handoff + the coupled multi-pass sweep, wired by
   the preprocess driver (lands with `refine`).
