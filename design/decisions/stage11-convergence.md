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

## Three cost axes, each its own single-responsibility `Plan -> Plan` step

The private sweeps three knobs (`g_max`, `sg_max`, `tilt_steps`). They control independent axes of
simulation cost, so — per "decompose by coupled home, not false independence" — each axis is one
single-responsibility step, mirroring the private's `optimize_gmax` / `optimize_sgmax` /
`optimize_tilt_steps`:

- **`converge_g_max`** — the Plan-level shared reciprocal-grid extent (the `ScatteringGrid` sizing).
- **`converge_beams`** — per-orientation beam-set inclusiveness (the `select_beams` active set).
- **`converge_sampling`** — the rocking-curve integration density
  (`rocking_curve_sampling` / the private `tilt_steps`).

Each step has the **same internal shape** as the private sweep: from the current value, repeatedly
increment by a fixed step, re-simulate, compute the consecutive-simulation R-factor, and stop the
first time it drops below threshold; raise if a hard iteration cap (`MAX_SWEEP_ITERATIONS = 100`) is
reached without converging — silent non-convergence is never returned (the same posture as
`iterate_until`).

### Open question — the `sg_max` → 2.0 knob mapping

The private's `sg_max` is a standalone scalar passed to `filter_hkls`. In 2.0 the Klar filter
*derives* its per-reflection `sg_max` from `integration_semiangle` inside `klar_beam_mask`, so there
is no standalone `sg_max` knob to sweep; the beam-set inclusiveness is governed by the
`BeamSelection` cutoffs (`rsg`, `integration_semiangle`). **Which `BeamSelection` field
`converge_beams` increments — and whether 2.0's derived `sg_max` changes the convergence behaviour —
is deferred to the implementation slice** (confirm against a private run rather than guess here).
`NumericsConfig.sg_max = 0.01` exists separately today; its role under the Klar filter must be
pinned before `converge_beams` is written.

## Composition — `pipeline` orders the suite, `iterate_until` is the cross-knob fixpoint

The three steps compose with the existing combinators, no new machinery:

- **Partial order (hard constraint):** the grid must contain any beam the selection might keep, so
  `converge_g_max` runs **before** `converge_beams`; `converge_sampling` is independent of grid
  extent. The private's pass-1 order (`g_max`, `tilt_steps`, `beams`) respects this; 2.0 follows it.
- **`pipeline([converge_g_max, converge_sampling, converge_beams])`** orders one pass.
- **Cross-knob fixpoint:** changing one knob can un-converge another, so the suite must revisit. The
  private hard-codes `num_passes = 2` and *varies the order* between passes (pass 2 leads with
  `tilt_steps`). 2.0 instead expresses the revisit as **`iterate_until(pipeline(...), until=...)`**
  — a fixpoint over the whole pass, driven to stability rather than a fixed count. This is a
  **two-level fixpoint**: each knob converges internally, the composer converges across knobs.

### Decision — `iterate_until`-until-stable (chosen), generalising the private's fixed 2 passes

The private's "two passes, with the order changed on the second" is an empirical detail with no
stated principle; the order-variation in particular looks like a hand-tuned heuristic. 2.0 uses a
single ordered `pipeline` driven by `iterate_until` to a genuine cross-knob fixpoint — the suite
repeats the ordered pass until a whole pass leaves every knob unchanged (or the
`ConvergenceTolerance` cap raises). This **generalises** the private's fixed count and is recorded
as a deliberate generalization in `DIVERGENCE.md` (like the `fit_orientation` iteration cap). It
reuses
the existing combinators with no new machinery, and removes the unprincipled per-pass order-swap.

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
(`r_factor_threshold` and the `max_iterations` / `MAX_SWEEP_ITERATIONS` cap) are a small invariant
bundle that crosses into the algorithm, so by the convention below they become **one frozen
value-type, `ConvergenceTolerance`** (`specs.py`), validated once at construction
(`r_factor_threshold > 0`, `max_iterations >= 1`). This also folds in `iterate_until`'s lone
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
| `converge_*` | `ConvergenceTolerance` | this slice |
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

1. `ConvergenceTolerance` value-type + the simulation-vs-simulation R-factor check
   (`ConvergenceCheck`).
2. `converge_g_max` / `converge_sampling` / `converge_beams` (resolving the `sg_max` mapping first).
3. The coverage `initial_minimum_param_sweep` step (match-count objective).
4. The operation discriminated union + the `both` pipeline, wired by the preprocess driver
   (lands with `refine`).
