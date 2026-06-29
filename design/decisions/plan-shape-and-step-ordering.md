# Plan shape & step ordering — self-describing values, ordering by construction, not by a step log

## Context

Stage 11 builds the `preprocess` pipeline as `Plan → Plan` transforms (`from_experiment`,
`select_beams`, `fit_orientation`, `fit_thickness`, `converge_numerics`). Two design questions came
up while wiring orientations into the `Plan`:

1. The rebuild inputs a step needs (`orientation`, `energy`, `u0` on `OrientationPlan`; the real
   `cell` on `ScatteringGrid`) aren't used by every consumer. Should they be **optional** — present
   before the step that needs them, absent after?
2. Steps depend on an ordering (`select_beams` needs physical orientations; `fit_orientation` runs
   after `select_beams`). **Where does that ordering live** — and if X depends on Y, should X look
   in something like `plan.completed_steps` for `"Y"`?

We checked how functional languages/frameworks answer both, because both have well-trodden
anti-patterns.

## Decision 1 — Plans are fully-populated, self-describing values (no optional/shed fields)

The rebuild inputs are **always present**, never optional and never shed. They are the plan's
*identity*, not transient scratch: `orientation` is the source of truth for the basis and the thing
`fit_orientation` adjusts; `energy`/`u0` are needed to rebuild *any* `BeamPlan`; `cell` is the
grid's geometric identity. "A Plan whose `orientation` is `None`" is a literal illegal state.

The cross-language verdict on optional-and-shed:

| Reference | Says |
|---|---|
| **Elm — make impossible states impossible** | `{ isLoading : Bool, data : Maybe Data }` is the textbook anti-pattern: optional fields encoding a phase **invent illegal states**. Their phantom process-flow page names "intermediate types with `Maybe` fields" as the thing to avoid. |
| **Haskell — Alexis King, *Parse, don't validate*** | "Don't stick a field in a record because the function you're writing needs it." "Avoid denormalized representations — duplicating data introduces a trivially representable illegal state: the copies getting out of sync. Strive for a single source of truth." |
| **Elm/OCaml typestate** | The rigorous way for fields to legitimately come-and-go is **phantom phase types** — but Python can't enforce that without `Generic[Phase]` + casts that mypy checks weakly and researchers find alien. Wrong cost/benefit. |
| **Phoenix `Plug.Conn`** | One immutable value threaded through the pipeline, *accumulating*; never bare `nil` for not-yet-populated data (explicit `Unfetched` sentinels). Doesn't shed fields, doesn't switch type per phase. |

**Role separation, not optionality.** The honest concern ("the `engine.simulate` hot path shouldn't
carry preprocess rebuild inputs") is solved by grouping by role — both groups always present:

- **compiled geometry** (`beam_plan`, `alignment`) → consumed by `engine.simulate`
- **source / rebuild inputs** (`orientation`, `energy`, `u0`; `cell` on the grid) → consumed by the
  preprocess `Plan → Plan` steps to recompile

**Sync boundary.** Storing `orientation` *and* the derived `beam_plan` (or `cell` *and*
`reciprocal_basis`) is denormalized and can desync — King's warning. The fix is the `Plan → Plan`
rebuild itself: it is the **single trusted constructor** that always sets source + compiled
together (frozen dataclasses; only `build`/rebuild ever construct them). That is the principled
justification for the rebuild step.

### The reshape (concrete)

- `ScatteringGrid` gains `cell` (real-space basis; companion of the `reciprocal_basis` it already
  derives), always present.
- `OrientationPlan` gains `orientation` (3×3), `energy`, `u0` — all required, set by `build`.

## Decision 2 — Ordering lives in the recipe + total construction, never in a `completed_steps` log

Making fields non-optional removes the type-level "too early to run" guard (no `None` to trip on).
The principled replacement is **not** to claw back optionality, nor to add a step registry — it is an
invariant that makes the guard unnecessary:

> **Every `Plan` is always complete and simulatable.** `from_experiment` produces a complete one;
> every `Plan → Plan` returns a complete one.

This is the ordering counterpart of non-optional fields: there is no half-built `Plan` for a step to
choke on, so most ordering stops being a *validity* question and becomes a *convergence/quality*
question. Our dependencies then split cleanly:

- **Hard dependencies satisfied by construction.** `select_beams` needs physical orientations →
  `from_experiment` always seeds them → satisfied before any `Plan` exists. No runtime check.
- **Recipe ordering (valid if reordered, just worse-converged).** `converge_numerics` tunes cutoffs
  `select_beams` applies; `fit_orientation` after `select_beams`. Lives in **one canonical recipe**
  (`pipeline([...])`), reviewed once.
- **Cyclic refinement.** `converge_numerics ⇄ select_beams` iterating to a fixpoint is already
  modelled by `iterate_until`.

### Does X look in `plan.completed_steps` for `"Y"`? — No.

The functional references are unanimous that dependencies are expressed **on data**, not on a
parallel "this step ran" log:

| Reference | How dependency is expressed |
|---|---|
| **Shake / Make / Nix** | Dependency = **demand on a value/artifact** (`need`). "Did Y run" is answered by "is Y's output present/up-to-date" — **the data is the proof**. Ordering is derived from the dependency graph, never a ran-registry. |
| **Ecto.Multi** (most registry-like) | Stores the **map of prior *results* keyed by name** (`changes_so_far`), read by dependents — **not a boolean ran-flag**. Order is append order (the recipe). |
| **Haskell / Elm** | X depends on the **type/value Y produces**. A separate "completed" boolean is King's denormalized state → the illegal state where flag and data disagree; and it is stringly-typed. |

For us it is cleaner than Ecto: every `Plan → Plan` returns a **complete `Plan`**, so the result is
already the threaded value — X reads the field it needs (always meaningfully present by the
invariant). No named-results map, no `completed_steps`. If "was Y applied?" is ever genuinely needed,
inspect Y's *output* (e.g. "are the beams already filtered?"), not a flag.

A history-of-applied-steps may exist only as **write-only provenance** for observability /
checkpointing — consistent with `effects-and-observability.md` (write-only reporter, never
re-imported as truth) — and must never gate control flow.

**No phase-tag / typestate machinery now.** Add a loud runtime precondition (Plug `Unfetched`-style)
*only if* a genuinely-illegal reorder ever appears. Don't build a phase registry speculatively.

## Status

**Accepted.** Realized incrementally in stage 11: the reshape (`ScatteringGrid.cell`,
`OrientationPlan.orientation/energy/u0`) lands with the `from_experiment` work; the rebuild
`Plan → Plan` (shared by `select_beams` / `fit_orientation`) is the single sync boundary; ordering
lives in the `from_experiment` / `pipeline` recipe.

**Invariants to hold:**
- No optional field encodes a phase; Plans are fully-populated and self-describing.
- Source and compiled geometry are only ever set together, by the rebuild constructor.
- Every `Plan` is complete and simulatable; ordering lives in the recipe; dependencies are read off
  the data, never a `completed_steps` registry.
