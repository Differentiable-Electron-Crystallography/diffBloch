# Decision: three composition shapes for `Plan -> Plan`, and the driver as a `State` runner

**Status:** accepted (stage 11; generalises `stage11-cross-lever-fixpoint.md`).
**Context:** the preprocess pipeline composes `Plan -> Plan` steps. Two combinators already exist
(`pipeline`, `iterate_until`); both thread *all* their loop state through the `Plan` itself. The
cross-lever convergence fixpoint (`stage11-cross-lever-fixpoint.md`) needed a *third* shape -- a
loop over several coupled levers whose coordination state deliberately does **not** live on the
`Plan` -- and that shape was recorded only as a one-off justification for convergence. This ADR
promotes it to a named pattern, states the test that selects between the three, and anchors the new
shape to a textbook functional construct so it reads as a *choice*, not an invention.
**Reference:** `pipeline.py` (`pipeline`, `iterate_until`, `identity`);
`plan-shape-and-step-ordering.md` (what a `Plan` carries; loop state on the `Plan`);
`stage11-cross-lever-fixpoint.md` (the worked convergence instance); `composable-methods.md` (prefer
our own combinators over a framework). `REFERENCES.md` for the State monad, `monad-loops`, and
coordinate descent.

## The three shapes

A `Plan -> Plan` step is a pure transform. There are exactly three ways we compose them, and they
differ on one axis: **where does the loop/coordination state live?**

| Shape | Steps | Feedback | Loop state lives... | Combinator |
|---|---|---|---|---|
| **Sequence** | many, order-dependent | none | on the `Plan` | `pipeline([...])` |
| **Self-fixpoint** | **one** (possibly a composed step) | re-applies itself | on the `Plan` | `iterate_until(step, until=...)` |
| **Driver** | **many, mutually coupled** | a coordinated multi-pass loop (fixed passes or a fixpoint) | **off the `Plan`** (held by the driver) | *hand-rolled* (this ADR) |

The first two are self-threading: the only state a loop needs is the `Plan` it produces, so the
combinator can carry it and return a `Plan -> Plan`. The third cannot, for the two reasons the pool
lever exposed (`stage11-cross-lever-fixpoint.md`): the coordination state (the un-pruned candidate
pool + the two live scalars `integration_semiangle` / `g_max_refine`) is **redundant-with and
transient-to** the `Plan`, so by `plan-shape-and-step-ordering.md`'s no-duplication rule it *must
not* live on the `Plan` -- which means no `Plan -> Plan` combinator can thread it.

## The selection test

> **Can this loop's state live on the `Plan` without duplicating or polluting it?**
> - **Yes, and it is one step** -> `iterate_until`.
> - **Yes, and it is a straight chain** -> `pipeline`.
> - **No (the state is redundant-with / transient-to the `Plan`, or several coupled steps must
>   share it)** -> a **driver** owns the state explicitly and calls the (still pure) levers.

The "no" branch is not a licence to bolt state onto the `Plan`. It is the signal that the loop has
an owner *other than* the value it transforms.

## The driver runs a `State` computation (it is `runState`, not the monad itself)

The shape is not bespoke: it is the standard functional decomposition of a stateful loop -- the
**State monad**, `State s a` (equivalently `s -> (a, s)`). Being precise about the three parts,
since they are easy to conflate:

- the **state type** `s` is a plain value record (here `ConvergenceState`) -- *not* a monad;
- the **monad** is `State ConvergenceState`; each *phase* is a `State ConvergenceState Plan` value
  (a function `ConvergenceState -> (Plan, ConvergenceState)`);
- the **driver** is the *runner* -- it plays `runState` / `evalState`, executing the phases and
  threading `s`. "Driver" is an informal role name (orchestrator / runner), **not** a monad and not
  a kind of monad; there is no "Driver monad".

We use plain `State`, not a transformer stack (`StateT s IO`): the levers are *pure* (expensive but
side-effect-free), so there is no base effect to stack over -- `State s = StateT s Identity`.

| Haskell | Ours |
|---|---|
| the state `s` in `State s a` | `ConvergenceState` = the live scalars (the un-pruned pool is *derived* from `grid` + `g_max_refine`, not stored) |
| the monad `State s` | `State ConvergenceState` |
| a computation `State s a` | a *phase* (`coverage`, `stability`): `State ConvergenceState Plan` |
| the threaded value `a` | the `Plan` |
| a pure step `a -> a` | each lever, a pure `Plan -> Plan` |
| the loop combinator (a bounded fold, or `iterateUntilM` for a fixpoint) | the phase's coordinated multi-pass sweep (fixed `num_passes`, faithful to the private) |
| `runState` / `evalState` | the **driver** (the runner) |

So the levers stay **pure and state-free** (they receive their scalar as a plain argument in their
spec), and the driver is the `runState` runner that (a) holds `s`, (b) reconstructs each lever with
the *other* lever's just-settled scalar, and (c) runs the loop (for convergence: a fixed
`num_passes` coordinate sweep, faithful to the private). This is precisely how Haskell keeps `State`
(the bookkeeping) separate from the loop combinator and composes them -- which is why our levers are
`Plan -> Plan` and the driver is the thing that owns the back-and-forth. Elm says the same less
formally (loop state goes in a `Model` the driver/runner owns, never in the domain record); OCaml
most bluntly (recurse with an explicit state record).

At its *outer boundary* the driver is `evalState`: it runs the phases and **discards** the final
`ConvergenceState`, returning just the converged `Plan`. So the public step
`converge_numerics : Plan -> Plan` is an ordinary `PlanStep` that nests in `pipeline` /
`iterate_until` like any other, even though its *phases* are `State` computations. Downstream fits
never see `s`.

### Abstraction stance: a general pattern, a concrete first instance

This pattern is **generic over the state type** `s` -- a later coupled loop (say a mosaicity <->
orientation coupling) is an instance with its own `FooState` record and its own runner. But we do
**not** ship a generic `State[S, A]` / `eval_state[S]` scaffold now: the first driver is written
concretely against `ConvergenceState`, exactly as we hand-roll `pipeline` rather than import a
combinator library. The state record stays a self-contained frozen value and the runner logic stays
separable from the phase bodies, so *if* a second stateful driver appears and genuinely shares
runner code (rule of three), lifting a generic `eval_state[S]` is a mechanical extraction -- done
then, when its true shape is known, not guessed for a single caller.

## Why hand-roll rather than import `returns.State`

Python *has* the named construct -- `returns` (dry-python) ships a typed `State` / `StateT`. We
hand-roll for the same reason we hand-roll `pipeline` and `iterate_until` (`composable-methods.md`):
a three-line loop typed to *our* domain (`Plan`, `ConvergenceState`) beats importing a monad
framework
and its `bind`/`do`-notation vocabulary, and avoids a dependency for a single call site. The value
of anchoring to `State` is *conceptual* -- it tells us the shape is correct and names its parts --
not a reason to take the import. Revisiting `returns` is a recorded *possible* roadmap task (if the
driver / `Result` story ever grows enough to earn it), not a commitment.

## Where `refine` fits (same family, different animal)

`refine` is *also* an iterative loop whose state lives off the threaded value (optimizer momentum,
iteration, loss history), so by the selection test it too is a "driver", and it is the terminal
exception to "can it be a `Plan -> Plan`?". But it should **not** share the convergence driver's
implementation, because it sits on the far side of two boundaries:

- **`Plan -> Plan` vs `Plan -> Result`.** The convergence driver stays *inside* the transformer
  world -- `evalState` hands back a `Plan` and it nests as one more step. `refine` *leaves* it: it
  consumes the frozen `Plan` and threads *differentiable parameters* (structure factors, ADPs,
  thickness) to a `Result`. It is the pipeline's exit, not a step in it.
- **Pure `State` vs effectful loop.** The convergence driver is a *pure* `State ConvergenceState
  Plan` runner coordinating several pure, swappable `Plan -> Plan` levers. `refine` has no such
  structure -- one loss, one autograd + torch-optimizer update rule, device + RNG. Its state is
  genuinely effectful (`StateT ... IO` -- an ordinary training loop). Dressing it as a pure `State`
  runner for uniformity would be the god-object anti-pattern this doc already warns against.

They share the abstract *pattern* ("iterate off-value state until a stopping rule") but not a
shape: different threaded value, effectful state. So `refine` is *not* the second same-shape caller
that would justify a generic runner (see the abstraction stance above); it stays the settled
effectful terminal (`stage10-refinement-loop.md`). A `StateT`-over-effects layer only earns its
place if several cross-cutting *effectful* concerns (device / RNG / telemetry / checkpoint) ever
thread the
whole pipeline -- the `returns` roadmap note, not now.

## Consequences

- **No `pipeline([window, pool])` fixpoint exists or should be added.** The coupled fixpoint is the
  driver's job (`stage11-cross-lever-fixpoint.md`); `pipeline`/`iterate_until` remain the tools for
  state-on-the-`Plan` loops only.
- **New coupled loops reuse this pattern, not re-derive it.** Any future "several levers coupled
  through state we don't want on the `Plan`" (e.g. a later mosaicity <-> orientation coupling)
  is a driver with its own `FooState` record and runner, not a new combinator and not `Plan` fields.
- **The levers stay independently testable.** Because each lever is a pure `Plan -> Plan` taking its
  scalar in its spec, it is unit-tested standalone; the driver is tested for the *coordination*
  (does a pass thread each settled scalar into the next lever correctly?), not re-testing each
  lever.

## Status

**Accepted.** Realized in the stage-11 preprocess driver (the same slice that owns the operation
discriminated union and default-pipeline assembly). `stage11-cross-lever-fixpoint.md` remains the
worked convergence instance; this ADR is the general pattern it is an instance of.
