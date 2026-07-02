# Decision: three composition shapes for `Plan -> Plan`, and the driver as a hand-rolled `State`

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

## The driver is a hand-rolled `State` / `StateT`

The driver shape is not bespoke: it is the standard functional decomposition of a stateful loop --
the **State monad**. In Haskell terms the convergence driver is

```
StateT DriverState Identity Plan
```

with the pieces mapping exactly onto ours:

| Haskell | Ours |
|---|---|
| the state `s` in `State s a` | `DriverState` = un-pruned pool + the two live scalars |
| the threaded value `a` | the `Plan` |
| a pure step `a -> a` | each lever, a pure `Plan -> Plan` |
| the loop combinator (a bounded fold, or `iterateUntilM` from `monad-loops` for a fixpoint) | the driver's coordinated multi-pass sweep (fixed `num_passes`, or repeat-until-stable) |
| `runStateT` | the driver function itself |

So the levers stay **pure and state-free** (they receive their scalar as a plain argument in their
spec), and the driver is the `runState` harness that (a) holds `s`, (b) reconstructs each lever with
the *other* lever's just-settled scalar, and (c) runs the loop (for convergence: a fixed
`num_passes` coordinate sweep, faithful to the private). This is precisely how Haskell
keeps `State` (the bookkeeping) separate from `fix` / `iterateUntilM` (the loop) and composes them
-- which is why our levers are `Plan -> Plan` and the driver is the thing that owns the back-and-
forth. Elm says the same less formally (loop state goes in a `Model` the driver owns, never in the
domain record); OCaml most bluntly (recurse with an explicit state record).

## Why hand-roll rather than import `returns.State`

Python *has* the named construct -- `returns` (dry-python) ships a typed `State` / `StateT`. We
hand-roll for the same reason we hand-roll `pipeline` and `iterate_until` (`composable-methods.md`):
a three-line loop typed to *our* domain (`Plan`, `DriverState`) beats importing a monad framework
and its `bind`/`do`-notation vocabulary, and avoids a dependency for a single call site. The value
of anchoring to `State` is *conceptual* -- it tells us the shape is correct and names its parts --
not a reason to take the import. Revisiting `returns` is a recorded *possible* roadmap task (if the
driver / `Result` story ever grows enough to earn it), not a commitment.

## Consequences

- **No `pipeline([window, pool])` fixpoint exists or should be added.** The coupled fixpoint is the
  driver's job (`stage11-cross-lever-fixpoint.md`); `pipeline`/`iterate_until` remain the tools for
  state-on-the-`Plan` loops only.
- **New coupled loops reuse this pattern, not re-derive it.** Any future "several levers coupled
  through state we don't want on the `Plan`" (e.g. a later mosaicity <-> orientation coupling)
  is a driver with its own `DriverState`, not a new combinator and not `Plan` fields.
- **The levers stay independently testable.** Because each lever is a pure `Plan -> Plan` taking its
  scalar in its spec, it is unit-tested standalone; the driver is tested for the *coordination*
  (does a pass thread each settled scalar into the next lever correctly?), not re-testing each
  lever.

## Status

**Accepted.** Realized in the stage-11 preprocess driver (the same slice that owns the operation
discriminated union and default-pipeline assembly). `stage11-cross-lever-fixpoint.md` remains the
worked convergence instance; this ADR is the general pattern it is an instance of.
