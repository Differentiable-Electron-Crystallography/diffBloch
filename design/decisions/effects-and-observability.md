# Effects & observability — how diffBloch keeps a pure core while still logging, plotting, saving

## Context

The differentiable core (`core/`, `preprocess/`, the pure parts of `engine/`) is meant to stay a
pure, testable, frozen-dataclass world: values in, values out, no I/O. But a real refinement run
still has to emit diagnostics, surface metrics, draw plots, and persist results. The question is how
to do that *without* threading file handles, loggers, or plotting libraries into the maths.

The answer, borrowed from a consistent lineage — OCaml's `Logs` (reporter installed at the app
boundary; "libraries must not install reporters"), OCaml 5 effect handlers / Elm `Cmd` /
redux-saga's `Effect` (the core *describes* effects as data, an interpreter at the boundary
*performs* them), and Phoenix `:telemetry` (emit named events; reporters attached in the supervision
tree) — is one rule:

> **The pure core only names, returns, or emits values. The imperative shell (`app/` + logging)
> interprets them. The core configures no sinks, performs no I/O, and runs correctly with zero
> sinks attached.**

We take the *idea*, not the machinery: no saga middleware, no effect-handler runtime — just returned
values, emitted events, and stdlib `logging`. (Simplest thing that holds the invariant.)

## Three channels (don't conflate them)

| Channel | Carries | Mechanism in diffBloch | Lives in |
|---|---|---|---|
| **Diagnostics** | debug/info/warn — "what is the solver doing" | stdlib `logging`: named loggers in the core; **library code only ever `getLogger(__name__)` + a `NullHandler`, never `basicConfig`/handlers** | core emits · `app/` installs handlers |
| **Domain observations** | refinement metrics, per-rotation `wR2`, step trajectory | core **returns / yields typed events** (effects-as-data); reporters (CSV, visualize, in-memory history) consume them at the edge | core emits · `app/` attaches reporters |
| **Persistence (truth)** | the resumable result | **serialize the whole `Plan` value** (+ a hash of its inputs), one boundary; never a per-facet CSV | `app/` / IO boundary |

## How each works, concretely

**Diagnostics.** `core` / `preprocess` / `engine` modules do `log = logging.getLogger(__name__)` and
`log.debug(...)`. They add a `NullHandler` and configure nothing else — exactly OCaml `Logs`'
"libraries don't install reporters." `app/` decides verbosity and where logs go. A "CSV of debug
logs" is just a `logging.Handler`.

**Domain observations (the effects-as-data part).** `refine()` does not call a logger or write a
file mid-loop. It *emits* named events as plain values — e.g. a `RefinementStep(iteration, loss,
r_factor, …)` per step — either by returning a history on the `Result` or yielding a stream. A thin
shell feeds those events to whatever **reporters** are attached. Your `CsvLogger` and
`VisualizeLogger` are both just reporters over the same event stream; adding a plotter is zero
changes to the core. (Phoenix's extra lesson, if/when we want it: put a declarative *metric
definition* — "track `last_value` of loss, `distribution` of per-rotation `wR2`" — between the
emitted event and the reporter, so the emitter never decides "this is a counter" and the CSV sink
never hardcodes which events it reads.)

**Persistence.** `fit_orientation` (and every other `Plan -> Plan` step) returns a `Plan` value — it
never writes `optim_orientation.csv`. If a run needs to be resumable or frozen for experimental
control, the shell *checkpoints the whole `Plan`* (orientations + thickness + beams + numerics are
co-fit and only meaningful together), stamped with an input hash — the ML "general checkpoint" idiom,
not a lossy per-facet dump. A human/interop CSV of orientations, if wanted, is a **write-only
reporter** (eyeballs / PETS2–Jana), never re-imported as truth.

## The one invariant to protect

Stated three ways by three sources, all the same: the differentiable core **must run with no sinks
attached, must install none, and must be mutated by none.** `logging` already gives us this for
diagnostics (loggers = sources, handlers = sinks, `NullHandler` = "no reporter installed"); for
domain metrics we return events as values and interpret them in the shell. The smell to avoid is
routing scientific results through the diagnostic logger, or threading a logger/file handle into the
maths.

## Status

Forward-looking. Realized in stage 12 (`logging` + `app/`). Recorded now because it settles two live
questions — *don't* write orientations to a CSV as state (return a `Plan`; checkpoint the `Plan`),
and *do* model CSV/visualization as interchangeable reporters at the boundary.
